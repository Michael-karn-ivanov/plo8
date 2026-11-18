"""The public library targets `//cli_server:protocol` and `//cli_server:e2e`
on their own (target `libraries_test`, built with those two targets only).

Other projects (e.g. `plo8/rl_env`) replay the recorded hands through their own
tables with `:e2e` and round-trip views through `:protocol`; this test makes the
same use of them, so a change to a case file or to a library is caught here
first.
"""

import json
import os
import random
import subprocess
import sys
import unittest

from engine import ActionType, Environment

from cli_server.e2e.case import Case, CaseError
from cli_server.e2e.cases import E2E_DIR, available_cases, load_cases
from cli_server.e2e.deck import StackedDeck, deal_script, engine_seats, join_offset, seating

SERVER_MODULES = (
    "cli_server.server",
    "cli_server.session",
    "cli_server.main",
)
RUNNER_MODULES = (
    "cli_server.e2e.driver",
    "cli_server.e2e.client",
    "cli_server.e2e.check",
    "cli_server.e2e.expect",
)
E2E_LIBRARY = ("cli_server.e2e.case", "cli_server.e2e.cases", "cli_server.e2e.deck")


def modules_after_import(*modules):
    """`sys.modules` of a fresh child interpreter after importing `modules`."""
    code = (
        "import json, sys\n"
        f"for m in {list(modules)!r}:\n"
        "    __import__(m)\n"
        "print(json.dumps(sorted(sys.modules)))\n"
    )
    env = dict(os.environ)
    # The child finds the same modules (inside the buck2 test too).
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=60,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"importing {modules} failed:\n{proc.stderr.decode('utf-8', 'replace')}"
        )
    return set(json.loads(proc.stdout.decode("utf-8").strip().splitlines()[-1]))


class ImportIsolationTest(unittest.TestCase):
    def test_e2e_library_loads_no_server_runner_engine_or_subprocess(self):
        loaded = modules_after_import(*E2E_LIBRARY)
        self.assertTrue(set(E2E_LIBRARY) <= loaded)
        forbidden = SERVER_MODULES + RUNNER_MODULES + ("engine",)
        self.assertEqual(sorted(loaded & set(forbidden)), [])
        # `subprocess`: the library adds nothing to what a bare interpreter
        # already loads (the buck2 test interpreter's startup may load it).
        if "subprocess" not in modules_after_import():
            self.assertNotIn("subprocess", loaded)

    def test_protocol_loads_no_server_or_e2e_module(self):
        loaded = modules_after_import("cli_server.protocol")
        self.assertIn("cli_server.protocol", loaded)
        forbidden = SERVER_MODULES + RUNNER_MODULES + E2E_LIBRARY + ("cli_server.e2e",)
        self.assertEqual(sorted(loaded & set(forbidden)), [])
        # `subprocess`: the protocol adds nothing to what its dependency `engine`
        # already loads (the standard library's multiprocessing may load it).
        if "subprocess" not in modules_after_import("engine", "entities"):
            self.assertNotIn("subprocess", loaded)


class CaseFilesTest(unittest.TestCase):
    def test_case_files_are_resources_of_the_library(self):
        self.assertTrue(E2E_DIR.is_dir(), E2E_DIR)
        names = available_cases()
        self.assertTrue(names)
        cases = load_cases()
        self.assertEqual([c.name for c in cases], names)
        self.assertTrue(all(isinstance(c, Case) for c in cases))


def replay(case: Case, seed: int = 0):
    """Replay `case` through an engine table without a server, as other projects
    do: chips_per_bb = case bb, players of `seating(case)` (button on engine seat
    0), a stacked deck, the case's actions in order (step mode). Returns the
    engine's HandResult."""
    bb = case.game.bb
    env = Environment(
        chips_per_bb=bb,
        seed=seed,
        deck=StackedDeck(deal_script(case), random.Random(seed)),
    )
    # Seated through joins (exact for whole chips at any chip unit).
    for k, p in enumerate(seating(case)):
        env.join(p.stack / bb, seat=join_offset(k), name=p.id)
    decision = env.start_hand()
    for street, a in case.actions:
        if a.is_blind:
            continue
        if decision is None:
            raise AssertionError(f"the hand ended before {a.player} {a.action} on the {street}")
        if decision.name != a.player:
            raise AssertionError(
                f"{street}: the engine asks {decision.name}, the case has {a.player} {a.action}"
            )
        decision = env.step(ActionType(a.engine_action))
    if decision is not None:
        raise AssertionError(f"the case's actions are over but the engine asks {decision.name}")
    return env, env.last_result


class ServerlessReplayTest(unittest.TestCase):
    def test_seating(self):
        for case in load_cases():
            with self.subTest(case=case.name):
                order = seating(case)
                self.assertEqual(order[0].id, case.button)
                self.assertEqual(engine_seats(case), {p.id: k for k, p in enumerate(order)})

    def test_every_case_replays_without_a_server(self):
        for case in load_cases():
            with self.subTest(case=case.name):
                self.check_replay(case)

    def check_replay(self, case: Case) -> None:
        bb = case.game.bb
        seats = engine_seats(case)

        def chips(x):
            return int(round(x * bb))

        env, result = replay(case)
        self.assertEqual([p.name for p in env.players], [p.id for p in seating(case)])
        self.assertIsNotNone(result)
        self.assertEqual(result.button, 0)
        names = dict(result.names)
        self.assertEqual(names, {s: pid for pid, s in seats.items()})
        self.assertEqual(result.showdown, case.result.ended_by == "showdown")

        final = {names[s]: chips(x) for s, x in result.stacks.items()}
        net = {names[s]: chips(x) for s, x in result.net.items()}
        self.assertEqual(final, dict(case.result.final_stacks))
        self.assertEqual(net, dict(case.result.net))

        self.assertEqual(len(result.pots), len(case.pots))
        for k, (want, got) in enumerate(zip(case.pots, result.pots)):
            high = {names[s]: chips(a) for s, a in got.high if chips(a)}
            low = {names[s]: chips(a) for s, a in got.low if chips(a)}
            per_winner = {}
            for pid, a in list(high.items()) + list(low.items()):
                per_winner[pid] = per_winner.get(pid, 0) + a
            self.assertEqual(
                (
                    chips(got.amount),
                    sorted(names[s] for s in got.eligible),
                    sorted(high),
                    sum(high.values()),
                    sorted(low),
                    sum(low.values()),
                    per_winner,
                ),
                (
                    want.amount,
                    sorted(want.eligible),
                    sorted(want.hi.winners),
                    want.hi.amount,
                    sorted(want.lo.winners) if want.lo else [],
                    want.lo.amount if want.lo else 0,
                    {pid: a for pid, a in want.awards.items() if a},
                ),
                f"pot {k} (amount, eligible, high winners, high, low winners, low, per winner)",
            )

    def test_a_wrong_recording_is_caught(self):
        """The replay compares for real: a case whose result is changed fails."""
        case = next(c for c in load_cases() if c.result.ended_by == "showdown")
        env, result = replay(case)
        bb = case.game.bb
        final = {result.names[s]: int(round(x * bb)) for s, x in result.stacks.items()}
        wrong = dict(case.result.final_stacks)
        a, b = sorted(wrong)[:2]
        wrong[a], wrong[b] = wrong[a] + 1, wrong[b] - 1
        self.assertNotEqual(final, wrong)

    def test_case_error_is_exported(self):
        self.assertTrue(issubclass(CaseError, ValueError))


if __name__ == "__main__":
    unittest.main()
