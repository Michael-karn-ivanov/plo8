"""Recorded hands: every recorded end-to-end hand of cli_server
(`cli_server/e2e_test_sets`) replayed through an rl_env Table, with the
recorded cards and actions mapped to masked actions, gives the recorded final
stacks and pot splits (in chips)."""

import random
import unittest

import numpy as np

from engine import ActionType
from rl_env import BET_RAISE, CHECK_CALL, FOLD, Table, encode_view, to_engine

from cli_server.e2e.cases import load_cases
from cli_server.e2e.deck import StackedDeck, deal_script, engine_seats, seating

N_CASES = 26

_MASKED = {
    ActionType.FOLD: FOLD,
    ActionType.CHECK: CHECK_CALL,
    ActionType.CALL: CHECK_CALL,
    ActionType.RAISE: BET_RAISE,
}


def replay(case):
    """Replay `case` through a Table: chips_per_bb = the case's big blind, case
    seating with the button on seat 0, a stacked deck from the case. Returns the
    table and how many actions needed `act_engine`."""
    bb = case.game.bb
    order = seating(case)
    seats = engine_seats(case)
    table = Table(chips_per_bb=bb)
    deck = StackedDeck(deal_script(case), random.Random(0))
    table.start_hand(len(order), [p.stack / bb for p in order], 0, 0, deck=deck)
    raw = 0
    for street, a in case.actions:
        if a.is_blind:
            continue
        if not table.decision_pending:
            raise AssertionError(f"the hand ended before {a.player} {a.action} on the {street}")
        if table.seat_to_act != seats[a.player]:
            raise AssertionError(
                f"{street}: the table asks seat {table.seat_to_act}, the case has {a.player}"
            )
        # Every decision of a recorded hand is a valid observation on both paths.
        obs = table.observation()
        assert obs.tobytes() == encode_view(table.view(), table.legal).tobytes()
        t = ActionType(a.engine_action)
        masked = _MASKED[t]
        if table.mask[masked]:
            assert to_engine(masked, table.legal) is t, (case.name, street, a)
            table.act(masked)
        else:
            # Only a fold when checking is free is not expressed by the mask.
            assert t is ActionType.FOLD and table.legal.can_check, (case.name, street, a)
            table.act_engine(t)
            raw += 1
    if table.decision_pending:
        raise AssertionError(f"the case's actions are over but seat {table.seat_to_act} is asked")
    return table, raw


class RecordedHandsTest(unittest.TestCase):
    def test_all_cases(self):
        cases = load_cases()
        self.assertEqual(len(cases), N_CASES)
        for case in cases:
            with self.subTest(case=case.name):
                self.check_case(case)

    def check_case(self, case):
        bb = case.game.bb
        seats = engine_seats(case)
        names = {s: pid for pid, s in seats.items()}

        def chips(x):
            return int(round(x * bb))

        table, _ = replay(case)
        self.assertTrue(table.hand_over)
        result = table.result
        self.assertEqual(result.button, 0)
        self.assertEqual(result.showdown, case.result.ended_by == "showdown")

        final = {names[s]: chips(x) for s, x in result.stacks.items()}
        net = {names[s]: chips(x) for s, x in result.net.items()}
        self.assertEqual(final, dict(case.result.final_stacks))
        self.assertEqual(net, dict(case.result.net))
        self.assertEqual({names[s]: chips(x) for s, x in enumerate(table.net_bb)}, net)
        np.testing.assert_array_equal(table.rewards, (table.net_bb / 2).astype(np.float32))

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
        case = next(c for c in load_cases() if c.result.ended_by == "showdown")
        table, _ = replay(case)
        bb = case.game.bb
        names = {s: pid for pid, s in engine_seats(case).items()}
        final = {names[s]: int(round(x * bb)) for s, x in table.result.stacks.items()}
        wrong = dict(case.result.final_stacks)
        a, b = sorted(wrong)[:2]
        wrong[a], wrong[b] = wrong[a] + 1, wrong[b] - 1
        self.assertNotEqual(final, wrong)


if __name__ == "__main__":
    unittest.main()
