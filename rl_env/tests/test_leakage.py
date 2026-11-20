"""Leakage (the most important test): changing the other players' hole cards
while keeping everything public the same leaves the acting player's
observations byte-for-byte identical, on both paths. Also: the observation code
does not import the critic input, and the privileged read is confined to
`table.py` (called by the critic input and `finished_hole_cards` only)."""

import ast
import importlib
import inspect
import random
import unittest

import rl_env.controllers
import rl_env.encoder
import rl_env.hints
import rl_env.tracker
from rl_env import Table, encode_view

try:
    from . import helpers
except ImportError:  # run as a top-level module
    import helpers  # type: ignore[no-redef]

PACKAGE_MODULES = (
    "rl_env",
    "rl_env.actions",
    "rl_env.batch",
    "rl_env.controllers",
    "rl_env.critic",
    "rl_env.encoder",
    "rl_env.hints",
    "rl_env.recorder",
    "rl_env.stacks",
    "rl_env.table",
    "rl_env.tracker",
)


def observations_of(table, spec, actions, deck, seat):
    """Replay; at every decision of `seat`, (training path, view path) bytes."""
    out = []

    def grab(t):
        if t.seat_to_act == seat:
            out.append((t.observation().tobytes(), encode_view(t.view(), t.legal).tobytes()))

    helpers.replay(table, spec, actions, grab, deck=deck)
    return out


class LeakageTest(unittest.TestCase):
    def test_other_hole_cards_do_not_change_the_observation(self):
        rng = random.Random(30)
        situations = 0
        changed_cards = 0
        for k in range(3000):
            n = 2 + k % 9
            spec = helpers.random_spec(rng, n)
            script = list(helpers.ALL_CARDS)
            rng.shuffle(script)
            table = Table()
            seats = []
            actions = helpers.play_random(
                table,
                spec,
                rng,
                lambda t: seats.append(t.seat_to_act),
                deck=helpers.StackedDeck(script),
            )
            if not actions:
                continue
            target = seats[rng.randrange(len(seats))]  # the actor of a random decision
            # Participant index (from the left of the button) of the target.
            ti = (target - spec.button - 1) % n
            # Other players' hole cards are redrawn from their cards and the
            # unused cards; the target's cards and the board stay.
            board = script[4 * n : 4 * n + 5]
            others = [i for i in range(n) if i != ti]
            pool = [script[r * n + i] for r in range(4) for i in others] + script[4 * n + 5 :]
            rng.shuffle(pool)
            new = list(script[: 4 * n + 5])
            it = iter(pool)
            for r in range(4):
                for i in others:
                    new[r * n + i] = next(it)
            for i in others:
                old_hand = {script[r * n + i] for r in range(4)}
                new_hand = {new[r * n + i] for r in range(4)}
                changed_cards += len(old_hand - new_hand)
            self.assertEqual(new[4 * n : 4 * n + 5], board)
            self.assertEqual(
                [new[r * n + ti] for r in range(4)], [script[r * n + ti] for r in range(4)]
            )

            first = observations_of(Table(), spec, actions, helpers.StackedDeck(script), target)
            second = observations_of(Table(), spec, actions, helpers.StackedDeck(new), target)
            self.assertTrue(first)
            self.assertEqual(len(first), len(second))
            for (a_train, a_view), (b_train, b_view) in zip(first, second):
                self.assertEqual(a_train, a_view)
                self.assertEqual(a_train, b_train)
                self.assertEqual(a_view, b_view)
                situations += 1
        self.assertGreater(situations, 2500)
        self.assertGreater(changed_cards, 10_000)


def imports_of(module):
    names = set()
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add("." * node.level + (node.module or ""))
            names.update(f"{'.' * node.level}{node.module or ''}:{a.name}" for a in node.names)
    return names


def references(module, name):
    """AST nodes of `module` that refer to `name` (attribute, name, definition or
    import; docstrings and comments do not count)."""
    out = []
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, ast.Attribute) and node.attr == name:
            out.append(node)
        elif isinstance(node, ast.Name) and node.id == name:
            out.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            out.append(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and any(
            a.name == name or a.name.endswith("." + name) for a in node.names
        ):
            out.append(node)
    return out


class SeparationTest(unittest.TestCase):
    def test_observation_code_does_not_import_the_critic(self):
        for module in (rl_env.encoder, rl_env.tracker, rl_env.hints):
            with self.subTest(module=module.__name__):
                names = imports_of(module)
                self.assertFalse([n for n in names if "critic" in n], names)
                self.assertFalse(hasattr(module, "critic_cards"))
                self.assertNotIn("privileged_hole_cards", inspect.getsource(module))

    def test_observation_and_controller_code_reads_no_hidden_cards(self):
        for module in (rl_env.encoder, rl_env.tracker, rl_env.hints, rl_env.controllers):
            with self.subTest(module=module.__name__):
                source = inspect.getsource(module)
                self.assertNotIn("privileged_hole_cards", source)
                self.assertNotIn("finished_hole_cards", source)

    def test_the_privileged_read_is_confined_to_table_py(self):
        users = {}
        for name in PACKAGE_MODULES:
            module = importlib.import_module(name)
            refs = references(module, "privileged_hole_cards")
            if refs:
                users[name] = refs
        # table.py defines it and calls the engine; critic.py calls the table's.
        self.assertEqual(set(users), {"rl_env.table", "rl_env.critic"})
        for node in users["rl_env.critic"]:
            self.assertIsInstance(node, ast.Attribute)
            self.assertIsInstance(node.value, ast.Name)
            self.assertEqual(node.value.id, "table")  # Table's read, not the engine's
        # Within table.py, the engine's accessor is called in one place only.
        engine_calls = [
            node
            for node in users["rl_env.table"]
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "env"
        ]
        self.assertEqual(len(engine_calls), 1)
        # finished_hole_cards: defined on Table, exposed by TableBatch, nowhere else.
        finished = {
            name
            for name in PACKAGE_MODULES
            if references(importlib.import_module(name), "finished_hole_cards")
        }
        self.assertEqual(finished, {"rl_env.table", "rl_env.batch"})


if __name__ == "__main__":
    unittest.main()
