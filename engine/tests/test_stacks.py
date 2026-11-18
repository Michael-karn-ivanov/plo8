"""Starting stacks are seated exactly (see plan.md, Units / chips -> Starting
stacks are seated exactly): a stack that is a whole number of the table's chips
gets exactly those chips, however the player arrives."""

import random
import unittest

from engine import (
    ActionType,
    Environment,
    HandStarted,
    Player,
    PlayerJoined,
    to_chips,
)

# Stacks (BB) that are whole numbers of chips at each unit; several are not whole
# at 100 chips/BB, and some have inexact float representations (0.29, 1.1, 2.3).
STACKS = {
    2: [0.5, 1.5, 3.0, 7.5, 100.0, 1.0, 250.5],
    10: [0.1, 0.3, 1.1, 2.3, 12.7, 0.9, 99.9],
    100: [0.29, 1.1, 2.3, 0.01, 100.0, 37.45, 0.07],
    1000: [1.234, 0.005, 0.29, 1.1, 2.3, 99.999, 0.001],
    10000: [0.0001, 1.2345, 0.29, 2.3, 1.1, 50.0001, 0.0007],
}
UNITS = sorted(STACKS)


def generated(cpb, n=40, seed=0):
    """Random stacks k / cpb (whole numbers of the table's chips)."""
    rng = random.Random(seed + cpb)
    return [rng.randrange(1, 300 * cpb) / cpb for _ in range(n)]


def all_stacks(cpb):
    return STACKS[cpb] + generated(cpb)


def chunks(values, size):
    return [values[i : i + size] for i in range(0, len(values), size)]


def passive(d):
    return ActionType.CHECK if d.legal.can_check else ActionType.CALL


def first_hand_started(env):
    """Start a hand in step mode and return its HandStarted event."""
    env.start_hand()
    started = env.history[0]
    assert isinstance(started, HandStarted)
    return started


class InitialPlayersTest(unittest.TestCase):
    def test_players_passed_to_environment_are_seated_exactly(self):
        for cpb in UNITS:
            for group in chunks(all_stacks(cpb), 6):
                if len(group) < 2:
                    group = group + [1.0]
                with self.subTest(chips_per_bb=cpb, stacks=group):
                    players = [Player(f"p{i}", s) for i, s in enumerate(group)]
                    env = Environment(players, chips_per_bb=cpb)
                    for p, s in zip(players, group):
                        self.assertEqual(p.chips, round(s * cpb), f"{s} BB at {cpb}")
                        self.assertEqual(p.stack, p.chips / cpb)
                        self.assertEqual(p.stack, s)
                    started = first_hand_started(env)
                    self.assertEqual(
                        [stack for _, _, stack in started.participants], [float(s) for s in group]
                    )

    def test_examples_from_the_plan(self):
        a, b = Player("A", 1.234), Player("B", 0.005)
        Environment([a, b, Player("C", 1)], chips_per_bb=1000)
        self.assertEqual(a.chips, 1234)  # not 1230
        self.assertEqual(b.chips, 5)  # not rejected as 0
        half = Player("H", 0.5)
        Environment([half, Player("X", 1)], chips_per_bb=2)
        self.assertEqual(half.chips, 1)

    def test_unseated_player_reports_its_bb_stack(self):
        p = Player("A", 1.234)
        self.assertEqual(p.stack, 1.234)
        self.assertEqual(p.chips, 123)  # display value at the default unit
        Environment([p, Player("B", 1)], chips_per_bb=1000)
        self.assertEqual((p.stack, p.chips), (1.234, 1234))

    def test_rebuy_before_seating(self):
        p = Player("A", 1.2)
        p.rebuy(0.034)
        self.assertAlmostEqual(p.stack, 1.234)
        self.assertAlmostEqual(p.rebought, 0.034)
        with self.assertRaises(ValueError):
            p.rebuy(0)
        with self.assertRaises(ValueError):
            p.rebuy(-1)
        with self.assertRaises(TypeError):
            p.rebuy("1")
        env = Environment([p, Player("B", 1)], chips_per_bb=1000)
        self.assertEqual(p.chips, 1234)
        self.assertEqual(p.rebuy_chips, 34)
        self.assertEqual(p.rebought, 0.034)
        self.assertEqual(first_hand_started(env).participants[0][2], 1.234)

    def test_rebuy_before_seating_tiny_amounts(self):
        # 0.0004 + 0.0003 BB: below one chip at 100/BB, 7 chips at 10,000/BB.
        p = Player("A", 0.0004)
        p.rebuy(0.0003)
        Environment([p, Player("B", 1)], chips_per_bb=10000)
        self.assertEqual(p.chips, 7)
        self.assertEqual(p.rebuy_chips, 3)

    def test_stack_below_one_chip_rejected(self):
        for cpb in UNITS:
            with self.subTest(chips_per_bb=cpb):
                tiny = 0.4 / cpb  # rounds to 0 chips
                with self.assertRaises(ValueError):
                    Environment([Player("A", tiny), Player("B", 1)], chips_per_bb=cpb)

    def test_one_chip_of_the_table_below_one_chip_at_100_accepted(self):
        for cpb, stack in ((1000, 0.005), (1000, 0.001), (10000, 0.0001), (10000, 0.0049)):
            with self.subTest(chips_per_bb=cpb, stack=stack):
                self.assertEqual(to_chips(stack, 100), 0)
                p = Player("A", stack)
                Environment([p, Player("B", 1)], chips_per_bb=cpb)
                self.assertEqual(p.chips, round(stack * cpb))
                self.assertGreater(p.chips, 0)

    def test_negative_stack_rejected_by_player(self):
        with self.assertRaises(ValueError):
            Player("A", -0.001)
        with self.assertRaises(TypeError):
            Player("A", "10")

    def test_non_whole_stack_rounded_once(self):
        p = Player("A", 1.237)
        Environment([p, Player("B", 1)], chips_per_bb=10)
        self.assertEqual(p.chips, 12)
        q = Player("A", 1.2351)  # 1235.1 chips at 1,000/BB: no detour via 100/BB
        Environment([q, Player("B", 1)], chips_per_bb=1000)
        self.assertEqual(q.chips, 1235)

    def test_default_unit_unchanged(self):
        rng = random.Random(5)
        values = [rng.uniform(0.01, 300) for _ in range(200)] + STACKS[100]
        for group in chunks(values, 8):
            players = [Player(f"p{i}", s) for i, s in enumerate(group)]
            Environment(players + [Player("z", 1)])
            for p, s in zip(players, group):
                self.assertEqual(p.chips, to_chips(s, 100))


class JoinersTest(unittest.TestCase):
    def joined_events(self, env):
        events = []

        class Listener:
            def on_turn(self, player, view, legal):  # pragma: no cover - step mode
                raise AssertionError

            def on_hand_end(self, player, result):
                pass

            def on_event(self, player, event):
                if isinstance(event, PlayerJoined):
                    events.append(event)

        env.join(100, name="listener", agent=Listener())
        return events

    def test_join_between_hands(self):
        for cpb in UNITS:
            stacks = all_stacks(cpb)
            for group in chunks(stacks, 8):
                with self.subTest(chips_per_bb=cpb, stacks=group):
                    env = Environment(chips_per_bb=cpb)
                    events = self.joined_events(env)
                    for i, s in enumerate(group):
                        env.join(s, name=f"j{i}")
                    by_name = {p.name: p for p in env.players}
                    for i, s in enumerate(group):
                        p = by_name[f"j{i}"]
                        self.assertEqual(p.chips, round(s * cpb))
                        self.assertEqual(p.stack, s)
                    joined = {e.name: e.stack_bb for e in events}
                    for i, s in enumerate(group):
                        self.assertEqual(joined[f"j{i}"], s)
                    started = first_hand_started(env)
                    stacks_by_name = {name: st for _, name, st in started.participants}
                    for i, s in enumerate(group):
                        self.assertEqual(stacks_by_name[f"j{i}"], s)

    def test_join_queued_mid_hand(self):
        for cpb in UNITS:
            group = STACKS[cpb][:5]
            with self.subTest(chips_per_bb=cpb):
                env = Environment(
                    [Player("a", 100), Player("b", 100), Player("c", 100)], chips_per_bb=cpb
                )
                events = self.joined_events(env)
                d = env.start_hand()
                self.assertIsNotNone(d)
                for i, s in enumerate(group):
                    env.join(s, name=f"q{i}")
                self.assertEqual(len(env.waiting), len(group))
                while d is not None:
                    d = env.step(passive(d))
                by_name = {p.name: p for p in env.players}
                for i, s in enumerate(group):
                    p = by_name[f"q{i}"]
                    self.assertEqual(p.chips, round(s * cpb))
                joined = {e.name: e.stack_bb for e in events}
                for i, s in enumerate(group):
                    self.assertEqual(joined[f"q{i}"], s)

    def test_join_below_one_chip_rejected(self):
        for cpb in UNITS:
            with self.subTest(chips_per_bb=cpb):
                env = Environment(chips_per_bb=cpb)
                with self.assertRaises(ValueError):
                    env.join(0.4 / cpb)
                env.join(1 / cpb)
                self.assertEqual(env.players[0].chips, 1)


if __name__ == "__main__":
    unittest.main()
