"""View-free reads in step mode: `Environment.history` and `hole_cards()`
(see plan.md, Environment -> Step mode)."""

import contextlib
import random
import unittest

from entities import Deck
from engine import (
    ActionType,
    Dealer,
    Environment,
    HandHistory,
    NoDecisionError,
    Player,
)


@contextlib.contextmanager
def counting_views():
    counter = {"n": 0}
    original = Dealer.table_view

    def table_view(self, p):
        counter["n"] += 1
        return original(self, p)

    Dealer.table_view = table_view
    try:
        yield counter
    finally:
        Dealer.table_view = original


def passive(d):
    return ActionType.CHECK if d.legal.can_check else ActionType.CALL


def random_stack(rng):
    r = rng.random()
    if r < 0.15:
        return rng.choice([0.3, 0.5, 0.75])  # below one big blind
    if r < 0.35:
        return rng.uniform(1, 3)
    return rng.uniform(3, 60)


def random_env(rng, seed):
    n = rng.randint(2, 10)
    players = [Player(f"p{i}", random_stack(rng)) for i in range(n)]
    return Environment(players, button=rng.randrange(n), deck=Deck(random.Random(seed)))


class HistoryAndHoleCardsTest(unittest.TestCase):
    def test_equal_to_the_view_at_every_decision(self):
        rng = random.Random(11)
        decisions = 0
        for seed in range(300):
            env = random_env(rng, seed)
            d = env.start_hand()
            while d is not None:
                # View-free reads first, on a fresh decision.
                history = tuple(env.history)
                hole = env.hole_cards()
                view = env.view()
                self.assertEqual(history, view.history)
                self.assertEqual(hole, view.hole_cards)
                self.assertIs(hole, d.player.hole_cards)
                decisions += 1
                d = env.step(rng.choice(d.legal.actions))
            self.assertEqual(tuple(env.history), env.last_result.history)
        self.assertGreater(decisions, 300)

    def test_reads_build_no_view(self):
        rng = random.Random(12)
        with counting_views() as count:
            decisions = 0
            for seed in range(150):
                env = random_env(rng, seed)
                d = env.start_hand()
                while d is not None:
                    env.history[:]
                    len(env.history)
                    env.hole_cards()
                    decisions += 1
                    d = env.step(rng.choice(d.legal.actions))
                env.history[:]
        self.assertGreater(decisions, 0)
        self.assertEqual(count["n"], 0)

    def test_cursor_reading(self):
        rng = random.Random(13)
        for seed in range(200):
            env = random_env(rng, seed)
            seen = 0
            collected = []
            d = env.start_hand()
            while d is not None:
                self.assertEqual(d.hand_number, 1)
                collected.extend(env.history[seen:])
                seen = len(env.history)
                d = env.step(rng.choice(d.legal.actions))
            collected.extend(env.history[seen:])  # once more after the hand ends
            self.assertEqual(tuple(collected), env.last_result.history)
            self.assertEqual(tuple(env.history), env.last_result.history)

    def test_history_after_busting_equals_the_result(self):
        # Short stacks: someone busts, so PlayerSatOut follows the settlement; the
        # window still shows exactly the hand's events.
        for seed in range(30):
            env = Environment([Player("a", 0.5), Player("b", 0.5)], deck=Deck(random.Random(seed)))
            d = env.start_hand()
            while d is not None:
                d = env.step(passive(d))
            self.assertEqual(tuple(env.history), env.last_result.history)

    def test_new_hand_new_history(self):
        env = Environment([Player(f"p{i}", 100) for i in range(4)], seed=3)
        d = env.start_hand()
        first = env.history
        while d is not None:
            d = env.step(passive(d))
        old_events = tuple(first)
        self.assertEqual(old_events, env.last_result.history)

        d = env.start_hand()
        self.assertEqual(d.hand_number, 2)
        second = env.history
        self.assertIsNot(second, first)
        self.assertLess(len(second), len(old_events))
        self.assertTrue(all(e.hand_no == 2 for e in second))
        self.assertEqual(tuple(first), old_events)  # the old one still shows hand 1
        self.assertTrue(all(e.hand_no == 1 for e in first))

    def test_history_is_a_live_read_only_window(self):
        env = Environment([Player(f"p{i}", 100) for i in range(3)], seed=4)
        self.assertEqual(len(env.history), 0)  # before the first hand
        self.assertEqual(tuple(env.history), ())
        d = env.start_hand()
        h = env.history
        self.assertIsInstance(h, HandHistory)
        lengths = [len(h)]
        while d is not None:
            self.assertIs(env.history, h)
            d = env.step(passive(d))
            lengths.append(len(h))
        self.assertEqual(lengths, sorted(lengths))
        self.assertGreater(lengths[-1], lengths[0])

        self.assertIsInstance(h[1:3], tuple)
        self.assertIsInstance(h[:], tuple)
        self.assertEqual(h[0], env.last_result.history[0])
        self.assertEqual(h[-1], env.last_result.history[-1])
        self.assertEqual(list(iter(h)), list(env.last_result.history))
        with self.assertRaises(AttributeError):
            h.append(h[0])
        with self.assertRaises(TypeError):
            h[0] = h[1]
        with self.assertRaises(TypeError):
            del h[0]
        with self.assertRaises(AttributeError):
            h.extra = 1  # __slots__: no other attributes

    def test_hole_cards_without_decision(self):
        env = Environment([Player("a", 100), Player("b", 100)], seed=1)
        with self.assertRaises(NoDecisionError):
            env.hole_cards()
        d = env.start_hand()
        while d is not None:
            d = env.step(passive(d))
        with self.assertRaises(NoDecisionError):
            env.hole_cards()

    def test_reads_change_no_state(self):
        env = Environment([Player(f"p{i}", 50) for i in range(5)], seed=6)

        def state():
            return (
                env._seq,
                len(env.history),
                [p.chips for p in env.players],
                env.decision,
                env.dealer.pending,
            )

        d = env.start_hand()
        while d is not None:
            before = state()
            env.history[:]
            env.hole_cards()
            self.assertEqual(before, state())
            self.assertIs(env.decision, d)
            d = env.step(passive(d))


if __name__ == "__main__":
    unittest.main()
