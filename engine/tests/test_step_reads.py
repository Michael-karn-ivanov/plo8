"""View-free reads in step mode: `Environment.history`, `hole_cards()` and
`privileged_hole_cards()` (see plan.md, Environment -> Step mode)."""

import contextlib
import dataclasses
import random
import types
import unittest

from entities import Card, Deck, Hand
from engine import (
    ActionType,
    Dealer,
    Environment,
    HandHistory,
    HandShown,
    NoDecisionError,
    Player,
    PlayerStatus,
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


ALL = [Card.from_index(i) for i in range(52)]


class StackedDeck(Deck):
    """`shuffle()` shuffles, then puts the next script (if any) on top."""

    def __init__(self, scripts, seed=0):
        super().__init__(random.Random(seed))
        self.scripts = list(scripts)

    def shuffle(self):
        super().shuffle()
        script = self.scripts.pop(0) if self.scripts else None
        if script:
            self.put_on_top(script)


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
                    env.privileged_hole_cards()
                    decisions += 1
                    d = env.step(rng.choice(d.legal.actions))
                env.history[:]
                env.privileged_hole_cards()
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
            env.privileged_hole_cards()
            self.assertEqual(before, state())
            self.assertIs(env.decision, d)
            d = env.step(passive(d))


class Watcher:
    """Plays check/call in callback mode, idles on its turns in `idle_hands`
    (and so sits out), records its hole cards and events."""

    def __init__(self, idle_hands=()):
        self.idle_hands = set(idle_hands)
        self.hole = []
        self.events = []

    def on_turn(self, player, view, legal):
        if view.hand_number in self.idle_hands:
            return
        if legal.can_check:
            player.check()
        else:
            player.call()

    def on_hand_end(self, player, result):
        pass

    def on_hole_cards(self, player, hand, view):
        self.hole.append((hand, view.hole_cards))

    def on_event(self, player, event):
        self.events.append(event)


class PrivilegedHoleCardsTest(unittest.TestCase):
    def test_empty_before_the_first_hand(self):
        env = Environment([Player("a", 100), Player("b", 100)])
        cards = env.privileged_hole_cards()
        self.assertEqual(dict(cards), {})
        self.assertIsInstance(cards, types.MappingProxyType)

    def test_participants_only_folded_included_read_only_cached(self):
        # Hand 1 (callback mode, unstacked): seat 1 idles and sits out.
        # Hand 2 (step mode) deals the script; seat 1 is skipped.
        script = random.Random(21).sample(ALL, 52)
        deck = StackedDeck([None, script], seed=5)
        watchers = [Watcher(), Watcher(idle_hands={1}), Watcher(), Watcher(), Watcher()]
        players = [Player(f"p{i}", 100, w) for i, w in enumerate(watchers)]
        env = Environment(players, deck=deck)
        env.start_round()
        self.assertIs(players[1].status, PlayerStatus.SITTING_OUT)
        for w in watchers:
            w.hole.clear()
            w.events.clear()

        d = env.start_hand()
        self.assertIsNotNone(d)
        button = env.button
        n = len(env.players)
        parts = sorted(
            (p.seat for p in players if p.status is PlayerStatus.ACTIVE),
            key=lambda s: (s - button - 1) % n,
        )
        k = len(parts)
        expected = {seat: Hand(script[r * k + j] for r in range(4)) for j, seat in enumerate(parts)}
        self.assertNotIn(1, expected)

        cards = env.privileged_hole_cards()
        self.assertIsInstance(cards, types.MappingProxyType)
        self.assertEqual(dict(cards), expected)
        with self.assertRaises(TypeError):
            cards[0] = cards[parts[0]]  # read-only

        first_actor = d.seat
        folded = False
        while d is not None:
            self.assertIs(env.privileged_hole_cards(), cards)  # same object all hand
            # Neither the decision nor the view carries other seats' cards.
            for f in dataclasses.fields(d):
                self.assertNotIsInstance(getattr(d, f.name), Hand)
            view = env.view()
            self.assertEqual(view.hole_cards, expected[d.seat])
            for seat_view in view.seats:
                for f in dataclasses.fields(seat_view):
                    self.assertNotIsInstance(getattr(seat_view, f.name), Hand)
            if not folded and d.seat == first_actor:
                action = ActionType.FOLD
                folded = True
            else:
                action = passive(d)
            d = env.step(action)

        self.assertTrue(players[first_actor].folded)
        self.assertIn(first_actor, env.privileged_hole_cards())  # folded included
        self.assertIs(env.privileged_hole_cards(), cards)  # still the last hand's
        self.assertEqual(dict(env.privileged_hole_cards()), expected)

        # on_hole_cards: only the agent's own cards; events: no hidden cards before
        # showdown (HandShown only for the players still in at showdown).
        for p, w in zip(players, watchers):
            if p.seat in expected:
                self.assertEqual(w.hole, [(expected[p.seat], expected[p.seat])])
            else:
                self.assertEqual(w.hole, [])
            shown_seen = False
            for e in w.events:
                if isinstance(e, HandShown):
                    shown_seen = True
                    self.assertFalse(players[e.seat].folded)
                    continue
                if shown_seen:
                    continue
                for f in dataclasses.fields(e):
                    self.assertNotIsInstance(getattr(e, f.name), Hand)

    def test_cleared_by_the_next_hand(self):
        env = Environment([Player(f"p{i}", 100) for i in range(3)], seed=8)
        d = env.start_hand()
        first = env.privileged_hole_cards()
        while d is not None:
            d = env.step(passive(d))
        env.start_hand()
        second = env.privileged_hole_cards()
        self.assertIsNot(second, first)
        self.assertEqual(set(second), {0, 1, 2})
        for seat, hand in second.items():
            self.assertIs(hand, env.players[seat].hole_cards)

    def test_seats_of_the_last_hand_kept_when_seats_change(self):
        players = [Player(f"p{i}", 100) for i in range(4)]
        env = Environment(players, seed=9)
        d = env.start_hand()
        while d is not None:
            d = env.step(passive(d))
        expected = {p.seat: p.hole_cards for p in players}
        players[0].leave()  # renumbers the seats between hands
        env.join(50, seat=1, name="new")
        self.assertEqual(dict(env.privileged_hole_cards()), expected)


if __name__ == "__main__":
    unittest.main()
