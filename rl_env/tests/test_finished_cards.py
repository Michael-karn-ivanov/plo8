"""Hole cards of finished hands (evaluation tools only): every seat's dealt
hand, readable from the end of the hand until that table's next hand, and
never part of anything a controller receives."""

import contextlib
import dataclasses
import random
import unittest
from enum import Enum

import numpy as np

from engine import Dealer, Environment, HandShown, Player
from entities import Card
from rl_env import (
    HandSampler,
    HandSpec,
    NoHandError,
    RandomController,
    Table,
    TableBatch,
)

try:
    from . import helpers
except ImportError:  # run as a top-level module
    import helpers  # type: ignore[no-redef]


@contextlib.contextmanager
def counting(cls, name):
    counter = {"n": 0}
    original = getattr(cls, name)

    def wrapper(self, *args, **kwargs):
        counter["n"] += 1
        return original(self, *args, **kwargs)

    setattr(cls, name, wrapper)
    try:
        yield counter
    finally:
        setattr(cls, name, original)


def mixed(spec):
    return ["r" if (spec.deck_seed + s) % 2 else "v" for s in range(spec.n_seats)]


def walk(x, cards, types, seen):
    """Collect every Card reachable from `x` (through dataclass fields and
    containers) and the types of the objects passed on the way."""
    if isinstance(x, Card):
        cards.add(x)
        return
    if x is None or isinstance(x, (bool, int, float, str, Enum, np.generic)):
        return
    if id(x) in seen:
        return
    seen.add(id(x))
    if isinstance(x, np.ndarray):
        assert x.dtype.kind in "biuf", x.dtype
    elif dataclasses.is_dataclass(x) and not isinstance(x, type):
        types.add(type(x))
        for f in dataclasses.fields(x):
            walk(getattr(x, f.name), cards, types, seen)
    elif isinstance(x, (list, tuple, set, frozenset)):
        for e in x:
            walk(e, cards, types, seen)
    elif isinstance(x, dict):
        for k, v in x.items():
            walk(k, cards, types, seen)
            walk(v, cards, types, seen)
    else:
        types.add(type(x))


def cards_in(x):
    cards, types = set(), set()
    walk(x, cards, types, set())
    return cards, types


class TableLevelTest(unittest.TestCase):
    def test_stacked_deck_hands(self):
        rng = random.Random(70)
        shown_checked = 0
        for k in range(400):
            n = 2 + k % 9
            spec = helpers.random_spec(rng, n)
            script = list(helpers.ALL_CARDS)
            rng.shuffle(script)
            table = Table()
            with self.assertRaises(NoHandError):
                table.finished_hole_cards()  # before the first hand
            last = {}

            def at_decision(t):
                with self.assertRaises(NoHandError):
                    t.finished_hole_cards()  # never while a decision is pending
                last.clear()
                last.update(t.privileged_hole_cards())

            helpers.play_random(table, spec, rng, at_decision, deck=helpers.StackedDeck(script))
            cards = table.finished_hole_cards()
            self.assertEqual(set(cards), set(range(n)))  # every seat, folded ones included
            for i in range(n):
                seat = helpers.seat_of(n, spec.button, i)
                self.assertEqual(set(cards[seat]), {script[r * n + i] for r in range(4)})
            if last:
                self.assertEqual(dict(cards), last)
            for seat, hand in table.result.shown.items():
                self.assertEqual(hand, cards[seat])
                shown_checked += 1
            board = tuple(table.result.board)
            self.assertEqual(board, tuple(script[4 * n : 4 * n + len(board)]))
            # Still readable after the hand; gone with the next one.
            self.assertEqual(dict(table.finished_hole_cards()), dict(cards))
            table.start_hand(n, [100.0] * n, spec.button, spec.deck_seed + 1)
            self.assertTrue(table.decision_pending)
            with self.assertRaises(NoHandError):
                table.finished_hole_cards()
        self.assertGreater(shown_checked, 100)

    def test_no_decision_hand(self):
        table = Table()
        self.assertFalse(table.start_hand(2, [0.5, 1.0], 0, 3))
        cards = table.finished_hole_cards()
        self.assertEqual(set(cards), {0, 1})
        for seat, hand in table.result.shown.items():
            self.assertEqual(hand, cards[seat])


class BatchLevelTest(unittest.TestCase):
    def test_random_batch_runs(self):
        batch = TableBatch(12)
        with self.assertRaises(NoHandError):
            batch.finished_hole_cards(0)
        sampler = HandSampler(71)
        ctrls = {"r": RandomController(71), "v": helpers.CheckCallController()}
        for t in batch.idle_tables():
            spec = sampler.sample()
            batch.start_hand(t, spec, mixed(spec))
        last = {}  # table -> privileged cards at its latest decision (engine read)
        hands = 0
        reads = 0  # successful finished_hole_cards reads (each is one privileged read)
        with counting(Table, "privileged_hole_cards") as priv:
            while hands < 500:
                for g in batch.pending(ctrls):
                    for t in g.tables.tolist():
                        last[t] = dict(batch.tables[t].env.privileged_hole_cards())
                        with self.assertRaises(RuntimeError) as cm:
                            batch.finished_hole_cards(t)  # a hand is running
                        self.assertNotIsInstance(cm.exception, NoHandError)
                    batch.submit(g, ctrls[g.key].act(g))
                self.assertEqual(priv["n"], reads)  # pending / act / submit never read it
                # Readable before collect_finished() returns the hand.
                for t in batch.idle_tables():
                    batch.finished_hole_cards(t)
                    reads += 1
                finished = batch.collect_finished()
                snap = {}
                for fh in finished:
                    n = fh.spec.n_seats
                    cards = dict(batch.finished_hole_cards(fh.table))
                    reads += 1
                    snap[fh.table] = cards
                    self.assertEqual(set(cards), set(range(n)))
                    dealt = [c for hand in cards.values() for c in hand]
                    self.assertEqual(len(set(dealt)), 4 * n)
                    self.assertFalse(set(dealt) & set(fh.result.board))
                    if fh.table in last:
                        self.assertEqual(cards, last.pop(fh.table))
                    for seat, hand in fh.result.shown.items():
                        self.assertEqual(hand, cards[seat])
                for fh in finished:
                    # Other tables' restarts do not touch it (`snap` holds the
                    # tables not restarted yet).
                    for t, cards in snap.items():
                        self.assertEqual(dict(batch.finished_hole_cards(t)), cards)
                        reads += 1
                    spec = sampler.sample()
                    batch.start_hand(fh.table, spec, mixed(spec))
                    if fh.table in batch.idle_tables():  # finished at once: the new deal
                        self.assertNotEqual(
                            dict(batch.finished_hole_cards(fh.table)), snap[fh.table]
                        )
                        reads += 1
                    else:
                        with self.assertRaises(RuntimeError):
                            batch.finished_hole_cards(fh.table)
                    del snap[fh.table]
                    hands += 1
                self.assertEqual(priv["n"], reads)
        self.assertGreater(reads, 500)

    def test_step_and_run_never_read_hidden_cards(self):
        batch = TableBatch(16)
        ctrls = {"r": RandomController(72), "v": helpers.CheckCallController()}
        with counting(Table, "privileged_hole_cards") as priv:
            done = batch.run(ctrls, HandSampler(72), 300, mixed)
            sampler = HandSampler(73)
            for t in batch.idle_tables():
                spec = sampler.sample()
                batch.start_hand(t, spec, mixed(spec))
            while batch.any_in_hand():
                batch.step(ctrls)
        self.assertEqual(done, 300)
        self.assertEqual(priv["n"], 0)

    def test_decision_batches_carry_no_other_seat_card(self):
        batch = TableBatch(12)
        sampler = HandSampler(74)
        ctrls = {"r": RandomController(74), "v": helpers.CheckCallController()}
        for t in batch.idle_tables():
            spec = sampler.sample()
            batch.start_hand(t, spec, mixed(spec))
        rows = 0
        viewed = 0
        for _ in range(300):
            for g in batch.pending(ctrls):
                own_all = set()
                public = set()
                for r, t in enumerate(g.tables.tolist()):
                    env = batch.tables[t].env
                    seat = int(g.seats[r])
                    own = set(env.privileged_hole_cards()[seat])
                    board = {c for e in env.history if hasattr(e, "board") for c in e.board}
                    own_all |= own
                    public |= board
                    # HandShown cannot precede a decision.
                    self.assertFalse(any(isinstance(e, HandShown) for e in env.history))
                    # obs: the hole block is the own hand; card blocks hold own + board only.
                    hole = set(np.flatnonzero(g.obs[r, :52]).tolist())
                    self.assertEqual(hole, {c.index for c in own})
                    in_cards = {i % 52 for i in np.flatnonzero(g.obs[r, :208]).tolist()}
                    self.assertTrue(in_cards <= {c.index for c in own | board})
                    if g.views is not None:
                        view = g.views[r]
                        self.assertEqual(set(view.hole_cards), own)
                        view_cards, _ = cards_in(view)
                        self.assertTrue(view_cards <= own | board)
                        viewed += 1
                    rows += 1
                # The whole batch: only the acting seats' own cards and the boards,
                # and no engine object that could reach the rest.
                batch_cards, types = cards_in(g)
                self.assertTrue(batch_cards <= own_all | public)
                for forbidden in (Player, Environment, Dealer, Table, TableBatch):
                    self.assertNotIn(forbidden, types)
                opaque = {tp.__name__ for tp in types if not dataclasses.is_dataclass(tp)}
                self.assertTrue(opaque <= {"_BatchToken"}, opaque)
                batch.submit(g, ctrls[g.key].act(g))
            for fh in batch.collect_finished():
                spec = sampler.sample()
                batch.start_hand(fh.table, spec, mixed(spec))
        self.assertGreater(rows, 1000)
        self.assertGreater(viewed, 100)

    def test_finished_hand_holds_only_public_cards(self):
        batch = TableBatch(8)
        hands = []
        batch.run(
            {"r": RandomController(75)},
            HandSampler(75),
            200,
            lambda spec: ["r"] * spec.n_seats,
            hands.append,
        )
        self.assertEqual(len(hands), 200)
        for fh in hands:
            self.assertEqual(
                {f.name for f in dataclasses.fields(fh)},
                {"table", "hand_id", "spec", "controllers", "net_bb", "rewards", "result"},
            )
            cards, _ = cards_in(fh)
            shown = {c for hand in fh.result.shown.values() for c in hand}
            self.assertTrue(cards <= shown | set(fh.result.board))

    def test_no_decision_hand_is_readable_at_once(self):
        batch = TableBatch(2)
        batch.start_hand(1, HandSpec(2, (0.5, 1.0), 0, 1), ["a", "b"])
        cards = dict(batch.finished_hole_cards(1))  # before collect_finished()
        self.assertEqual(set(cards), {0, 1})
        (fh,) = batch.collect_finished()
        self.assertEqual(fh.table, 1)
        self.assertEqual(dict(batch.finished_hole_cards(1)), cards)
        with self.assertRaises(NoHandError):
            batch.finished_hole_cards(0)


if __name__ == "__main__":
    unittest.main()
