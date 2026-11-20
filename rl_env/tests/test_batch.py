"""TableBatch: grouping, ordering, lazy views, atomic submit, finished hands,
idle tables, DecisionBatch.single."""

import contextlib
import random
import unittest

import numpy as np

from engine import Dealer
from rl_env import (
    BET_RAISE,
    CHECK_CALL,
    FOLD,
    OBS_DIM,
    DecisionBatch,
    HandSampler,
    IllegalMaskedActionError,
    RandomController,
    Recorder,
    TableBatch,
    encode_view,
)

try:
    from . import helpers
except ImportError:  # run as a top-level module
    import helpers  # type: ignore[no-redef]


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


def alternating(spec):
    return ["a" if s % 2 == 0 else "b" for s in range(spec.n_seats)]


def fill(batch, sampler, seat_controllers=alternating, tables=None):
    for t in batch.idle_tables() if tables is None else tables:
        spec = sampler.sample()
        batch.start_hand(t, spec, seat_controllers(spec))


def table_state(batch):
    out = []
    for tbl in batch.tables:
        env = tbl.env
        if env is None:
            out.append(None)
        else:
            out.append((env.decision, len(env.history), tuple(p.chips for p in env.players)))
    return out


class GroupingTest(unittest.TestCase):
    def test_groups_rows_and_order(self):
        batch = TableBatch(12)
        sampler = HandSampler(1)
        controllers = {"a": RandomController(1), "b": RandomController(2)}
        fill(batch, sampler)
        for _ in range(200):
            groups = batch.pending(controllers)
            seen_tables = []
            firsts = []
            keys = []
            for g in groups:
                self.assertIsInstance(g, DecisionBatch)
                keys.append(g.key)
                rows = g.tables.tolist()
                self.assertEqual(rows, sorted(rows))
                self.assertEqual(g.obs.shape, (len(g), OBS_DIM))
                self.assertEqual(g.obs.dtype, np.float32)
                self.assertEqual(g.mask.shape, (len(g), 3))
                self.assertEqual(g.mask.dtype, np.bool_)
                self.assertIsNone(g.views)
                firsts.append(rows[0])
                for r, t in enumerate(rows):
                    tbl = batch.tables[t]
                    self.assertTrue(tbl.decision_pending)
                    self.assertEqual(int(g.seats[r]), tbl.seat_to_act)
                    self.assertEqual(batch.controllers_of(t)[tbl.seat_to_act], g.key)
                    self.assertEqual(g.legal[r], tbl.legal)
                    self.assertEqual(g.mask[r].tolist(), list(tbl.mask))
                    self.assertEqual(g.obs[r].tobytes(), tbl.observation().tobytes())
                seen_tables.extend(rows)
            self.assertEqual(len(keys), len(set(keys)))  # one group per controller
            self.assertEqual(firsts, sorted(firsts))  # by smallest table index
            self.assertEqual(len(seen_tables), len(set(seen_tables)))  # one decision per table
            self.assertEqual(
                sorted(seen_tables),
                [t for t, tbl in enumerate(batch.tables) if tbl.decision_pending],
            )
            for fh in batch.step(controllers):
                spec = sampler.sample()
                batch.start_hand(fh.table, spec, alternating(spec))

    def test_views_only_for_controllers_that_want_them(self):
        batch = TableBatch(16)
        sampler = HandSampler(2)
        viewer = helpers.CheckCallController()
        controllers = {"a": RandomController(3), "b": viewer}
        fill(batch, sampler)
        with counting_views() as counter:
            for _ in range(300):
                for fh in batch.step(controllers):
                    spec = sampler.sample()
                    batch.start_hand(fh.table, spec, alternating(spec))
        self.assertGreater(viewer.views_seen, 0)
        self.assertEqual(counter["n"], viewer.views_seen)
        # Only random controllers: no view at all.
        batch = TableBatch(8)
        fill(batch, HandSampler(3), lambda spec: ["a"] * spec.n_seats)
        with counting_views() as counter:
            for _ in range(100):
                batch.step({"a": RandomController(4)})
        self.assertEqual(counter["n"], 0)

    def test_views_in_the_batch_are_the_tables_views(self):
        batch = TableBatch(6)
        fill(batch, HandSampler(4), lambda spec: ["v"] * spec.n_seats)
        (group,) = batch.pending({"v": helpers.CheckCallController()})
        self.assertEqual(len(group.views), len(group))
        for r, t in enumerate(group.tables.tolist()):
            self.assertIs(group.views[r], batch.tables[t].view())


class SubmitTest(unittest.TestCase):
    def test_one_masked_action_raises_and_changes_no_table(self):
        batch = TableBatch(16, recorder=Recorder())
        sampler = HandSampler(5)
        controllers = {"a": RandomController(5), "b": RandomController(6)}
        fill(batch, sampler)
        tested = 0
        for _ in range(300):
            for g in batch.pending(controllers):
                rows = [r for r in range(len(g)) if not g.mask[r].all()]
                if not rows or len(g) < 2:
                    continue
                r = rows[-1]
                actions = np.full(len(g), CHECK_CALL, dtype=np.int64)
                actions[r] = FOLD if not g.mask[r, FOLD] else BET_RAISE
                before = table_state(batch)
                pending = batch.recorder.n_pending
                decisions = batch.n_decisions
                with self.assertRaises(IllegalMaskedActionError):
                    batch.submit(g, actions)
                self.assertEqual(table_state(batch), before)
                self.assertEqual(batch.recorder.n_pending, pending)
                self.assertEqual(batch.n_decisions, decisions)
                tested += 1
                for bad in (np.array([CHECK_CALL] * (len(g) + 1)), np.full(len(g), 3)):
                    with self.assertRaises(IllegalMaskedActionError):
                        batch.submit(g, bad)
                    self.assertEqual(table_state(batch), before)
            for fh in batch.step(controllers):
                spec = sampler.sample()
                batch.start_hand(fh.table, spec, alternating(spec))
        self.assertGreater(tested, 10)

    def test_stale_batch_is_rejected(self):
        batch = TableBatch(4)
        fill(batch, HandSampler(6), lambda spec: ["a"] * spec.n_seats)
        (group,) = batch.pending()
        batch.submit(group, np.full(len(group), CHECK_CALL))
        with self.assertRaises(ValueError):
            batch.submit(group, np.full(len(group), CHECK_CALL))

    def test_batch_of_another_table_batch_is_rejected(self):
        # Same seeds: same tables, seats and decisions, but a different batch.
        one, other = TableBatch(4), TableBatch(4)
        fill(one, HandSampler(7), lambda spec: ["a"] * spec.n_seats)
        fill(other, HandSampler(7), lambda spec: ["a"] * spec.n_seats)
        (group,) = one.pending()
        before = table_state(other)
        with self.assertRaises(ValueError):
            other.submit(group, np.full(len(group), CHECK_CALL))
        self.assertEqual(table_state(other), before)
        one.submit(group, np.full(len(group), CHECK_CALL))  # its own batch accepts it


class FinishedTest(unittest.TestCase):
    def test_finished_once_with_rewards_and_idle_tables(self):
        batch = TableBatch(10)
        sampler = HandSampler(7)
        controllers = {"a": RandomController(7), "b": RandomController(8)}
        active = [0, 2, 3, 7]
        fill(batch, sampler, tables=active)
        ids = [batch.hand_id(t) for t in active]
        self.assertEqual(ids, sorted(ids))
        self.assertEqual(len(set(ids)), len(ids))
        idle = [t for t in range(10) if t not in active]
        seen = set()
        started = len(active)
        while batch.any_in_hand() or started < 200:
            self.assertEqual(
                batch.any_in_hand(), len(batch.idle_tables()) < len(batch.tables)
            )
            for t in idle:
                self.assertIsNone(batch.tables[t].env)  # never advanced
            for fh in batch.step(controllers):
                self.assertNotIn(fh.hand_id, seen)
                seen.add(fh.hand_id)
                self.assertIn(fh.table, active)
                self.assertIn(fh.table, batch.idle_tables())
                self.assertEqual(fh.spec.n_seats, len(fh.controllers))
                self.assertEqual(fh.controllers, tuple(alternating(fh.spec)))
                np.testing.assert_array_equal(fh.rewards, (fh.net_bb / 2).astype(np.float32))
                self.assertEqual(sum(int(round(x * 100)) for x in fh.net_bb), 0)
                self.assertEqual([fh.result.net[s] for s in range(fh.spec.n_seats)], fh.net_bb.tolist())
                self.assertEqual(fh.hand_id, batch.hand_id(fh.table))
                if started < 200:
                    spec = sampler.sample()
                    batch.start_hand(fh.table, spec, alternating(spec))
                    started += 1
        self.assertEqual(len(seen), 200)
        self.assertFalse(batch.any_in_hand())
        self.assertEqual(batch.idle_tables(), list(range(10)))
        self.assertEqual(batch.collect_finished(), [])

    def test_no_decision_hand_is_reported(self):
        batch = TableBatch(2)
        spec = helpers.HandSpec(2, (0.5, 1.0), 0, 1)
        hid = batch.start_hand(1, spec, ["a", "b"])
        self.assertFalse(batch.any_in_hand())
        self.assertEqual(batch.pending(), [])
        (fh,) = batch.step({})
        self.assertEqual((fh.table, fh.hand_id), (1, hid))
        with self.assertRaises(RuntimeError):
            batch.start_hand(0, helpers.HandSpec(2, (100.0, 100.0), 0, 1), ["a", "b"])
            batch.start_hand(0, helpers.HandSpec(2, (100.0, 100.0), 0, 1), ["a", "b"])
        with self.assertRaises(ValueError):
            batch.start_hand(1, helpers.HandSpec(2, (100.0, 100.0), 0, 1), ["a"])

    def test_run_helper(self):
        batch = TableBatch(8)
        finished = []
        done = batch.run(
            {"r": RandomController(9)},
            HandSampler(9),
            100,
            lambda spec: ["r"] * spec.n_seats,
            finished.append,
        )
        self.assertEqual(done, 100)
        self.assertEqual(len(finished), 100)
        self.assertEqual(len({fh.hand_id for fh in finished}), 100)
        self.assertFalse(batch.any_in_hand())
        self.assertEqual(batch.n_hands, 100)
        self.assertGreater(batch.n_decisions, 100)


class SingleTest(unittest.TestCase):
    def test_single_equals_the_batch_row(self):
        batch = TableBatch(8)
        sampler = HandSampler(10)
        controllers = {"a": RandomController(10), "b": RandomController(11)}
        fill(batch, sampler)
        compared = 0
        for _ in range(100):
            for g in batch.pending(controllers):
                for r, t in enumerate(g.tables.tolist()):
                    tbl = batch.tables[t]
                    single = DecisionBatch.single(tbl.view(), tbl.legal, key=g.key)
                    self.assertEqual(len(single), 1)
                    self.assertEqual(single.obs[0].tobytes(), g.obs[r].tobytes())
                    self.assertEqual(single.mask[0].tolist(), g.mask[r].tolist())
                    self.assertEqual(int(single.seats[0]), int(g.seats[r]))
                    self.assertEqual(int(single.tables[0]), -1)
                    self.assertEqual(single.views, [tbl.view()])
                    self.assertEqual(
                        single.obs[0].tobytes(), encode_view(tbl.view(), tbl.legal).tobytes()
                    )
                    compared += 1
            for fh in batch.step(controllers):
                spec = sampler.sample()
                batch.start_hand(fh.table, spec, alternating(spec))
        self.assertGreater(compared, 300)
        # A controller answers a single engine turn the same way.
        (out,) = [RandomController(0).act(single)]
        self.assertEqual(out.shape, (1,))
        self.assertTrue(single.mask[0, int(out[0])])


if __name__ == "__main__":
    unittest.main()
