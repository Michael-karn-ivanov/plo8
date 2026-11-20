"""Batches without observations: same play as with observations, no encoder
call, views still for the controllers that want them, controllers that need
vectors rejected loudly, no recorder."""

import contextlib
import unittest

import numpy as np

from engine import Dealer
from rl_env import (
    OBS_DIM,
    DecisionBatch,
    HandSampler,
    RandomController,
    Recorder,
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


class NoWantsObs:
    """A controller that forgets `wants_obs` (so it counts as wanting vectors)."""

    wants_views = False

    def act(self, batch):
        return RandomController(0).act(batch)


class WantsObs(NoWantsObs):
    wants_obs = True


def mixed(spec):
    # Random and view-reading controllers mixed over the seats.
    return ["r" if (spec.deck_seed + s) % 2 else "v" for s in range(spec.n_seats)]


def table_state(batch):
    out = []
    for tbl in batch.tables:
        env = tbl.env
        if env is None:
            out.append(None)
        else:
            out.append((env.decision, len(env.history), tuple(p.chips for p in env.players)))
    return out


def play(observations, *, n_tables=16, hands=400, seed=40):
    batch = TableBatch(n_tables, observations=observations)
    sampler = HandSampler(seed)
    viewer = helpers.CheckCallController()
    ctrls = {"r": RandomController(seed), "v": viewer}
    steps = []
    results = []
    started = 0
    for t in batch.idle_tables():
        spec = sampler.sample()
        batch.start_hand(t, spec, mixed(spec))
        started += 1
    while batch.any_in_hand() or started < hands:
        for g in batch.pending(ctrls):
            if observations:
                assert g.obs is not None and g.obs.shape == (len(g), OBS_DIM)
            else:
                assert g.obs is None
            assert (g.views is not None) == (g.key == "v")
            actions = np.asarray(ctrls[g.key].act(g))
            steps.append(
                (
                    g.key,
                    g.tables.tolist(),
                    g.seats.tolist(),
                    g.mask.tobytes(),
                    list(g.legal),
                    None if g.views is None else [v.seat for v in g.views],
                    actions.tolist(),
                )
            )
            batch.submit(g, actions)
        for fh in batch.collect_finished():
            results.append(
                (
                    fh.table,
                    fh.hand_id,
                    fh.spec,
                    fh.controllers,
                    fh.net_bb.tolist(),
                    fh.rewards.tolist(),
                    fh.result.history,
                )
            )
            if started < hands:
                spec = sampler.sample()
                batch.start_hand(fh.table, spec, mixed(spec))
                started += 1
    return batch, viewer, steps, results


class SamePlayTest(unittest.TestCase):
    def test_identical_to_a_batch_with_observations(self):
        _, _, steps1, res1 = play(True)
        with counting(Table, "collect_observation") as collected, counting(
            Table, "observation"
        ) as observed, counting(Dealer, "table_view") as views:
            batch, viewer, steps2, res2 = play(False)
        self.assertGreater(len(steps1), 200)
        self.assertEqual(len(res1), 400)
        self.assertEqual(steps1, steps2)
        self.assertEqual(res1, res2)
        self.assertEqual({spec.n_seats for _, _, spec, *_ in res1}, set(range(2, 11)))
        # No encoder call, no tracker fed.
        self.assertEqual(collected["n"], 0)
        self.assertEqual(observed["n"], 0)
        for tbl in batch.tables:
            self.assertEqual(tbl.events_fed, 0)
        # Views still built, exactly for the view-reading controller.
        self.assertGreater(viewer.views_seen, 0)
        self.assertEqual(views["n"], viewer.views_seen)

    def test_with_observations_every_controller_gets_them(self):
        # wants_obs = False does not change what a batch with observations builds.
        batch = TableBatch(8)
        sampler = HandSampler(41)
        for t in batch.idle_tables():
            spec = sampler.sample()
            batch.start_hand(t, spec, ["r"] * spec.n_seats)
        (g,) = batch.pending({"r": RandomController(1)})
        self.assertEqual(g.obs.shape, (len(g), OBS_DIM))
        for r, t in enumerate(g.tables.tolist()):
            self.assertEqual(g.obs[r].tobytes(), batch.tables[t].observation().tobytes())

    def test_run_helper_without_observations(self):
        batch = TableBatch(8, observations=False)
        with counting(Table, "collect_observation") as collected:
            done = batch.run(
                {"r": RandomController(42)},
                HandSampler(42),
                200,
                lambda spec: ["r"] * spec.n_seats,
            )
        self.assertEqual(done, 200)
        self.assertEqual(collected["n"], 0)

    def test_single_always_builds_obs(self):
        batch = TableBatch(2, observations=False)
        batch.start_hand(0, helpers.HandSpec(3, (100.0,) * 3, 0, 5), ["r"] * 3)
        tbl = batch.tables[0]
        single = DecisionBatch.single(tbl.view(), tbl.legal)
        self.assertEqual(single.obs.shape, (1, OBS_DIM))


class RejectTest(unittest.TestCase):
    def _batch(self):
        # Even tables: the view-reading controller "v" (group first, so a late
        # check would already have built its views); odd tables: "x".
        batch = TableBatch(6, observations=False)
        for t in batch.idle_tables():
            spec = helpers.HandSpec(3, (100.0,) * 3, t % 3, 100 + t)
            batch.start_hand(t, spec, ["v" if t % 2 == 0 else "x"] * spec.n_seats)
        return batch

    def test_controllers_that_want_obs_raise_before_any_table_changes(self):
        for bad in (NoWantsObs(), WantsObs()):
            with self.subTest(controller=type(bad).__name__):
                batch = self._batch()
                ctrls = {"v": helpers.CheckCallController(), "x": bad}
                self.assertEqual([g.key for g in batch.pending()], ["v", "x"])
                before = table_state(batch)
                with counting(Dealer, "table_view") as views:
                    with self.assertRaises(ValueError):
                        batch.pending(ctrls)
                    with self.assertRaises(ValueError):
                        batch.step(ctrls)
                self.assertEqual(views["n"], 0)
                self.assertEqual(table_state(batch), before)
                self.assertEqual(batch.n_decisions, 0)
                self.assertEqual(ctrls["v"].views_seen, 0)
                # Without controllers the check is skipped (the caller knows).
                for g in batch.pending():
                    self.assertIsNone(g.obs)
                    self.assertIsNone(g.views)
                # Controllers that declare wants_obs = False are fine.
                ok = {"v": helpers.CheckCallController(), "x": RandomController(1)}
                batch.step(ok)
                self.assertEqual(batch.n_decisions, 6)

    def test_recorder_needs_observations(self):
        with self.assertRaises(ValueError):
            TableBatch(4, recorder=Recorder(), observations=False)
        TableBatch(4, recorder=Recorder(), observations=True)  # fine

    def test_shipped_controllers_declare_no_obs(self):
        self.assertFalse(RandomController.wants_obs)
        self.assertFalse(RandomController.wants_views)
        self.assertFalse(helpers.CheckCallController.wants_obs)
        self.assertTrue(helpers.CheckCallController.wants_views)


if __name__ == "__main__":
    unittest.main()
