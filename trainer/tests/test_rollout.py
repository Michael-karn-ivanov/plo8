"""RolloutWorker: construction, lineups, collection (records only from finished
hands, rewards, masks, logp, the stop rule, nothing left pending, one learner
call per step), weights, statistics and determinism."""

import collections
import dataclasses
import math
import unittest

import numpy as np
import torch

from rl_env import TrainingRecords
from trainer import (
    LEARNER_KEY,
    CriticNet,
    NetConfigError,
    PolicyNet,
    ResultStats,
    RolloutStats,
    RolloutWorker,
)

try:
    from . import helpers
except ImportError:  # run as a top-level module
    import helpers  # type: ignore[no-redef]

N_MAIN = 3000
MIN_HANDS = 400  # hands of the main collection (typically ~600)


def param_snapshot(worker):
    return {k: v.detach().clone() for k, v in worker.policy.state_dict().items() if k != "_extra_state"}


def records_equal(test, a: TrainingRecords, b: TrainingRecords):
    for f in dataclasses.fields(TrainingRecords):
        x, y = getattr(a, f.name), getattr(b, f.name)
        test.assertEqual(x.dtype, y.dtype, f.name)
        test.assertTrue(np.array_equal(x, y, equal_nan=x.dtype.kind == "f"), f.name)
    # logp bit-identical (NaN never appears for the learner)
    test.assertEqual(a.logp.tobytes(), b.logp.tobytes())


def _same_float(x: float, y: float) -> bool:
    return (math.isnan(x) and math.isnan(y)) or x == y


def result_equal(a: ResultStats, b: ResultStats) -> bool:
    return all(
        _same_float(getattr(a, f.name), getattr(b, f.name))
        if isinstance(getattr(a, f.name), float)
        else getattr(a, f.name) == getattr(b, f.name)
        for f in dataclasses.fields(ResultStats)
    )


def stats_equal_but_time(a: RolloutStats, b: RolloutStats) -> bool:
    """Equal apart from wall_time and the two rates (NaN rates of empty groups
    compare equal)."""
    if not result_equal(a.overall, b.overall) or a.all_decisions != b.all_decisions:
        return False
    for da, db in ((a.by_size, b.by_size), (a.by_opponent, b.by_opponent)):
        if list(da) != list(db) or not all(result_equal(da[k], db[k]) for k in da):
            return False
    return True


def collect_keep(worker, weights, n):
    hands = []
    recs, stats = worker.collect(weights, n, on_finished=hands.append)
    return recs, stats, hands


def learner_seats(fh):
    return [s for s, k in enumerate(fh.controllers) if k == LEARNER_KEY]


class RolloutMain(unittest.TestCase):
    """One larger collection against all four default bots, shared by the
    lineup, collection and statistics checks."""

    @classmethod
    def setUpClass(cls):
        cls.rng_before = torch.random.get_rng_state()
        cls.weights = helpers.small_weights(0, peaked=True)
        cls.worker = helpers.small_worker(1)
        cls.rng_after_init = torch.random.get_rng_state()
        cls.recs, cls.stats, cls.hands = collect_keep(cls.worker, cls.weights, N_MAIN)
        cls.rng_after_collect = torch.random.get_rng_state()
        cls.by_id = {fh.hand_id: fh for fh in cls.hands}

    # --- construction / RNG ----------------------------------------------------

    def test_global_torch_rng_untouched(self):
        self.assertTrue(torch.equal(self.rng_before, self.rng_after_init))
        self.assertTrue(torch.equal(self.rng_before, self.rng_after_collect))

    def test_properties(self):
        w = self.worker
        self.assertEqual(w.opponents, ("random", "calling", "maniac", "rule"))
        self.assertEqual(w.sizes, tuple(range(2, 11)))
        self.assertEqual(len(w.batch), 32)
        self.assertEqual(w.recorder.keys, frozenset([LEARNER_KEY]))

    # --- lineups ---------------------------------------------------------------

    def test_lineups(self):
        self.assertGreaterEqual(len(self.hands), MIN_HANDS)
        sizes = {fh.spec.n_seats for fh in self.hands}
        self.assertEqual(sizes, set(range(2, 11)))
        names = set()
        for fh in self.hands:
            n = fh.spec.n_seats
            k = fh.controllers.count(LEARNER_KEY)
            self.assertEqual(len(fh.controllers), n)
            self.assertTrue(1 <= k <= min(3, n - 1), (n, k))
            if n == 2:
                self.assertEqual(k, 1)
            others = {c for c in fh.controllers if c != LEARNER_KEY}
            self.assertTrue(others <= set(self.worker.opponents), others)
            names |= others
        self.assertEqual(names, {"random", "calling", "maniac", "rule"})

    # --- collection ------------------------------------------------------------

    def test_records_learner_finished(self):
        recs = self.recs
        self.assertGreaterEqual(len(recs), N_MAIN)
        code = self.worker.recorder.codes[LEARNER_KEY]
        self.assertTrue((recs.controller == code).all())
        for h, s in zip(recs.hand_id.tolist(), recs.seat.tolist()):
            self.assertIn(h, self.by_id)
            self.assertEqual(self.by_id[h].controllers[s], LEARNER_KEY)

    def test_rewards(self):
        recs = self.recs
        pairs = list(zip(recs.hand_id.tolist(), recs.seat.tolist()))
        expected = np.array([self.by_id[h].rewards[s] for h, s in pairs], dtype=np.float32)
        self.assertTrue(np.array_equal(recs.reward, expected))
        net = np.array([self.by_id[h].net_bb[s] for h, s in pairs])
        np.testing.assert_allclose(recs.reward, net / 2, rtol=1e-6, atol=1e-6)

    def test_actions_masked(self):
        recs = self.recs
        self.assertTrue(recs.mask[np.arange(len(recs)), recs.action].all())

    def test_logp_matches_weights(self):
        recs = self.recs
        self.assertFalse(np.isnan(recs.logp).any())
        want = helpers.recorded_logp_under(helpers.net_from_weights(self.weights), recs)
        np.testing.assert_allclose(recs.logp, want, atol=1e-5, rtol=0)

    def test_nothing_pending(self):
        w = self.worker
        self.assertFalse(w.batch.any_in_hand())
        self.assertEqual(w.batch.collect_finished(), [])
        self.assertEqual(w.recorder.n_pending, 0)
        self.assertEqual(w.recorder.n_ready, 0)

    # --- statistics ------------------------------------------------------------

    def test_stats_counts(self):
        st, recs = self.stats, self.recs
        self.assertEqual(st.overall.hands, len(self.hands))
        self.assertEqual(st.hands, len(self.hands))
        self.assertEqual(st.overall.decisions, len(recs))
        self.assertEqual(st.decisions, len(recs))
        self.assertGreaterEqual(st.overall.hands, len(set(recs.hand_id.tolist())))
        self.assertEqual(list(st.by_size), list(range(2, 11)))
        self.assertEqual(list(st.by_opponent), list(self.worker.opponents))
        self.assertEqual(sum(g.hands for g in st.by_size.values()), st.overall.hands)
        self.assertEqual(sum(g.decisions for g in st.by_size.values()), st.overall.decisions)
        self.assertEqual(
            st.overall.learner_seats, sum(len(learner_seats(fh)) for fh in self.hands)
        )
        dec_per_hand = collections.Counter(recs.hand_id.tolist())
        for size, g in st.by_size.items():
            hs = [fh for fh in self.hands if fh.spec.n_seats == size]
            self.assertEqual(g.hands, len(hs), size)
            self.assertEqual(g.decisions, sum(dec_per_hand[fh.hand_id] for fh in hs), size)
        for name, g in st.by_opponent.items():
            hs = [fh for fh in self.hands if name in fh.controllers]
            self.assertEqual(g.hands, len(hs), name)  # once per hand, not per copy
            self.assertEqual(g.decisions, sum(dec_per_hand[fh.hand_id] for fh in hs), name)
        # some hand has two copies of one bot (and still counts once, above)
        self.assertTrue(
            any(
                max(collections.Counter(c for c in fh.controllers if c != LEARNER_KEY).values()) > 1
                for fh in self.hands
                if fh.spec.n_seats >= 5
            )
        )

    def test_stats_results(self):
        st, recs = self.stats, self.recs
        total = sum(
            int(round(float(fh.net_bb[s]) * 100)) for fh in self.hands for s in learner_seats(fh)
        )
        self.assertEqual(st.overall.total_chips, total)
        # From the records: each learner seat once (not once per decision) ...
        seat_reward = {}
        for h, s, r in zip(recs.hand_id.tolist(), recs.seat.tolist(), recs.reward.tolist()):
            seat_reward[(h, s)] = int(round(r * 200))
        from_records = sum(seat_reward.values())
        # ... plus the learner seats that made no decision.
        silent = sum(
            int(round(float(fh.net_bb[s]) * 100))
            for fh in self.hands
            for s in learner_seats(fh)
            if (fh.hand_id, s) not in seat_reward
        )
        self.assertEqual(from_records + silent, st.overall.total_chips)
        self.assertEqual(st.overall.total_bb, st.overall.total_chips / 200)
        self.assertEqual(st.total_bb, st.overall.total_bb)
        self.assertEqual(st.overall.bb_per_100, 100 * st.overall.total_bb / st.overall.hands)
        self.assertEqual(st.bb_per_100, st.overall.bb_per_100)
        self.assertEqual(sum(g.total_chips for g in st.by_size.values()), st.overall.total_chips)

    def test_stats_rates(self):
        st = self.stats
        self.assertGreater(st.wall_time, 0)
        self.assertEqual(st.decisions_per_second, st.decisions / st.wall_time)
        self.assertEqual(st.all_decisions_per_second, st.all_decisions / st.wall_time)
        self.assertGreaterEqual(st.all_decisions, st.decisions)


class RolloutConstruction(unittest.TestCase):
    def test_invalid(self):
        bad = [
            dict(n_tables=0),
            dict(opponents=()),
            dict(opponents=("random", "random")),
            dict(opponents=("random", "nobody")),
            dict(opponents=("learner",)),
            dict(sizes=(1,)),
            dict(sizes=(2, 11)),
            dict(sizes=()),
            dict(max_learner_seats=0),
            dict(seed=-1),
        ]
        for kw in bad:
            seed = kw.pop("seed", 0)
            with self.assertRaises((ValueError, TypeError), msg=str(kw)):
                helpers.small_worker(seed, **kw)
        with self.assertRaises(TypeError):
            helpers.small_worker(0, opponents="random")
        with self.assertRaises(TypeError):
            RolloutWorker(seed=1.5)

    def test_subset_lineups(self):
        w = helpers.small_worker(2, opponents=("calling", "maniac"), sizes=(2, 6))
        recs, stats, hands = collect_keep(w, helpers.small_weights(3), 800)
        self.assertGreaterEqual(len(recs), 800)
        self.assertEqual({fh.spec.n_seats for fh in hands}, {2, 6})
        names = {c for fh in hands for c in fh.controllers if c != LEARNER_KEY}
        self.assertEqual(names, {"calling", "maniac"})
        self.assertEqual(list(stats.by_size), [2, 6])
        self.assertEqual(list(stats.by_opponent), ["calling", "maniac"])


class RolloutLoop(unittest.TestCase):
    def test_stop_rule_and_second_collection(self):
        w = helpers.small_worker(4)
        rec = w.recorder
        original = w.batch.start_hand
        limit = [0]
        calls = [0]

        def start_hand(table, spec, controllers):
            calls[0] += 1
            recorded = rec.n_ready + rec.n_pending
            assert recorded < limit[0], (recorded, limit[0])
            return original(table, spec, controllers)

        w.batch.start_hand = start_hand  # instance attribute, delegating
        weights = helpers.small_weights(5)
        limit[0] = 1000
        r1, _ = w.collect(weights, 1000)
        self.assertGreater(calls[0], 32)  # tables were refilled
        self.assertFalse(w.batch.any_in_hand())
        self.assertEqual(w.batch.collect_finished(), [])
        self.assertEqual((rec.n_pending, rec.n_ready), (0, 0))
        limit[0] = 900
        r2, _ = w.collect(weights, 900)
        self.assertGreaterEqual(len(r2), 900)
        self.assertGreater(int(r2.hand_id.min()), int(r1.hand_id.max()))
        self.assertFalse(w.batch.any_in_hand())
        self.assertEqual((rec.n_pending, rec.n_ready), (0, 0))

    def test_one_decision_on_eight_tables(self):
        w = helpers.small_worker(6, n_tables=8)
        rec = w.recorder
        original = w.batch.start_hand
        started = set()

        def start_hand(table, spec, controllers):
            recorded = rec.n_ready + rec.n_pending
            assert recorded < 1, recorded
            hand_id = original(table, spec, controllers)
            started.add(hand_id)
            return hand_id

        w.batch.start_hand = start_hand
        recs, stats = w.collect(helpers.small_weights(5), 1)
        self.assertGreaterEqual(len(recs), 1)
        # every record comes from a hand started while no learner decision existed
        self.assertTrue(set(recs.hand_id.tolist()) <= started)
        self.assertGreaterEqual(len(started), 8)
        self.assertEqual(stats.overall.hands, len(started))

    def test_one_learner_call_per_step(self):
        w = helpers.small_worker(7)
        policy, batch = w.policy, w.batch
        calls = []  # rows per learner call
        log = []  # (learner calls, rows, pending learner decisions) per step
        orig_act = policy.act
        orig_step = batch.step

        def act(b):
            calls.append(len(b))
            return orig_act(b)

        def step(controllers):
            idle = set(batch.idle_tables())
            pending = 0
            for t, tbl in enumerate(batch.tables):
                if t in idle:
                    continue
                d = tbl.decision
                if d is not None and batch.controllers_of(t)[d.seat] == LEARNER_KEY:
                    pending += 1
            before = len(calls)
            out = orig_step(controllers)
            log.append((len(calls) - before, sum(calls[before:]), pending))
            return out

        policy.act = act
        batch.step = step
        w.collect(helpers.small_weights(5), 1000)
        self.assertTrue(log)
        for n_calls, rows, pending in log:
            self.assertLessEqual(n_calls, 1)
            self.assertEqual(rows, pending)
        self.assertGreater(max(entry[1] for entry in log), 1)  # really batched


class RolloutWeights(unittest.TestCase):
    def test_switch_weights(self):
        a = helpers.small_weights(10, peaked=True)
        b = helpers.small_weights(11, peaked=True)
        w = helpers.small_worker(8)
        w.collect(a, 600)
        recs, _ = w.collect(b, 600)
        want_b = helpers.recorded_logp_under(helpers.net_from_weights(b), recs)
        want_a = helpers.recorded_logp_under(helpers.net_from_weights(a), recs)
        np.testing.assert_allclose(recs.logp, want_b, atol=1e-5, rtol=0)
        self.assertGreater(float(np.abs(recs.logp - want_a).max()), 0.1)
        sd = w.policy.state_dict()
        self.assertEqual(sorted(sd), sorted(b))
        for k in b:
            if k == "_extra_state":
                self.assertEqual(sd[k], b[k])
            else:
                self.assertTrue(torch.equal(sd[k], b[k]), k)

    def test_mismatched_weights(self):
        good = helpers.small_weights(12)
        with torch.random.fork_rng(devices=[]):
            other_hidden = PolicyNet(hidden=32, layers=2).state_dict()
            critic = CriticNet(hidden=64, layers=2).state_dict()
        altered = dict(good)
        altered["_extra_state"] = dict(good["_extra_state"], encoder_version=-7)
        missing = {k: v for k, v in good.items() if k != "head.bias"}
        w = helpers.small_worker(9)
        before = param_snapshot(w)
        for bad in (other_hidden, critic, altered, missing):
            with self.assertRaises(NetConfigError):
                w.collect(bad, 500)
            self.assertEqual(w.batch.n_hands, 0)
            self.assertEqual(w.batch.n_decisions, 0)
            self.assertFalse(w.batch.any_in_hand())
            self.assertEqual((w.recorder.n_pending, w.recorder.n_ready), (0, 0))
            after = param_snapshot(w)
            for k in before:
                self.assertTrue(torch.equal(before[k], after[k]), k)
        # no generator was advanced: same records as a fresh worker
        r1, _ = w.collect(good, 500)
        r2, _ = helpers.small_worker(9).collect(good, 500)
        records_equal(self, r1, r2)

    def test_invalid_n_decisions(self):
        w = helpers.small_worker(9)
        cases = ((0, ValueError), (-3, ValueError), (1.5, TypeError), (True, TypeError), ("9", TypeError))
        for n, exc in cases:
            with self.assertRaises(exc) as cm:
                w.collect({}, n)  # bad weights too: n is checked first
            self.assertNotIsInstance(cm.exception, NetConfigError, n)
        self.assertEqual(w.batch.n_hands, 0)


class RolloutDeterminism(unittest.TestCase):
    def test_same_seed_same_everything(self):
        weights = helpers.small_weights(20, peaked=True)
        w1, w2 = helpers.small_worker(30), helpers.small_worker(30)
        for n in (800, 700):
            r1, s1, h1 = collect_keep(w1, weights, n)
            r2, s2, h2 = collect_keep(w2, weights, n)
            records_equal(self, r1, r2)
            self.assertTrue(stats_equal_but_time(s1, s2))
            self.assertEqual([fh.spec for fh in h1], [fh.spec for fh in h2])
            self.assertEqual([fh.controllers for fh in h1], [fh.controllers for fh in h2])

    def test_other_seed_differs(self):
        weights = helpers.small_weights(20, peaked=True)
        r3, _, h3 = collect_keep(helpers.small_worker(31), weights, 800)
        r4, _, h4 = collect_keep(helpers.small_worker(30), weights, 800)
        self.assertNotEqual([fh.spec for fh in h3], [fh.spec for fh in h4])
        self.assertFalse(len(r3) == len(r4) and np.array_equal(r3.obs, r4.obs))


if __name__ == "__main__":
    unittest.main()
