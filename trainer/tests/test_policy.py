"""NeuralPolicy: the rl_env Controller over a PolicyNet - protocol, what it
reads, gradient-free acting on a private copy, its own generator, sampling
frequencies, weight loading, and play at real tables."""

import math
import os
import tempfile
import unittest

import numpy as np
import torch
from torch.distributions import Categorical

from rl_env import (
    CHECK_CALL,
    ENCODER_VERSION,
    N_ACTIONS,
    OBS_DIM,
    PRIV_DIM,
    DecisionBatch,
    HandSampler,
    RandomController,
    Recorder,
    TableBatch,
)
from trainer import CriticNet, NetConfigError, NeuralPolicy, PolicyNet, critic_input, log_probs

try:
    from . import helpers
except ImportError:  # run as a top-level module
    import helpers  # type: ignore[no-redef]


class Poison:
    """Raises on any access: proves a field is never read."""

    def _fail(self, *_args, **_kw):
        raise AssertionError("NeuralPolicy read a field it must not read")

    def __getattr__(self, name):
        self._fail()

    __getitem__ = _fail
    __len__ = _fail
    __iter__ = _fail
    __bool__ = _fail
    __array__ = _fail


def snapshot(net):
    return {k: v.detach().clone() for k, v in net.state_dict().items() if k != "_extra_state"}


def same_params(test, a: dict, b: dict):
    test.assertEqual(sorted(a), sorted(b))
    for k in a:
        test.assertTrue(torch.equal(a[k], b[k]), k)


def logp_under(net, batch, actions) -> np.ndarray:
    """log_probs(net(obs, mask)) at `actions` - the policy's own computation."""
    with torch.inference_mode():
        lp = log_probs(net(torch.from_numpy(batch.obs), torch.from_numpy(batch.mask)))
        return lp.gather(1, torch.from_numpy(actions)[:, None]).squeeze(1).numpy().copy()


def all_nn(spec, rng):
    return ["nn"] * spec.n_seats


def mixed(spec, rng):
    keys = ["nn" if x else "rnd" for x in rng.random(spec.n_seats) < 0.5]
    if "nn" not in keys:
        keys[int(rng.integers(spec.n_seats))] = "nn"
    elif "rnd" not in keys:
        keys[int(rng.integers(spec.n_seats))] = "rnd"
    return keys


class PolicyTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recs = helpers.records()

    def setUp(self):
        self._threads = torch.get_num_threads()

    def tearDown(self):
        torch.set_num_threads(self._threads)

    def rows(self, n, start=0, key="nn"):
        r = self.recs
        return helpers.decision_batch(r.obs[start : start + n], r.mask[start : start + n], key)

    def all_masks(self, n=16):
        return helpers.decision_batch(*helpers.all_mask_batch(self.recs.obs[:n]))

    def assert_valid(self, batch, out):
        self.assertIsInstance(out, tuple)
        actions, logp = out
        b = batch.mask.shape[0]
        self.assertIsInstance(actions, np.ndarray)
        self.assertIsInstance(logp, np.ndarray)
        self.assertEqual(actions.dtype, np.int64)
        self.assertEqual(logp.dtype, np.float32)
        self.assertEqual(actions.shape, (b,))
        self.assertEqual(logp.shape, (b,))
        if b:
            self.assertTrue(np.all((actions >= 0) & (actions < N_ACTIONS)))
            self.assertTrue(batch.mask[np.arange(b), actions].all())
            self.assertTrue(np.isfinite(logp).all())
            self.assertTrue(np.all(logp <= 0))


class UnitTest(PolicyTestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.net = helpers.small_policy_net(1)

    def test_protocol(self):
        self.assertIs(NeuralPolicy.wants_views, False)
        self.assertIs(NeuralPolicy.wants_obs, True)
        p = NeuralPolicy(self.net, seed=0)
        self.assertIs(p.wants_views, False)
        self.assertIs(p.wants_obs, True)
        for b in (1, 7, 512):
            self.assert_valid(self.rows(b), p.act(self.rows(b)))
        batch = self.all_masks()
        self.assert_valid(batch, p.act(batch))

    def test_default_size_net(self):
        torch.manual_seed(2)
        p = NeuralPolicy(PolicyNet(), seed=0)
        batch = self.rows(64)
        self.assert_valid(batch, p.act(batch))

    def test_single(self):
        tb = TableBatch(8)
        sampler = HandSampler(1)
        for t in range(8):
            spec = sampler.sample()
            tb.start_hand(t, spec, ["a"] * spec.n_seats)
        p = NeuralPolicy(self.net, seed=0)
        n = 0
        for tbl in tb.tables:
            if not tbl.decision_pending:
                continue
            batch = DecisionBatch.single(tbl.view(), tbl.legal)
            out = p.act(batch)
            self.assert_valid(batch, out)
            np.testing.assert_array_equal(out[1], logp_under(p.net, batch, out[0]))
            n += 1
        self.assertGreater(n, 0)

    def test_empty_batch(self):
        empty = helpers.decision_batch(
            np.zeros((0, OBS_DIM), dtype=np.float32), np.zeros((0, N_ACTIONS), dtype=bool)
        )
        self.assertEqual(len(empty), 0)
        p = NeuralPolicy(self.net, seed=9)
        out = p.act(empty)
        self.assert_valid(empty, out)
        # The generator was not used: the next draw equals a fresh policy's.
        batch = self.rows(64)
        a, lp = p.act(batch)
        fa, flp = NeuralPolicy(self.net, seed=9).act(batch)
        np.testing.assert_array_equal(a, fa)
        np.testing.assert_array_equal(lp, flp)

    def test_logp_matches_network(self):
        p = NeuralPolicy(self.net, seed=3)
        for batch in (self.rows(512), self.all_masks()):
            actions, logp = p.act(batch)
            np.testing.assert_array_equal(logp, logp_under(p.net, batch, actions))
            with torch.no_grad():
                logits = self.net(torch.from_numpy(batch.obs), torch.from_numpy(batch.mask))
                cat = Categorical(logits=logits).log_prob(torch.from_numpy(actions)).numpy()
            np.testing.assert_allclose(logp, cat, atol=1e-6, rtol=0)
        # The single-legal rows of the all-mask batch: probability 1.
        batch = self.all_masks()
        actions, logp = p.act(batch)
        single = (batch.mask == helpers.SINGLE_LEGAL).all(1)
        self.assertTrue(np.all(actions[single] == CHECK_CALL))
        np.testing.assert_allclose(logp[single], 0.0, atol=1e-6)

    def test_reads_only_obs_and_mask(self):
        batch = self.rows(64)
        poisoned = DecisionBatch(
            key="nn",
            tables=Poison(),
            seats=Poison(),
            obs=batch.obs,
            mask=batch.mask,
            legal=Poison(),
            views=Poison(),
        )
        a, lp = NeuralPolicy(self.net, seed=4).act(poisoned)
        ea, elp = NeuralPolicy(self.net, seed=4).act(batch)
        np.testing.assert_array_equal(a, ea)
        np.testing.assert_array_equal(lp, elp)

    def test_refuses_missing_or_privileged_obs(self):
        p = NeuralPolicy(self.net, seed=0)
        no_obs = helpers.decision_batch(None, self.recs.mask[:4])
        with self.assertRaises(ValueError):
            p.act(no_obs)
        r = self.recs
        x = critic_input(np.ascontiguousarray(r.obs[:4]), r.critic[:4])
        self.assertEqual(x.shape[1], OBS_DIM + PRIV_DIM)
        with self.assertRaises(ValueError):
            p.act(helpers.decision_batch(x, r.mask[:4]))

    def test_no_gradients_eval_mode(self):
        p = NeuralPolicy(self.net, seed=0)
        self.assertFalse(p.net.training)
        self.assertFalse(any(q.requires_grad for q in p.net.parameters()))
        seen = {}

        def hook(_mod, _inp, out):
            seen["requires_grad"] = out.requires_grad
            seen["inference"] = torch.is_inference_mode_enabled()

        handle = p.net.register_forward_hook(hook)
        try:
            with torch.enable_grad():
                p.act(self.rows(32))
        finally:
            handle.remove()
        self.assertEqual(seen, {"requires_grad": False, "inference": True})
        # The source network is left as it was.
        self.assertTrue(all(q.requires_grad for q in self.net.parameters()))

    def test_private_copy(self):
        net = helpers.small_policy_net(5)
        p = NeuralPolicy(net, seed=1)
        self.assertIsNot(p.net, net)
        own = {q.data_ptr() for q in p.net.parameters()}
        self.assertFalse(own & {q.data_ptr() for q in net.parameters()})
        batch = self.rows(256)
        a1, l1 = p.act(batch)
        before = snapshot(net)
        obs, mask = torch.from_numpy(batch.obs), torch.from_numpy(batch.mask)
        opt = torch.optim.SGD(net.parameters(), lr=0.5)
        for _ in range(3):
            opt.zero_grad()
            loss = -log_probs(net(obs, mask))[:, CHECK_CALL].sum()
            loss.backward()
            opt.step()
        self.assertFalse(torch.equal(before["head.weight"], net.head.weight.detach()))
        p.seed(1)
        a2, l2 = p.act(batch)
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(l1, l2)

    def test_threads(self):
        NeuralPolicy(self.net, seed=0)
        self.assertEqual(torch.get_num_threads(), 1)
        NeuralPolicy(self.net, seed=0, num_threads=2)
        self.assertEqual(torch.get_num_threads(), 2)
        NeuralPolicy(self.net, seed=0, num_threads=None)
        self.assertEqual(torch.get_num_threads(), 2)
        NeuralPolicy.from_state_dict(self.net.state_dict(), seed=0)
        self.assertEqual(torch.get_num_threads(), 1)

    def test_argument_checks(self):
        with self.assertRaises(TypeError):
            NeuralPolicy(CriticNet(hidden=8, layers=1), seed=0)
        with self.assertRaises(TypeError):
            NeuralPolicy(self.net.state_dict(), seed=0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            NeuralPolicy(self.net, seed=-1)
        with self.assertRaises(TypeError):
            NeuralPolicy(self.net, seed=1.5)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            NeuralPolicy(self.net, seed=True)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            NeuralPolicy(self.net)  # type: ignore[call-arg]
        with self.assertRaises(ValueError):
            NeuralPolicy(self.net, seed=0, num_threads=0)
        p = NeuralPolicy(self.net, seed=0)
        with self.assertRaises(ValueError):
            p.seed(-3)

    def test_global_rng_untouched(self):
        batch = self.rows(64)
        torch.manual_seed(123)
        state = torch.random.get_rng_state()
        p1 = NeuralPolicy(self.net, seed=0)
        p2 = NeuralPolicy.from_state_dict(self.net.state_dict(), seed=0)
        for _ in range(20):
            p1.act(batch)
            p2.act(batch)
        p1.seed(5)
        p1.load_state_dict(p2.state_dict())
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))
        # The global seed does not change the policy's actions.
        torch.manual_seed(1)
        a1, l1 = NeuralPolicy(self.net, seed=3).act(batch)
        torch.manual_seed(2)
        a2, l2 = NeuralPolicy(self.net, seed=3).act(batch)
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(l1, l2)

    def test_reproducible(self):
        batches = [self.rows(7), self.rows(512, 7), self.rows(1, 600), self.all_masks()]

        def run(p):
            return [p.act(b) for b in batches]

        p, q = NeuralPolicy(self.net, seed=21), NeuralPolicy(self.net, seed=21)
        first = run(p)
        for (a, la), (b, lb) in zip(first, run(q)):
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(la, lb)
        p.seed(21)
        for (a, la), (b, lb) in zip(first, run(p)):
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(la, lb)
        other = NeuralPolicy(self.net, seed=22).act(batches[1])[0]
        self.assertFalse(np.array_equal(first[1][0], other))


class FrequencyTest(PolicyTestBase):
    N_CALLS = 5000
    Z_1E6 = 4.753424  # upper 1e-6 quantile of the standard normal

    def test_frequencies_match_probabilities(self):
        r = self.recs
        idx = helpers.rows_with_masks(r, 12, per_mask=4)
        obs4, mask4 = helpers.all_mask_batch(r.obs[idx[:1]])  # one row per ALL_MASKS entry
        obs = np.concatenate([r.obs[idx], obs4])
        mask = np.concatenate([r.mask[idx], mask4])
        batch = helpers.decision_batch(obs, mask)
        self.assertEqual(len(batch), 16)
        self.assertEqual(helpers.mask_rows(mask), helpers.mask_rows(helpers.ALL_MASKS))

        net = helpers.peaked_policy_net(7)
        with torch.no_grad():
            p = torch.softmax(
                net(torch.from_numpy(batch.obs), torch.from_numpy(batch.mask)), -1
            ).double().numpy()
        policy = NeuralPolicy(net, seed=8)
        counts = np.zeros((16, N_ACTIONS), dtype=np.int64)
        rows = np.arange(16)
        for _ in range(self.N_CALLS):
            actions, _ = policy.act(batch)
            counts[rows, actions] += 1
        n = self.N_CALLS

        self.assertTrue(np.all(counts[~mask] == 0), "illegal action sampled")
        freq = counts / n
        # 5 standard deviations (+1 draw for the discreteness at tiny p).
        tol = 5.0 * np.sqrt(p * (1.0 - p) / n) + 1.0 / n
        bad = np.abs(freq - p) > tol
        self.assertFalse(bad.any(), f"freq {freq[bad]} vs p {p[bad]}")

        # Chi-square over the rows whose legal cells all expect >= 5 draws.
        expected = n * p
        chi2, df = 0.0, 0
        for i in range(16):
            legal = mask[i]
            k = int(legal.sum())
            if k < 2 or np.any(expected[i, legal] < 5):
                continue
            chi2 += float(((counts[i, legal] - expected[i, legal]) ** 2 / expected[i, legal]).sum())
            df += k - 1
        self.assertGreater(df, 0)
        # Wilson-Hilferty approximation of the chi-square 1e-6 upper quantile.
        h = 2.0 / (9.0 * df)
        limit = df * (1.0 - h + self.Z_1E6 * math.sqrt(h)) ** 3
        self.assertLess(chi2, limit)


class WeightsTest(PolicyTestBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.a = helpers.small_policy_net(40)
        cls.b = helpers.small_policy_net(41)

    def check_acts_like(self, policy, net):
        batch = self.rows(256)
        actions, logp = policy.act(batch)
        self.assert_valid(batch, (actions, logp))
        np.testing.assert_array_equal(logp, logp_under(net, batch, actions))

    def test_load_state_dict_sources(self):
        p = NeuralPolicy(self.a, seed=0)
        p.load_state_dict(self.b.state_dict())
        self.check_acts_like(p, self.b)

        q = NeuralPolicy(self.b, seed=5)
        p2 = NeuralPolicy(self.a, seed=0)
        p2.load_state_dict(q.state_dict())
        self.check_acts_like(p2, self.b)

        p3 = NeuralPolicy(self.a, seed=0)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "policy.pt")
            torch.save(self.b.state_dict(), path)
            state = torch.load(path, weights_only=True)
        p3.load_state_dict(state)
        self.check_acts_like(p3, self.b)

    def test_from_state_dict_like_constructor(self):
        batches = [self.rows(512), self.all_masks(), self.rows(3, 900)]
        p = NeuralPolicy.from_state_dict(self.b.state_dict(), seed=17)
        q = NeuralPolicy(self.b, seed=17)
        self.assertEqual(p.net.hidden, self.b.hidden)
        self.assertEqual(p.net.layers, self.b.layers)
        self.assertFalse(p.net.training)
        self.assertFalse(any(x.requires_grad for x in p.net.parameters()))
        for batch in batches:
            a1, l1 = p.act(batch)
            a2, l2 = q.act(batch)
            np.testing.assert_array_equal(a1, a2)
            np.testing.assert_array_equal(l1, l2)

    def test_load_replaces_completely(self):
        b = helpers.small_policy_net(41)
        p = NeuralPolicy(self.a, seed=0)
        p.load_state_dict(b.state_dict())
        loaded = snapshot(p.net)
        same_params(self, loaded, snapshot(b))
        # No tensor of the state dict is aliased: training b does not change p.
        own = {x.data_ptr() for x in p.net.parameters()}
        self.assertFalse(own & {x.data_ptr() for x in b.parameters()})
        batch = self.rows(128)
        opt = torch.optim.SGD(b.parameters(), lr=0.5)
        opt.zero_grad()
        loss = -log_probs(b(torch.from_numpy(batch.obs), torch.from_numpy(batch.mask)))[:, 0].sum()
        loss.backward()
        opt.step()
        self.assertFalse(torch.equal(loaded["head.weight"], b.head.weight.detach()))
        same_params(self, loaded, snapshot(p.net))

    def test_refused_state_dicts_leave_weights(self):
        p = NeuralPolicy(self.a, seed=0)
        batch = self.rows(128)
        before = snapshot(p.net)
        a0, l0 = p.act(batch)

        altered = self.b.state_dict()
        altered["_extra_state"] = dict(altered["_extra_state"], encoder_version=ENCODER_VERSION + 1)
        missing = self.b.state_dict()
        del missing["head.bias"]
        bad = {
            "hidden": helpers.small_policy_net(42, hidden=32).state_dict(),
            "layers": helpers.small_policy_net(43, layers=3).state_dict(),
            "critic": CriticNet(hidden=64, layers=2).state_dict(),
            "encoder_version": altered,
            "missing key": missing,
        }
        for name, state in bad.items():
            with self.assertRaises(NetConfigError, msg=name):
                p.load_state_dict(state)
            same_params(self, before, snapshot(p.net))
            p.seed(0)
            a, lp = p.act(batch)
            np.testing.assert_array_equal(a, a0)
            np.testing.assert_array_equal(lp, l0)

        with self.assertRaises(NetConfigError):
            NeuralPolicy.from_state_dict(altered, seed=0)
        with self.assertRaises(NetConfigError):
            NeuralPolicy.from_state_dict(CriticNet(hidden=64, layers=2).state_dict(), seed=0)
        with self.assertRaises(NetConfigError):
            NeuralPolicy.from_state_dict(missing, seed=0)
        no_config = self.b.state_dict()
        del no_config["_extra_state"]
        with self.assertRaises(NetConfigError):
            NeuralPolicy.from_state_dict(no_config, seed=0)


class TableTest(PolicyTestBase):
    N_HANDS = 300

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.net = helpers.small_policy_net(50)

    def check_run(self, finished, log, rec, policy, recorder_key="nn"):
        self.assertEqual(len(finished), self.N_HANDS)
        self.assertEqual(len({fh.hand_id for fh in finished}), self.N_HANDS)
        for fh in finished:
            self.assertAlmostEqual(float(fh.net_bb.sum()), 0.0, places=6)
        recs = rec.drain()
        self.assertGreater(len(recs), 0)
        n = len(recs)
        self.assertTrue(recs.mask[np.arange(n), recs.action].all())
        code = rec.codes[recorder_key]
        nn_rows = recs.controller == code
        self.assertTrue(nn_rows.any())
        self.assertFalse(np.isnan(recs.logp[nn_rows]).any())
        # The recorded rows are exactly the policy's calls, in decision order.
        nn_log = [entry for entry in log if entry[0] == "nn"]
        np.testing.assert_array_equal(
            recs.action[nn_rows], np.concatenate([e[1] for e in nn_log])
        )
        np.testing.assert_array_equal(recs.logp[nn_rows], np.concatenate([e[2] for e in nn_log]))
        # Recomputed in one large batch: equal up to rounding.
        obs = np.ascontiguousarray(recs.obs[nn_rows])
        mask = np.ascontiguousarray(recs.mask[nn_rows])
        with torch.inference_mode():
            lp = log_probs(policy.net(torch.from_numpy(obs), torch.from_numpy(mask)))
            acts = torch.from_numpy(np.ascontiguousarray(recs.action[nn_rows]))
            expected = lp.gather(1, acts[:, None]).squeeze(1).numpy()
        np.testing.assert_allclose(recs.logp[nn_rows], expected, atol=1e-5, rtol=0)
        return recs

    def test_view_counter_counts(self):
        # The counter really sees views built by the batch.
        class WantsViews:
            wants_views = True

        tb = TableBatch(4)
        sampler = HandSampler(2)
        for t in range(4):
            spec = sampler.sample()
            tb.start_hand(t, spec, ["v"] * spec.n_seats)
        with helpers.count_views() as views:
            batches = tb.pending({"v": WantsViews()})
        self.assertEqual(views.n, sum(len(b) for b in batches))
        self.assertGreater(views.n, 0)

    def test_all_seats(self):
        policy = NeuralPolicy(self.net, seed=11)
        rec = Recorder()
        with helpers.count_views() as views:
            finished, log = helpers.play(
                {"nn": policy}, all_nn, self.N_HANDS, seed=12, recorder=rec
            )
        self.assertEqual(views.n, 0)
        self.assertEqual({fh.spec.n_seats for fh in finished}, set(range(2, 11)))
        recs = self.check_run(finished, log, rec, policy)
        self.assertTrue(np.all(recs.controller == rec.codes["nn"]))
        acts = np.concatenate([e[1] for e in log])
        self.assertTrue(np.all(np.bincount(acts, minlength=N_ACTIONS) > 0))

    def test_mixed(self):
        policy = NeuralPolicy(self.net, seed=13)
        rec = Recorder(keys={"nn"})
        controllers = {"nn": policy, "rnd": RandomController(14)}
        with helpers.count_views() as views:
            finished, log = helpers.play(controllers, mixed, self.N_HANDS, seed=15, recorder=rec)
        self.assertEqual(views.n, 0)
        for fh in finished:
            self.assertIn("nn", fh.controllers)
            self.assertIn("rnd", fh.controllers)
        self.assertTrue(any(e[0] == "rnd" for e in log))
        recs = self.check_run(finished, log, rec, policy)
        self.assertTrue(np.all(recs.controller == rec.codes["nn"]))

    def test_determinism(self):
        def run(policy_seed):
            policy = NeuralPolicy(self.net, seed=policy_seed)
            return helpers.play({"nn": policy}, all_nn, 150, seed=16)

        f1, log1 = run(7)
        f2, log2 = run(7)
        self.assertEqual(len(log1), len(log2))
        for (k1, a1, l1), (k2, a2, l2) in zip(log1, log2):
            self.assertEqual(k1, k2)
            np.testing.assert_array_equal(a1, a2)
            np.testing.assert_array_equal(l1, l2)
        self.assertEqual([fh.hand_id for fh in f1], [fh.hand_id for fh in f2])
        for x, y in zip(f1, f2):
            np.testing.assert_array_equal(x.net_bb, y.net_bb)

        _, log3 = run(8)
        a1 = np.concatenate([e[1] for e in log1])
        a3 = np.concatenate([e[1] for e in log3])
        self.assertFalse(len(a1) == len(a3) and np.array_equal(a1, a3))


if __name__ == "__main__":
    unittest.main()
