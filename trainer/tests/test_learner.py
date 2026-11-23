"""Learner: configuration, advantages computed once before the update, the
update itself (all parameters move, records untouched), steps and minibatch
sizes, gradient clipping, statistics, refusals that change nothing, weights for
collection, torch thread handling and determinism."""

import copy
import dataclasses
import math
import os
import tempfile
import unittest

import numpy as np
import torch

import trainer.ppo as ppo_mod
from rl_env import Recorder, TrainingRecords
from trainer import (
    Learner,
    NeuralPolicy,
    PPOConfig,
    UpdateStats,
    critic_input,
    log_probs,
)

try:
    from . import helpers
except ImportError:  # run as a top-level module
    import helpers  # type: ignore[no-redef]


# --- comparison helpers ------------------------------------------------------------


def params(learner):
    return [p.detach().clone() for p in (*learner.policy.parameters(), *learner.critic.parameters())]


def same_tree(a, b) -> bool:
    """Deep equality of state dicts (tensors with torch.equal)."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and torch.equal(a, b)
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return list(a) == list(b) and all(same_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(same_tree(x, y) for x, y in zip(a, b))
    return a == b


def records_copy(r: TrainingRecords) -> dict:
    return {f.name: getattr(r, f.name).copy() for f in dataclasses.fields(TrainingRecords)}


def no_time(stats: UpdateStats) -> UpdateStats:
    return dataclasses.replace(stats, wall_time=0.0)


class WrapLoss:
    """Replaces trainer.ppo.ppo_loss (the module global the Learner calls) with
    a recording wrapper."""

    def __init__(self, on_call=None):
        self.calls = []
        self.on_call = on_call

    def __enter__(self):
        self.original = ppo_mod.ppo_loss

        def wrapper(policy, critic, batch, adv, idx, cfg):
            self.calls.append(
                {"adv": adv, "idx": idx.clone(), "threads": torch.get_num_threads(), "batch": batch}
            )
            if self.on_call is not None:
                self.on_call(len(self.calls))
            return self.original(policy, critic, batch, adv, idx, cfg)

        ppo_mod.ppo_loss = wrapper
        return self

    def __exit__(self, *exc):
        ppo_mod.ppo_loss = self.original
        return False


def global_grad_norm(learner) -> float:
    norms = [
        torch.linalg.vector_norm(p.grad.detach())
        for p in (*learner.policy.parameters(), *learner.critic.parameters())
        if p.grad is not None
    ]
    return float(torch.linalg.vector_norm(torch.stack(norms)))


class LearnerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        worker = helpers.small_worker(7)
        w = helpers.small_learner(7).get_weights()
        cls.r1, _ = worker.collect(w, 2500)
        cls.r2, _ = worker.collect(w, 2000)

    # --- construction ----------------------------------------------------------

    def test_config_validation(self):
        bad = {
            "lr": [0.0, -1e-3, math.nan, math.inf, True, "1"],
            "clip": [0.0, 1.0, -0.1, math.nan],
            "epochs": [0, -1, True, 1.5],
            "minibatch": [0, False, 2.0],
            "vf_coef": [-0.1, math.inf],
            "ent_coef": [-0.1, math.nan],
            "max_grad_norm": [0.0, -1.0],
            "hidden": [0, True, 64.0],
            "layers": [0, False],
            "adam_eps": [0.0, -1e-5],
            "adv_eps": [0.0, -1e-8],
        }
        for name, values in bad.items():
            for v in values:
                with self.assertRaises((ValueError, TypeError), msg=f"{name}={v!r}"):
                    PPOConfig(**{name: v})

    def test_defaults(self):
        cfg = PPOConfig()
        self.assertEqual(
            (cfg.lr, cfg.clip, cfg.epochs, cfg.minibatch, cfg.vf_coef, cfg.ent_coef, cfg.max_grad_norm),
            (3e-4, 0.2, 3, 4096, 0.5, 0.01, 1.0),
        )
        self.assertEqual((cfg.hidden, cfg.layers), (512, 3))
        for f in dataclasses.fields(cfg):
            self.assertIn(type(getattr(cfg, f.name)), (int, float), f.name)
        learner = helpers.small_learner(0, hidden=32, layers=1)
        for net in (learner.policy, learner.critic):
            self.assertEqual((net.hidden, net.layers), (32, 1))
        self.assertEqual(learner.n_updates, 0)
        self.assertEqual(learner.seed, 0)
        with self.assertRaises((ValueError, TypeError)):
            Learner(helpers.small_config(), seed=-1)
        with self.assertRaises(TypeError):
            Learner(helpers.small_config(), seed=True)
        with self.assertRaises(ValueError):
            Learner(helpers.small_config(), seed=0, num_threads=0)

    def test_global_rng_untouched(self):
        torch.manual_seed(1234)
        before = torch.random.get_rng_state()
        learner = helpers.small_learner(3)
        learner.update(self.r1)
        learner.get_weights()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.pt")
            learner.save(path)
            Learner.load(path)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before))

    # --- advantages ------------------------------------------------------------

    def test_advantages_once_before_update(self):
        learner = helpers.small_learner(1)
        critic_before = copy.deepcopy(learner.critic)
        with WrapLoss() as w:
            learner.update(self.r1)
        self.assertGreater(len(w.calls), 1)
        adv = w.calls[0]["adv"]
        for c in w.calls:
            self.assertIs(c["adv"], adv)
        self.assertFalse(adv.requires_grad)
        batch = w.calls[0]["batch"]
        with torch.no_grad():
            expected = batch.reward - critic_before(batch.critic_x)
        self.assertTrue(torch.allclose(adv, expected, atol=1e-6, rtol=0))

    # --- the update ------------------------------------------------------------

    def test_update_moves_every_parameter_and_leaves_records(self):
        learner = helpers.small_learner(2)
        before = params(learner)
        saved = records_copy(self.r1)
        learner.update(self.r1)
        after = params(learner)
        for b, a in zip(before, after):
            self.assertFalse(torch.equal(a, b))
            self.assertTrue(torch.isfinite(a).all())
        for name, arr in saved.items():
            self.assertTrue(np.array_equal(getattr(self.r1, name), arr, equal_nan=True), name)
        # not aliased: the learner's tensors share no memory with the records
        sd = learner.policy.state_dict()
        for name in ("obs", "mask", "action", "logp", "critic", "reward"):
            arr = getattr(self.r1, name)
            for t in sd.values():
                if isinstance(t, torch.Tensor):
                    self.assertFalse(np.shares_memory(arr, t.numpy()), name)

    def test_steps_and_counts(self):
        n = len(self.r1)
        for mb in (256, 300 if n % 300 else 299):
            learner = helpers.small_learner(4, minibatch=mb)
            with WrapLoss() as w:
                stats = learner.update(self.r1)
            epochs = learner.config.epochs
            per_epoch = math.ceil(n / mb)
            self.assertEqual(stats.decisions, n)
            self.assertEqual(stats.steps, epochs * per_epoch)
            self.assertEqual(len(w.calls), stats.steps)
            sizes = [len(c["idx"]) for c in w.calls]
            expected = [mb] * (n // mb) + ([n % mb] if n % mb else [])
            self.assertEqual(sizes, expected * epochs)
            for e in range(epochs):
                rows = torch.cat([c["idx"] for c in w.calls[e * per_epoch : (e + 1) * per_epoch]])
                self.assertTrue(torch.equal(rows.sort().values, torch.arange(n)))
            self.assertEqual(stats.update, 1)
            self.assertEqual(learner.n_updates, 1)
        learner = helpers.small_learner(4)
        stats = learner.update(self.r1, self.r2)
        self.assertEqual(stats.decisions, len(self.r1) + len(self.r2))

    # --- gradient clipping -----------------------------------------------------

    def step_norms(self, learner):
        opt = learner.optimizer
        original = opt.step
        norms = []

        def step(*args, **kw):
            norms.append(global_grad_norm(learner))
            return original(*args, **kw)

        opt.step = step
        return norms

    def test_gradient_clipping(self):
        learner = helpers.small_learner(5, max_grad_norm=1e-3)
        norms = self.step_norms(learner)
        stats = learner.update(self.r1)
        self.assertEqual(len(norms), stats.steps)
        for g in norms:
            self.assertLessEqual(g, 1e-3 * (1 + 1e-5))
        self.assertGreater(stats.grad_norm, 1e-3)
        self.assertGreater(stats.grad_norm_max, 1e-3)

        learner = helpers.small_learner(5, max_grad_norm=1e9)
        norms = self.step_norms(learner)
        stats = learner.update(self.r1)
        per_epoch = stats.steps // learner.config.epochs
        self.assertAlmostEqual(stats.grad_norm_max, max(norms), delta=1e-5 * max(norms))
        last = float(np.mean(norms[-per_epoch:]))
        self.assertAlmostEqual(stats.grad_norm, last, delta=1e-5 * last)

    # --- statistics ------------------------------------------------------------

    def test_statistics(self):
        learner = helpers.small_learner(6)
        critic_before = copy.deepcopy(learner.critic)
        stats = learner.update(self.r1)
        for f in dataclasses.fields(stats):
            v = getattr(stats, f.name)
            self.assertTrue(math.isfinite(v), f.name)
        self.assertGreater(stats.wall_time, 0)
        self.assertGreaterEqual(stats.approx_kl, 0.0)
        self.assertGreaterEqual(stats.clip_frac, 0.0)
        self.assertLessEqual(stats.clip_frac, 1.0)
        self.assertGreater(stats.entropy, 0.0)
        self.assertLessEqual(stats.entropy, math.log(3) + 1e-6)
        x = torch.from_numpy(critic_input(self.r1.obs, self.r1.critic))
        with torch.no_grad():
            v = critic_before(x).double().numpy()
        r = self.r1.reward.astype(np.float64)
        ev = 1.0 - np.var(r - v) / np.var(r)
        self.assertAlmostEqual(stats.explained_variance, ev, delta=1e-5)
        self.assertAlmostEqual(stats.reward_mean, float(r.mean()), delta=1e-6)
        self.assertAlmostEqual(stats.value_mean, float(v.mean()), delta=1e-5)

    # --- refusals --------------------------------------------------------------

    def state_of(self, learner):
        return (
            params(learner),
            copy.deepcopy(learner.optimizer.state_dict()),
            learner._gen.get_state().clone(),
            learner.n_updates,
        )

    def assert_same_state(self, a, b):
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a[0], b[0])))
        self.assertTrue(same_tree(a[1], b[1]))
        self.assertTrue(torch.equal(a[2], b[2]))
        self.assertEqual(a[3], b[3])

    def test_refusals_change_nothing(self):
        learner = helpers.small_learner(8)
        learner.update(self.r2)  # non-empty Adam state
        r = self.r1
        illegal_row = int(np.flatnonzero(~r.mask.all(axis=1))[0])
        illegal_action = int(np.flatnonzero(~r.mask[illegal_row])[0])
        action = r.action.copy()
        action[illegal_row] = illegal_action
        cases = {
            "nan logp": (helpers.records(),),
            "narrow obs": (helpers.replace_records(r, obs=np.ascontiguousarray(r.obs[:, :-1])),),
            "empty": (Recorder().drain(),),
            "no argument": (),
            "not a record": ({"obs": r.obs},),
            "illegal action": (helpers.replace_records(r, action=action),),
            "one bad of two": (r, helpers.records()),
        }
        for name, args in cases.items():
            before = self.state_of(learner)
            with self.assertRaises((ValueError, TypeError), msg=name):
                learner.update(*args)
            self.assert_same_state(before, self.state_of(learner))
        with self.assertRaisesRegex(ValueError, "log-probability"):
            learner.update(helpers.records())
        with self.assertRaisesRegex(ValueError, "no decisions"):
            learner.update(Recorder().drain())

    # --- weights ---------------------------------------------------------------

    def test_weights_for_collection(self):
        learner = helpers.small_learner(9)
        learner.update(self.r1)
        w = learner.get_weights()
        policy = NeuralPolicy.from_state_dict(w, seed=3, num_threads=None)
        obs, mask = self.r2.obs[:500], self.r2.mask[:500]
        actions, logp = policy.act(helpers.decision_batch(obs, mask))
        with torch.no_grad():
            lp = log_probs(learner.policy(torch.from_numpy(obs), torch.from_numpy(mask)))
            ref = lp.gather(1, torch.from_numpy(actions).unsqueeze(1)).squeeze(1).numpy()
        self.assertTrue(np.allclose(logp, ref, atol=1e-6, rtol=0))

        worker = helpers.small_worker(11)
        recs, _ = worker.collect(w, 500)
        self.assertTrue(np.allclose(recs.logp, helpers.recorded_logp_under(learner.policy, recs), atol=1e-5, rtol=0))

        kept = copy.deepcopy(w)
        learner.update(self.r2)
        self.assertTrue(same_tree(w, kept))
        self.assertFalse(same_tree(w, learner.get_weights()))

        live = learner.policy.head.weight.detach().clone()
        w2 = learner.get_weights()
        w2["head.weight"].add_(1.0)
        self.assertTrue(torch.equal(learner.policy.head.weight.detach(), live))

    # --- threads ---------------------------------------------------------------

    def test_threads(self):
        original = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            learner = helpers.small_learner(10, num_threads=2)
            with WrapLoss() as w:
                learner.update(self.r1)
            self.assertTrue(all(c["threads"] == 2 for c in w.calls))
            self.assertEqual(torch.get_num_threads(), 1)

            def boom(i):
                raise RuntimeError("boom")

            with WrapLoss(on_call=boom):
                with self.assertRaises(RuntimeError):
                    learner.update(self.r1)
            self.assertEqual(torch.get_num_threads(), 1)

            learner = helpers.small_learner(10, num_threads=None)
            with WrapLoss() as w:
                learner.update(self.r1)
            self.assertTrue(all(c["threads"] == 1 for c in w.calls))
            self.assertEqual(torch.get_num_threads(), 1)
        finally:
            torch.set_num_threads(original)

    # --- determinism -----------------------------------------------------------

    def test_determinism(self):
        a, b = helpers.small_learner(12), helpers.small_learner(12)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(params(a), params(b))))
        sa = [no_time(a.update(self.r1)), no_time(a.update(self.r2))]
        sb = [no_time(b.update(self.r1)), no_time(b.update(self.r2))]
        self.assertEqual(sa, sb)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(params(a), params(b))))

        c = helpers.small_learner(13)
        d = helpers.small_learner(12)
        self.assertFalse(all(torch.equal(x, y) for x, y in zip(params(c), params(d))))
        self.assertNotEqual(no_time(c.update(self.r1)), sa[0])


if __name__ == "__main__":
    unittest.main()
