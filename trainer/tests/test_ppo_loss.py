"""ppo_loss: ratio 1 for an unchanged policy, finite values and gradients (also
with one legal action and one-row minibatches), entropy, the parts of the loss,
clipping, advantage normalisation and the privileged-input boundary."""

import dataclasses
import math
import unittest

import numpy as np
import torch

from rl_env import OBS_DIM, PRIV_DIM
from trainer import PolicyNet, compute_advantages, log_probs, make_batch, ppo_loss

try:
    from . import helpers
except ImportError:  # run as a top-level module
    import helpers  # type: ignore[no-redef]

TTT = np.array([True, True, True])
FTF = np.array([False, True, False])

# approx_kl = mean((r - 1) - log r) is computed in float32 next to 1.0 (ulp
# 1.2e-7), so for ratios within float rounding of 1 it is ~1e-8, not exactly 0.
KL_TOL = 1e-6


def rows_of(batch, mask_row):
    return np.flatnonzero((batch.mask.numpy() == mask_row).all(axis=1))


def all_finite_grads(test, nets):
    for net in nets:
        for name, p in net.named_parameters():
            if p.grad is not None:
                test.assertTrue(torch.isfinite(p.grad).all(), name)


class PolicyWrapper:
    """Asserts that the policy only ever receives OBS_DIM-wide inputs."""

    def __init__(self, test, net):
        self.test, self.net, self.calls = test, net, 0

    def __call__(self, obs, mask):
        self.calls += 1
        self.test.assertEqual(obs.shape[1], OBS_DIM)
        return self.net(obs, mask)


class CriticWrapper:
    def __init__(self, net):
        self.net, self.widths = net, []

    def __call__(self, x):
        self.widths.append(int(x.shape[1]))
        return self.net(x)


class PPOLossTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.learner = helpers.small_learner(5)
        cls.cfg = cls.learner.config
        worker = helpers.small_worker(5)
        recs, _ = worker.collect(cls.learner.get_weights(), 2000)
        real = helpers.records()
        obs, mask = helpers.all_mask_batch(real.obs[:64])
        critic = np.repeat(real.critic[:64], len(helpers.ALL_MASKS), axis=0)
        rewards = np.random.default_rng(0).normal(size=len(obs))
        acted = helpers.acted_records(
            cls.learner.get_weights(), obs, mask, critic, lambda a, r: rewards[r], seed=1
        )
        cls.batch = make_batch(recs, acted)
        cls.n = len(cls.batch)
        cls.adv, cls.values = compute_advantages(cls.learner.critic, cls.batch)

    def setUp(self):
        self.learner.optimizer.zero_grad(set_to_none=True)

    def tearDown(self):
        self.learner.optimizer.zero_grad(set_to_none=True)

    def loss(self, idx, cfg=None, batch=None, adv=None):
        return ppo_loss(
            self.learner.policy,
            self.learner.critic,
            self.batch if batch is None else batch,
            self.adv if adv is None else adv,
            idx,
            self.cfg if cfg is None else cfg,
        )

    # --- unchanged policy ------------------------------------------------------

    def test_unchanged_policy_ratio_one(self):
        idxs = [torch.arange(self.n)]
        perm = torch.randperm(self.n, generator=torch.Generator().manual_seed(0))
        idxs += [perm[s : s + 256] for s in range(0, self.n, 256)]
        for idx in idxs:
            _, parts = self.loss(idx)
            self.assertTrue(torch.allclose(parts.ratio, torch.ones_like(parts.ratio), atol=1e-5, rtol=0))
            self.assertLess(abs(float(parts.approx_kl)), KL_TOL)
            self.assertEqual(float(parts.clip_frac), 0.0)
        with torch.no_grad():
            lp = log_probs(self.learner.policy(self.batch.obs, self.batch.mask))
            new = lp.gather(1, self.batch.action.unsqueeze(1)).squeeze(1)
        self.assertTrue(torch.allclose(new, self.batch.logp, atol=1e-5, rtol=0))

    # --- finite values ---------------------------------------------------------

    def test_finite_loss_and_gradients(self):
        nets = (self.learner.policy, self.learner.critic)
        cases = [
            torch.arange(self.n),
            torch.from_numpy(rows_of(self.batch, FTF)),
            torch.tensor([3]),
        ]
        self.assertGreater(len(cases[1]), 0)
        for idx in cases:
            self.learner.optimizer.zero_grad(set_to_none=True)
            loss, parts = self.loss(idx)
            self.assertTrue(torch.isfinite(loss))
            for f in dataclasses.fields(parts):
                self.assertTrue(torch.isfinite(getattr(parts, f.name)).all(), f.name)
            loss.backward()
            all_finite_grads(self, nets)
        # one legal action: entropy 0, ratio 1
        _, parts = self.loss(cases[1])
        self.assertAlmostEqual(float(parts.entropy), 0.0, places=6)
        self.assertTrue(torch.allclose(parts.ratio, torch.ones_like(parts.ratio), atol=1e-5, rtol=0))
        # one row: normalised advantage 0, so no policy loss
        _, parts = self.loss(cases[2])
        self.assertEqual(float(parts.policy_loss), 0.0)

    # --- entropy ---------------------------------------------------------------

    def test_entropy(self):
        idx = torch.arange(self.n)
        _, parts = self.loss(idx)
        with torch.no_grad():
            logits = self.learner.policy(self.batch.obs, self.batch.mask)
            ref = torch.distributions.Categorical(logits=logits).entropy().mean()
        self.assertAlmostEqual(float(parts.entropy), float(ref), delta=1e-6)
        ttt = torch.from_numpy(rows_of(self.batch, TTT))
        self.assertGreater(len(ttt), 0)
        _, parts = self.loss(ttt)
        self.assertAlmostEqual(float(parts.entropy), math.log(3), delta=0.01)

    # --- the parts -------------------------------------------------------------

    def test_parts_match_formulas(self):
        idx = torch.arange(0, self.n, 3)
        loss, parts = self.loss(idx)
        cfg = self.cfg
        with torch.no_grad():
            lp = log_probs(self.learner.policy(self.batch.obs[idx], self.batch.mask[idx]))
            new = lp.gather(1, self.batch.action[idx].unsqueeze(1)).squeeze(1)
            r = torch.exp(new - self.batch.logp[idx])
            a = self.adv[idx]
            a = (a - a.mean()) / (a.std(correction=0) + cfg.adv_eps)
            pg = -torch.minimum(r * a, r.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
            v = 0.5 * ((self.learner.critic(self.batch.critic_x[idx]) - self.batch.reward[idx]) ** 2).mean()
        self.assertAlmostEqual(float(parts.policy_loss), float(pg), delta=1e-6)
        self.assertAlmostEqual(float(parts.value_loss), float(v), delta=1e-5 * max(1.0, float(v)))
        expected = parts.policy_loss + cfg.vf_coef * parts.value_loss - cfg.ent_coef * parts.entropy
        self.assertAlmostEqual(float(loss), float(expected), delta=1e-6)

    def test_gradient_routing(self):
        idx = torch.arange(0, self.n, 2)
        cfg = dataclasses.replace(self.cfg, ent_coef=0.0, vf_coef=0.0)
        loss, _ = self.loss(idx, cfg=cfg)
        loss.backward()
        for p in self.learner.critic.parameters():
            self.assertTrue(p.grad is None or bool((p.grad == 0).all()))
        self.assertTrue(
            any(p.grad is not None and bool((p.grad != 0).any()) for p in self.learner.policy.parameters())
        )
        self.learner.optimizer.zero_grad(set_to_none=True)
        _, parts = self.loss(idx)
        parts.value_loss.backward()
        for p in self.learner.policy.parameters():
            self.assertIsNone(p.grad)
        self.assertTrue(any(p.grad is not None for p in self.learner.critic.parameters()))

    # --- clipping --------------------------------------------------------------

    def clipped_case(self, ratio_i, ratio_j):
        """Batch copy where rows i, j (TTT masks) have recorded logp shifted to
        give the requested ratios; advantages +1 for i, -1 for j."""
        ttt = rows_of(self.batch, TTT)
        i, j = int(ttt[0]), int(ttt[1])
        idx = torch.tensor([i, j])
        with torch.no_grad():
            lp = log_probs(self.learner.policy(self.batch.obs[idx], self.batch.mask[idx]))
            new = lp.gather(1, self.batch.action[idx].unsqueeze(1)).squeeze(1)
        logp = self.batch.logp.clone()
        logp[i] = new[0] - math.log(ratio_i)
        logp[j] = new[1] - math.log(ratio_j)
        batch = dataclasses.replace(self.batch, logp=logp)
        adv = torch.zeros(self.n)
        adv[i], adv[j] = 1.0, -1.0
        return batch, adv, idx

    def policy_grads(self):
        return [
            torch.zeros_like(p) if p.grad is None else p.grad.clone()
            for p in self.learner.policy.parameters()
        ]

    def test_clipping_favoured_direction_has_no_gradient(self):
        batch, adv, idx = self.clipped_case(1.5, 0.5)
        _, parts = self.loss(idx, batch=batch, adv=adv)
        self.assertTrue(torch.allclose(parts.ratio, torch.tensor([1.5, 0.5]), atol=1e-5))
        self.assertEqual(float(parts.clip_frac), 1.0)
        parts.policy_loss.backward()
        for g in self.policy_grads():
            self.assertTrue(bool((g == 0).all()))

    def test_clipping_unfavoured_direction_keeps_gradient(self):
        batch, adv, idx = self.clipped_case(0.5, 1.5)
        _, parts = self.loss(idx, batch=batch, adv=adv)
        parts.policy_loss.backward()
        self.assertTrue(any(bool((g != 0).any()) for g in self.policy_grads()))

    def test_one_clipped_one_not(self):
        # i: ratio 1.5 with a > 0 (clipped); j: ratio 1.5 with a < 0 (not clipped)
        batch, adv, idx = self.clipped_case(1.5, 1.5)
        _, parts = self.loss(idx, batch=batch, adv=adv)
        parts.policy_loss.backward()
        g_loss = self.policy_grads()
        self.learner.optimizer.zero_grad(set_to_none=True)
        cfg = self.cfg
        lp = log_probs(self.learner.policy(batch.obs[idx], batch.mask[idx]))
        new = lp.gather(1, batch.action[idx].unsqueeze(1)).squeeze(1)
        ratio = torch.exp(new - batch.logp[idx])
        a = adv[idx]
        a = (a - a.mean()) / (a.std(correction=0) + cfg.adv_eps)
        term = -(ratio[1] * a[1]) / 2  # the unclipped row's term of the mean
        term.backward()
        g_ref = self.policy_grads()
        for g, r in zip(g_loss, g_ref):
            self.assertTrue(torch.allclose(g, r, atol=1e-7, rtol=0))
        self.assertTrue(any(bool((r != 0).any()) for r in g_ref))

    # --- advantage normalisation -----------------------------------------------

    def test_advantage_normalisation(self):
        idx = torch.arange(0, self.n, 5)
        a = self.adv[idx]
        a = (a - a.mean()) / (a.std(correction=0) + self.cfg.adv_eps)
        self.assertAlmostEqual(float(a.mean()), 0.0, delta=1e-5)
        self.assertAlmostEqual(float(a.std(correction=0)), 1.0, delta=1e-4)
        # unchanged policy: ratio ~1, so the policy loss is -mean(a) ~ 0
        _, parts = self.loss(idx)
        self.assertAlmostEqual(float(parts.policy_loss), 0.0, delta=1e-4)
        # constant advantage: policy loss 0, nothing NaN
        const = torch.full((self.n,), 2.5)
        loss, parts = self.loss(idx, adv=const)
        self.assertEqual(float(parts.policy_loss), 0.0)
        self.assertTrue(torch.isfinite(loss))

    # --- privileged input ------------------------------------------------------

    def test_policy_never_receives_critic_input(self):
        pw = PolicyWrapper(self, self.learner.policy)
        cw = CriticWrapper(self.learner.critic)
        for idx in (torch.arange(self.n), torch.arange(0, 100)):
            ppo_loss(pw, cw, self.batch, self.adv, idx, self.cfg)
        self.assertEqual(pw.calls, 2)
        self.assertEqual(cw.widths, [OBS_DIM + PRIV_DIM] * 2)
        self.assertIsInstance(self.learner.policy, PolicyNet)


if __name__ == "__main__":
    unittest.main()
