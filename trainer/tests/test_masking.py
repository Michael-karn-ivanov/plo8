"""Masking: illegal actions get exactly MASK_VALUE and zero probability, legal
probabilities sum to 1, log-probabilities (Categorical and `log_probs`), entropy
and their gradients are finite - for every mask rl_env produces and for extreme
inputs."""

import itertools
import math
import unittest

import numpy as np
import torch
from torch.distributions import Categorical

from rl_env import CHECK_CALL, N_ACTIONS
from trainer import LOGIT_LIMIT, MASK_VALUE, PolicyNet, log_probs, masked_logits

try:
    from . import helpers
except ImportError:  # run as a top-level module
    import helpers  # type: ignore[no-redef]

_EPS32 = float(torch.finfo(torch.float32).eps)


class MaskingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(1)
        cls.recs = helpers.records()
        cls.net = PolicyNet()

    def check(self, logits: torch.Tensor, mask: torch.Tensor, params) -> None:
        """All invariants on masked `logits`; `params` must receive finite
        gradients from entropy + log-probabilities of the legal actions."""
        self.assertFalse(torch.isnan(logits).any())
        self.assertTrue(torch.all(logits[~mask] == MASK_VALUE))
        self.assertTrue(torch.all(logits[mask].abs() <= LOGIT_LIMIT))
        probs = torch.softmax(logits, dim=-1)
        self.assertTrue(torch.all(probs[~mask] == 0))
        self.assertTrue(torch.all(probs[~mask] <= 1e-30))
        legal_sum = (probs * mask).sum(-1)
        torch.testing.assert_close(legal_sum, torch.ones_like(legal_sum), atol=1e-6, rtol=0)

        dist = Categorical(logits=logits)
        ent = dist.entropy()
        self.assertTrue(torch.isfinite(ent).all())
        lps = log_probs(logits).detach()
        self.assertEqual(tuple(lps.shape), tuple(logits.shape))
        self.assertEqual(lps.dtype, torch.float32)
        self.assertTrue(torch.isfinite(lps).all())
        # log_softmax and Categorical round differently at the scale of the
        # largest legal logit (1 ulp of 1e4 is ~1e-3 for the clamped extremes):
        # 1e-6 for normal logits, a few ulps of that scale beyond.
        scale = logits.detach().masked_fill(~mask, 0.0).abs().amax(-1)
        tol = 1e-6 + 4.0 * _EPS32 * scale
        loss = ent.sum()
        b = logits.shape[0]
        for a in range(N_ACTIONS):
            lp = dist.log_prob(torch.full((b,), a, dtype=torch.long))
            self.assertTrue(torch.isfinite(lp).all(), f"log_prob of action {a}")
            legal = mask[:, a]
            err = (lps[:, a] - lp.detach()).abs()
            self.assertTrue(
                torch.all(err[legal] <= tol[legal]),
                f"log_probs differs from Categorical for action {a}: max {err[legal].max()}",
            )
            loss = loss + (lp * mask[:, a]).sum()

        single = (mask == torch.from_numpy(helpers.SINGLE_LEGAL)).all(-1)
        if single.any():
            p = probs[single][:, CHECK_CALL]
            self.assertTrue(torch.all(p == 1))
            lp = dist.log_prob(torch.full((b,), CHECK_CALL, dtype=torch.long))[single]
            torch.testing.assert_close(lp, torch.zeros_like(lp), atol=1e-6, rtol=0)
            torch.testing.assert_close(
                lps[single][:, CHECK_CALL], torch.zeros_like(lp), atol=1e-6, rtol=0
            )
            torch.testing.assert_close(ent[single], torch.zeros_like(ent[single]), atol=1e-6, rtol=0)

        params = list(params)
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        self.assertTrue(any(g is not None for g in grads))
        for g in grads:
            if g is not None:
                self.assertTrue(torch.isfinite(g).all())

    def run_net(self, obs: np.ndarray, mask: np.ndarray, net=None) -> None:
        net = net if net is not None else self.net
        m = torch.from_numpy(mask)
        logits = net(torch.from_numpy(obs), m)
        self.check(logits, m, net.parameters())

    def test_all_masks_on_real_observations(self):
        obs, mask = helpers.all_mask_batch(self.recs.obs[:200])
        self.run_net(obs, mask)

    def test_real_masks(self):
        found = helpers.mask_rows(self.recs.mask)
        self.assertLessEqual(found, helpers.mask_rows(helpers.ALL_MASKS))
        self.assertLessEqual(helpers.mask_rows(helpers.COMMON_MASKS), found)
        self.run_net(self.recs.obs, self.recs.mask)

    def test_extreme_observations(self):
        obs, mask = helpers.all_mask_batch(self.recs.obs[:64])
        rng = np.random.default_rng(0)
        for x in (
            obs * 1e3,
            obs * -1e3,
            rng.standard_normal(obs.shape) * 1e3,
        ):
            self.run_net(np.ascontiguousarray(x, dtype=np.float32), mask)

    def test_extreme_head(self):
        torch.manual_seed(2)
        net = PolicyNet()
        with torch.no_grad():
            net.head.weight.mul_(1e6)
        obs, mask = helpers.all_mask_batch(self.recs.obs[:64])
        self.run_net(obs, mask, net)
        self.run_net(np.ascontiguousarray(obs * 1e3, dtype=np.float32), mask, net)

    def test_masked_logits_extreme_raw(self):
        values = [1e30, -1e30, math.inf, -math.inf, 0.0]
        raw_rows = list(itertools.product(values, repeat=N_ACTIONS))
        raw = torch.tensor(raw_rows * len(helpers.ALL_MASKS), dtype=torch.float32)
        mask = torch.from_numpy(np.repeat(helpers.ALL_MASKS, len(raw_rows), axis=0))
        raw.requires_grad_(True)
        logits = masked_logits(raw, mask)
        self.check(logits, mask, [raw])

    def test_masked_logits_rejects_bad_masks(self):
        raw = torch.zeros((4, N_ACTIONS))
        mask = torch.from_numpy(helpers.ALL_MASKS)
        with self.assertRaises(TypeError):
            masked_logits(raw, mask.to(torch.uint8))
        with self.assertRaises(ValueError):
            masked_logits(raw, mask[:3])

    def test_log_probs_normal_logits(self):
        # Real (small) logits: log_probs matches Categorical to 1e-6.
        obs, mask = helpers.all_mask_batch(self.recs.obs[:100])
        m = torch.from_numpy(mask)
        with torch.no_grad():
            logits = self.net(torch.from_numpy(obs), m)
        lps = log_probs(logits)
        cat = Categorical(logits=logits).logits  # normalised log-probabilities
        torch.testing.assert_close(lps[m], cat[m], atol=1e-6, rtol=0)

    def test_initial_policy_near_uniform(self):
        m = torch.from_numpy(self.recs.mask)
        with torch.no_grad():
            ent = Categorical(logits=self.net(torch.from_numpy(self.recs.obs), m)).entropy()
        expected = torch.log(m.sum(-1).float())
        torch.testing.assert_close(ent, expected, atol=0.05, rtol=0)


if __name__ == "__main__":
    unittest.main()
