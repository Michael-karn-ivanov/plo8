"""Sampling from the masked policy never picks an illegal action - with
`Categorical` and with the sampler NeuralPolicy uses,
`torch.multinomial(log_probs(logits).exp(), ...)`."""

import itertools
import math
import unittest

import numpy as np
import torch
from torch.distributions import Categorical

from rl_env import N_ACTIONS
from trainer import PolicyNet, log_probs, masked_logits

try:
    from . import helpers
except ImportError:  # run as a top-level module
    import helpers  # type: ignore[no-redef]

N_SAMPLES = 100_000


class SamplingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(3)
        cls.net = PolicyNet()
        recs = helpers.records()
        n_rows = 1000
        obs, mask = helpers.all_mask_batch(recs.obs[: n_rows // len(helpers.ALL_MASKS)])
        cls.obs = np.ascontiguousarray(np.concatenate([obs, recs.obs[:n_rows]]))
        cls.mask = np.ascontiguousarray(np.concatenate([mask, recs.mask[:n_rows]]))

    def real_logits(self):
        with torch.no_grad():
            return self.net(torch.from_numpy(self.obs), torch.from_numpy(self.mask))

    def check_samples(self, samples: torch.Tensor, m: torch.Tensor, every_legal: bool) -> None:
        self.assertGreaterEqual(samples.numel(), N_SAMPLES)
        # No illegal action, ever.
        self.assertTrue(torch.all(m.gather(1, samples)))
        if every_legal:
            # Every legal action appears in every row (initial policy ~ uniform).
            seen = torch.zeros_like(m)
            seen.scatter_(1, samples, True)
            self.assertTrue(torch.equal(seen, m))

    def test_categorical_on_real_decisions(self):
        m = torch.from_numpy(self.mask)
        probs = Categorical(logits=self.real_logits()).probs
        per_row = N_SAMPLES // len(self.obs) + 1
        gen = torch.Generator().manual_seed(4)
        samples = torch.multinomial(probs, per_row, replacement=True, generator=gen)
        self.check_samples(samples, m, every_legal=True)

    def test_log_probs_sampler_on_real_decisions(self):
        m = torch.from_numpy(self.mask)
        p = log_probs(self.real_logits()).exp()
        per_row = N_SAMPLES // len(self.obs) + 1
        gen = torch.Generator().manual_seed(5)
        samples = torch.multinomial(p, per_row, replacement=True, generator=gen)
        self.check_samples(samples, m, every_legal=True)
        # One draw per row, as NeuralPolicy does, many times.
        for _ in range(per_row):
            a = torch.multinomial(p, 1, generator=gen)
            self.assertTrue(torch.all(m.gather(1, a)))

    def test_log_probs_sampler_on_extreme_logits(self):
        gen = torch.Generator().manual_seed(6)
        # masked_logits with raw logits +-1e30, +-inf and 0, every mask.
        values = [1e30, -1e30, math.inf, -math.inf, 0.0]
        raw_rows = list(itertools.product(values, repeat=N_ACTIONS))
        raw = torch.tensor(raw_rows * len(helpers.ALL_MASKS), dtype=torch.float32)
        mask = torch.from_numpy(np.repeat(helpers.ALL_MASKS, len(raw_rows), axis=0))
        cases = [(masked_logits(raw, mask), mask)]
        # A network with an extreme head, on extreme observations.
        torch.manual_seed(2)
        net = PolicyNet()
        with torch.no_grad():
            net.head.weight.mul_(1e6)
        obs, m = helpers.all_mask_batch(helpers.records().obs[:64])
        rng = np.random.default_rng(0)
        mt = torch.from_numpy(m)
        for x in (obs, obs * 1e3, obs * -1e3, rng.standard_normal(obs.shape) * 1e3):
            x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
            with torch.no_grad():
                cases.append((net(x, mt), mt))
        for logits, mk in cases:
            p = log_probs(logits).exp()
            self.assertTrue(torch.isfinite(p).all())
            self.assertTrue(torch.all(p[~mk] == 0))
            per_row = N_SAMPLES // len(p) + 1
            samples = torch.multinomial(p, per_row, replacement=True, generator=gen)
            self.check_samples(samples, mk, every_legal=False)


if __name__ == "__main__":
    unittest.main()
