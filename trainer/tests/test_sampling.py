"""Sampling from the masked policy never picks an illegal action."""

import unittest

import numpy as np
import torch
from torch.distributions import Categorical

from rl_env import N_ACTIONS, HandSampler, TableBatch
from trainer import PolicyNet

try:
    from . import helpers
except ImportError:  # run as a top-level module
    import helpers  # type: ignore[no-redef]


class SamplingTest(unittest.TestCase):
    def test_many_samples_from_real_decisions(self):
        torch.manual_seed(3)
        net = PolicyNet()
        recs = helpers.records()
        n_rows = 1000
        obs, mask = helpers.all_mask_batch(recs.obs[: n_rows // len(helpers.ALL_MASKS)])
        obs = np.concatenate([obs, recs.obs[:n_rows]])
        mask = np.concatenate([mask, recs.mask[:n_rows]])
        m = torch.from_numpy(mask)
        with torch.no_grad():
            probs = Categorical(logits=net(torch.from_numpy(obs), m)).probs
        per_row = 100_000 // len(obs) + 1
        gen = torch.Generator().manual_seed(4)
        samples = torch.multinomial(probs, per_row, replacement=True, generator=gen)
        self.assertGreaterEqual(samples.numel(), 100_000)
        # No illegal action, ever.
        self.assertTrue(torch.all(m.gather(1, samples)))
        # Every legal action appears in every row (initial policy ~ uniform).
        seen = torch.zeros_like(m)
        seen.scatter_(1, samples, True)
        self.assertTrue(torch.equal(seen, m))

    def test_table_batch_with_sampling_controllers(self):
        torch.manual_seed(5)
        net = PolicyNet()
        controllers = {k: helpers.SamplingController(net, seed=10 + k) for k in range(3)}
        batch = TableBatch(128)
        sampler = HandSampler(6, range(2, 11))
        n_hands = 2000
        sizes = set()

        def seats(spec):
            sizes.add(spec.n_seats)
            return [s % len(controllers) for s in range(spec.n_seats)]

        # TableBatch.submit re-checks every action against its mask and raises on
        # an illegal one; the controllers also check before returning.
        done = batch.run(controllers, sampler, n_hands, seats)
        self.assertEqual(done, n_hands)
        self.assertEqual(sizes, set(range(2, 11)))
        decisions = sum(c.decisions for c in controllers.values())
        self.assertEqual(decisions, batch.n_decisions)
        counts = sum(c.counts for c in controllers.values())
        self.assertTrue(np.all(counts > 0), counts)
        self.assertEqual(len(counts), N_ACTIONS)


if __name__ == "__main__":
    unittest.main()
