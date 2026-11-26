"""The learner learns: a synthetic one-step problem (raising is best; once the
policy always raises, an iteration's rewards are all equal and its explained
variance is NaN) and a few collect -> update iterations against the arena bots
(everything finite, logp consistent, no entropy collapse)."""

import math
import unittest

import numpy as np
import torch

from rl_env import BET_RAISE, CHECK_CALL, FOLD
from trainer import log_probs

try:
    from . import helpers
except ImportError:  # run as a top-level module
    import helpers  # type: ignore[no-redef]

TTT = np.array([True, True, True])


def non_finite(obj, skip=()) -> list:
    """Names of the float fields of a dataclass that are not finite."""
    bad = []
    for name in obj.__dataclass_fields__:
        v = getattr(obj, name)
        if name not in skip and isinstance(v, float) and not math.isfinite(v):
            bad.append(name)
    return bad


def all_finite(obj) -> bool:
    return not non_finite(obj)


def ttt_entropy(learner, obs) -> float:
    mask = torch.ones((len(obs), 3), dtype=torch.bool)
    with torch.no_grad():
        lp = log_probs(learner.policy(torch.from_numpy(obs), mask))
        return float(-(lp.exp() * lp).sum(-1).mean())


class SyntheticTest(unittest.TestCase):
    def test_learns_to_raise(self):
        real = helpers.records()
        rows = np.flatnonzero((real.mask == TTT).all(axis=1))[:512]
        self.assertEqual(len(rows), 512)
        obs = np.ascontiguousarray(real.obs[rows])
        mask = np.ascontiguousarray(real.mask[rows])
        critic = np.ascontiguousarray(real.critic[rows])
        table = np.zeros(3, dtype=np.float32)
        table[BET_RAISE], table[CHECK_CALL], table[FOLD] = 1.0, 0.0, -0.5

        learner = helpers.small_learner(0, lr=1e-3, minibatch=128)

        def p_raise() -> float:
            with torch.no_grad():
                lp = log_probs(learner.policy(torch.from_numpy(obs), torch.from_numpy(mask)))
            return float(lp.exp()[:, BET_RAISE].mean())

        start = p_raise()
        self.assertAlmostEqual(start, 1 / 3, delta=0.05)
        value_losses = []
        all_equal_iterations = 0
        for it in range(40):
            recs = helpers.acted_records(
                learner.get_weights(), obs, mask, critic, lambda a, r: table[a], seed=1000 + it
            )
            stats = learner.update(recs)
            if np.all(recs.reward == recs.reward[0]):
                # Every row raised: every reward is 1.0, the explained variance is
                # undefined (NaN by design); the update still ran and every other
                # figure is finite.
                all_equal_iterations += 1
                self.assertTrue(math.isnan(stats.explained_variance), it)
                self.assertEqual(non_finite(stats, skip=("explained_variance",)), [], it)
            else:
                self.assertEqual(non_finite(stats), [], it)
            value_losses.append(stats.value_loss)
        self.assertGreater(p_raise(), 0.8)
        self.assertLess(np.mean(value_losses[-5:]), np.mean(value_losses[:5]))
        # Once learned, some iteration samples BET_RAISE for all 512 rows.
        self.assertGreater(all_equal_iterations, 0)


class ArenaTest(unittest.TestCase):
    def test_collect_and_update(self):
        worker = helpers.small_worker(0)
        learner = helpers.small_learner(0, minibatch=1024)
        first_obs = None
        update_stats = []
        for it in range(5):
            recs, rstats = worker.collect(learner.get_weights(), 3000)
            self.assertTrue(all_finite(rstats.overall), it)
            self.assertTrue(math.isfinite(rstats.wall_time))
            if it == 0:
                ref = helpers.recorded_logp_under(learner.policy, recs)
                self.assertTrue(np.allclose(recs.logp, ref, atol=1e-5, rtol=0))
                first_obs = np.ascontiguousarray(recs.obs[(recs.mask == TTT).all(axis=1)])
                self.assertGreater(len(first_obs), 0)
                self.assertAlmostEqual(ttt_entropy(learner, first_obs), math.log(3), delta=0.01)
            stats = learner.update(recs)
            # Real rewards vary: the explained variance is finite too.
            self.assertEqual(non_finite(stats), [], it)
            update_stats.append(stats)
            print(
                f"iter {it}: bb/100 {rstats.bb_per_100:+.1f}  entropy {stats.entropy:.3f}  "
                f"approx_kl {stats.approx_kl:.4f}  clip_frac {stats.clip_frac:.3f}  "
                f"ev {stats.explained_variance:+.3f}"
            )
        self.assertGreater(ttt_entropy(learner, first_obs), 0.8)
        self.assertGreaterEqual(update_stats[-1].entropy, 0.5 * update_stats[0].entropy)


if __name__ == "__main__":
    unittest.main()
