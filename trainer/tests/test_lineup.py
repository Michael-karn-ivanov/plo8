"""draw_lineup: learner count and positions, opponent draws, determinism and a
draw count independent of the outcome."""

import collections
import functools
import math
import unittest

import numpy as np

from rl_env import HandSampler
from trainer import DEFAULT_OPPONENTS, LEARNER_KEY, draw_lineup

N = 20_000


def sigma_bound(p: float, n: int, k: float = 5.0) -> float:
    return k * math.sqrt(p * (1.0 - p) / n)


@functools.lru_cache(maxsize=None)
def lineups(n_seats: int, opponents: tuple = DEFAULT_OPPONENTS, n: int = N, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed + 1000 * n_seats)
    return tuple(draw_lineup(rng, n_seats, opponents) for _ in range(n))


class LineupTest(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(LEARNER_KEY, "learner")
        self.assertEqual(DEFAULT_OPPONENTS, ("random", "calling", "maniac", "rule"))

    def test_learner_count(self):
        for n_seats in range(2, 11):
            ls = lineups(n_seats)
            kmax = min(3, n_seats - 1)
            counts = collections.Counter(lu.count(LEARNER_KEY) for lu in ls)
            for lu in ls:
                self.assertEqual(len(lu), n_seats)
            self.assertEqual(set(counts), set(range(1, kmax + 1)), n_seats)
            p = 1.0 / kmax
            for k in range(1, kmax + 1):
                self.assertLess(abs(counts[k] / N - p), sigma_bound(p, N) + 1e-12, (n_seats, k))
        # heads-up: exactly one learner seat
        self.assertTrue(all(lu.count(LEARNER_KEY) == 1 for lu in lineups(2)))

    def test_learner_positions(self):
        ls = lineups(6)
        mean_k = sum(lu.count(LEARNER_KEY) for lu in ls) / N
        self.assertAlmostEqual(mean_k, 2.0, delta=0.05)
        p = 2.0 / 6
        for s in range(6):
            f = sum(lu[s] == LEARNER_KEY for lu in ls) / N
            self.assertLess(abs(f - p), sigma_bound(p, N), s)

    def test_opponents_uniform(self):
        for n_seats in range(2, 11):
            ls = lineups(n_seats)
            others = [k for lu in ls for k in lu if k != LEARNER_KEY]
            self.assertTrue(set(others) <= set(DEFAULT_OPPONENTS))
            c = collections.Counter(others)
            m = len(others)
            for name in DEFAULT_OPPONENTS:
                self.assertLess(abs(c[name] / m - 0.25), sigma_bound(0.25, m), (n_seats, name))
        # per seat, at 6 seats
        ls = lineups(6)
        for s in range(6):
            col = [lu[s] for lu in ls if lu[s] != LEARNER_KEY]
            c = collections.Counter(col)
            for name in DEFAULT_OPPONENTS:
                self.assertLess(abs(c[name] / len(col) - 0.25), sigma_bound(0.25, len(col)), (s, name))

    def test_opponent_subset(self):
        opp = ("calling", "maniac")
        for n_seats in (2, 6, 10):
            ls = lineups(n_seats, opp, 5000)
            others = {k for lu in ls for k in lu if k != LEARNER_KEY}
            self.assertEqual(others, set(opp))

    def test_worker_sizes(self):
        sampler = HandSampler(3)
        rng = np.random.default_rng(3)
        seen = set()
        for _ in range(5000):
            n = sampler.sample().n_seats
            lu = draw_lineup(rng, n, DEFAULT_OPPONENTS)
            seen.add(n)
            self.assertEqual(len(lu), n)
            self.assertTrue(1 <= lu.count(LEARNER_KEY) <= min(3, n - 1))
        self.assertEqual(seen, set(range(2, 11)))

    def test_max_learner_seats(self):
        rng = np.random.default_rng(4)
        for _ in range(2000):
            self.assertEqual(draw_lineup(rng, 8, DEFAULT_OPPONENTS, 1).count(LEARNER_KEY), 1)
        ks = {draw_lineup(rng, 8, DEFAULT_OPPONENTS, 5).count(LEARNER_KEY) for _ in range(2000)}
        self.assertEqual(ks, set(range(1, 6)))

    def test_same_state_same_lineup(self):
        a, b = np.random.default_rng(9), np.random.default_rng(9)
        for n in [2, 3, 10, 6, 6, 4]:
            self.assertEqual(draw_lineup(a, n, DEFAULT_OPPONENTS), draw_lineup(b, n, DEFAULT_OPPONENTS))

    def test_draws_independent_of_outcome(self):
        # Same generator state, different names (a permutation): the outcomes
        # differ but the generators stay in lockstep after every lineup.
        perm = ("rule", "maniac", "random", "calling")
        rename = dict(zip(DEFAULT_OPPONENTS, perm))
        a, b = np.random.default_rng(11), np.random.default_rng(11)
        sizes = np.random.default_rng(12).integers(2, 11, size=2000)
        for n in sizes:
            la = draw_lineup(a, int(n), DEFAULT_OPPONENTS)
            lb = draw_lineup(b, int(n), perm)
            self.assertEqual(tuple(rename.get(k, k) for k in la), lb)
            self.assertEqual(a.bit_generator.state, b.bit_generator.state)
        # And the bot draw always covers every seat: learner seats do not change
        # how many draws a lineup takes.
        c = np.random.default_rng(13)
        d = np.random.default_rng(13)
        draw_lineup(c, 7, DEFAULT_OPPONENTS)
        k = int(d.integers(1, 4))
        d.choice(7, size=k, replace=False)
        d.integers(4, size=7)
        self.assertEqual(c.bit_generator.state, d.bit_generator.state)

    def test_invalid(self):
        rng = np.random.default_rng(0)
        with self.assertRaises(ValueError):
            draw_lineup(rng, 1, DEFAULT_OPPONENTS)
        with self.assertRaises(ValueError):
            draw_lineup(rng, 6, ())
        with self.assertRaises(ValueError):
            draw_lineup(rng, 6, DEFAULT_OPPONENTS, 0)
        with self.assertRaises(TypeError):
            draw_lineup(rng, 6.0, DEFAULT_OPPONENTS)


if __name__ == "__main__":
    unittest.main()
