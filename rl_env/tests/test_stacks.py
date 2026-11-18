"""Training stacks: deep share, short range and resolution, reproducibility."""

import unittest

import numpy as np

from rl_env import HandSampler, HandSpec, StackConfig, Table, draw_stacks


class DrawStacksTest(unittest.TestCase):
    def test_deep_share_and_short_range(self):
        rng = np.random.Generator(np.random.PCG64(0))
        draws = 20_000
        deep = 0
        for _ in range(draws):
            stacks = draw_stacks(rng, 6)
            self.assertEqual(len(stacks), 6)
            if stacks == (100.0,) * 6:
                deep += 1
                continue
            for s in stacks:
                self.assertGreaterEqual(s, 5.0)
                self.assertLessEqual(s, 150.0)
                self.assertAlmostEqual(s * 100, round(s * 100), places=6)  # 0.01 BB
        share = deep / draws
        self.assertAlmostEqual(share, 0.8, delta=0.015)

    def test_short_stacks_are_whole_chips_for_a_table(self):
        rng = np.random.Generator(np.random.PCG64(1))
        cfg = StackConfig(deep_share=0.0)
        for _ in range(200):
            stacks = draw_stacks(rng, 9, cfg)
            Table().start_hand(9, stacks, 0, 1)  # accepted: whole chips

    def test_custom_config(self):
        rng = np.random.Generator(np.random.PCG64(2))
        cfg = StackConfig(deep_share=0.0, short_min_bb=1.0, short_max_bb=2.0)
        for _ in range(500):
            for s in draw_stacks(rng, 3, cfg):
                self.assertTrue(1.0 <= s <= 2.0)
        cfg = StackConfig(deep_share=1.0, deep_bb=40.0)
        self.assertEqual(draw_stacks(rng, 4, cfg), (40.0,) * 4)
        for bad in (
            dict(deep_share=1.5),
            dict(deep_bb=0.0),
            dict(short_min_bb=0.0),
            dict(short_min_bb=10.0, short_max_bb=5.0),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                StackConfig(**bad)


class SamplerTest(unittest.TestCase):
    def test_same_seed_same_specs(self):
        a = [HandSampler(5).sample() for _ in range(1)]
        s1, s2 = HandSampler(5), HandSampler(5)
        specs1 = [s1.sample() for _ in range(500)]
        specs2 = [s2.sample() for _ in range(500)]
        self.assertEqual(specs1, specs2)
        self.assertEqual(specs1[0], a[0])
        self.assertNotEqual(specs1, [HandSampler(6).sample() for _ in range(500)])

    def test_specs(self):
        sampler = HandSampler(7)
        sizes = set()
        for _ in range(2000):
            spec = sampler.sample()
            self.assertIsInstance(spec, HandSpec)
            self.assertTrue(2 <= spec.n_seats <= 10)
            self.assertEqual(len(spec.stacks_bb), spec.n_seats)
            self.assertTrue(0 <= spec.button < spec.n_seats)
            self.assertIsInstance(spec.deck_seed, int)
            self.assertTrue(0 <= spec.deck_seed < 2**63)
            sizes.add(spec.n_seats)
        self.assertEqual(sizes, set(range(2, 11)))
        fixed = HandSampler(8, sizes=[6])
        self.assertTrue(all(fixed.sample().n_seats == 6 for _ in range(50)))
        self.assertEqual(HandSampler(9).sample(n_seats=4).n_seats, 4)
        with self.assertRaises(ValueError):
            HandSampler(1, sizes=[1])

    def test_spec_is_frozen(self):
        spec = HandSampler(1).sample()
        with self.assertRaises(Exception):
            spec.button = 0  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
