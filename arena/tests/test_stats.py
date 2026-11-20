"""Statistics: bb/100 and the 95% interval against hand computations (sign,
units in big bets, one sample per rotation group)."""

import math
import unittest
from collections import defaultdict

import numpy as np

from arena import CHIPS_PER_BIG_BET, play_match, summarize


class SummarizeTest(unittest.TestCase):
    def test_hand_computed_example(self):
        s = summarize([400, -200, 0, 200], k=1, n=2)
        np.testing.assert_allclose(s.samples, [1.0, -0.5, 0.0, 0.5])
        self.assertEqual(s.deals, 4)
        self.assertEqual(s.hands, 8)
        self.assertEqual(s.total_chips, 400)
        self.assertAlmostEqual(s.total_bb, 2.0)
        self.assertAlmostEqual(s.bb_per_100, 25.0)
        mean = 0.25
        var = sum((x - mean) ** 2 for x in (1.0, -0.5, 0.0, 0.5)) / 3
        self.assertAlmostEqual(s.ci95, 100 * 1.96 * math.sqrt(var) / math.sqrt(4))

    def test_units_and_sign(self):
        self.assertEqual(CHIPS_PER_BIG_BET, 200)
        s = summarize([200, 200], k=1, n=1)  # one big bet per hand
        self.assertAlmostEqual(s.bb_per_100, 100.0)
        self.assertAlmostEqual(s.ci95, 0.0)
        s = summarize([-400, -400, -400], k=1, n=2)  # -1 big bet per hand
        self.assertAlmostEqual(s.bb_per_100, -100.0)
        self.assertAlmostEqual(s.total_bb, -6.0)

    def test_pooled_name(self):
        s = summarize([600, -300], k=3, n=6)
        self.assertEqual(s.hands, 36)
        np.testing.assert_allclose(s.samples, [600 / 18 / 200, -300 / 18 / 200])
        self.assertAlmostEqual(s.total_bb, 1.5)
        self.assertAlmostEqual(s.bb_per_100, 100 * 1.5 / 36)
        self.assertAlmostEqual(s.bb_per_100, 100 * float(np.mean(s.samples)))

    def test_one_deal_has_no_interval(self):
        s = summarize([100], k=1, n=2)
        self.assertTrue(math.isnan(s.ci95))

    def test_rejects_floats_and_bad_shapes(self):
        with self.assertRaises(ValueError):
            summarize([1.5, 2.0], k=1, n=2)
        with self.assertRaises(ValueError):
            summarize([], k=1, n=2)
        with self.assertRaises(ValueError):
            summarize([1, 2], k=3, n=2)


class RealMatchTest(unittest.TestCase):
    def test_stats_recomputed_from_hand_records(self):
        lineup = ["rule", "random", "calling", "random"]
        n, deals = len(lineup), 15
        res = play_match(lineup, deals, seed=3, record_hands=True)
        per_name = defaultdict(lambda: np.zeros(deals, dtype=np.int64))
        for rec in res.hands:
            self.assertEqual(sum(rec.chips), 0)
            for s in range(n):
                per_name[rec.names[s]][rec.deal] += rec.chips[s]
        for name, st in res.stats.items():
            k = len(st.positions)
            chips = per_name[name]
            self.assertEqual(st.total_chips, int(chips.sum()))
            self.assertEqual(st.hands, k * n * deals)
            samples = chips / (k * n) / 200
            np.testing.assert_allclose(st.samples, samples)
            self.assertAlmostEqual(st.bb_per_100, 100 * samples.mean())
            sd = float(np.std(samples, ddof=1))
            self.assertAlmostEqual(st.ci95, 100 * 1.96 * sd / math.sqrt(deals))


if __name__ == "__main__":
    unittest.main()
