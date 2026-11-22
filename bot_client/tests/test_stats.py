"""Hand lines and the summary."""

import unittest

from bot_client.stats import Stats, format_bb, hand_line, summary


class StatsTest(unittest.TestCase):
    def test_format(self):
        self.assertEqual(format_bb(1.5, sign=True), "+1.50")
        self.assertEqual(format_bb(-2.0, sign=True), "-2.00")
        self.assertEqual(format_bb(0.0, sign=True), "+0.00")
        self.assertEqual(format_bb(-0.001, sign=True), "+0.00")
        self.assertEqual(format_bb(98.0), "98.00")

    def test_hand_line(self):
        self.assertEqual(
            hand_line(12, -2.0, 98.0, 2.0), "hand 12: net -2.00 BB, stack 98.00 BB, rebuy 2.00 BB"
        )
        self.assertEqual(hand_line(3, 1.5, 101.5), "hand 3: net +1.50 BB, stack 101.50 BB")
        self.assertEqual(hand_line(3, 1.5, 101.5, None), "hand 3: net +1.50 BB, stack 101.50 BB")

    def test_summary(self):
        s = Stats()
        nets = [34.5] + [0.0] * 199
        for i, n in enumerate(nets):
            s.add_hand(i + 1, n)
        s.add_rebuy(100.0, 3)
        s.add_rebuy(20.0, 9)
        # 100 * (34.5 / 2) / 200 = 8.625, rounded half up
        self.assertEqual(
            summary("RuleBot", "rule", s, 100.0),
            "RuleBot (rule): 200 hands, net +34.50 BB = +8.63 big bets/100 hands, "
            "rebought 120.00 BB, final stack 100.00 BB",
        )
        self.assertEqual(s.rebuys, [(3, 10000), (9, 2000)])
        self.assertEqual(s.rebought_bb, 120.0)

    def test_rate_and_zero_hands(self):
        s = Stats()
        self.assertIsNone(s.big_bets_per_100())
        self.assertEqual(
            summary("X", "random", s, None),
            "X (random): 0 hands, net +0.00 BB = n/a big bets/100 hands, "
            "rebought 0.00 BB, final stack n/a",
        )
        s.add_hand(1, -3.0)
        s.add_hand(2, 1.0)
        # 100 * (-2 / 2) / 2 = -50
        self.assertEqual(float(s.big_bets_per_100()), -50.0)
        self.assertIn("net -2.00 BB = -50.00 big bets/100 hands", summary("X", "random", s, 1.0))

    def test_integer_chip_totals(self):
        s = Stats()
        for i in range(10000):
            s.add_hand(i, 0.1)
        self.assertEqual(s.net_chips, 100000)
        self.assertEqual(s.net_bb, 1000.0)
        self.assertEqual(s.hands, 10000)
        self.assertEqual(s.nets[0], (0, 10))


if __name__ == "__main__":
    unittest.main()
