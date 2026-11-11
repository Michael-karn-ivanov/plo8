import unittest

from entities import Board, Card, Hand
from engine.evaluator import (
    FLUSH,
    FULL_HOUSE,
    HIGH_CARD,
    PAIR,
    QUADS,
    STRAIGHT,
    STRAIGHT_FLUSH,
    TRIPS,
    TWO_PAIR,
    best_omaha,
    rank_high,
    rank_low,
)


def cards(s):
    return [Card.parse(t) for t in s.split()]


class RankHighTest(unittest.TestCase):
    def test_category_ordering(self):
        hands = [
            ("2c 5d 9h Js Kc", HIGH_CARD),
            ("2c 2d 9h Js Kc", PAIR),
            ("2c 2d 9h 9s Kc", TWO_PAIR),
            ("2c 2d 2h Js Kc", TRIPS),
            ("5c 6d 7h 8s 9c", STRAIGHT),
            ("2h 5h 9h Jh Kh", FLUSH),
            ("2c 2d 2h Ks Kc", FULL_HOUSE),
            ("2c 2d 2h 2s Kc", QUADS),
            ("5h 6h 7h 8h 9h", STRAIGHT_FLUSH),
        ]
        ranks = [rank_high(cards(h)) for h, _ in hands]
        for (h, cat), r in zip(hands, ranks):
            self.assertEqual(r[0], cat, h)
        self.assertEqual(ranks, sorted(ranks))

    def test_wheel_is_lowest_straight(self):
        wheel = rank_high(cards("Ac 2d 3h 4s 5c"))
        six_high = rank_high(cards("2d 3h 4s 5c 6c"))
        self.assertEqual(wheel, (STRAIGHT, 5))
        self.assertLess(wheel, six_high)
        self.assertGreater(wheel, rank_high(cards("Ac Ad Kh Qs Jc")))
        self.assertEqual(rank_high(cards("Ah 2h 3h 4h 5h")), (STRAIGHT_FLUSH, 5))

    def test_kickers_and_pairs(self):
        self.assertGreater(rank_high(cards("Kc Kd 2h 2s 3c")), rank_high(cards("Qc Qd Jh Js Ac")))
        self.assertGreater(rank_high(cards("9c 9d Ah 3s 2c")), rank_high(cards("9h 9s Kh Qs Jc")))
        self.assertEqual(rank_high(cards("9c 9d Ah 3s 2c")), rank_high(cards("9h 9s Ad 3c 2d")))

    def test_needs_five_cards(self):
        with self.assertRaises(ValueError):
            rank_high(cards("Ac Kd Qh Js"))


class RankLowTest(unittest.TestCase):
    def test_qualifying(self):
        self.assertEqual(rank_low(cards("Ac 2d 3h 4s 5c")), (5, 4, 3, 2, 1))
        self.assertEqual(rank_low(cards("8c 2d 3h 4s 5c")), (8, 5, 4, 3, 2))

    def test_not_qualifying(self):
        self.assertIsNone(rank_low(cards("9c 2d 3h 4s 5c")))  # nine
        self.assertIsNone(rank_low(cards("Ac Ad 3h 4s 5c")))  # pair

    def test_ordering(self):
        self.assertLess(rank_low(cards("Ac 2d 3h 4s 5c")), rank_low(cards("Ac 2d 3h 4s 6c")))
        self.assertLess(rank_low(cards("7c 5d 4h 3s 2c")), rank_low(cards("8c 2d 3h 4s 5c")))
        self.assertLess(rank_low(cards("8c 5d 4h 3s 2c")), rank_low(cards("8c 6d 3h 2s Ac")))


class BestOmahaTest(unittest.TestCase):
    def test_exactly_two_hole_cards_for_flush(self):
        # Four spades on board plus one in hand is not a flush in Omaha.
        hi, _ = best_omaha(Hand.parse("AsKdKhQc"), Board.parse("2s5s9s Js 3d"))
        self.assertEqual(hi[0], PAIR)
        self.assertEqual(hi, (PAIR, 13, 11, 9, 5))

    def test_exactly_three_board_cards(self):
        # Quads in hand do not count: only two hole cards can play.
        hi, _ = best_omaha(Hand.parse("AcAdAhAs"), Board.parse("2c7d9h Js 4d"))
        self.assertEqual(hi[0], PAIR)
        # A board straight needs two hole cards to complete it.
        hi, _ = best_omaha(Hand.parse("KcKdQhQs"), Board.parse("5c6d7h 8s 9d"))
        self.assertEqual(hi[0], PAIR)

    def test_low_qualification(self):
        _, lo = best_omaha(Hand.parse("Ah2dKsKc"), Board.parse("3c4d8h Qs Jd"))
        self.assertEqual(lo, (8, 4, 3, 2, 1))
        _, lo = best_omaha(Hand.parse("Ah2dKsKc"), Board.parse("9cTd8h Qs Jd"))
        self.assertIsNone(lo)

    def test_counterfeited_low(self):
        board = Board.parse("Ac2s5h 7d Jc")
        _, lo = best_omaha(Hand.parse("Ah2dKsKc"), board)
        self.assertIsNone(lo)  # A2 duplicated on board: no three other low board cards
        _, lo = best_omaha(Hand.parse("Ah3dKsKc"), board)
        self.assertEqual(lo, (7, 5, 3, 2, 1))

    def test_a2_beats_a3(self):
        board = Board.parse("4c5d8h Ks Qd")
        _, lo_a2 = best_omaha(Hand.parse("Ah2dKcQc"), board)
        _, lo_a3 = best_omaha(Hand.parse("As3dJcJd"), board)
        self.assertEqual(lo_a2, (8, 5, 4, 2, 1))
        self.assertEqual(lo_a3, (8, 5, 4, 3, 1))
        self.assertLess(lo_a2, lo_a3)

    def test_high_and_low_independent(self):
        # Quad kings (KsKc + KhKd) for high, A2 + 345 for the nut low.
        hi, lo = best_omaha(Hand.parse("Ah2dKsKc"), Board.parse("3c4d5h Kh Kd"))
        self.assertEqual(hi, (QUADS, 13, 5))
        self.assertEqual(lo, (5, 4, 3, 2, 1))

    def test_needs_three_board_cards(self):
        with self.assertRaises(ValueError):
            best_omaha(Hand.parse("Ah2dKsKc"), Board.empty())


if __name__ == "__main__":
    unittest.main()
