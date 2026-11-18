"""Hand-strength hints from the engine's evaluator."""

import unittest
from unittest import mock

from entities import Board, Hand
from rl_env import CHECK_CALL, Table, hints
from rl_env.hints import (
    ACE_DEUCE,
    HAS_LOW,
    HOLE_LOW_RANKS,
    LOW_STRENGTH,
    LOW_TOP,
    LOWS,
    evaluate,
    hole_hints,
    low_index,
    street_hints,
)


def hand(text):
    return Hand.parse(text)


def board(text):
    return Board.parse(text)


class CategoryTest(unittest.TestCase):
    def test_each_high_category(self):
        cases = [
            ("AsKd7c2h", "Qc9d4s3hJc", 0),  # high card
            ("AsAd7c2h", "Qc9d4s3hJc", 1),  # pair
            ("AsQd7c2h", "Qc9dAh3hJc", 2),  # two pair
            ("9s9hKd2c", "9d5cJs3hQc", 3),  # three of a kind
            ("AhTd9c2s", "KcQdJs4h3c", 4),  # straight A-K-Q-J-T
            ("Ah9hKc2d", "3h7hJhQs4c", 5),  # flush
            ("KsKd4c9h", "Kh7c7d2sJc", 6),  # full house
            ("QsQd3c8h", "QhQc5d9s2c", 7),  # four of a kind
            ("9s8s2d2c", "7s6s5sKdAc", 8),  # straight flush
        ]
        for h, b, category in cases:
            with self.subTest(hand=h, board=b):
                self.assertEqual(evaluate(hand(h), board(b))[0], category)
                pairs = dict(street_hints(hand(h), board(b)))
                self.assertEqual(pairs.get(category), 1.0)
                self.assertEqual([c for c in range(9) if c in pairs], [category])

    def test_wheel_straight(self):
        self.assertEqual(evaluate(hand("As2dKcKh"), board("3c4h5s9dJc"))[0], 4)

    def test_no_flush_with_one_suited_hole_card(self):
        # Four hearts on the board, one in the hole: exactly two hole cards needed.
        category, _ = evaluate(hand("AhKc7d2s"), board("3h6h9hJhQc"))
        self.assertEqual(category, 0)


class LowTest(unittest.TestCase):
    def test_low_table(self):
        self.assertEqual(len(LOWS), 56)
        self.assertEqual(LOWS[0], (5, 4, 3, 2, 1))
        self.assertEqual(LOWS[-1], (8, 7, 6, 5, 4))
        self.assertEqual(low_index((8, 4, 3, 2, 1)), 21)
        self.assertEqual(sorted(LOWS), list(LOWS))

    def test_qualifying_low(self):
        _, low = evaluate(hand("As2dKcQh"), board("3c5h8sJdKd"))
        self.assertEqual(low, (8, 5, 3, 2, 1))
        pairs = dict(street_hints(hand("As2dKcQh"), board("3c5h8sJdKd")))
        self.assertEqual(pairs[HAS_LOW], 1.0)
        self.assertEqual(pairs[LOW_TOP + 3], 1.0)
        self.assertEqual(pairs[LOW_STRENGTH], 1.0 - low_index((8, 5, 3, 2, 1)) / 55)

    def test_wheel_is_strength_one(self):
        pairs = dict(street_hints(hand("As2dKcKh"), board("3c4h5s9dJc")))
        self.assertEqual(pairs[LOW_TOP + 0], 1.0)
        self.assertEqual(pairs[LOW_STRENGTH], 1.0)

    def test_no_low(self):
        h, b = hand("As2dKcQh"), board("3c9hTsJdKd")
        self.assertIsNone(evaluate(h, b)[1])
        pairs = dict(street_hints(h, b))
        for off in range(HAS_LOW, HOLE_LOW_RANKS):
            self.assertNotIn(off, pairs)

    def test_counterfeited_low(self):
        h = hand("As2dKcKh")
        self.assertIsNotNone(evaluate(h, board("7s8h3d"))[1])
        # The board's ace and deuce duplicate the hole low cards: no low.
        self.assertIsNone(evaluate(h, board("7s8hAd2c"))[1])
        self.assertNotIn(HAS_LOW, dict(street_hints(h, board("7s8hAd2c"))))

    def test_low_only_on_the_river(self):
        h = hand("As2dKcKh")
        self.assertIsNone(evaluate(h, board("3s9hJd"))[1])
        self.assertIsNone(evaluate(h, board("3s9hJd7c"))[1])
        self.assertEqual(evaluate(h, board("3s9hJd7c8h"))[1], (8, 7, 3, 2, 1))

    def test_a2_beats_a3(self):
        b = board("4c5h8sJdTc")
        low2 = evaluate(hand("As2dKcKh"), b)[1]
        low3 = evaluate(hand("Ah3dQcQh"), b)[1]
        self.assertEqual(low2, (8, 5, 4, 2, 1))
        self.assertEqual(low3, (8, 5, 4, 3, 1))
        self.assertLess(low_index(low2), low_index(low3))
        s2 = dict(street_hints(hand("As2dKcKh"), b))[LOW_STRENGTH]
        s3 = dict(street_hints(hand("Ah3dQcQh"), b))[LOW_STRENGTH]
        self.assertGreater(s2, s3)


class HoleHintsTest(unittest.TestCase):
    def test_distinct_low_ranks(self):
        cases = [
            ("AsAd2c2h", 2, True),  # paired low cards count once
            ("KsQdJc9h", 0, False),
            ("As2d3c4h", 4, True),
            ("8s8d7c9h", 2, False),
            ("AsKdQc3h", 2, False),
            ("2s3d4c5h", 4, False),
        ]
        for h, count, ace_deuce in cases:
            with self.subTest(hand=h):
                pairs = dict(hole_hints(hand(h)))
                self.assertEqual(pairs.get(HOLE_LOW_RANKS + count), 1.0)
                self.assertEqual(
                    [k for k in range(HOLE_LOW_RANKS, HOLE_LOW_RANKS + 5) if k in pairs],
                    [HOLE_LOW_RANKS + count],
                )
                self.assertEqual(ACE_DEUCE in pairs, ace_deuce)

    def test_zeros_preflop(self):
        self.assertEqual(street_hints(hand("As2dKcKh"), Board.empty()), ())


class CacheTest(unittest.TestCase):
    def test_evaluator_called_once_per_player_per_street(self):
        calls = []
        real = hints.best_omaha

        def counting(h, b):
            calls.append((str(h), len(b)))
            return real(h, b)

        with mock.patch.object(hints, "best_omaha", counting):
            table = Table()
            table.start_hand(2, [100.0, 100.0], 0, 17)
            table.observation()
            self.assertEqual(calls, [])  # preflop: not called
            table.act(CHECK_CALL)  # button calls
            table.act(CHECK_CALL)  # BB checks; flop
            seat = table.seat_to_act
            table.observation()
            table.observation()
            self.assertEqual(len(calls), 1)
            table.act(CHECK_CALL)  # check
            table.observation()  # the other player, flop
            self.assertEqual(len(calls), 2)
            table.act(CHECK_CALL)  # check; turn
            self.assertEqual(table.seat_to_act, seat)
            table.observation()
            table.observation()
            self.assertEqual(len(calls), 3)
            self.assertEqual(calls[-1][1], 4)  # refreshed with the turn card


if __name__ == "__main__":
    unittest.main()
