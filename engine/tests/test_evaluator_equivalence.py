"""The table-driven evaluator gives exactly the results of the original one
(frozen in reference_evaluator.py, the oracle)."""

import random
import unittest
from collections import Counter
from itertools import combinations, combinations_with_replacement

from entities import Board, Card, Hand, Rank, Suit
from engine import best_omaha, rank_high, rank_low
from engine.evaluator import best_omaha_many

try:
    from . import reference_evaluator as ref
except ImportError:  # run as a top-level module
    import reference_evaluator as ref  # type: ignore[no-redef]

SUITS = (Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES)
ALL = [Card.from_index(i) for i in range(52)]


def board_of(cards):
    cards = list(cards)
    return Board(
        flop=tuple(cards[:3]),
        turn=cards[3] if len(cards) > 3 else None,
        river=cards[4] if len(cards) > 4 else None,
    )


def cmp(a, b):
    return (a > b) - (a < b)


def cmp_low(a, b):
    """-1 if low `a` is better than `b`; None (no low) is worst."""
    if a is None and b is None:
        return 0
    if a is None:
        return 1
    if b is None:
        return -1
    return cmp(a, b)


class FiveCardTest(unittest.TestCase):
    def check(self, five):
        self.assertEqual(rank_high(five), ref.rank_high(five), five)
        self.assertEqual(rank_low(five), ref.rank_low(five), five)

    def test_every_rank_multiset_not_suited(self):
        count = 0
        for combo in combinations_with_replacement(range(2, 15), 5):
            if max(Counter(combo).values()) > 4:
                continue
            seen = Counter()
            five = []
            for r in combo:
                five.append(Card(Rank(r), SUITS[seen[r]]))
                seen[r] += 1
            if len({c.suit for c in five}) == 1:  # five distinct ranks: break the flush
                five[1] = Card(five[1].rank, Suit.DIAMONDS)
            self.assertGreater(len({c.suit for c in five}), 1)
            self.check(tuple(five))
            count += 1
        self.assertEqual(count, 6175)

    def test_every_rank_set_as_a_flush(self):
        count = 0
        for k, combo in enumerate(combinations(range(2, 15), 5)):
            suit = SUITS[k % 4]
            five = tuple(Card(Rank(r), suit) for r in combo)
            self.check(five)
            # Any order of the same cards ranks the same.
            self.check(five[::-1])
            count += 1
        self.assertEqual(count, 1287)

    def test_random_hands(self):
        rng = random.Random(1)
        for _ in range(200_000):
            self.check(tuple(rng.sample(ALL, 5)))

    def test_stacked_cards_that_are_not_interned(self):
        # Cards built by a caller (not the ALL_CARDS instances) rank the same.
        rng = random.Random(2)
        for _ in range(2000):
            five = tuple(Card(c.rank, c.suit) for c in rng.sample(ALL, 5))
            self.check(five)

    def test_wrong_card_counts(self):
        rng = random.Random(3)
        for n in (0, 1, 4, 6, 7):
            cards = rng.sample(ALL, n)
            for fast, slow in ((rank_high, ref.rank_high), (rank_low, ref.rank_low)):
                with self.assertRaises(ValueError) as fast_error:
                    fast(cards)
                with self.assertRaises(ValueError) as slow_error:
                    slow(cards)
                self.assertEqual(str(fast_error.exception), str(slow_error.exception))


class OmahaTest(unittest.TestCase):
    def check(self, hand, board):
        fast = best_omaha(hand, board)
        slow = ref.best_omaha(hand, board)
        self.assertEqual(fast, slow, (hand, board))
        return fast, slow

    def run_random(self, n_boards, board_size, seed):
        rng = random.Random(seed)
        for _ in range(n_boards):
            cards = rng.sample(ALL, 8 + board_size)
            board = board_of(cards[8:])
            hands = [Hand(cards[0:4]), Hand(cards[4:8])]
            (f1, s1), (f2, s2) = (self.check(h, board) for h in hands)
            # The same winner of each half, whichever evaluator decides.
            self.assertEqual(cmp(f1[0], f2[0]), cmp(s1[0], s2[0]))
            self.assertEqual(cmp_low(f1[1], f2[1]), cmp_low(s1[1], s2[1]))
            self.assertEqual(best_omaha_many(hands, board), [f1, f2])

    def test_random_five_card_boards(self):
        self.run_random(50_000, 5, seed=10)  # 100,000 (hand, board) pairs

    def test_random_flops(self):
        self.run_random(5_000, 3, seed=11)  # 10,000 pairs

    def test_random_turns(self):
        self.run_random(5_000, 4, seed=12)  # 10,000 pairs

    def test_exactly_two_hole_cards(self):
        # Four hearts on the board and one in hand: no flush with exactly 2 + 3.
        hand = Hand.parse("Ah2c3d9s")
        board = Board.parse("Kh Qh 7h 4h 9c")
        fast, _ = self.check(hand, board)
        self.assertNotEqual(fast[0][0], 5)

    def test_wrong_card_counts(self):
        rng = random.Random(4)
        cards = rng.sample(ALL, 9)

        class Loose:
            """An iterable of cards of any length (Hand insists on 4)."""

            def __init__(self, cards):
                self.cards = tuple(cards)

            def __iter__(self):
                return iter(self.cards)

        cases = [
            (Loose(cards[:3]), board_of(cards[4:9])),
            (Loose(cards[:5]), board_of(cards[5:8])),
            (Hand(cards[:4]), Board()),
            (Hand(cards[:4]), Loose(cards[4:6])),
        ]
        for hand, board in cases:
            with self.assertRaises(ValueError) as fast_error:
                best_omaha(hand, board)
            with self.assertRaises(ValueError) as slow_error:
                ref.best_omaha(hand, board)
            self.assertEqual(str(fast_error.exception), str(slow_error.exception))
            with self.assertRaises(ValueError):
                best_omaha_many([hand], board)


if __name__ == "__main__":
    unittest.main()
