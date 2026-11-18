"""The table-driven evaluator gives exactly the results of the original one
(frozen in reference_evaluator.py, the oracle)."""

import random
import unittest
from collections import Counter
from itertools import combinations, combinations_with_replacement

from entities import Board, Card, Hand, Rank, Suit
from engine import rank_high, rank_low

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


if __name__ == "__main__":
    unittest.main()
