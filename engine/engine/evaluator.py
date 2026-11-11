"""Hand evaluation: high ranking, 8-or-better low ranking and Omaha 2+3 selection."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Iterable, Optional

from entities import Board, Card, Hand

__all__ = [
    "HIGH_CARD",
    "PAIR",
    "TWO_PAIR",
    "TRIPS",
    "STRAIGHT",
    "FLUSH",
    "FULL_HOUSE",
    "QUADS",
    "STRAIGHT_FLUSH",
    "CATEGORY_NAMES",
    "HighRank",
    "LowRank",
    "rank_high",
    "rank_low",
    "best_omaha",
]

HIGH_CARD = 0
PAIR = 1
TWO_PAIR = 2
TRIPS = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
QUADS = 7
STRAIGHT_FLUSH = 8

CATEGORY_NAMES = (
    "high card",
    "pair",
    "two pair",
    "three of a kind",
    "straight",
    "flush",
    "full house",
    "four of a kind",
    "straight flush",
)

# (category, tiebreak ranks...): a higher tuple is a better high hand.
HighRank = tuple[int, ...]
# Low card values sorted descending (ace = 1): a *lower* tuple is a better low.
LowRank = tuple[int, ...]

_WHEEL = [14, 5, 4, 3, 2]


def _five(cards: Iterable[Card]) -> tuple[Card, ...]:
    five = tuple(cards)
    if len(five) != 5:
        raise ValueError(f"exactly 5 cards are needed, got {len(five)}")
    return five


def rank_high(cards: Iterable[Card]) -> HighRank:
    """Rank a 5-card high hand. A-2-3-4-5 (the wheel) is a 5-high straight."""
    five = _five(cards)
    ranks = sorted((int(c.rank) for c in five), reverse=True)
    flush = len({c.suit for c in five}) == 1

    straight_high: Optional[int] = None
    if len(set(ranks)) == 5:
        if ranks[0] - ranks[4] == 4:
            straight_high = ranks[0]
        elif ranks == _WHEEL:
            straight_high = 5

    if straight_high is not None and flush:
        return (STRAIGHT_FLUSH, straight_high)

    # Groups ordered by size, then by rank: e.g. two pair -> high pair, low pair, kicker.
    groups = sorted(Counter(ranks).items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    shape = [n for _, n in groups]
    order = tuple(r for r, _ in groups)

    if shape == [4, 1]:
        return (QUADS, *order)
    if shape == [3, 2]:
        return (FULL_HOUSE, *order)
    if flush:
        return (FLUSH, *ranks)
    if straight_high is not None:
        return (STRAIGHT, straight_high)
    if shape == [3, 1, 1]:
        return (TRIPS, *order)
    if shape == [2, 2, 1]:
        return (TWO_PAIR, *order)
    if shape == [2, 1, 1, 1]:
        return (PAIR, *order)
    return (HIGH_CARD, *ranks)


def rank_low(cards: Iterable[Card]) -> Optional[LowRank]:
    """Rank a 5-card 8-or-better low, or return None if it does not qualify.

    A qualifying low has five distinct ranks, all 8 or lower (ace counts as 1).
    The result is sorted descending, so tuples compare directly: lower is better.
    """
    five = _five(cards)
    values = sorted({c.rank.low_value for c in five}, reverse=True)
    if len(values) != 5 or values[0] > 8:
        return None
    return tuple(values)


def best_omaha(hand: Hand, board: Board) -> tuple[HighRank, Optional[LowRank]]:
    """Best Omaha high and low using exactly two hole cards and exactly three board cards.

    The high and the low are chosen independently (they may use different cards).
    """
    hole = tuple(hand)
    community = tuple(board)
    if len(hole) != 4:
        raise ValueError(f"an Omaha hand has 4 hole cards, got {len(hole)}")
    if len(community) < 3:
        raise ValueError(f"at least 3 board cards are needed, got {len(community)}")

    best_hi: Optional[HighRank] = None
    best_lo: Optional[LowRank] = None
    for two in combinations(hole, 2):
        for three in combinations(community, 3):
            five = two + three
            hi = rank_high(five)
            if best_hi is None or hi > best_hi:
                best_hi = hi
            lo = rank_low(five)
            if lo is not None and (best_lo is None or lo < best_lo):
                best_lo = lo
    assert best_hi is not None
    return best_hi, best_lo
