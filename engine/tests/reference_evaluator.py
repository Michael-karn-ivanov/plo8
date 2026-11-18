"""Frozen copy of the original (pre table-driven) evaluator.

Used only as the oracle of `test_evaluator_equivalence`; never imported at
runtime. Do not change it.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Iterable, Optional

from entities import Board, Card, Hand

HIGH_CARD = 0
PAIR = 1
TWO_PAIR = 2
TRIPS = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
QUADS = 7
STRAIGHT_FLUSH = 8

HighRank = tuple[int, ...]
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
    five = _five(cards)
    values = sorted({c.rank.low_value for c in five}, reverse=True)
    if len(values) != 5 or values[0] > 8:
        return None
    return tuple(values)


def best_omaha(hand: Hand, board: Board) -> tuple[HighRank, Optional[LowRank]]:
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
