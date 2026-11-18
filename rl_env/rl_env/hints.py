"""Hand-strength hints from the engine's evaluator.

High and low come from `engine.evaluator.best_omaha(hand, board)` (exactly two
hole cards and three board cards - the engine's rule); the category is the
first element of the high rank. The low index is the position of the best low
among the 56 possible 8-or-better lows, sorted the way the evaluator compares
them (a lower descending tuple is a better low; the wheel is first).

Offsets returned here are relative to the hints block of the observation
(21 values, see plan.md):

| +0..+8   | high-hand category one-hot (flop on)             |
| +9       | has a qualifying low (flop on)                   |
| +10..+13 | highest card of the best low, 5 / 6 / 7 / 8      |
| +14      | low strength 1 - i / 55 (wheel = 1.0)            |
| +15..+19 | distinct low ranks in the hole cards, 0..4       |
| +20      | hole cards contain an ace and a deuce            |
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional

from entities import Board, Hand, Rank
from engine.evaluator import best_omaha

__all__ = [
    "N_HINTS",
    "N_CATEGORIES",
    "HAS_LOW",
    "LOW_TOP",
    "LOW_STRENGTH",
    "HOLE_LOW_RANKS",
    "ACE_DEUCE",
    "LOWS",
    "LOW_INDEX",
    "low_index",
    "evaluate",
    "street_hints",
    "hole_hints",
]

N_HINTS = 21
N_CATEGORIES = 9  # high card ... straight flush (engine.evaluator categories 0..8)
HAS_LOW = 9
LOW_TOP = 10
LOW_STRENGTH = 14
HOLE_LOW_RANKS = 15
ACE_DEUCE = 20

# All 56 five-card 8-or-better lows as descending tuples (ace = 1), best first.
LOWS: tuple[tuple[int, ...], ...] = tuple(
    sorted(tuple(sorted(c, reverse=True)) for c in combinations(range(1, 9), 5))
)
LOW_INDEX: dict[tuple[int, ...], int] = {low: i for i, low in enumerate(LOWS)}
assert len(LOWS) == 56 and LOWS[0] == (5, 4, 3, 2, 1)

_LAST_LOW = len(LOWS) - 1  # 55


def low_index(low: tuple[int, ...]) -> int:
    """Rank of a qualifying low among the 56 possible lows (0 = the wheel)."""
    return LOW_INDEX[tuple(low)]


def evaluate(hand: Hand, board: Board) -> tuple[int, Optional[tuple[int, ...]]]:
    """(high category 0..8, best low tuple or None) from the engine's evaluator."""
    high, low = best_omaha(hand, board)
    return high[0], low


def street_hints(hand: Hand, board: Board) -> tuple[tuple[int, float], ...]:
    """The flop-on hints as (offset, value) pairs (non-zero values only).
    Preflop (no flop on the board) the evaluator is not called: ()."""
    if board.flop is None:
        return ()
    high, low = best_omaha(hand, board)
    out: list[tuple[int, float]] = [(high[0], 1.0)]
    if low is not None:
        out.append((HAS_LOW, 1.0))
        out.append((LOW_TOP + low[0] - 5, 1.0))
        strength = 1.0 - LOW_INDEX[tuple(low)] / _LAST_LOW
        if strength > 0.0:
            out.append((LOW_STRENGTH, strength))
    return tuple(out)


def hole_hints(hand: Hand) -> tuple[tuple[int, float], ...]:
    """The every-street hints of the hole cards as (offset, value) pairs."""
    lows = set()
    ace = deuce = False
    for c in hand:
        rank = c.rank
        v = rank.low_value
        if v <= 8:
            lows.add(v)
        if rank is Rank.ACE:
            ace = True
        elif rank is Rank.TWO:
            deuce = True
    out: list[tuple[int, float]] = [(HOLE_LOW_RANKS + len(lows), 1.0)]
    if ace and deuce:
        out.append((ACE_DEUCE, 1.0))
    return tuple(out)
