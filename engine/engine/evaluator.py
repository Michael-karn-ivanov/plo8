"""Hand evaluation: high ranking, 8-or-better low ranking and Omaha 2+3 selection.

Table-driven implementation (pure Python, tables built once at import):

- Per-card tables indexed by `Card.index`: `PRIME` (a distinct prime per rank),
  `SUIT` (0..3) and `LOWBIT` (`1 << low_value` for low values <= 8, else 0).
- High ranks are handled as integers: within one category every rank tuple has
  the same length, so `category * 16**5 + sum(tiebreak_k * 16**(4 - k))` keeps
  the tuple order exactly. `HIGH_INT` (non-flush) and `FLUSH_INT` (flush /
  straight flush) map the product of the five rank primes to that integer;
  `DECODE_HIGH` gives the public tuple back. The tables are built by running
  the original ranking logic over every rank multiset, so the tuples are the
  original ones by construction.
- A qualifying low is five distinct low values <= 8; for such sets comparing
  the descending tuples equals comparing the bitmasks `sum(1 << v)`, so the
  best low is the smallest mask. `DECODE_LOW` gives the public tuple.

Results are identical to the original implementation (kept in
`tests/reference_evaluator.py` as the test oracle).
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations, combinations_with_replacement
from typing import Iterable, Optional, Sequence

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
    "best_omaha_many",
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


# --- the original ranking logic (builds the tables; fallback for odd input) ------


def _rank_tuple(ranks: list[int], flush: bool) -> HighRank:
    """High rank of five ranks (sorted descending); `flush`: all one suit."""
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


def _slow_high(five: Sequence[Card]) -> HighRank:
    ranks = sorted((int(c.rank) for c in five), reverse=True)
    return _rank_tuple(ranks, len({c.suit for c in five}) == 1)


def _slow_low(five: Sequence[Card]) -> Optional[LowRank]:
    values = sorted({c.rank.low_value for c in five}, reverse=True)
    if len(values) != 5 or values[0] > 8:
        return None
    return tuple(values)


# --- tables -----------------------------------------------------------------------

_RANK_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)  # ranks 2..14


def _low_value(rank: int) -> int:
    return 1 if rank == 14 else rank


def _low_bit(index: int) -> int:
    v = _low_value((index >> 2) + 2)
    return 1 << v if v <= 8 else 0


# Card index = (rank - 2) * 4 + suit index.
PRIME: tuple[int, ...] = tuple(_RANK_PRIMES[i >> 2] for i in range(52))
SUIT: tuple[int, ...] = tuple(i & 3 for i in range(52))
LOWBIT: tuple[int, ...] = tuple(_low_bit(i) for i in range(52))


def _encode(t: HighRank) -> int:
    v = t[0] * 16**5
    for k, r in enumerate(t[1:]):
        v += r * 16 ** (4 - k)
    return v


def _rank_product(ranks: Iterable[int]) -> int:
    p = 1
    for r in ranks:
        p *= _RANK_PRIMES[r - 2]
    return p


HIGH_INT: dict[int, int] = {}
FLUSH_INT: dict[int, int] = {}
DECODE_HIGH: dict[int, HighRank] = {}
DECODE_LOW: dict[int, LowRank] = {}


def _build_tables() -> None:
    # The 6,175 rank multisets (no rank more than 4 times), not suited.
    for combo in combinations_with_replacement(range(2, 15), 5):
        if max(Counter(combo).values()) > 4:
            continue
        ranks = sorted(combo, reverse=True)
        t = _rank_tuple(ranks, False)
        v = _encode(t)
        HIGH_INT[_rank_product(ranks)] = v
        DECODE_HIGH[v] = t
    # The 1,287 sets of five distinct ranks, suited.
    for combo in combinations(range(2, 15), 5):
        ranks = sorted(combo, reverse=True)
        t = _rank_tuple(ranks, True)
        v = _encode(t)
        FLUSH_INT[_rank_product(ranks)] = v
        DECODE_HIGH[v] = t
    # The 56 qualifying lows.
    for values in combinations(range(1, 9), 5):
        mask = 0
        for x in values:
            mask |= 1 << x
        DECODE_LOW[mask] = tuple(sorted(values, reverse=True))


_build_tables()

_NO_LOW = 1 << 10  # bigger than every low mask


# --- public functions ---------------------------------------------------------------


def _five(cards: Iterable[Card]) -> tuple[Card, ...]:
    five = tuple(cards)
    if len(five) != 5:
        raise ValueError(f"exactly 5 cards are needed, got {len(five)}")
    return five


def rank_high(cards: Iterable[Card]) -> HighRank:
    """Rank a 5-card high hand. A-2-3-4-5 (the wheel) is a 5-high straight."""
    five = _five(cards)
    a, b, c, d, e = (x.index for x in five)
    prod = PRIME[a] * PRIME[b] * PRIME[c] * PRIME[d] * PRIME[e]
    s = SUIT[a]
    flush = SUIT[b] == s and SUIT[c] == s and SUIT[d] == s and SUIT[e] == s
    v = (FLUSH_INT if flush else HIGH_INT).get(prod)
    if v is None:  # not five distinct cards
        return _slow_high(five)
    return DECODE_HIGH[v]


def rank_low(cards: Iterable[Card]) -> Optional[LowRank]:
    """Rank a 5-card 8-or-better low, or return None if it does not qualify.

    A qualifying low has five distinct ranks, all 8 or lower (ace counts as 1).
    The result is sorted descending, so tuples compare directly: lower is better.
    """
    five = _five(cards)
    mask = 0
    for x in five:
        mask |= LOWBIT[x.index]
    # Five cards set five bits only if all are low and of distinct values.
    return DECODE_LOW.get(mask)


_Triples = tuple[list[int], dict[int, list[int]], list[int]]


def _triples(idx: Sequence[int]) -> _Triples:
    """Board triples: all rank products, the products per flush suit, and the
    low masks of the triples with three distinct low values."""
    prods: list[int] = []
    flush: dict[int, list[int]] = {}
    lows: list[int] = []
    for a, b, c in combinations(idx, 3):
        p = PRIME[a] * PRIME[b] * PRIME[c]
        prods.append(p)
        s = SUIT[a]
        if SUIT[b] == s and SUIT[c] == s:
            flush.setdefault(s, []).append(p)
        m = LOWBIT[a] | LOWBIT[b] | LOWBIT[c]
        if bin(m).count("1") == 3:
            lows.append(m)
    return prods, flush, lows


def _best(hole: Sequence[int], triples: _Triples) -> tuple[HighRank, Optional[LowRank]]:
    prods, flush, lows = triples
    H = HIGH_INT
    F = FLUSH_INT
    best = 0
    lo = _NO_LOW
    for a, b in combinations(hole, 2):
        pp = PRIME[a] * PRIME[b]
        for tp in prods:
            v = H[pp * tp]
            if v > best:
                best = v
        sa = SUIT[a]
        if sa == SUIT[b]:
            fl = flush.get(sa)
            if fl:
                for tp in fl:
                    v = F[pp * tp]
                    if v > best:
                        best = v
        la = LOWBIT[a]
        lb = LOWBIT[b]
        if la and lb and la != lb:
            pm = la | lb
            for tm in lows:
                if not pm & tm:
                    m = pm | tm
                    if m < lo:
                        lo = m
    return DECODE_HIGH[best], DECODE_LOW.get(lo)


def _slow_omaha(
    hole: Sequence[Card], community: Sequence[Card]
) -> tuple[HighRank, Optional[LowRank]]:
    """The original algorithm, for input the tables do not cover (repeated cards)."""
    best_hi: Optional[HighRank] = None
    best_lo: Optional[LowRank] = None
    for two in combinations(hole, 2):
        for three in combinations(community, 3):
            five = two + three
            hi = _slow_high(five)
            if best_hi is None or hi > best_hi:
                best_hi = hi
            lo = _slow_low(five)
            if lo is not None and (best_lo is None or lo < best_lo):
                best_lo = lo
    assert best_hi is not None
    return best_hi, best_lo


def _check(hole: tuple, community: tuple) -> None:
    if len(hole) != 4:
        raise ValueError(f"an Omaha hand has 4 hole cards, got {len(hole)}")
    if len(community) < 3:
        raise ValueError(f"at least 3 board cards are needed, got {len(community)}")


def best_omaha(hand: Hand, board: Board) -> tuple[HighRank, Optional[LowRank]]:
    """Best Omaha high and low using exactly two hole cards and exactly three board cards.

    The high and the low are chosen independently (they may use different cards).
    """
    hole = tuple(hand)
    community = tuple(board)
    _check(hole, community)
    try:
        return _best([c.index for c in hole], _triples([c.index for c in community]))
    except KeyError:  # repeated cards
        return _slow_omaha(hole, community)


def best_omaha_many(
    hands: Iterable[Hand], board: Board
) -> list[tuple[HighRank, Optional[LowRank]]]:
    """`best_omaha` for several hands on one board; the board triples are built once."""
    community = tuple(board)
    triples: Optional[_Triples] = None
    out = []
    for hand in hands:
        hole = tuple(hand)
        _check(hole, community)
        if triples is None:
            triples = _triples([c.index for c in community])
        try:
            out.append(_best([c.index for c in hole], triples))
        except KeyError:  # repeated cards
            out.append(_slow_omaha(hole, community))
    return out
