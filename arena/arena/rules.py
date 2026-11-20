"""RuleBot: a simple, readable tight-aggressive baseline.

All numbers live in `RULE_PARAMS` below: tuning means editing one block.

Preflop it scores its four hole cards (`preflop_score`) and raises / calls /
folds on the score. After the flop it asks the engine's evaluator
(`best_omaha`: exactly two hole cards and three board cards) for the high
category and whether it has a qualifying low, and looks up the nut low, the
nut-low draw and the nut-flush draw in tables over ranks and suits
(`postflop_class`). No hand comparison is re-implemented: the nut-low table is
built from `engine.evaluator.rank_low`.

Design decisions (spec 1.4):
- Nut low: the hand's low equals the best low any two hole cards could make on
  this board. Card removal by own hole cards is not considered; a nut low that
  may be tied (quartered) still counts as the nut.
- Nut-low draw (flop and turn): no nut low now, and some low rank not on the
  board would give the hand the nut low.
- Nut-flush draw (flop and turn): a suit with exactly two board cards and at
  least two hole cards, one of which is the highest rank of that suit not on
  the board (own cards not removed).

RuleBot reads only `batch.views[i]` (own hole cards, board, street, `raises`)
and `batch.legal[i]`; it is stateless and deterministic.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np

from engine import LegalActions, TableView
from engine.evaluator import best_omaha, rank_low
from entities import Board, Card, Hand, Rank, Street, Suit
from rl_env import BET_RAISE, CHECK_CALL, FOLD

from .bots import Bot, _masked

__all__ = [
    "RuleParams",
    "RULE_PARAMS",
    "PostflopClass",
    "NUT_LOW_PAIRS",
    "nut_low_pairs",
    "preflop_score",
    "postflop_class",
    "decide",
    "RuleBot",
]


@dataclass(frozen=True)
class RuleParams:
    # preflop: base low pair, only the best one counts
    a2: int = 30
    a3: int = 20
    two_three: int = 15
    extra_low: int = 6  # per distinct low rank (A..8) beyond 2, only with a base low pair
    suited_ace: int = 10  # per suit holding an ace and another hole card (max 2 suits)
    suited_other: int = 4  # per other suit with >= 2 hole cards
    aces: int = 10  # a pair of aces
    big_two: int = 4  # exactly 2 distinct big ranks (T..A)
    big_three: int = 10  # 3 or more distinct big ranks
    connected: int = 3  # per adjacent pair of distinct ranks (ace both high and low)
    trips: int = -20  # three of one rank
    quads: int = -40  # four of one rank
    nine: int = -6  # per nine
    middle: int = -3  # per T/J/Q when the hand has no ace and no other card >= T
    # preflop thresholds
    raise_at: int = 35
    call_at: int = 18
    # postflop
    strong_high_cat: int = 3  # evaluator category: 3 = three of a kind, 4 = straight, ...
    two_pair_cat: int = 2
    value_raise_cap: int = 3  # one-sided strong hand: bet/raise while bets this street < 3


RULE_PARAMS = RuleParams()


# --- nut-low table ------------------------------------------------------------------

_LOW_VALUES = tuple(range(1, 9))  # ace = 1 .. eight


def _low_card(v: int) -> Card:
    """A card of low value `v` (suit irrelevant for the low)."""
    rank = Rank.ACE if v == 1 else Rank(v)
    return Card(rank, Suit.CLUBS)


_LOW_CARD = {v: _low_card(v) for v in _LOW_VALUES}


def _build_nut_low_pairs() -> Mapping[frozenset, frozenset]:
    """For every set of 3..5 distinct board low values: the hole low-value pairs
    whose best low on that board equals the best low any two hole cards could
    make. Pairs sharing a value with the board use the same card object, so
    `rank_low` sees fewer than five distinct values and rejects them, exactly
    like a counterfeited low."""
    table: dict[frozenset, frozenset] = {}
    pairs = list(combinations(_LOW_VALUES, 2))
    for size in (3, 4, 5):
        for board in combinations(_LOW_VALUES, size):
            triples = list(combinations(board, 3))
            best = None
            winners: list[tuple[int, int]] = []
            for pair in pairs:
                low = None
                for tri in triples:
                    r = rank_low([_LOW_CARD[v] for v in pair + tri])
                    if r is not None and (low is None or r < low):
                        low = r
                if low is None:
                    continue
                if best is None or low < best:
                    best = low
                    winners = [pair]
                elif low == best:
                    winners.append(pair)
            table[frozenset(board)] = frozenset(frozenset(p) for p in winners)
    return MappingProxyType(table)


NUT_LOW_PAIRS: Mapping[frozenset, frozenset] = _build_nut_low_pairs()
_NO_PAIRS: frozenset = frozenset()


def nut_low_pairs(board_low_values: Iterable[int]) -> frozenset:
    """The nut hole low-value pairs for a board's distinct low values (empty
    when the board has fewer than three)."""
    return NUT_LOW_PAIRS.get(frozenset(board_low_values), _NO_PAIRS)


def _hole_low_pairs(hand: Hand) -> list[frozenset]:
    values = sorted({c.rank.low_value for c in hand if c.rank.low_value <= 8})
    return [frozenset(p) for p in combinations(values, 2)]


def _board_low_values(board: Board) -> frozenset:
    return frozenset(c.rank.low_value for c in board if c.rank.low_value <= 8)


def _has_nut_low(pairs: list[frozenset], board_lows: frozenset) -> bool:
    nut = NUT_LOW_PAIRS.get(board_lows)
    return bool(nut) and any(p in nut for p in pairs)


def _has_nut_low_draw(pairs: list[frozenset], board_lows: frozenset) -> bool:
    for r in _LOW_VALUES:
        if r in board_lows:
            continue
        if _has_nut_low(pairs, board_lows | {r}):
            return True
    return False


def _has_nut_flush_draw(hand: Hand, board: Board) -> bool:
    for suit in Suit:
        on_board = {int(c.rank) for c in board if c.suit is suit}
        if len(on_board) != 2:
            continue
        mine = [int(c.rank) for c in hand if c.suit is suit]
        if len(mine) < 2:
            continue
        top = next(r for r in range(14, 1, -1) if r not in on_board)
        if top in mine:
            return True
    return False


# --- preflop ------------------------------------------------------------------------


def preflop_score(hand: Hand, params: RuleParams = RULE_PARAMS) -> int:
    """Score of four hole cards (see RuleParams for the terms)."""
    p = params
    cards = tuple(hand)
    ranks = [int(c.rank) for c in cards]
    counts = Counter(ranks)
    ace = Rank.ACE in counts
    lows = {c.rank.low_value for c in cards if c.rank.low_value <= 8}

    if ace and Rank.TWO in counts:
        base = p.a2
    elif ace and Rank.THREE in counts:
        base = p.a3
    elif Rank.TWO in counts and Rank.THREE in counts:
        base = p.two_three
    else:
        base = 0
    score = base
    if base:
        score += p.extra_low * max(0, len(lows) - 2)

    suited_aces = suited_others = 0
    for suit in Suit:
        group = [c for c in cards if c.suit is suit]
        if len(group) >= 2:
            if any(c.rank is Rank.ACE for c in group):
                suited_aces += 1
            else:
                suited_others += 1
    score += p.suited_ace * min(suited_aces, 2) + p.suited_other * suited_others

    if counts[Rank.ACE] >= 2:
        score += p.aces

    big = sum(1 for r in counts if r >= Rank.TEN)
    if big == 2:
        score += p.big_two
    elif big >= 3:
        score += p.big_three

    values = set(counts)
    if ace:
        values.add(1)
    score += p.connected * sum(1 for v in values if v + 1 in values)

    for c in counts.values():
        if c == 3:
            score += p.trips
        elif c == 4:
            score += p.quads

    score += p.nine * counts[Rank.NINE]

    if not ace:
        for i, r in enumerate(ranks):
            if Rank.TEN <= r <= Rank.QUEEN and not any(
                ranks[j] >= Rank.TEN for j in range(len(ranks)) if j != i
            ):
                score += p.middle
    return score


# --- postflop -----------------------------------------------------------------------


@dataclass(frozen=True)
class PostflopClass:
    high_cat: int  # evaluator category 0 (high card) .. 8 (straight flush)
    has_low: bool  # a qualifying 8-or-better low
    nut_low: bool
    nut_low_draw: bool  # flop and turn only
    nut_flush_draw: bool  # flop and turn only


def postflop_class(hand: Hand, board: Board) -> PostflopClass:
    """The hand's class on a board of 3..5 cards (one evaluator call)."""
    high, low = best_omaha(hand, board)
    pairs = _hole_low_pairs(hand)
    board_lows = _board_low_values(board)
    nut = _has_nut_low(pairs, board_lows)
    drawing = len(board) < 5
    return PostflopClass(
        high_cat=high[0],
        has_low=low is not None,
        nut_low=nut,
        nut_low_draw=drawing and not nut and _has_nut_low_draw(pairs, board_lows),
        nut_flush_draw=drawing and _has_nut_flush_draw(hand, board),
    )


# --- decision -----------------------------------------------------------------------


def _allowed(action: int, legal: LegalActions) -> int:
    """Fold when checking is free -> check; raise when not allowed -> call."""
    if action == FOLD and legal.can_check:
        return CHECK_CALL
    if action == BET_RAISE and not legal.can_raise:
        return CHECK_CALL
    return action


def _check_or_fold(legal: LegalActions) -> int:
    return CHECK_CALL if legal.can_check else FOLD


def _preflop(view: TableView, legal: LegalActions, p: RuleParams) -> int:
    score = preflop_score(view.hole_cards, p)  # type: ignore[arg-type]
    if score >= p.raise_at:
        return BET_RAISE
    if score >= p.call_at:
        if legal.can_check or view.raises <= 1:
            return CHECK_CALL
        return FOLD
    return _check_or_fold(legal)


def _postflop(view: TableView, legal: LegalActions, p: RuleParams) -> int:
    cls = postflop_class(view.hole_cards, view.board)  # type: ignore[arg-type]
    n = view.raises
    strong_high = cls.high_cat >= p.strong_high_cat
    if strong_high and cls.nut_low:
        return BET_RAISE
    if strong_high or cls.nut_low:
        return BET_RAISE if n < p.value_raise_cap else CHECK_CALL
    made = cls.has_low or cls.high_cat >= p.two_pair_cat
    if view.board.street is not Street.RIVER:
        if made or cls.nut_low_draw or cls.nut_flush_draw:
            return CHECK_CALL
        return _check_or_fold(legal)
    if made:
        if legal.can_check or n == 1:
            return CHECK_CALL
        return FOLD
    return _check_or_fold(legal)


def decide(view: TableView, legal: LegalActions, params: RuleParams = RULE_PARAMS) -> int:
    """RuleBot's action for one decision, always allowed by the mask."""
    if view.hole_cards is None:
        raise ValueError("RuleBot needs the acting player's hole cards in the view")
    if view.board.street is Street.PREFLOP:
        action = _preflop(view, legal, params)
    else:
        action = _postflop(view, legal, params)
    return _allowed(action, legal)


class RuleBot(Bot):
    """Tight-aggressive rule-based baseline (reads views)."""

    name = "rule"
    wants_views = True
    wants_obs = False

    def __init__(self, params: RuleParams = RULE_PARAMS) -> None:
        self.params = params

    def act(self, batch: Any) -> np.ndarray:
        views = batch.views
        if views is None:
            raise ValueError("RuleBot needs DecisionBatch.views (wants_views = True)")
        params = self.params
        out = np.array(
            [decide(v, lg, params) for v, lg in zip(views, batch.legal)], dtype=np.int64
        )
        return _masked(out, batch.mask)
