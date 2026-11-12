"""Pot construction (side pots) and hi/lo pot splitting. Amounts are integer chips."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, NamedTuple, Optional, Protocol, Sequence

from .evaluator import HighRank, LowRank

__all__ = ["Pot", "PotSplit", "build_pots", "split_pot"]


class _Contributor(Protocol):
    seat: Optional[int]
    total_chips: int
    folded: bool


@dataclass
class Pot:
    amount: int
    eligible: tuple[int, ...]  # seats of non-folded players who can win this pot

    @property
    def uncontested(self) -> bool:
        return len(self.eligible) == 1


class PotSplit(NamedTuple):
    high: dict[int, int]
    low: dict[int, int]

    @property
    def shares(self) -> tuple[tuple[int, str, int], ...]:
        """Individual shares as (seat, "high" | "low", chips): the high winners
        first, then the low winners, each in odd-chip order (left of the button
        first). Zero-chip shares are left out. The dealer credits them one by one."""
        return tuple((s, "high", a) for s, a in self.high.items() if a > 0) + tuple(
            (s, "low", a) for s, a in self.low.items() if a > 0
        )


def build_pots(players: Iterable[_Contributor]) -> list[Pot]:
    """Build the main pot and side pots from each player's total contribution.

    Folded players' chips stay in the pots, but they are not eligible. Adjacent
    levels with the same eligible players are merged into one pot. A pot with a
    single eligible player is an uncalled bet / uncontested pot for that player.
    """
    contribs = [(p.seat, p.total_chips, p.folded) for p in players if p.total_chips > 0]
    levels = sorted({c for _, c, _ in contribs})
    pots: list[Pot] = []
    prev = 0
    carry = 0
    for level in levels:
        amount = carry + sum(min(c, level) - min(c, prev) for _, c, _ in contribs)
        eligible = tuple(sorted(s for s, c, folded in contribs if not folded and c >= level))
        prev = level
        if not eligible:
            # Only folded players reached this level: carry their chips along and
            # add them to the last pot below.
            carry = amount
            continue
        carry = 0
        if pots and pots[-1].eligible == eligible:
            pots[-1].amount += amount
        else:
            pots.append(Pot(amount, eligible))
    if carry:
        if pots:
            pots[-1].amount += carry
        else:  # pragma: no cover - everyone folded, which the dealer never allows
            pots.append(Pot(carry, ()))
    return pots


def _divide(amount: int, winners: Sequence[int]) -> dict[int, int]:
    """Split evenly; odd chips go one each to the first winners (left of the button)."""
    if not winners:
        return {}
    share, rest = divmod(amount, len(winners))
    return {seat: share + (1 if i < rest else 0) for i, seat in enumerate(winners)}


def split_pot(
    pot: Pot,
    hands: Mapping[int, tuple[HighRank, Optional[LowRank]]],
    order: Sequence[int],
) -> PotSplit:
    """Split a pot between its eligible players.

    `hands` maps seat -> (high rank, low rank or None) from `best_omaha`.
    `order` lists seats starting with the first seat left of the button; it
    decides who gets odd chips.

    If any eligible player has a qualifying low, half goes to the best high and
    half to the best low (the odd chip from halving goes to the high half);
    otherwise the best high takes the whole pot. Ties split each half.
    `PotSplit.shares` lists the individual shares.
    """
    position = {seat: i for i, seat in enumerate(order)}
    contenders = sorted(
        (s for s in pot.eligible if s in hands),
        key=lambda s: (position.get(s, len(position)), s),
    )
    if not contenders:
        raise ValueError("no eligible player has a hand for this pot")

    best_hi = max(hands[s][0] for s in contenders)
    hi_winners = [s for s in contenders if hands[s][0] == best_hi]

    lows = [lo for lo in (hands[s][1] for s in contenders) if lo is not None]
    if lows:
        best_lo = min(lows)
        lo_winners = [s for s in contenders if hands[s][1] == best_lo]
        low_half = pot.amount // 2
        high_half = pot.amount - low_half
    else:
        lo_winners = []
        low_half = 0
        high_half = pot.amount

    return PotSplit(high=_divide(high_half, hi_winners), low=_divide(low_half, lo_winners))
