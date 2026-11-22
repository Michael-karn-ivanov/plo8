"""Per-hand log lines and the final summary. Pure.

Amounts are kept in integer chips (100 per BB, as the server and arena count),
so totals do not drift over thousands of hands; they are converted to BB / big
bets only for output, rounded half away from zero to two decimals.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Optional

__all__ = [
    "CHIPS_PER_BB",
    "to_chips",
    "format_chips",
    "format_bb",
    "Stats",
    "hand_line",
    "summary",
]

CHIPS_PER_BB = 100
_CENT = Decimal("0.01")


def to_chips(amount_bb: float) -> int:
    return int(round(amount_bb * CHIPS_PER_BB))


def _fmt(value: Decimal, sign: bool) -> str:
    q = value.quantize(_CENT, rounding=ROUND_HALF_UP)
    if q == 0:
        q = abs(q)  # no "-0.00"
    return f"{q:+}" if sign else f"{q}"


def format_chips(chips: int, sign: bool = False) -> str:
    """Chips as BB with two decimals (`+1.50` with `sign`)."""
    return _fmt(Decimal(chips) / CHIPS_PER_BB, sign)


def format_bb(amount_bb: float, sign: bool = False) -> str:
    return format_chips(to_chips(amount_bb), sign)


class Stats:
    """Nets of the hands the bot was dealt into, and its rebuys."""

    def __init__(self) -> None:
        self.hands = 0
        self.net_chips = 0
        self.rebought_chips = 0
        self.nets: list[tuple[int, int]] = []  # (hand number, net chips)
        # (hand number the rebuy followed, chips); None before the first hand
        self.rebuys: list[tuple[Optional[int], int]] = []

    def add_hand(self, hand_number: int, net_bb: float) -> None:
        chips = to_chips(net_bb)
        self.hands += 1
        self.net_chips += chips
        self.nets.append((hand_number, chips))

    def add_rebuy(self, amount_bb: float, after_hand: Optional[int] = None) -> None:
        chips = to_chips(amount_bb)
        self.rebought_chips += chips
        self.rebuys.append((after_hand, chips))

    @property
    def net_bb(self) -> float:
        return self.net_chips / CHIPS_PER_BB

    @property
    def rebought_bb(self) -> float:
        return self.rebought_chips / CHIPS_PER_BB

    def big_bets_per_100(self) -> Optional[Decimal]:
        """`100 * (net_bb / 2) / hands` (one big bet = 2 BB); None with no hands."""
        if self.hands == 0:
            return None
        # 100 * (chips / 100 / 2) / hands = chips / (2 * hands)
        return Decimal(self.net_chips) / Decimal(2 * self.hands)


def hand_line(
    hand_number: int, net_bb: float, stack_bb: float, rebuy_bb: Optional[float] = None
) -> str:
    """`hand 12: net -2.00 BB, stack 98.00 BB[, rebuy 2.00 BB]`."""
    line = f"hand {hand_number}: net {format_bb(net_bb, sign=True)} BB, stack {format_bb(stack_bb)} BB"
    if rebuy_bb:
        line += f", rebuy {format_bb(rebuy_bb)} BB"
    return line


def summary(name: str, label: str, stats: Stats, final_stack_bb: Optional[float]) -> str:
    """`RuleBot (rule): 200 hands, net +34.50 BB = +8.63 big bets/100 hands,
    rebought 120.00 BB, final stack 100.00 BB`."""
    rate = stats.big_bets_per_100()
    rate_text = "n/a" if rate is None else _fmt(rate, sign=True)
    final = "n/a" if final_stack_bb is None else f"{format_bb(final_stack_bb)} BB"
    return (
        f"{name} ({label}): {stats.hands} hands, net {format_chips(stats.net_chips, sign=True)} BB"
        f" = {rate_text} big bets/100 hands, rebought {format_chips(stats.rebought_chips)} BB,"
        f" final stack {final}"
    )
