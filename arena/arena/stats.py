"""Pure accounting: per-deal chips -> big bets per 100 hands and a 95% interval.

No engine, no tables. Results are integers (chips) until here; they are
converted to big bets (1 big bet = 2 BB = 200 chips at 100 chips/BB) only in
`summarize`.

One sample = one rotation group (all `table_size` hands of a deal), expressed
per hand, so the correlation between the rotations of one deal is accounted
for: `ci95 = 100 * 1.96 * sd(samples) / sqrt(deals)` (in bb/100).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "CHIPS_PER_BB",
    "CHIPS_PER_BIG_BET",
    "Z95",
    "AccountingError",
    "Summary",
    "summarize",
    "to_chips",
]

CHIPS_PER_BB = 100  # rl_env tables run at 100 chips per big blind
CHIPS_PER_BIG_BET = 2 * CHIPS_PER_BB  # one big bet = 2 BB
Z95 = 1.96


class AccountingError(RuntimeError):
    """Nets of a hand, or the totals of a match, do not sum to zero in chips.
    Never "fixed up": it is a bug somewhere and must be looked at."""


def to_chips(net_bb: float, chips_per_bb: int = CHIPS_PER_BB) -> int:
    """A net amount in BB as integer chips (rounded once)."""
    return int(round(float(net_bb) * chips_per_bb))


@dataclass(frozen=True)
class Summary:
    hands: int  # k * n * deals: hands played by the name's seats
    deals: int
    total_chips: int
    total_bb: float  # big bets
    bb_per_100: float  # big bets per 100 hands
    ci95: float  # half-width of the 95% interval, in bb/100 (nan with one deal)
    samples: np.ndarray  # float64 (deals,), big bets per hand per rotation group


def summarize(
    chips_per_deal: Sequence[int] | np.ndarray,
    k: int,
    n: int,
    chips_per_big_bet: int = CHIPS_PER_BIG_BET,
) -> Summary:
    """Statistics of one report name holding `k` of the `n` lineup positions.

    `chips_per_deal[d]` is the name's total over the `n` rotations of deal `d`
    (integer chips). Every rotation group weighs `k * n` hands, so
    `bb_per_100 = 100 * mean(samples) = 100 * total_bb / hands`."""
    if k < 1 or n < 1 or k > n:
        raise ValueError(f"need 1 <= k <= n, got k={k}, n={n}")
    chips = np.asarray(chips_per_deal)
    if chips.ndim != 1 or chips.shape[0] < 1:
        raise ValueError("need a 1-d array with at least one deal")
    if not np.issubdtype(chips.dtype, np.integer):
        raise ValueError(f"chips must be integers, got dtype {chips.dtype}")
    chips = chips.astype(np.int64)
    deals = int(chips.shape[0])
    per_group = k * n
    hands = per_group * deals
    total_chips = int(chips.sum())
    total_bb = total_chips / chips_per_big_bet
    samples = chips.astype(np.float64) / per_group / chips_per_big_bet
    bb_per_100 = 100.0 * total_chips / (chips_per_big_bet * hands)
    if deals > 1:
        ci95 = 100.0 * Z95 * float(np.std(samples, ddof=1)) / math.sqrt(deals)
    else:
        ci95 = float("nan")
    samples.setflags(write=False)
    return Summary(
        hands=hands,
        deals=deals,
        total_chips=total_chips,
        total_bb=total_bb,
        bb_per_100=bb_per_100,
        ci95=ci95,
        samples=samples,
    )
