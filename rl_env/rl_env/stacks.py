"""Starting stacks used for training, and the inputs of a hand.

With probability `deep_share` all seats start at `deep_bb` (100 BB = 50 big
bets, effectively deep in limit poker); otherwise each seat gets an
independent uniform stack in `[short_min_bb, short_max_bb]`, rounded to
0.01 BB (whole chips at 100 chips/BB), so all-ins and side pots appear in
training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

__all__ = ["StackConfig", "draw_stacks", "HandSpec", "HandSampler"]

_SEED_HIGH = 2**63 - 1  # deck seeds are drawn in [0, 2**63 - 1)


@dataclass(frozen=True)
class StackConfig:
    deep_share: float = 0.8
    deep_bb: float = 100.0
    short_min_bb: float = 5.0
    short_max_bb: float = 150.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.deep_share <= 1.0:
            raise ValueError(f"deep_share must be in [0, 1], got {self.deep_share!r}")
        if not self.deep_bb > 0:
            raise ValueError(f"deep_bb must be positive, got {self.deep_bb!r}")
        if not 0 < self.short_min_bb <= self.short_max_bb:
            raise ValueError(
                f"need 0 < short_min_bb <= short_max_bb, got "
                f"{self.short_min_bb!r}, {self.short_max_bb!r}"
            )


def draw_stacks(
    rng: np.random.Generator, n: int, cfg: StackConfig = StackConfig()
) -> tuple[float, ...]:
    """Starting stacks (BB) of `n` seats, drawn from `rng` (see the module docstring)."""
    if rng.random() < cfg.deep_share:
        return (float(cfg.deep_bb),) * n
    draws = rng.uniform(cfg.short_min_bb, cfg.short_max_bb, size=n)
    return tuple(max(0.01, round(float(x), 2)) for x in draws)


@dataclass(frozen=True)
class HandSpec:
    """The four inputs of a hand."""

    n_seats: int
    stacks_bb: tuple[float, ...]
    button: int
    deck_seed: int


class HandSampler:
    """Draws `HandSpec`s from one `np.random.Generator(np.random.PCG64(seed))`:
    the size (if not given), the stacks, a uniform button and a 63-bit deck seed,
    always in that order, so a run is reproducible from the sampler's seed."""

    def __init__(
        self,
        seed: int,
        sizes: Iterable[int] = range(2, 11),
        cfg: StackConfig = StackConfig(),
    ) -> None:
        self.sizes = tuple(int(s) for s in sizes)
        if not self.sizes or any(not 2 <= s <= 10 for s in self.sizes):
            raise ValueError(f"table sizes must be in 2..10, got {self.sizes}")
        self.cfg = cfg
        self.rng = np.random.Generator(np.random.PCG64(seed))

    def sample(self, n_seats: Optional[int] = None) -> HandSpec:
        rng = self.rng
        if n_seats is None:
            n = self.sizes[int(rng.integers(len(self.sizes)))]
        else:
            n = int(n_seats)
            if not 2 <= n <= 10:
                raise ValueError(f"n_seats must be 2-10, got {n}")
        stacks = draw_stacks(rng, n, self.cfg)
        button = int(rng.integers(n))
        deck_seed = int(rng.integers(0, _SEED_HIGH, dtype=np.int64))
        return HandSpec(n, stacks, button, deck_seed)
