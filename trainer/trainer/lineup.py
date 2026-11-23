"""Who sits where: one controller key per seat of a new hand.

`draw_lineup(rng, n_seats, opponents)` puts the learner (`LEARNER_KEY`) in 1 to
`min(max_learner_seats, n_seats - 1)` seats (uniform count, random positions,
never every seat) and an opponent name, drawn uniformly and independently, in
every other seat. Later opponent kinds (frozen snapshots, league entries) only
add other keys and their controllers; the table batch and the recorder already
work per key.

The draw order is fixed - count, positions, one opponent draw for EVERY seat,
then the learner seats overwritten - so the number of draws does not depend on
the outcome and a lineup is a function of the generator state and `n_seats`.
"""

from __future__ import annotations

import numbers
from typing import Any, Sequence

import numpy as np

__all__ = ["LEARNER_KEY", "DEFAULT_OPPONENTS", "MAX_LEARNER_SEATS", "draw_lineup"]

LEARNER_KEY = "learner"
DEFAULT_OPPONENTS = ("random", "calling", "maniac", "rule")  # = arena.BOT_NAMES
MAX_LEARNER_SEATS = 3


def _int_at_least(name: str, value: Any, low: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an int, got {value!r}")
    value = int(value)
    if value < low:
        raise ValueError(f"{name} must be >= {low}, got {value}")
    return value


def draw_lineup(
    rng: np.random.Generator,
    n_seats: int,
    opponents: Sequence[str],
    max_learner_seats: int = MAX_LEARNER_SEATS,
) -> tuple[str, ...]:
    """Controller key per seat (`LEARNER_KEY` or a name of `opponents`)."""
    n_seats = _int_at_least("n_seats", n_seats, 2)
    max_learner_seats = _int_at_least("max_learner_seats", max_learner_seats, 1)
    if len(opponents) == 0:
        raise ValueError("opponents must not be empty")
    k = int(rng.integers(1, min(max_learner_seats, n_seats - 1) + 1))  # never every seat
    learner = rng.choice(n_seats, size=k, replace=False)  # random positions
    keys = [opponents[int(i)] for i in rng.integers(len(opponents), size=n_seats)]
    for s in learner:
        keys[int(s)] = LEARNER_KEY
    return tuple(keys)
