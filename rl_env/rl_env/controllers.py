"""Controllers shipped with rl_env.

Only `RandomController` (needed by rl_env's own bench, fuzz test and tests).
The baseline bots (RandomBot, CallingStation, Maniac, RuleBot) belong to arena.
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np

__all__ = ["RandomController"]


class RandomController:
    """Uniform over the masked actions, from its own seeded `np.random.Generator`.

    Reads only `mask` (no views, no observation vectors), so it may act in a
    batch without observations.

    With `with_logp=True` `act` also returns the log-probability of each chosen
    action (`-log(number of allowed actions)`, float32)."""

    wants_views = False
    wants_obs = False

    def __init__(self, seed: int = 0, *, with_logp: bool = False) -> None:
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.with_logp = with_logp

    def act(self, batch: Any) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        mask = np.asarray(batch.mask, dtype=bool)
        counts = mask.sum(axis=1)
        if mask.shape[0] and counts.min() == 0:
            raise ValueError("a row has no allowed action")
        u = self.rng.random(mask.shape[0])
        k = np.minimum((u * counts).astype(np.int64), counts - 1)
        cums = np.cumsum(mask, axis=1)
        actions = np.argmax(cums > k[:, None], axis=1).astype(np.int64)
        if self.with_logp:
            return actions, (-np.log(counts)).astype(np.float32)
        return actions
