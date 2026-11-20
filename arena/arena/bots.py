"""Baseline bots: RandomBot, CallingStation, Maniac (RuleBot is in rules.py).

Every bot implements the rl_env `Controller` protocol (`wants_views`,
`act(batch) -> np.ndarray[int64, (B,)]`), has a short `name` and is stateless
between calls (RandomBot's only state is its generator). They see only what a
player sees: `DecisionBatch.mask`, `legal`, `views` (RuleBot only).

Every bot finishes with `_masked(actions, batch.mask)`: an action whose mask
bit is off becomes CHECK_CALL (always allowed), so mask safety never depends on
a rule being written correctly. For the three simple bots it never triggers.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rl_env import BET_RAISE, CHECK_CALL, RandomController

__all__ = [
    "Bot",
    "RandomBot",
    "CallingStation",
    "Maniac",
    "BOT_NAMES",
    "make_bot",
]


def _masked(actions: Any, mask: Any) -> np.ndarray:
    """`actions` with every action outside its row's mask replaced by CHECK_CALL."""
    a = np.asarray(actions, dtype=np.int64).reshape(-1)
    m = np.asarray(mask, dtype=bool)
    if a.shape[0] == 0:
        return a
    ok = m[np.arange(a.shape[0]), a]
    return np.where(ok, a, CHECK_CALL).astype(np.int64)


class Bot:
    """Base of the arena bots (a `Controller`). Reads neither views nor
    observation vectors unless a subclass says so."""

    name: str = ""
    wants_views: bool = False
    wants_obs: bool = False

    def act(self, batch: Any) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class RandomBot(Bot):
    """Uniform over the actions allowed by the mask (never folds when checking
    is free), from rl_env's `RandomController` seeded at creation."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self._random = RandomController(self.seed)

    def act(self, batch: Any) -> np.ndarray:
        return _masked(self._random.act(batch), batch.mask)

    def __repr__(self) -> str:
        return f"RandomBot(seed={self.seed})"


class CallingStation(Bot):
    """Always checks or calls."""

    name = "calling"

    def act(self, batch: Any) -> np.ndarray:
        n = len(batch.mask)
        return _masked(np.full(n, CHECK_CALL, dtype=np.int64), batch.mask)


class Maniac(Bot):
    """Bets or raises whenever allowed, otherwise checks or calls. Never folds."""

    name = "maniac"

    def act(self, batch: Any) -> np.ndarray:
        mask = np.asarray(batch.mask, dtype=bool).reshape(-1, 3)
        actions = np.where(mask[:, BET_RAISE], BET_RAISE, CHECK_CALL).astype(np.int64)
        return _masked(actions, mask)


BOT_NAMES = ("random", "calling", "maniac", "rule")


def make_bot(name: str, seed: int = 0) -> Bot:
    """A bot from its short name; `seed` is used by `random`, ignored otherwise."""
    if name == "random":
        return RandomBot(seed)
    if name == "calling":
        return CallingStation()
    if name == "maniac":
        return Maniac()
    if name == "rule":
        from .rules import RuleBot  # rules imports this module

        return RuleBot()
    raise ValueError(f"unknown bot {name!r}; known bots: {', '.join(BOT_NAMES)}")
