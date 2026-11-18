"""Training records: every decision with what training needs, matched to the
result of its own hand.

`Recorder(keys=None)` records the decisions of the given controller keys (all
when None). Per submitted decision it keeps: table, hand id, seat, controller
(a small int code, `Recorder.codes` maps keys to codes), observation, mask,
action, log-probability (NaN when the controller returned actions only) and
the critic input (taken before the action is applied). Rows of a running hand
stay pending; when the hand finishes, each of its rows gets
`reward = rewards[seat]` of that hand's `FinishedHand` and becomes ready.
`drain()` returns the ready rows (finished hands only) in decision order. The
hand id is unique within the batch, so a record is never matched to the wrong
or an unfinished hand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Hashable, Iterable, Optional

import numpy as np

from .critic import N_SLOTS, critic_cards
from .encoder import OBS_DIM

__all__ = ["Recorder", "TrainingRecords"]


@dataclass(frozen=True, eq=False)
class TrainingRecords:
    """One row per decision (decision order), numpy arrays."""

    table: np.ndarray  # int32 (N,)
    hand_id: np.ndarray  # int64 (N,)
    seat: np.ndarray  # int32 (N,)
    controller: np.ndarray  # int32 (N,), codes of Recorder.codes
    obs: np.ndarray  # float32 (N, OBS_DIM)
    mask: np.ndarray  # bool (N, 3)
    action: np.ndarray  # int64 (N,)
    logp: np.ndarray  # float32 (N,), NaN when not given
    critic: np.ndarray  # int8 (N, 9, 4)
    reward: np.ndarray  # float32 (N,), the seat's net result in big bets

    def __len__(self) -> int:
        return int(self.table.shape[0])


# One pending / ready row.
_Row = tuple  # (seq, table, hand_id, seat, code, obs, mask, action, logp, critic)


class Recorder:
    def __init__(self, keys: Optional[Iterable[Hashable]] = None) -> None:
        self.keys: Optional[frozenset] = None if keys is None else frozenset(keys)
        self.codes: dict[Hashable, int] = {}
        if keys is not None:
            for k in keys:
                self.codes.setdefault(k, len(self.codes))
        self._pending: dict[int, list[_Row]] = {}
        self._ready: list[tuple[_Row, float]] = []
        self._seq = 0
        self._n_pending = 0

    # --- recording ---------------------------------------------------------------

    def wants(self, key: Hashable) -> bool:
        return self.keys is None or key in self.keys

    def code(self, key: Hashable) -> int:
        code = self.codes.get(key)
        if code is None:
            code = self.codes[key] = len(self.codes)
        return code

    def record(
        self,
        table: Any,
        table_index: int,
        hand_id: int,
        seat: int,
        key: Hashable,
        obs_row: np.ndarray,
        mask_row: np.ndarray,
        action: int,
        logp: Optional[float] = None,
    ) -> None:
        """Record one decision before its action is applied (the critic input is
        read from `table` now)."""
        critic = critic_cards(table, seat)
        row = (
            self._seq,
            int(table_index),
            int(hand_id),
            int(seat),
            self.code(key),
            np.array(obs_row, dtype=np.float32, copy=True),
            np.array(mask_row, dtype=bool, copy=True),
            int(action),
            math.nan if logp is None else float(logp),
            critic,
        )
        self._seq += 1
        self._pending.setdefault(int(hand_id), []).append(row)
        self._n_pending += 1

    def finish(self, hand_id: int, rewards: np.ndarray) -> None:
        """The hand `hand_id` is over: its pending rows get their seat's reward."""
        rows = self._pending.pop(int(hand_id), None)
        if not rows:
            return
        self._n_pending -= len(rows)
        for row in rows:
            self._ready.append((row, float(rewards[row[3]])))

    # --- reading -----------------------------------------------------------------

    @property
    def n_ready(self) -> int:
        return len(self._ready)

    @property
    def n_pending(self) -> int:
        return self._n_pending

    def drain(self) -> TrainingRecords:
        """The ready rows (finished hands only), in decision order; they are
        removed from the recorder."""
        ready = sorted(self._ready, key=lambda item: item[0][0])
        self._ready = []
        n = len(ready)
        if n:
            obs = np.stack([r[5] for r, _ in ready])
            mask = np.stack([r[6] for r, _ in ready])
            critic = np.stack([r[9] for r, _ in ready])
        else:
            obs = np.zeros((0, OBS_DIM), dtype=np.float32)
            mask = np.zeros((0, 3), dtype=bool)
            critic = np.zeros((0, N_SLOTS, 4), dtype=np.int8)
        return TrainingRecords(
            table=np.array([r[1] for r, _ in ready], dtype=np.int32),
            hand_id=np.array([r[2] for r, _ in ready], dtype=np.int64),
            seat=np.array([r[3] for r, _ in ready], dtype=np.int32),
            controller=np.array([r[4] for r, _ in ready], dtype=np.int32),
            obs=obs.astype(np.float32, copy=False),
            mask=mask.astype(bool, copy=False),
            action=np.array([r[7] for r, _ in ready], dtype=np.int64),
            logp=np.array([r[8] for r, _ in ready], dtype=np.float32),
            critic=critic.astype(np.int8, copy=False),
            reward=np.array([w for _, w in ready], dtype=np.float32),
        )
