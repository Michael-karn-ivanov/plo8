"""Critic input: the other seats' hole cards, relative to a seat.

PRIVILEGED. This is the only way hidden information leaves rl_env during a
hand (after a hand, evaluation tools read `finished_hole_cards`; both go
through `Table.privileged_hole_cards`). It is a separate function and a
separate array; `encoder.py` and `tracker.py` never import this module
(checked by a test), and nothing a playing policy receives may be derived from
it.

`critic_cards(table, seat)` -> int8 (9, 4): row `k-1` holds the hole cards of
relative slot `k = 1..9` (the same slot order as the observation: the k-th
participant clockwise after `seat`), as `Card.index`, sorted ascending;
`NO_CARD = 52` marks a slot beyond the table size (a valid row of a 53-row
card embedding).

`expand_critic(cards)` -> float32 (B, PRIV_DIM): per slot a 52-wide multi-hot
of its hole cards (`NO_CARD` contributes nothing) - the reference for the
learner's expansion.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["NO_CARD", "N_SLOTS", "PRIV_DIM", "critic_cards", "expand_critic"]

NO_CARD = 52
N_SLOTS = 9
PRIV_DIM = N_SLOTS * 52  # 468


def critic_cards(table: Any, seat: int, out: Any = None) -> np.ndarray:
    """The other participants' hole cards relative to `seat` (see the module
    docstring). Read through `Table.privileged_hole_cards()` (the engine's
    privileged step-mode accessor)."""
    cards = table.privileged_hole_cards()
    seats = sorted(cards)
    if out is None:
        out = np.full((N_SLOTS, 4), NO_CARD, dtype=np.int8)
    else:
        out[:] = NO_CARD
    try:
        me = seats.index(seat)
    except ValueError:
        raise ValueError(f"seat {seat} is not dealt into the hand") from None
    n = len(seats)
    for k in range(1, n):
        hand = cards[seats[(me + k) % n]]
        out[k - 1] = sorted(c.index for c in hand)
    return out


def expand_critic(cards: Any) -> np.ndarray:
    """(B, 9, 4) card indices -> (B, PRIV_DIM) float32 multi-hot (a single
    (9, 4) array is treated as B = 1)."""
    arr = np.asarray(cards)
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3 or arr.shape[1:] != (N_SLOTS, 4):
        raise ValueError(f"expected (B, {N_SLOTS}, 4) card indices, got shape {arr.shape}")
    idx = arr.astype(np.int64)
    if idx.size and (idx.min() < 0 or idx.max() > NO_CARD):
        raise ValueError("card indices must be in 0..52")
    b = arr.shape[0]
    wide = np.zeros((b, N_SLOTS, NO_CARD + 1), dtype=np.float32)
    rows = np.arange(b)[:, None, None]
    slots = np.arange(N_SLOTS)[None, :, None]
    wide[rows, slots, idx] = 1.0
    return np.ascontiguousarray(wide[:, :, :NO_CARD]).reshape(b, PRIV_DIM)
