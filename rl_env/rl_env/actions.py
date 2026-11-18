"""The three actions of a learning player and their link to the engine.

A learning player has exactly three actions: fold, check-or-call, bet-or-raise
(fixed limit: no bet size to choose). The mask is derived from the engine's
`LegalActions`; no betting rule is re-implemented here.
"""

from __future__ import annotations

import operator
from typing import Any

import numpy as np

from engine import ActionType, LegalActions

__all__ = [
    "FOLD",
    "CHECK_CALL",
    "BET_RAISE",
    "N_ACTIONS",
    "ACTION_NAMES",
    "IllegalMaskedActionError",
    "action_mask",
    "mask_array",
    "to_engine",
    "action_kind",
    "check_action",
]

FOLD, CHECK_CALL, BET_RAISE = 0, 1, 2
N_ACTIONS = 3
ACTION_NAMES = ("fold", "check_call", "bet_raise")

# Engine action -> action kind (for the observation's action history).
_KIND = {
    ActionType.FOLD: FOLD,
    ActionType.CHECK: CHECK_CALL,
    ActionType.CALL: CHECK_CALL,
    ActionType.RAISE: BET_RAISE,
}


class IllegalMaskedActionError(ValueError):
    """An action outside the mask (or not one of 0, 1, 2) was submitted.
    Raised before anything is sent to the engine, so nothing changes."""


def action_mask(legal: LegalActions) -> tuple[bool, bool, bool]:
    """(fold, check-or-call, bet-or-raise) allowed for this decision.

    - fold: only when facing a bet (`not legal.can_check`); folding when checking
      is free is masked although the engine accepts it;
    - check-or-call: always (the engine always offers check or call to a player
      who has a decision; a violation is an engine bug and raises);
    - bet-or-raise: exactly when the engine allows a raise.
    """
    can_check = legal.can_check
    if not (can_check or legal.call_amount > 0):
        raise AssertionError(f"engine offered neither check nor call: {legal!r}")
    return (not can_check, True, bool(legal.can_raise))


def mask_array(legal: LegalActions) -> np.ndarray:
    """The mask of one decision as a numpy bool row of shape (3,)."""
    return np.array(action_mask(legal), dtype=bool)


def check_action(action: Any, mask: Any) -> int:
    """`action` as an int in 0..2 allowed by `mask`; IllegalMaskedActionError
    otherwise (bools, non-integers, out of range, masked out)."""
    if isinstance(action, (bool, np.bool_)):
        raise IllegalMaskedActionError(f"not an action: {action!r}")
    try:
        a = operator.index(action)
    except TypeError:
        raise IllegalMaskedActionError(f"not an action: {action!r}") from None
    if not 0 <= a < N_ACTIONS:
        raise IllegalMaskedActionError(f"action must be 0, 1 or 2, got {a}")
    if not mask[a]:
        raise IllegalMaskedActionError(f"{ACTION_NAMES[a]} is masked out for this decision")
    return a


def to_engine(action: Any, legal: LegalActions) -> ActionType:
    """Masked action -> engine `ActionType` (check-or-call becomes CHECK when
    checking is free, CALL otherwise). Raises IllegalMaskedActionError for an
    action outside the mask."""
    a = check_action(action, action_mask(legal))
    if a == FOLD:
        return ActionType.FOLD
    if a == CHECK_CALL:
        return ActionType.CHECK if legal.can_check else ActionType.CALL
    return ActionType.RAISE


def action_kind(action_type: ActionType) -> int:
    """Engine action -> action kind: FOLD -> FOLD, CHECK / CALL -> CHECK_CALL,
    RAISE -> BET_RAISE."""
    return _KIND[action_type]
