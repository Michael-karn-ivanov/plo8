"""The reply to a `hand_end`: rebuy, sit in, leave. Pure, no I/O."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

__all__ = ["POLICIES", "HandEndChoice", "hand_end_reply", "is_busted"]

POLICIES = ("topup", "bust", "never")


@dataclass(frozen=True)
class HandEndChoice:
    rebuy_bb: Optional[float]  # None = no rebuy
    sit_in: bool
    leave: bool
    reason: str = ""  # why the bot leaves (empty when it stays)


def is_busted(stack: float) -> bool:
    """No chips left (in whole cents of a BB, the server's chip unit)."""
    return round(stack, 2) <= 0


def hand_end_reply(
    *,
    stack: float,
    sitting_out: bool,
    target: float,
    max_buy_in: Optional[float],
    policy: str,
    hands_played: int,
    max_hands: Optional[int],
    stopping: bool,
) -> HandEndChoice:
    """What to answer in the rebuy window, from the stack after the hand.

    - leave when stopping, when `max_hands` hands were played, or when busted
      under `never` (no rebuy and no sit-in then);
    - `topup`: rebuy up to `target` (capped by `max_buy_in`) when below it;
      `bust`: the same, only when busted; `never`: no rebuy;
    - sit in when sitting out (the engine decides whether it can: a busted bot
      comes back only with the rebuy of the same reply).
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown rebuy policy {policy!r}; known: {', '.join(POLICIES)}")
    busted = is_busted(stack)
    reason = ""
    if stopping:
        reason = "stopping"
    elif max_hands is not None and hands_played >= max_hands:
        reason = f"{hands_played} hands played"
    elif policy == "never" and busted:
        reason = "busted, --rebuy never"
    if reason:
        return HandEndChoice(None, False, True, reason)

    cap = max_buy_in if max_buy_in is not None else math.inf
    rebuy: Optional[float] = None
    if policy == "topup" or (policy == "bust" and busted):
        missing = round(min(target, cap) - stack, 2)
        if missing > 0:
            rebuy = missing
    return HandEndChoice(rebuy, bool(sitting_out), False)
