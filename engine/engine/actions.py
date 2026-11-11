"""Actions available to a player in fixed-limit Omaha Hi-Lo."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

__all__ = ["ActionType", "Action", "LegalActions", "IllegalActionError"]


class IllegalActionError(Exception):
    """An action (or rebuy) was attempted that the rules do not allow right now."""


class ActionType(Enum):
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    # A bet into an unopened round is a RAISE from 0.
    RAISE = "raise"


@dataclass(frozen=True)
class Action:
    """A player decision. There is no amount: limit bets and raises are one bet unit."""

    type: ActionType

    @classmethod
    def fold(cls) -> "Action":
        return cls(ActionType.FOLD)

    @classmethod
    def check(cls) -> "Action":
        return cls(ActionType.CHECK)

    @classmethod
    def call(cls) -> "Action":
        return cls(ActionType.CALL)

    @classmethod
    def raise_(cls) -> "Action":
        return cls(ActionType.RAISE)


@dataclass(frozen=True)
class LegalActions:
    """What the acting player may do. Amounts are in big blinds.

    - `call_amount`: chips (BB) needed to call, capped by the player's stack; 0 if
      the player can check.
    - `raise_to`: the player's total street bet after raising (current bet plus one
      bet unit, capped by the player's stack).
    - `raises_left`: bets/raises still allowed on this street, or None when the cap
      is lifted (only two players left in the hand).
    """

    can_check: bool
    call_amount: float
    can_raise: bool
    raise_to: float
    bet_unit: float
    raises_left: Optional[int]

    @property
    def can_call(self) -> bool:
        return self.call_amount > 0

    def allows(self, action_type: ActionType) -> bool:
        if action_type is ActionType.FOLD:
            return True
        if action_type is ActionType.CHECK:
            return self.can_check
        if action_type is ActionType.CALL:
            return self.can_call
        if action_type is ActionType.RAISE:
            return self.can_raise
        return False

    @property
    def actions(self) -> tuple[ActionType, ...]:
        return tuple(t for t in ActionType if self.allows(t))
