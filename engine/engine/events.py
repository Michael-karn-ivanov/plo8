"""Public table events, broadcast to every seated player. Amounts are in big blinds.

Every event derives from `TableEvent`, which carries `hand_no` (the hand in
progress, None for table events between hands) and `seq` (monotonic per table).
Both are stamped by the table when the event is broadcast. Events about a
player carry the player's `seat`, `name` and `stack_bb` - the stack right
*after* the event.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from entities import Board, Card, Hand, Street

from .actions import ActionType

__all__ = [
    "TableEvent",
    "HandStarted",
    "BlindPosted",
    "PlayerActed",
    "BoardDealt",
    "HandShown",
    "BetReturned",
    "PotAwarded",
    "PlayerJoined",
    "PlayerSatOut",
    "PlayerReturned",
    "PlayerRebought",
    "PlayerLeft",
    "FOLD_LEAVE",
    "FOLD_TIMEOUT",
]

# `PlayerActed.reason` for forced folds.
FOLD_LEAVE = "leave"
FOLD_TIMEOUT = "timeout"


@dataclass(frozen=True)
class TableEvent:
    """Base of all public events (frozen and picklable)."""

    hand_no: Optional[int] = field(default=None, kw_only=True)
    seq: int = field(default=0, kw_only=True)


# --- hand events (emitted by the dealer) ----------------------------------------


@dataclass(frozen=True)
class HandStarted(TableEvent):
    """A hand starts (before the blinds). `participants` lists
    (seat, name, stack_bb) of the players dealt in, in seat order."""

    button: int
    sb_seat: int
    bb_seat: int
    participants: tuple[tuple[int, str, float], ...]


@dataclass(frozen=True)
class BlindPosted(TableEvent):
    seat: int
    name: str
    blind: str  # "SB" | "BB"
    amount_bb: float
    all_in: bool
    stack_bb: float


@dataclass(frozen=True)
class PlayerActed(TableEvent):
    """One applied action. `amount_bb` is what this action put in (0 for fold /
    check), `to_bb` the player's street bet after it. Forced folds carry a
    `reason` ("leave" or "timeout"); a voluntary action has None."""

    seat: int
    name: str
    street: Street
    action: ActionType
    amount_bb: float
    to_bb: float
    all_in: bool
    stack_bb: float
    reason: Optional[str] = None


@dataclass(frozen=True)
class BoardDealt(TableEvent):
    """Flop (3 cards), turn or river (1 card); also for an all-in run-out."""

    street: Street
    cards: tuple[Card, ...]
    board: Board


@dataclass(frozen=True)
class HandShown(TableEvent):
    seat: int
    name: str
    hand: Hand
    stack_bb: float


@dataclass(frozen=True)
class BetReturned(TableEvent):
    """The uncalled part of a bet given back to the player."""

    seat: int
    name: str
    amount_bb: float
    stack_bb: float


@dataclass(frozen=True)
class PotAwarded(TableEvent):
    """One share of pot `pot_index` credited to a winner. `share` is "high" or
    "low" for a contested pot, "whole" for a pot nobody else is eligible for
    (e.g. everyone else folded)."""

    seat: int
    name: str
    pot_index: int
    share: str  # "high" | "low" | "whole"
    amount_bb: float
    stack_bb: float


# --- table events (emitted by the environment) -----------------------------------


@dataclass(frozen=True)
class PlayerJoined(TableEvent):
    seat: int
    name: str
    stack_bb: float


@dataclass(frozen=True)
class PlayerSatOut(TableEvent):
    seat: int
    name: str
    stack_bb: float


@dataclass(frozen=True)
class PlayerReturned(TableEvent):
    seat: int
    name: str
    stack_bb: float


@dataclass(frozen=True)
class PlayerRebought(TableEvent):
    seat: int
    name: str
    amount_bb: float
    stack_bb: float


@dataclass(frozen=True)
class PlayerLeft(TableEvent):
    """`seat` is the seat the player had before being removed."""

    seat: int
    name: str
    stack_bb: float
