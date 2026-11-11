"""Read-only, picklable snapshots handed to agents. Amounts are in big blinds."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from entities import Board, Card, Hand, Street

__all__ = [
    "EventType",
    "Event",
    "SeatView",
    "PotView",
    "TableView",
    "PotResult",
    "HandResult",
]


class EventType(Enum):
    HAND_START = "hand_start"
    POST_SMALL_BLIND = "post_small_blind"
    POST_BIG_BLIND = "post_big_blind"
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    RAISE = "raise"
    BOARD = "board"
    SHOW = "show"
    AWARD = "award"


@dataclass(frozen=True)
class Event:
    """Something public that happened in the hand.

    `amount` is what this event moved (chips put in, or won for AWARD);
    `to` is the seat's total street bet after a blind/bet action.
    """

    type: EventType
    street: Street
    seat: Optional[int] = None
    amount: float = 0.0
    to: float = 0.0
    all_in: bool = False
    cards: tuple[Card, ...] = ()


@dataclass(frozen=True)
class SeatView:
    seat: int
    name: str
    stack: float
    street_bet: float
    total_bet: float
    folded: bool
    all_in: bool
    in_hand: bool


@dataclass(frozen=True)
class PotView:
    amount: float
    eligible: tuple[int, ...]


@dataclass(frozen=True)
class TableView:
    """What the acting player can see. Only its own hole cards are included."""

    hand_number: int
    seat: int
    hole_cards: Optional[Hand]
    street: Street
    board: Board
    pot: float
    side_pots: tuple[PotView, ...]
    current_bet: float
    bet_unit: float
    raises: int  # bets made on this street (the big blind counts preflop)
    button: int
    sb_seat: Optional[int]
    bb_seat: Optional[int]
    seats: tuple[SeatView, ...]
    history: tuple[Event, ...]

    @property
    def me(self) -> SeatView:
        return self.seats[self.seat]

    @property
    def to_call(self) -> float:
        return max(0.0, self.current_bet - self.me.street_bet)


@dataclass(frozen=True)
class PotResult:
    """How one pot was awarded. `high`/`low` list (seat, amount) shares."""

    amount: float
    eligible: tuple[int, ...]
    high: tuple[tuple[int, float], ...]
    low: tuple[tuple[int, float], ...]
    uncontested: bool  # a single eligible player (everyone else folded / uncalled bet)


@dataclass(frozen=True)
class HandResult:
    hand_number: int
    button: int
    sb_seat: Optional[int]
    bb_seat: Optional[int]
    board: Board
    showdown: bool
    shown: dict[int, Hand]  # seat -> hand, only for hands shown at showdown
    pots: tuple[PotResult, ...]
    net: dict[int, float]  # chip change per seat in this hand (BB)
    stacks: dict[int, float]  # stacks after the hand, before any rebuy (BB)
    history: tuple[Event, ...]

    @property
    def winnings(self) -> dict[int, float]:
        """Total amount awarded to each seat from all pots."""
        out: dict[int, float] = {}
        for pot in self.pots:
            for seat, amount in pot.high + pot.low:
                out[seat] = out.get(seat, 0.0) + amount
        return out
