"""Read-only, picklable snapshots handed to agents. Amounts are in big blinds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from entities import Board, Hand, Street

from .events import TableEvent
from .player import PlayerStatus

__all__ = [
    "SeatView",
    "PotView",
    "TableView",
    "PotResult",
    "HandResult",
]


@dataclass(frozen=True)
class SeatView:
    seat: int
    name: str
    stack: float
    street_bet: float
    total_bet: float
    folded: bool
    all_in: bool
    in_hand: bool  # dealt into the current (or last) hand
    status: PlayerStatus = PlayerStatus.ACTIVE  # ACTIVE or SITTING_OUT
    leaving: bool = False  # asked to leave; removed when the hand is over

    @property
    def sitting_out(self) -> bool:
        return self.status is PlayerStatus.SITTING_OUT


@dataclass(frozen=True)
class PotView:
    amount: float
    eligible: tuple[int, ...]


@dataclass(frozen=True)
class TableView:
    """What the receiving player can see. Only its own hole cards are included
    (None if it is not dealt in). `history` holds the public events of the
    current hand, the same ones delivered through `on_event`."""

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
    button: Optional[int]
    sb_seat: Optional[int]
    bb_seat: Optional[int]
    seats: tuple[SeatView, ...]
    history: tuple[TableEvent, ...]
    waiting: tuple[str, ...] = ()  # names of players queued to be seated

    @property
    def me(self) -> SeatView:
        return self.seats[self.seat]

    @property
    def to_call(self) -> float:
        return max(0.0, self.current_bet - self.me.street_bet)


@dataclass(frozen=True)
class PotResult:
    """How one pot was awarded. `high`/`low` list (seat, amount) shares; an
    uncontested pot (one eligible player) is listed under `high`."""

    amount: float
    eligible: tuple[int, ...]
    high: tuple[tuple[int, float], ...]
    low: tuple[tuple[int, float], ...]
    uncontested: bool  # a single eligible player (everyone else folded)


@dataclass(frozen=True)
class HandResult:
    """End of a hand. Per-seat dicts cover every seated player, including the
    sitting-out ones (`dealt_in` False, net change 0).

    Summarises the hand's `HandShown` (`shown`), `BetReturned` (`returned`) and
    `PotAwarded` (`pots`) events; `history` holds all public events of the hand
    up to the settlement.
    """

    hand_number: int
    button: int
    sb_seat: Optional[int]
    bb_seat: Optional[int]
    board: Board
    showdown: bool
    shown: dict[int, Hand]  # seat -> hand, only for hands shown at showdown
    pots: tuple[PotResult, ...]  # after uncalled bets were returned
    net: dict[int, float]  # chip change per seat in this hand (BB)
    stacks: dict[int, float]  # stacks after the hand, before any rebuy (BB)
    history: tuple[TableEvent, ...]
    names: dict[int, str] = field(default_factory=dict)  # seat -> player name
    dealt_in: dict[int, bool] = field(default_factory=dict)  # seat -> took part
    returned: dict[int, float] = field(default_factory=dict)  # seat -> uncalled bet

    @property
    def winnings(self) -> dict[int, float]:
        """Total amount awarded to each seat from all pots (uncalled bets excluded)."""
        out: dict[int, float] = {}
        for pot in self.pots:
            for seat, amount in pot.high + pot.low:
                out[seat] = out.get(seat, 0.0) + amount
        return out
