"""TableState: the client's local picture of the table, built from server messages.

Seat numbers change between hands (the engine renumbers seats on joins and
leaves), so players are identified by name, which is unique at the table, and
seat indices are positions in the current `seats` list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Union

from entities import Board, Hand, Street
from engine import (
    FOLD_LEAVE,
    FOLD_TIMEOUT,
    ActionType,
    BetReturned,
    BlindPosted,
    BoardDealt,
    HandResult,
    HandShown,
    HandStarted,
    PlayerActed,
    PlayerJoined,
    PlayerLeft,
    PlayerRebought,
    PlayerReturned,
    PlayerSatOut,
    PlayerStatus,
    PotAwarded,
    PotView,
    TableEvent,
    TableView,
)

from cli_server.protocol import TableSettings

from .cards import Segment, card_segments, describe_hand, fmt_amount

__all__ = [
    "DEFAULT_SETTINGS",
    "MAX_LOG_LINES",
    "STREET_NAMES",
    "SeatState",
    "LogLine",
    "TableState",
]

MAX_LOG_LINES = 500

# The engine's fixed structure, used when the server sends no table settings.
DEFAULT_SETTINGS = TableSettings(
    small_blind_bb=0.5,
    big_blind_bb=1.0,
    small_bet_bb=1.0,
    big_bet_bb=2.0,
    max_seats=10,
    max_buy_in_bb=None,
    action_timeout=None,
    grace_period=0.0,
    hand_pause=0.0,
)

STREET_NAMES = {
    Street.PREFLOP: "Preflop",
    Street.FLOP: "Flop",
    Street.TURN: "Turn",
    Street.RIVER: "River",
}

_EPS = 1e-9


def _add(a: float, b: float) -> float:
    return round(a + b, 2)


def _sub(a: float, b: float) -> float:
    return max(0.0, round(a - b, 2))


@dataclass
class SeatState:
    name: str
    stack: float
    status: PlayerStatus = PlayerStatus.ACTIVE
    street_bet: float = 0.0
    total_bet: float = 0.0
    folded: bool = False
    all_in: bool = False
    leaving: bool = False
    in_hand: bool = False
    last_action: str = ""
    shown: Optional[Hand] = None
    won: float = 0.0

    @property
    def sitting_out(self) -> bool:
        return self.status is PlayerStatus.SITTING_OUT

    def reset_for_hand(self) -> None:
        self.street_bet = 0.0
        self.total_bet = 0.0
        self.folded = False
        self.all_in = False
        self.in_hand = False
        self.last_action = ""
        self.shown = None
        self.won = 0.0


@dataclass
class LogLine:
    """One line of the action log. Segments with style None take the line's
    style (`log_me` for my own lines, else `log`)."""

    segments: list[Segment] = field(default_factory=list)
    mine: bool = False

    @property
    def text(self) -> str:
        return "".join(text for text, _ in self.segments)


class TableState:
    """The table model. Updated only from server messages, on the UI thread."""

    def __init__(
        self, my_name: str, settings: Optional[TableSettings] = None, ascii: bool = False
    ) -> None:
        self.my_name = my_name
        self.ascii = ascii
        self.settings: TableSettings = settings or DEFAULT_SETTINGS
        self.has_settings = settings is not None
        # table
        self.hand_number = 0
        self.street = Street.PREFLOP
        self.board = Board()
        self.button: Optional[int] = None
        self.sb_seat: Optional[int] = None
        self.bb_seat: Optional[int] = None
        self.pot = 0.0
        self.side_pots: tuple[PotView, ...] = ()
        self.current_bet = 0.0
        self.bet_unit = self.settings.small_bet_bb
        self.waiting: tuple[str, ...] = ()
        self.seats: list[SeatState] = []
        self.hand_running = False  # from hand_started to the hand_end message
        # me
        self.my_cards: Optional[Hand] = None
        self.queued = False
        self.sitting_out = False
        self.left = False
        # whose turn it is: (seat, monotonic deadline or None)
        self.acting: Optional[tuple[int, Optional[float]]] = None
        self.log: list[LogLine] = []
        self._handlers: dict[type, Callable[[TableEvent], None]] = {
            HandStarted: self._on_hand_started,
            BlindPosted: self._on_blind,
            PlayerActed: self._on_acted,
            BoardDealt: self._on_board,
            HandShown: self._on_shown,
            BetReturned: self._on_bet_returned,
            PotAwarded: self._on_awarded,
            PlayerJoined: self._on_joined,
            PlayerLeft: self._on_left,
            PlayerSatOut: self._on_sat_out,
            PlayerReturned: self._on_returned,
            PlayerRebought: self._on_rebought,
        }

    # --- lookups ---------------------------------------------------------------

    def set_settings(self, settings: Optional[TableSettings]) -> None:
        if settings is not None:
            self.settings = settings
            self.has_settings = True

    def seat_of(self, name: Optional[str]) -> Optional[int]:
        for i, s in enumerate(self.seats):
            if s.name == name:
                return i
        return None

    def find(self, name: Optional[str]) -> Optional[SeatState]:
        i = self.seat_of(name)
        return None if i is None else self.seats[i]

    @property
    def my_seat(self) -> Optional[int]:
        return self.seat_of(self.my_name)

    @property
    def me(self) -> Optional[SeatState]:
        return self.find(self.my_name)

    def seat_name(self, seat: Optional[int]) -> Optional[str]:
        if seat is None or not 0 <= seat < len(self.seats):
            return None
        return self.seats[seat].name

    @property
    def in_hand(self) -> bool:
        """I am dealt into the running hand and have not folded."""
        me = self.me
        return self.hand_running and me is not None and me.in_hand and not me.folded

    # --- log -------------------------------------------------------------------

    def add_log(self, *parts: Union[str, Segment, list], mine: bool = False) -> LogLine:
        """Append a log line. Parts are plain strings (line style), (text, style)
        segments, or lists of segments."""
        segments: list[Segment] = []
        for part in parts:
            if isinstance(part, str):
                segments.append((part, None))
            elif isinstance(part, tuple):
                segments.append(part)
            else:
                segments.extend(part)
        line = LogLine(segments, mine)
        self.log.append(line)
        if len(self.log) > MAX_LOG_LINES:
            del self.log[: len(self.log) - MAX_LOG_LINES]
        return line

    def _mine(self, name: Optional[str]) -> bool:
        return name == self.my_name

    # --- snapshots -------------------------------------------------------------

    def apply_view(self, view: TableView, hand_running: Optional[bool] = None) -> None:
        """Replace the table part of the model with a server snapshot. With
        `hand_running` False (a `seated` view between hands) the per-hand bets
        are not taken from the view."""
        if hand_running is not None:
            self.hand_running = hand_running
        running = self.hand_running
        same_hand = view.hand_number == self.hand_number
        old = {s.name: s for s in self.seats}
        seats: list[SeatState] = []
        for sv in view.seats:
            prev = old.get(sv.name)
            s = SeatState(name=sv.name, stack=sv.stack, status=sv.status, leaving=sv.leaving)
            if running:
                s.street_bet = sv.street_bet
                s.total_bet = sv.total_bet
                s.folded = sv.folded
                s.all_in = sv.all_in
                s.in_hand = sv.in_hand
                if prev is not None and same_hand:
                    s.last_action, s.shown, s.won = prev.last_action, prev.shown, prev.won
                elif sv.folded:
                    s.last_action = "Fold"
            elif prev is not None:
                s.last_action, s.shown, s.won = prev.last_action, prev.shown, prev.won
            seats.append(s)
        self.seats = seats
        self.hand_number = view.hand_number
        self.button, self.sb_seat, self.bb_seat = view.button, view.sb_seat, view.bb_seat
        self.waiting = tuple(view.waiting)
        if running:
            self.street = view.street
            self.board = view.board
            self.pot = view.pot
            self.side_pots = tuple(view.side_pots)
            self.current_bet = view.current_bet
            self.bet_unit = view.bet_unit
        else:
            self.pot = 0.0
            self.side_pots = ()
            self.current_bet = 0.0
        if view.hole_cards is not None:
            self.my_cards = view.hole_cards
        if 0 <= view.seat < len(seats) and seats[view.seat].name == self.my_name:
            self.sitting_out = seats[view.seat].sitting_out
            self.queued = False

    def apply_result(self, result: HandResult) -> None:
        """Reconcile names, stacks, shown hands and winnings with a hand result
        (the `hand_end` message). The hand is over afterwards."""
        old = {s.name: s for s in self.seats}
        winnings = result.winnings
        if result.names:
            seats: list[SeatState] = []
            for idx in sorted(result.names):
                name = result.names[idx]
                s = old.get(name) or SeatState(name, 0.0)
                seats.append(s)
            self.seats = seats
        for idx, s in enumerate(self.seats):
            if idx in result.stacks:
                s.stack = result.stacks[idx]
            if idx in result.shown:
                s.shown = result.shown[idx]
            s.won = round(winnings.get(idx, 0.0), 2)
            s.street_bet = 0.0
            s.total_bet = 0.0
            if result.dealt_in.get(idx) is not None:
                s.in_hand = bool(result.dealt_in[idx])
        self.hand_number = result.hand_number
        self.button, self.sb_seat, self.bb_seat = result.button, result.sb_seat, result.bb_seat
        self.board = result.board
        self.street = result.board.street
        self.pot = 0.0
        self.side_pots = ()
        self.current_bet = 0.0
        self.hand_running = False
        self.acting = None

    # --- events ----------------------------------------------------------------

    def apply_event(self, event: TableEvent) -> bool:
        """Update the model (and the log) from one table event. Returns False for
        an unknown event type, which is ignored."""
        handler = self._handlers.get(type(event))
        if handler is None:
            return False
        handler(event)
        return True

    def _on_hand_started(self, e: HandStarted) -> None:  # type: ignore[override]
        self.hand_number = e.hand_no if e.hand_no is not None else self.hand_number + 1
        self.hand_running = True
        self.board = Board()
        self.street = Street.PREFLOP
        self.pot = 0.0
        self.side_pots = ()
        self.current_bet = 0.0
        self.bet_unit = self.settings.small_bet_bb
        self.my_cards = None
        self.acting = None
        for s in self.seats:
            s.reset_for_hand()
        for seat, name, stack in e.participants:
            s = self.find(name)
            if s is None:
                s = SeatState(name, stack)
                self.seats.insert(max(0, min(seat, len(self.seats))), s)
            s.stack = stack
            s.in_hand = True
        self.button, self.sb_seat, self.bb_seat = e.button, e.sb_seat, e.bb_seat
        self.add_log((f"── Hand #{self.hand_number} ──", "log_hand"))
        button = self.seat_name(self.button)
        if button is not None:
            self.add_log((f"Button: {button}", "dim"))

    def _on_blind(self, e: BlindPosted) -> None:  # type: ignore[override]
        s = self.find(e.name)
        if s is not None:
            s.stack = e.stack_bb
            s.total_bet = _add(s.total_bet, e.amount_bb)
            s.street_bet = e.amount_bb
            s.all_in = e.all_in
            s.in_hand = True
            s.last_action = f"{e.blind} {fmt_amount(e.amount_bb)}"
        self.pot = _add(self.pot, e.amount_bb)
        floor = self.settings.big_blind_bb if e.blind == "BB" else 0.0
        self.current_bet = max(self.current_bet, e.amount_bb, floor)
        text = f"{e.name} posts {e.blind} {fmt_amount(e.amount_bb)}"
        if e.all_in:
            text += " (all-in)"
        self.add_log(text, mine=self._mine(e.name))

    def _on_acted(self, e: PlayerActed) -> None:  # type: ignore[override]
        before = self.current_bet
        s = self.find(e.name)
        label, verb = _action_text(e, before)
        if s is not None:
            s.stack = e.stack_bb
            s.total_bet = _add(s.total_bet, e.amount_bb)
            s.street_bet = e.to_bb
            s.all_in = e.all_in
            if e.action is ActionType.FOLD:
                s.folded = True
            s.last_action = label
        self.pot = _add(self.pot, e.amount_bb)
        self.current_bet = max(self.current_bet, e.to_bb)
        self.acting = None
        text = f"{e.name} {verb}"
        if e.all_in and e.action is not ActionType.FOLD:
            text += " (all-in)"
        self.add_log(text, mine=self._mine(e.name))

    def _on_board(self, e: BoardDealt) -> None:  # type: ignore[override]
        self.board = e.board
        self.street = e.street
        for s in self.seats:
            s.street_bet = 0.0
            if not s.folded:
                s.last_action = ""
        self.current_bet = 0.0
        self.bet_unit = (
            self.settings.small_bet_bb if e.street is Street.FLOP else self.settings.big_bet_bb
        )
        self.acting = None
        self.add_log(
            (f"{STREET_NAMES.get(e.street, e.street.name.title())}: ", "log_board"),
            card_segments(e.cards, self.ascii),
        )

    def _on_shown(self, e: HandShown) -> None:  # type: ignore[override]
        s = self.find(e.name)
        if s is not None:
            s.shown = e.hand
            s.stack = e.stack_bb
        if e.name == self.my_name and self.my_cards is None:
            self.my_cards = e.hand
        parts: list = [f"{e.name} shows ", card_segments(e.hand, self.ascii)]
        description = describe_hand(e.hand, self.board)
        if description:
            parts.append(f" – {description}")
        self.add_log(*parts, mine=self._mine(e.name))

    def _on_bet_returned(self, e: BetReturned) -> None:  # type: ignore[override]
        s = self.find(e.name)
        if s is not None:
            s.stack = e.stack_bb
            s.total_bet = _sub(s.total_bet, e.amount_bb)
            s.street_bet = _sub(s.street_bet, e.amount_bb)
        self.pot = _sub(self.pot, e.amount_bb)
        self.add_log(f"{fmt_amount(e.amount_bb)} returned to {e.name}", mine=self._mine(e.name))

    def _on_awarded(self, e: PotAwarded) -> None:  # type: ignore[override]
        s = self.find(e.name)
        if s is not None:
            s.stack = e.stack_bb
            s.won = _add(s.won, e.amount_bb)
        self.pot = _sub(self.pot, e.amount_bb)
        share = "uncontested" if e.share == "whole" else e.share
        if e.pot_index > 0:
            share = f"side pot {e.pot_index}, {share}"
        self.add_log(
            (f"{e.name} wins {fmt_amount(e.amount_bb)} ({share})", "log_me" if self._mine(e.name) else "log_win"),
            mine=self._mine(e.name),
        )

    def _on_joined(self, e: PlayerJoined) -> None:  # type: ignore[override]
        s = self.find(e.name)
        if s is None:
            idx = max(0, min(e.seat, len(self.seats)))
            self.seats.insert(idx, SeatState(e.name, e.stack_bb))
            if self.button is not None and len(self.seats) > 1 and self.button >= idx:
                self.button += 1
            self.sb_seat = self.bb_seat = None  # they refer to the old numbering
        else:
            s.stack = e.stack_bb
        self.waiting = tuple(n for n in self.waiting if n != e.name)
        self.add_log(
            f"{e.name} joins the table ({fmt_amount(e.stack_bb)} BB)", mine=self._mine(e.name)
        )

    def _on_left(self, e: PlayerLeft) -> None:  # type: ignore[override]
        idx = self.seat_of(e.name)
        if idx is not None:
            self.seats.pop(idx)
            if not self.seats:
                self.button = None
            elif self.button is not None:
                if self.button > idx:
                    self.button -= 1
                elif self.button == idx:
                    self.button = idx % len(self.seats)  # passes clockwise
            self.sb_seat = self.bb_seat = None
        self.add_log(f"{e.name} leaves the table", mine=self._mine(e.name))

    def _on_sat_out(self, e: PlayerSatOut) -> None:  # type: ignore[override]
        s = self.find(e.name)
        if s is not None:
            s.status = PlayerStatus.SITTING_OUT
            s.stack = e.stack_bb
        if e.name == self.my_name:
            self.sitting_out = True
        self.add_log(f"{e.name} sits out", mine=self._mine(e.name))

    def _on_returned(self, e: PlayerReturned) -> None:  # type: ignore[override]
        s = self.find(e.name)
        if s is not None:
            s.status = PlayerStatus.ACTIVE
            s.stack = e.stack_bb
        if e.name == self.my_name:
            self.sitting_out = False
        self.add_log(f"{e.name} is back", mine=self._mine(e.name))

    def _on_rebought(self, e: PlayerRebought) -> None:  # type: ignore[override]
        s = self.find(e.name)
        if s is not None:
            s.stack = e.stack_bb
        self.add_log(f"{e.name} rebuys {fmt_amount(e.amount_bb)}", mine=self._mine(e.name))


def _action_text(e: PlayerActed, current_bet_before: float) -> tuple[str, str]:
    """(seat label, log verb) for an action, e.g. ("Raise to 4", "raises to 4")."""
    t = e.action
    if t is ActionType.FOLD:
        if e.reason == FOLD_TIMEOUT:
            return "Fold (timeout)", "folds (timeout)"
        if e.reason == FOLD_LEAVE:
            return "Fold (left)", "folds (leaving)"
        return "Fold", "folds"
    if t is ActionType.CHECK:
        return "Check", "checks"
    if t is ActionType.CALL:
        amount = fmt_amount(e.amount_bb)
        return f"Call {amount}", f"calls {amount}"
    to = fmt_amount(e.to_bb)
    if current_bet_before <= _EPS:
        return f"Bet {to}", f"bets {to}"
    return f"Raise to {to}", f"raises to {to}"
