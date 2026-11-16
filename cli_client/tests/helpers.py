"""Test helpers: builders for engine views / results and server messages, a fake
server connection and a fake clock."""

from __future__ import annotations

import queue
from typing import Any, Callable, Optional

from entities import Board, Hand, Street
from engine import HandResult, LegalActions, PlayerStatus, PotResult, SeatView, TableView

from cli_server.protocol import (
    Acting,
    LobbyStatus,
    TableSettings,
    encode_acting,
    encode_event,
    encode_lobby,
    encode_server_message,
)

ME = "Alice"

SETTINGS = TableSettings(
    small_blind_bb=0.5,
    big_blind_bb=1.0,
    small_bet_bb=1.0,
    big_bet_bb=2.0,
    max_seats=6,
    max_buy_in_bb=None,
    action_timeout=30.0,
    grace_period=1.0,
    hand_pause=0.0,
)


def seat_view(
    seat: int,
    name: str,
    stack: float,
    street_bet: float = 0.0,
    total_bet: float = 0.0,
    folded: bool = False,
    all_in: bool = False,
    in_hand: bool = True,
    status: PlayerStatus = PlayerStatus.ACTIVE,
    leaving: bool = False,
) -> SeatView:
    return SeatView(seat, name, stack, street_bet, total_bet, folded, all_in, in_hand, status, leaving)


def table_view(
    seat: int,
    seats: list[SeatView],
    hole: Optional[str] = None,
    street: Street = Street.PREFLOP,
    board: str = "",
    pot: float = 1.5,
    current_bet: float = 1.0,
    bet_unit: float = 1.0,
    raises: int = 1,
    button: Optional[int] = 0,
    sb: Optional[int] = 0,
    bb: Optional[int] = 1,
    hand_number: int = 1,
    waiting: tuple = (),
) -> TableView:
    return TableView(
        hand_number=hand_number,
        seat=seat,
        hole_cards=None if hole is None else Hand.parse(hole),
        street=street,
        board=Board.parse(board),
        pot=pot,
        side_pots=(),
        current_bet=current_bet,
        bet_unit=bet_unit,
        raises=raises,
        button=button,
        sb_seat=sb,
        bb_seat=bb,
        seats=tuple(seats),
        history=(),
        waiting=waiting,
    )


def legal(
    can_check: bool = False,
    call_amount: float = 0.5,
    can_raise: bool = True,
    raise_to: float = 2.0,
    bet_unit: float = 1.0,
    raises_left: Optional[int] = 3,
) -> LegalActions:
    return LegalActions(can_check, call_amount, can_raise, raise_to, bet_unit, raises_left)


def hand_result(
    names: dict[int, str],
    stacks: dict[int, float],
    hand_number: int = 1,
    board: str = "",
    shown: Optional[dict[int, str]] = None,
    pots: tuple = (),
    button: int = 0,
    sb: Optional[int] = 0,
    bb: Optional[int] = 1,
) -> HandResult:
    return HandResult(
        hand_number=hand_number,
        button=button,
        sb_seat=sb,
        bb_seat=bb,
        board=Board.parse(board),
        showdown=bool(shown),
        shown={s: Hand.parse(h) for s, h in (shown or {}).items()},
        pots=tuple(pots),
        net={s: 0.0 for s in names},
        stacks=dict(stacks),
        history=(),
        names=dict(names),
        dealt_in={s: True for s in names},
    )


def pot_result(amount: float, eligible: tuple, high: tuple, low: tuple = (), uncontested: bool = False) -> PotResult:
    return PotResult(amount, eligible, high, low, uncontested)


# --- messages ---------------------------------------------------------------------


def seated_msg(view: TableView, to: str = ME) -> dict:
    return encode_server_message(("seated", view), to)


def hole_cards_msg(cards: str, view: TableView, to: str = ME) -> dict:
    return encode_server_message(("hole_cards", Hand.parse(cards), view), to)


def event_msg(event: Any, to: str = ME) -> dict:
    return {"type": "event", "to": to, "event": encode_event(event)}


def turn_msg(seq: int, view: TableView, legal_actions: LegalActions, timeout: Optional[float] = 30.0, to: str = ME) -> dict:
    return encode_server_message(("turn", seq, view, legal_actions), to, turn_timeout=timeout)


def hand_end_msg(seq: int, seat: int, result: HandResult, to: str = ME) -> dict:
    return encode_server_message(("hand_end", seq, seat, result), to)


def simple_msg(kind: str, to: str = ME) -> dict:
    return encode_server_message((kind,), to)


def left_msg(final_stack: float, to: str = ME) -> dict:
    return encode_server_message(("left", final_stack), to)


def acting_msg(hand_number: int, seat: int, name: str, timeout: Optional[float], to: str = ME) -> dict:
    return encode_acting(Acting(hand_number, seat, name, timeout), to)


def lobby_msg(state: str, active: int, needed: int = 2, starts_in: Optional[float] = None, to: str = ME) -> dict:
    return encode_lobby(LobbyStatus(state, active, needed, starts_in), to)


# --- fakes ------------------------------------------------------------------------


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


class FakeConnection:
    """Records requests; answers with `responder(msg)` (default: ok)."""

    def __init__(self, responder: Optional[Callable[[dict], Any]] = None) -> None:
        self.requests: list[dict] = []
        self.responder = responder or (lambda msg: {"type": "response", "ok": True})
        self.closed = False

    def request(self, msg: dict) -> dict:
        self.requests.append(msg)
        response = self.responder(msg)
        if isinstance(response, BaseException):
            raise response
        return response

    def of_type(self, kind: str) -> list[dict]:
        return [r for r in self.requests if r["type"] == kind]

    def close(self) -> None:
        self.closed = True


def new_inbox() -> "queue.Queue[Any]":
    return queue.Queue()
