"""JSON-lines wire protocol between cli_server and its clients.

Every message is one UTF-8 JSON object on one line, with a `"type"` field.

Two kinds of TCP connection:
- client -> server (the server's listening port): requests (`join`, `action`,
  `hand_end`, `rebuy`, `sit_in`, `leave`), each answered by exactly one
  `{"type": "response", "ok": ...}` line. Every request carries the player `name`.
- server -> client (the client's callback port, given in `join`): one-way
  notifications and requests mirroring the engine's `RemoteAgent` messages
  (`seated`, `hole_cards`, `event`, `turn`, `hand_end`, `sat_out`, `returned`,
  `left`, `close`), each with `"to": <player name>`. Answers to `turn` /
  `hand_end` go back as requests on the server port.

This module also encodes / decodes the engine types (cards, board, events, views,
results) so that a Python client can rebuild the engine dataclasses.
"""

from __future__ import annotations

import json
import math
from dataclasses import MISSING, fields
from typing import Any, BinaryIO, Callable, Optional, Union

from entities import Board, Card, Hand, Street
from engine import (
    Action,
    ActionType,
    BetReturned,
    BlindPosted,
    BoardDealt,
    HandResult,
    HandShown,
    HandStarted,
    JoinReply,
    LegalActions,
    Leave,
    PlayerActed,
    PlayerJoined,
    PlayerLeft,
    PlayerRebought,
    PlayerReturned,
    PlayerSatOut,
    PlayerStatus,
    PotAwarded,
    PotResult,
    PotView,
    SeatView,
    TableEvent,
    TableView,
)

__all__ = [
    "MAX_LINE_BYTES",
    "ProtocolError",
    "LineTooLongError",
    "encode_line",
    "decode_line",
    "read_message",
    "write_message",
    # engine types
    "encode_card",
    "decode_card",
    "encode_cards",
    "decode_cards",
    "encode_hand",
    "decode_hand",
    "encode_board",
    "decode_board",
    "encode_street",
    "decode_street",
    "encode_action_type",
    "decode_action_type",
    "encode_status",
    "decode_status",
    "ACTION_NAMES",
    "encode_action",
    "decode_action",
    "EVENT_TYPES",
    "encode_event",
    "decode_event",
    "encode_legal",
    "decode_legal",
    "encode_seat_view",
    "decode_seat_view",
    "encode_table_view",
    "decode_table_view",
    "encode_hand_result",
    "decode_hand_result",
    # server -> client
    "SERVER_MESSAGE_TYPES",
    "encode_server_message",
    "decode_server_message",
    # client -> server
    "REQUEST_TYPES",
    "join_request",
    "action_request",
    "hand_end_request",
    "rebuy_request",
    "sit_in_request",
    "leave_request",
    "validate_request",
    "ok_response",
    "error_response",
    "encode_join_response",
    "decode_join_response",
]

MAX_LINE_BYTES = 1 << 20  # longest accepted message line (1 MiB)


class ProtocolError(ValueError):
    """A malformed message: invalid JSON, no `type`, wrong or missing fields."""


class LineTooLongError(ProtocolError):
    """A message line exceeds `MAX_LINE_BYTES`; the stream cannot be resynchronised."""


# --- framing ----------------------------------------------------------------------


def encode_line(msg: dict) -> bytes:
    """One message as a newline-terminated UTF-8 JSON line."""
    if not isinstance(msg, dict) or not isinstance(msg.get("type"), str):
        raise ProtocolError(f"a message must be an object with a 'type', got {msg!r}")
    try:
        # ensure_ascii (the default) guarantees there is no raw newline inside.
        text = json.dumps(msg, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as e:
        raise ProtocolError(f"cannot encode message: {e}") from None
    return (text + "\n").encode("utf-8")


def decode_line(line: Union[bytes, str]) -> dict:
    """Parse one JSON line; it must be an object with a non-empty string `type`."""
    if isinstance(line, bytes):
        try:
            line = line.decode("utf-8")
        except UnicodeDecodeError:
            raise ProtocolError("message is not valid UTF-8") from None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as e:
        raise ProtocolError(f"invalid JSON: {e.msg}") from None
    if not isinstance(obj, dict):
        raise ProtocolError("a message must be a JSON object")
    kind = obj.get("type")
    if not isinstance(kind, str) or not kind:
        raise ProtocolError("message has no 'type'")
    return obj


def read_message(file: BinaryIO) -> Optional[dict]:
    """Read the next message from a binary file-like object; None at end of stream.
    Blank lines are skipped."""
    while True:
        line = file.readline(MAX_LINE_BYTES + 1)
        if not line:
            return None
        if len(line) > MAX_LINE_BYTES:
            raise LineTooLongError(f"message longer than {MAX_LINE_BYTES} bytes")
        if line.strip():
            return decode_line(line)


def write_message(file: BinaryIO, msg: dict) -> None:
    file.write(encode_line(msg))
    file.flush()


# --- helpers ----------------------------------------------------------------------


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _require(d: Any, keys: tuple[str, ...], what: str) -> dict:
    if not isinstance(d, dict):
        raise ProtocolError(f"{what} must be an object, got {d!r}")
    missing = [k for k in keys if k not in d]
    if missing:
        raise ProtocolError(f"{what} is missing {', '.join(missing)}")
    return d


def _list(x: Any, what: str) -> list:
    if not isinstance(x, list):
        raise ProtocolError(f"{what} must be a list, got {x!r}")
    return x


def _opt(fn: Callable[[Any], Any], x: Any) -> Any:
    return None if x is None else fn(x)


def _pairs(x: Any, what: str) -> tuple[tuple[Any, Any], ...]:
    out = []
    for item in _list(x, what):
        if not isinstance(item, list) or len(item) != 2:
            raise ProtocolError(f"{what} entries must be [seat, amount], got {item!r}")
        out.append((item[0], item[1]))
    return tuple(out)


# --- cards, hands, board ----------------------------------------------------------


def encode_card(card: Card) -> str:
    return str(card)


def decode_card(text: Any) -> Card:
    if not isinstance(text, str):
        raise ProtocolError(f"a card must be a string such as 'As', got {text!r}")
    try:
        return Card.parse(text)
    except ValueError as e:
        raise ProtocolError(str(e)) from None


def encode_cards(cards: Any) -> list[str]:
    return [encode_card(c) for c in cards]


def decode_cards(items: Any) -> tuple[Card, ...]:
    return tuple(decode_card(c) for c in _list(items, "cards"))


def encode_hand(hand: Hand) -> list[str]:
    return encode_cards(hand.cards)


def decode_hand(items: Any) -> Hand:
    try:
        return Hand(decode_cards(items))
    except ValueError as e:
        raise ProtocolError(str(e)) from None


def encode_board(board: Board) -> dict:
    return {
        "flop": None if board.flop is None else encode_cards(board.flop),
        "turn": _opt(encode_card, board.turn),
        "river": _opt(encode_card, board.river),
    }


def decode_board(d: Any) -> Board:
    if not isinstance(d, dict):
        raise ProtocolError(f"a board must be an object, got {d!r}")
    try:
        return Board(
            flop=_opt(decode_cards, d.get("flop")),
            turn=_opt(decode_card, d.get("turn")),
            river=_opt(decode_card, d.get("river")),
        )
    except ValueError as e:
        raise ProtocolError(str(e)) from None


# --- enums ------------------------------------------------------------------------


def encode_street(street: Street) -> str:
    # Street values are ints; the wire uses the readable lower-case names.
    return street.name.lower()


def decode_street(x: Any) -> Street:
    if isinstance(x, str) and x.islower():
        try:
            return Street[x.upper()]
        except KeyError:
            pass
    raise ProtocolError(f"unknown street {x!r}")


def encode_action_type(action: ActionType) -> str:
    return action.value


def decode_action_type(x: Any) -> ActionType:
    try:
        return ActionType(x)
    except (ValueError, TypeError):
        raise ProtocolError(f"unknown action {x!r}") from None


def encode_status(status: PlayerStatus) -> str:
    return status.value


def decode_status(x: Any) -> PlayerStatus:
    try:
        return PlayerStatus(x)
    except (ValueError, TypeError):
        raise ProtocolError(f"unknown player status {x!r}") from None


ACTION_NAMES = ("fold", "check", "call", "raise", "leave")


def encode_action(action: Union[Action, Leave]) -> str:
    if isinstance(action, Leave):
        return "leave"
    if isinstance(action, Action):
        return encode_action_type(action.type)
    raise ProtocolError(f"not an action: {action!r}")


def decode_action(x: Any) -> Union[Action, Leave]:
    """'fold' / 'check' / 'call' / 'raise' -> Action, 'leave' -> Leave."""
    if x == "leave":
        return Leave()
    return Action(decode_action_type(x))


# --- events -----------------------------------------------------------------------

EVENT_TYPES: dict[str, type] = {
    "hand_started": HandStarted,
    "blind_posted": BlindPosted,
    "player_acted": PlayerActed,
    "board_dealt": BoardDealt,
    "hand_shown": HandShown,
    "bet_returned": BetReturned,
    "pot_awarded": PotAwarded,
    "player_joined": PlayerJoined,
    "player_sat_out": PlayerSatOut,
    "player_returned": PlayerReturned,
    "player_rebought": PlayerRebought,
    "player_left": PlayerLeft,
}
_EVENT_NAMES = {cls: name for name, cls in EVENT_TYPES.items()}


def _encode_participants(participants: Any) -> list:
    return [[seat, name, stack] for seat, name, stack in participants]


def _decode_participants(x: Any) -> tuple:
    out = []
    for item in _list(x, "participants"):
        if not isinstance(item, list) or len(item) != 3:
            raise ProtocolError(f"participants entries must be [seat, name, stack_bb], got {item!r}")
        out.append((item[0], item[1], item[2]))
    return tuple(out)


# Event fields that are not plain JSON values: field name -> (encode, decode).
_EVENT_FIELD_CODECS: dict[str, tuple[Callable[[Any], Any], Callable[[Any], Any]]] = {
    "street": (encode_street, decode_street),
    "action": (encode_action_type, decode_action_type),
    "cards": (encode_cards, decode_cards),
    "board": (encode_board, decode_board),
    "hand": (encode_hand, decode_hand),
    "participants": (_encode_participants, _decode_participants),
}


def encode_event(event: TableEvent) -> dict:
    """`{"type": <snake_case class name>, "hand_no", "seq", ...dataclass fields}`."""
    name = _EVENT_NAMES.get(type(event))
    if name is None:
        raise ProtocolError(f"unknown event {event!r}")
    out: dict[str, Any] = {"type": name, "hand_no": event.hand_no, "seq": event.seq}
    for f in fields(event):
        if f.name in out:
            continue
        value = getattr(event, f.name)
        codec = _EVENT_FIELD_CODECS.get(f.name)
        out[f.name] = codec[0](value) if codec is not None and value is not None else value
    return out


def decode_event(d: Any) -> TableEvent:
    if not isinstance(d, dict):
        raise ProtocolError(f"an event must be an object, got {d!r}")
    kind = d.get("type")
    cls = EVENT_TYPES.get(kind) if isinstance(kind, str) else None
    if cls is None:
        raise ProtocolError(f"unknown event type {kind!r}")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name in d:
            value = d[f.name]
            codec = _EVENT_FIELD_CODECS.get(f.name)
            kwargs[f.name] = codec[1](value) if codec is not None and value is not None else value
        elif f.default is MISSING and f.default_factory is MISSING:
            raise ProtocolError(f"event {kind!r} is missing {f.name!r}")
    try:
        return cls(**kwargs)
    except (TypeError, ValueError) as e:
        raise ProtocolError(f"invalid {kind!r} event: {e}") from None


# --- legal actions and views -------------------------------------------------------

_LEGAL_FIELDS = ("can_check", "call_amount", "can_raise", "raise_to", "bet_unit", "raises_left")


def encode_legal(legal: LegalActions) -> dict:
    return {k: getattr(legal, k) for k in _LEGAL_FIELDS}


def decode_legal(d: Any) -> LegalActions:
    _require(d, _LEGAL_FIELDS, "legal actions")
    return LegalActions(**{k: d[k] for k in _LEGAL_FIELDS})


_SEAT_FIELDS = (
    "seat",
    "name",
    "stack",
    "street_bet",
    "total_bet",
    "folded",
    "all_in",
    "in_hand",
    "status",
    "leaving",
)


def encode_seat_view(seat: SeatView) -> dict:
    out = {k: getattr(seat, k) for k in _SEAT_FIELDS}
    out["status"] = encode_status(seat.status)
    return out


def decode_seat_view(d: Any) -> SeatView:
    _require(d, _SEAT_FIELDS, "seat")
    kwargs = {k: d[k] for k in _SEAT_FIELDS}
    kwargs["status"] = decode_status(d["status"])
    return SeatView(**kwargs)


def _encode_pot_view(pot: PotView) -> dict:
    return {"amount": pot.amount, "eligible": list(pot.eligible)}


def _decode_pot_view(d: Any) -> PotView:
    _require(d, ("amount", "eligible"), "side pot")
    return PotView(d["amount"], tuple(_list(d["eligible"], "eligible")))


_VIEW_FIELDS = (
    "hand_number",
    "seat",
    "hole_cards",
    "street",
    "board",
    "pot",
    "side_pots",
    "current_bet",
    "bet_unit",
    "raises",
    "button",
    "sb_seat",
    "bb_seat",
    "seats",
    "history",
    "waiting",
)


def encode_table_view(view: TableView) -> dict:
    return {
        "hand_number": view.hand_number,
        "seat": view.seat,
        "hole_cards": _opt(encode_hand, view.hole_cards),
        "street": encode_street(view.street),
        "board": encode_board(view.board),
        "pot": view.pot,
        "side_pots": [_encode_pot_view(p) for p in view.side_pots],
        "current_bet": view.current_bet,
        "bet_unit": view.bet_unit,
        "raises": view.raises,
        "button": view.button,
        "sb_seat": view.sb_seat,
        "bb_seat": view.bb_seat,
        "seats": [encode_seat_view(s) for s in view.seats],
        "history": [encode_event(e) for e in view.history],
        "waiting": list(view.waiting),
    }


def decode_table_view(d: Any) -> TableView:
    _require(d, _VIEW_FIELDS, "view")
    return TableView(
        hand_number=d["hand_number"],
        seat=d["seat"],
        hole_cards=_opt(decode_hand, d["hole_cards"]),
        street=decode_street(d["street"]),
        board=decode_board(d["board"]),
        pot=d["pot"],
        side_pots=tuple(_decode_pot_view(p) for p in _list(d["side_pots"], "side_pots")),
        current_bet=d["current_bet"],
        bet_unit=d["bet_unit"],
        raises=d["raises"],
        button=d["button"],
        sb_seat=d["sb_seat"],
        bb_seat=d["bb_seat"],
        seats=tuple(decode_seat_view(s) for s in _list(d["seats"], "seats")),
        history=tuple(decode_event(e) for e in _list(d["history"], "history")),
        waiting=tuple(_list(d["waiting"], "waiting")),
    )


# --- hand result --------------------------------------------------------------------


def _encode_pot_result(pot: PotResult) -> dict:
    return {
        "amount": pot.amount,
        "eligible": list(pot.eligible),
        "high": [[seat, amount] for seat, amount in pot.high],
        "low": [[seat, amount] for seat, amount in pot.low],
        "uncontested": pot.uncontested,
    }


def _decode_pot_result(d: Any) -> PotResult:
    _require(d, ("amount", "eligible", "high", "low", "uncontested"), "pot")
    return PotResult(
        amount=d["amount"],
        eligible=tuple(_list(d["eligible"], "eligible")),
        high=_pairs(d["high"], "high"),
        low=_pairs(d["low"], "low"),
        uncontested=d["uncontested"],
    )


_RESULT_FIELDS = (
    "hand_number",
    "button",
    "sb_seat",
    "bb_seat",
    "board",
    "showdown",
    "pots",
    "history",
    "seats",
)


def encode_hand_result(result: HandResult) -> dict:
    """The round summary. The per-seat dicts of `HandResult` (`names`, `dealt_in`,
    `net`, `stacks`, `shown`, `returned`) are merged into one `seats` list,
    because JSON object keys cannot be ints."""
    seats = sorted(
        set(result.stacks)
        | set(result.names)
        | set(result.net)
        | set(result.dealt_in)
        | set(result.shown)
        | set(result.returned)
    )
    return {
        "hand_number": result.hand_number,
        "button": result.button,
        "sb_seat": result.sb_seat,
        "bb_seat": result.bb_seat,
        "board": encode_board(result.board),
        "showdown": result.showdown,
        "pots": [_encode_pot_result(p) for p in result.pots],
        "history": [encode_event(e) for e in result.history],
        "seats": [
            {
                "seat": s,
                "name": result.names.get(s),
                "dealt_in": result.dealt_in.get(s),
                "net": result.net.get(s),
                "stack": result.stacks.get(s),
                "shown": _opt(encode_hand, result.shown.get(s)),
                "returned": result.returned.get(s),
            }
            for s in seats
        ],
    }


def decode_hand_result(d: Any) -> HandResult:
    _require(d, _RESULT_FIELDS, "hand result")
    names: dict[int, str] = {}
    dealt_in: dict[int, bool] = {}
    net: dict[int, float] = {}
    stacks: dict[int, float] = {}
    shown: dict[int, Hand] = {}
    returned: dict[int, float] = {}
    for entry in _list(d["seats"], "seats"):
        _require(entry, ("seat",), "result seat")
        s = entry["seat"]
        if entry.get("name") is not None:
            names[s] = entry["name"]
        if entry.get("dealt_in") is not None:
            dealt_in[s] = entry["dealt_in"]
        if entry.get("net") is not None:
            net[s] = entry["net"]
        if entry.get("stack") is not None:
            stacks[s] = entry["stack"]
        if entry.get("shown") is not None:
            shown[s] = decode_hand(entry["shown"])
        if entry.get("returned") is not None:
            returned[s] = entry["returned"]
    return HandResult(
        hand_number=d["hand_number"],
        button=d["button"],
        sb_seat=d["sb_seat"],
        bb_seat=d["bb_seat"],
        board=decode_board(d["board"]),
        showdown=d["showdown"],
        shown=shown,
        pots=tuple(_decode_pot_result(p) for p in _list(d["pots"], "pots")),
        net=net,
        stacks=stacks,
        history=tuple(decode_event(e) for e in _list(d["history"], "history")),
        names=names,
        dealt_in=dealt_in,
        returned=returned,
    )


# --- server -> client messages --------------------------------------------------------

SERVER_MESSAGE_TYPES = (
    "seated",
    "hole_cards",
    "event",
    "turn",
    "hand_end",
    "sat_out",
    "returned",
    "left",
    "close",
)


def encode_server_message(msg: tuple, to: str) -> dict:
    """Engine `RemoteAgent` message tuple -> callback-channel JSON object.

    ("joined", JoinReply) is not a callback message (it is the `join` response)
    and is rejected here.
    """
    if not isinstance(msg, tuple) or not msg:
        raise ProtocolError(f"not an engine message: {msg!r}")
    kind = msg[0]
    out: dict[str, Any] = {"type": kind, "to": to}
    if kind == "seated":
        _, view = msg
        out["view"] = encode_table_view(view)
    elif kind == "hole_cards":
        _, hand, view = msg
        out["hand_number"] = view.hand_number
        out["cards"] = encode_hand(hand)
        out["view"] = encode_table_view(view)
    elif kind == "event":
        _, event = msg
        out["event"] = encode_event(event)
    elif kind == "turn":
        _, seq, view, legal = msg
        out["seq"] = seq
        out["view"] = encode_table_view(view)
        out["legal"] = encode_legal(legal)
    elif kind == "hand_end":
        _, seq, seat, result = msg
        out["seq"] = seq
        out["seat"] = seat
        out["result"] = encode_hand_result(result)
    elif kind in ("sat_out", "returned", "close"):
        if len(msg) != 1:
            raise ProtocolError(f"unexpected payload in {kind!r}: {msg!r}")
    elif kind == "left":
        _, final_stack_bb = msg
        out["final_stack_bb"] = final_stack_bb
    else:
        raise ProtocolError(f"cannot forward engine message {kind!r}")
    return out


def decode_server_message(d: Any) -> tuple:
    """Callback-channel JSON object -> the engine `RemoteAgent` message tuple."""
    if not isinstance(d, dict):
        raise ProtocolError(f"a message must be an object, got {d!r}")
    kind = d.get("type")
    if kind == "seated":
        _require(d, ("view",), kind)
        return ("seated", decode_table_view(d["view"]))
    if kind == "hole_cards":
        _require(d, ("cards", "view"), kind)
        return ("hole_cards", decode_hand(d["cards"]), decode_table_view(d["view"]))
    if kind == "event":
        _require(d, ("event",), kind)
        return ("event", decode_event(d["event"]))
    if kind == "turn":
        _require(d, ("seq", "view", "legal"), kind)
        return ("turn", d["seq"], decode_table_view(d["view"]), decode_legal(d["legal"]))
    if kind == "hand_end":
        _require(d, ("seq", "seat", "result"), kind)
        return ("hand_end", d["seq"], d["seat"], decode_hand_result(d["result"]))
    if kind in ("sat_out", "returned", "close"):
        return (kind,)
    if kind == "left":
        _require(d, ("final_stack_bb",), kind)
        return ("left", d["final_stack_bb"])
    raise ProtocolError(f"unknown server message type {kind!r}")


# --- client -> server requests ----------------------------------------------------------

REQUEST_TYPES = ("join", "action", "hand_end", "rebuy", "sit_in", "leave")


def join_request(
    name: str, stack_bb: float, callback_port: int, seat: Optional[int] = None
) -> dict:
    return {
        "type": "join",
        "name": name,
        "stack_bb": stack_bb,
        "seat": seat,
        "callback_port": callback_port,
    }


def action_request(name: str, seq: int, action: Union[str, Action, Leave]) -> dict:
    if not isinstance(action, str):
        action = encode_action(action)
    return {"type": "action", "name": name, "seq": seq, "action": action}


def hand_end_request(
    name: str,
    seq: int,
    rebuy_bb: Optional[float] = None,
    sit_in: bool = False,
    leave: bool = False,
) -> dict:
    return {
        "type": "hand_end",
        "name": name,
        "seq": seq,
        "rebuy_bb": rebuy_bb,
        "sit_in": sit_in,
        "leave": leave,
    }


def rebuy_request(name: str, amount_bb: float) -> dict:
    return {"type": "rebuy", "name": name, "amount_bb": amount_bb}


def sit_in_request(name: str) -> dict:
    return {"type": "sit_in", "name": name}


def leave_request(name: str) -> dict:
    return {"type": "leave", "name": name}


def _seq(msg: dict) -> int:
    seq = msg.get("seq")
    if not _is_int(seq):
        raise ProtocolError(f"'seq' must be an int, got {seq!r}")
    return seq


def _flag(msg: dict, key: str) -> bool:
    value = msg.get(key, False)
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ProtocolError(f"{key!r} must be true or false, got {value!r}")
    return value


def validate_request(msg: Any) -> dict:
    """Check a client request and return it normalised (optional fields filled in).
    Raises ProtocolError for anything malformed."""
    if not isinstance(msg, dict):
        raise ProtocolError("a request must be a JSON object")
    kind = msg.get("type")
    if not isinstance(kind, str) or kind not in REQUEST_TYPES:
        raise ProtocolError(f"unknown request type {kind!r}")
    name = msg.get("name")
    if not isinstance(name, str) or not name:
        raise ProtocolError("'name' must be a non-empty string")
    out: dict[str, Any] = {"type": kind, "name": name}
    if kind == "join":
        stack = msg.get("stack_bb")
        if not _is_number(stack):
            raise ProtocolError(f"'stack_bb' must be a number, got {stack!r}")
        seat = msg.get("seat")
        if seat is not None and not _is_int(seat):
            raise ProtocolError(f"'seat' must be an int offset from the button or null, got {seat!r}")
        port = msg.get("callback_port")
        if not _is_int(port) or not 0 < port < 65536:
            raise ProtocolError(f"'callback_port' must be a port number, got {port!r}")
        out.update(stack_bb=stack, seat=seat, callback_port=port)
    elif kind == "action":
        seq = _seq(msg)
        action = msg.get("action")
        if not isinstance(action, str) or action not in ACTION_NAMES:
            raise ProtocolError(
                f"unknown action {action!r} (expected one of {', '.join(ACTION_NAMES)})"
            )
        out.update(seq=seq, action=action)
    elif kind == "hand_end":
        seq = _seq(msg)
        rebuy = msg.get("rebuy_bb")
        if rebuy is not None and not _is_number(rebuy):
            raise ProtocolError(f"'rebuy_bb' must be a number or null, got {rebuy!r}")
        out.update(seq=seq, rebuy_bb=rebuy, sit_in=_flag(msg, "sit_in"), leave=_flag(msg, "leave"))
    elif kind == "rebuy":
        amount = msg.get("amount_bb")
        if not _is_number(amount):
            raise ProtocolError(f"'amount_bb' must be a number, got {amount!r}")
        out["amount_bb"] = amount
    return out


def ok_response(**extra: Any) -> dict:
    return {"type": "response", "ok": True, **extra}


def error_response(error: str) -> dict:
    return {"type": "response", "ok": False, "error": error}


def encode_join_response(reply: JoinReply) -> dict:
    """The response to a `join` request, built from the engine's `JoinReply`."""
    out = {
        "type": "response",
        "ok": reply.accepted,
        "accepted": reply.accepted,
        "queued": reply.queued,
        "name": reply.name,
        "seat": reply.seat,
    }
    if not reply.accepted:
        out["error"] = reply.error or "join rejected"
    return out


def decode_join_response(d: Any) -> JoinReply:
    if not isinstance(d, dict):
        raise ProtocolError(f"a response must be an object, got {d!r}")
    accepted = bool(d.get("accepted", d.get("ok", False)))
    return JoinReply(
        accepted=accepted,
        queued=bool(d.get("queued", False)),
        name=d.get("name"),
        seat=d.get("seat"),
        error=None if accepted else d.get("error"),
    )
