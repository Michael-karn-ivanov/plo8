"""BotClient: the message loop of one bot at the table.

One thread (the one calling `run()`) reads the inbox filled by cli_client's
`CallbackListener` and answers at once: every `turn` with the player's decision,
every `hand_end` with the rebuy / sit-in / leave reply of `policy.py`. Every
request to the server goes out from this thread through the one
`ServerConnection`, so requests never interleave. Signal handlers only call
`request_stop()`.

The player sees only the `turn` view (own hole cards, public state) and the
legal actions. `stack`, `seat` and `status` are client bookkeeping that the
player never reads.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable, Optional

from cli_client.net import DISCONNECTED, ConnectionLost
from cli_server.protocol import (
    ProtocolError,
    TableSettings,
    action_request,
    decode_event,
    decode_lobby,
    decode_server_message,
    decode_table_view,
    hand_end_request,
    leave_request,
    rebuy_request,
    sit_in_request,
)
from engine import ActionType, HandResult, HandStarted, LegalActions, PlayerRebought, TableView

from .players import Player
from .policy import HandEndChoice, hand_end_reply, is_busted
from .stats import Stats, format_bb, hand_line, summary

__all__ = [
    "STOP",
    "WAITING",
    "SEATED",
    "SITTING_OUT",
    "LEAVING",
    "LEFT",
    "LEFT_GRACE",
    "BotClient",
    "is_stale_turn_error",
    "fallback_action",
]

STOP = ("stop",)  # inbox marker: user stop (signal handler)

WAITING = "waiting"
SEATED = "seated"
SITTING_OUT = "sitting_out"
LEAVING = "leaving"
LEFT = "left"

LEFT_GRACE = 1.0  # seconds to wait for `close` / the end of the channel after `left`
POLL = 0.2  # inbox timeout: lets the loop see deadlines

# Decoding errors of a malformed message (the protocol raises ProtocolError; a
# bad value can also surface from an engine dataclass).
_DECODE_ERRORS = (ProtocolError, ValueError, TypeError, KeyError, IndexError)


def is_stale_turn_error(text: Any) -> bool:
    """The server rejected an `action` because its `seq` is not the pending turn
    (the turn timed out or was already answered) - as opposed to an illegal
    action for the pending turn. The one place that knows the server's wording
    (cli_server `ClientSession.submit_action`); pinned by test_integration."""
    text = str(text or "")
    return "is not the pending turn of" in text or "has no pending turn" in text


def fallback_action(legal: LegalActions) -> ActionType:
    """Check-or-call: always legal."""
    return ActionType.CHECK if legal.can_check else ActionType.CALL


class _Done(Exception):
    """Internal: the loop ends with `code`."""

    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


class BotClient:
    """One bot at the table (see the module docstring)."""

    def __init__(
        self,
        name: str,
        player: Player,
        conn: Any,
        inbox: "queue.Queue[Any]",
        *,
        stack_bb: float,
        rebuy: str = "topup",
        max_hands: Optional[int] = None,
        settings: Optional[TableSettings] = None,
        queued: bool = False,
        clock: Callable[[], float] = time.monotonic,
        leave_timeout: float = 5.0,
        listener: Any = None,
        logger: Optional[logging.Logger] = None,
        poll: float = POLL,
    ) -> None:
        self.name = name
        self.player = player
        self.conn = conn
        self.inbox = inbox
        self.listener = listener
        self.target = stack_bb
        self.rebuy_policy = rebuy
        self.max_hands = max_hands
        self.settings = settings
        self.clock = clock
        self.leave_timeout = leave_timeout
        self.poll = poll
        self.log = logger if logger is not None else logging.getLogger("bot_client")

        # State, readable by tests.
        self.status = WAITING if queued else SEATED
        self.stack: Optional[float] = stack_bb
        self.seat: Optional[int] = None
        self.hands_played = 0
        self.stats = Stats()
        self.rejected_actions = 0
        self.rejected_hand_ends = 0
        self.fallbacks = 0
        self.sat_out_count = 0
        self.turns_answered = 0
        self.hand_ends_answered = 0
        self.problems: list[str] = []
        self.final_stack: Optional[float] = None
        self.exit_code: Optional[int] = None

        self.stop_requested = False
        self._stop_lock = threading.Lock()
        self._stops = 0  # STOP markers queued by request_stop()
        self._stops_handled = 0
        self._leave_sent = False  # the stop's leave request was sent and accepted
        self._leave_deadline: Optional[float] = None
        self._left_at: Optional[float] = None
        self._pending_rebuy: Optional[float] = None  # asked at hand_end, not yet seen applied
        self._last_hand: Optional[int] = None  # number of the last hand seen
        self._retry_key: Optional[tuple] = None  # lobby status of the last between-hands retry

    # --- public API -------------------------------------------------------------

    def run(self) -> int:
        """Handle messages until an end condition; returns the exit status."""
        while self.exit_code is None:
            try:
                msg = self.inbox.get(timeout=self.poll)
            except queue.Empty:
                msg = None
            self.step(msg)
        return self.exit_code

    def step(self, msg: Any) -> Optional[int]:
        """Handle one inbox item (None: nothing arrived) and check the deadlines.
        Returns the exit status once the loop is over, else None."""
        if self.exit_code is not None:
            return self.exit_code
        try:
            if msg is not None:
                self.handle(msg)
            self.tick()
        except _Done as done:
            self.exit_code = done.code
        except ConnectionLost as e:
            self.log.error("connection to the server lost: %s", e)
            self.exit_code = 1
        return self.exit_code

    def request_stop(self) -> None:
        """User stop (signal handlers; any thread): leave the table, then exit."""
        with self._stop_lock:
            self._stops += 1
        self.inbox.put(STOP)

    def close(self) -> None:
        """Close the callback listener and the server connection (idempotent)."""
        if self.listener is not None:
            self.listener.close()
        self.conn.close()

    def summary(self) -> str:
        final = self.final_stack if self.final_stack is not None else self.stack
        return summary(self.name, self.player.label, self.stats, final)

    # --- dispatch -----------------------------------------------------------------

    def handle(self, msg: Any) -> None:
        if msg is STOP or msg == STOP:
            self._on_stop()
            return
        if msg is DISCONNECTED or msg == DISCONNECTED:
            self._on_disconnected()
            return
        if not isinstance(msg, dict):
            self._problem(f"unexpected inbox item {msg!r}")
            return
        kind = msg.get("type")
        handler = self._HANDLERS.get(kind)
        if handler is None:
            self.log.debug("ignoring message type %r", kind)
            return
        handler(self, msg)

    def tick(self) -> None:
        now = self.clock()
        if self._left_at is not None and now - self._left_at >= LEFT_GRACE:
            raise _Done(0)
        if self._leave_deadline is not None and now >= self._leave_deadline:
            self.log.warning("no confirmation of the leave after %gs: exiting", self.leave_timeout)
            raise _Done(0)

    def _problem(self, text: str, level: int = logging.WARNING) -> None:
        self.problems.append(text)
        self.log.log(level, "%s", text)

    def _request(self, msg: dict) -> dict:
        """One request to the server (raises ConnectionLost)."""
        response = self.conn.request(msg)
        if not isinstance(response, dict):
            return {"ok": False, "error": f"bad response {response!r}"}
        return response

    # --- simple messages ------------------------------------------------------------

    def _on_seated(self, msg: dict) -> None:
        try:
            view = decode_table_view(msg.get("view"))
        except _DECODE_ERRORS as e:
            self._problem(f"undecodable 'seated' message: {e}")
            self.status = SEATED
            return
        self.status = SEATED
        self._update_from_view(view)
        self.log.info("seated at seat %s", view.seat)

    def _on_hole_cards(self, msg: dict) -> None:
        try:
            _, hand, view = decode_server_message(msg)
        except _DECODE_ERRORS as e:
            self._problem(f"undecodable 'hole_cards' message: {e}")
            return
        self._update_from_view(view)
        self.log.debug("hand %d: hole cards %s", view.hand_number, hand)

    def _update_from_view(self, view: TableView) -> None:
        self.seat = view.seat
        try:
            self.stack = view.me.stack
        except (IndexError, TypeError):
            pass

    def _on_event(self, msg: dict) -> None:
        try:
            event = decode_event(msg.get("event"))
        except _DECODE_ERRORS as e:
            # An unknown event type (a newer server) is skipped like a bad one.
            self.log.debug("event skipped: %s", e)
            if not str(e).startswith("unknown event type"):
                self._problem(f"undecodable event: {e}")
            return
        if isinstance(event, HandStarted):
            self._on_hand_started(event)
            return
        if getattr(event, "name", None) != self.name:
            return
        stack = getattr(event, "stack_bb", None)
        if isinstance(stack, (int, float)):
            self.stack = float(stack)
        seat = getattr(event, "seat", None)
        if isinstance(seat, int):
            self.seat = seat
        if isinstance(event, PlayerRebought):
            self.stats.add_rebuy(event.amount_bb, self._last_hand)
            self._pending_rebuy = None
            self.log.debug("rebought %s BB", format_bb(event.amount_bb))

    def _on_hand_started(self, event: HandStarted) -> None:
        self._check_rebuy_applied()
        if event.hand_no is not None:
            self._last_hand = event.hand_no
        self._retry_key = None  # the next lobby status may retry again
        for seat, name, stack in event.participants:
            if name == self.name:
                self.seat = seat
                self.stack = stack

    def _check_rebuy_applied(self) -> None:
        if self._pending_rebuy is not None:
            self.log.warning("rebuy of %s BB was not applied", format_bb(self._pending_rebuy))
            self._pending_rebuy = None

    def _on_sat_out(self, msg: dict) -> None:
        self.status = SITTING_OUT if self.status not in (LEAVING, LEFT) else self.status
        self.sat_out_count += 1
        self.log.warning("sat out by the server")

    def _on_returned(self, msg: dict) -> None:
        if self.status not in (LEAVING, LEFT):
            self.status = SEATED
        self.log.info("back in the game")

    def _on_acting(self, msg: dict) -> None:
        pass

    def _on_left(self, msg: dict) -> None:
        self.status = LEFT
        final = msg.get("final_stack_bb")
        if isinstance(final, (int, float)) and not isinstance(final, bool):
            self.final_stack = float(final)
            self.stack = float(final)
            self.log.info("left the table with %s BB", format_bb(final))
        else:
            self.log.info("left the table")
        self._left_at = self.clock()

    def _on_close(self, msg: dict) -> None:
        if self.status != LEFT:
            self.log.info("the server closed the table")
        raise _Done(0)

    def _on_disconnected(self) -> None:
        if self.status == LEFT:
            raise _Done(0)
        if self.stop_requested and self._leave_sent:
            self.log.info("callback channel closed after the leave")
            raise _Done(0)
        self.log.error("connection to the server lost (callback channel closed)")
        raise _Done(1)

    # --- turns ------------------------------------------------------------------------

    def _on_turn(self, msg: dict) -> None:
        seq = msg.get("seq")
        try:
            _, seq, view, legal = decode_server_message(msg)
        except _DECODE_ERRORS as e:
            # Cannot be decided: the server's timeout applies.
            self._problem(f"undecodable turn {seq!r}, not answered: {e}", logging.ERROR)
            return
        try:
            t = self.player.decide(view, legal)
            if not isinstance(t, ActionType) or not legal.allows(t):
                raise ValueError(f"the player chose {t!r}, not legal now")
        except Exception as e:  # noqa: BLE001 - a bot bug must not lose the turn
            self.log.exception("decision failed on turn %s", seq)
            self.problems.append(f"decision failed on turn {seq}: {e!r}")
            t = fallback_action(legal)
            self.fallbacks += 1
        if self.log.isEnabledFor(logging.DEBUG):
            self.log.debug(
                "turn %s: hand %d %s board [%s] cards %s pot %g to call %g; legal %s -> %s",
                seq,
                view.hand_number,
                view.street.name.lower(),
                view.board,
                view.hole_cards,
                view.pot,
                legal.call_amount,
                "/".join(a.value for a in legal.actions),
                t.value,
            )
        response = self._request(action_request(self.name, seq, t.value))
        if response.get("ok"):
            self.turns_answered += 1
            return
        error = response.get("error")
        self.rejected_actions += 1
        self.log.warning("action %s for turn %s rejected: %s", t.value, seq, error)
        if is_stale_turn_error(error):
            return  # the turn timed out or was answered: nothing to retry
        fallback = fallback_action(legal)
        self.fallbacks += 1
        response = self._request(action_request(self.name, seq, fallback.value))
        if response.get("ok"):
            self.turns_answered += 1
            self.log.warning("fell back to %s for turn %s", fallback.value, seq)
        else:
            self.log.warning(
                "fallback %s for turn %s rejected too: %s", fallback.value, seq, response.get("error")
            )

    # --- hand end ------------------------------------------------------------------------

    def _on_hand_end(self, msg: dict) -> None:
        seq = msg.get("seq")
        try:
            _, seq, seat, result = decode_server_message(msg)
        except _DECODE_ERRORS as e:
            # Answer anyway (no rebuy / sit-in): a missing reply only costs the window.
            self._problem(f"undecodable hand_end {seq!r}: {e}")
            if isinstance(seq, int) and not isinstance(seq, bool):
                self._send_hand_end(seq, HandEndChoice(None, self.status == SITTING_OUT, self.stop_requested))
            return
        self._check_rebuy_applied()
        seat = self._own_seat(seat, result)
        dealt_in = bool(result.dealt_in.get(seat, False)) if seat is not None else False
        net = result.net.get(seat, 0.0) if seat is not None else 0.0
        stack = result.stacks.get(seat) if seat is not None else None
        if stack is None:
            stack = self.stack if self.stack is not None else 0.0
        self.seat = seat
        self.stack = stack
        self._last_hand = result.hand_number
        if dealt_in:
            self.hands_played += 1
            self.stats.add_hand(result.hand_number, net)
        reply = self._reply_for(stack)
        self._send_hand_end(seq, reply)
        if reply.rebuy_bb:
            self._pending_rebuy = reply.rebuy_bb
        if dealt_in:
            self.log.info("%s", hand_line(result.hand_number, net, stack, reply.rebuy_bb))
        if reply.leave and self.status != LEAVING:
            self.status = LEAVING
            self.log.info("leaving: %s", reply.reason)

    def _own_seat(self, seat: Any, result: HandResult) -> Optional[int]:
        if isinstance(seat, int) and result.names.get(seat, self.name) == self.name:
            return seat
        for s, n in result.names.items():
            if n == self.name:
                return s
        return seat if isinstance(seat, int) else None

    def _reply_for(self, stack: float) -> HandEndChoice:
        return hand_end_reply(
            stack=stack,
            sitting_out=self.status == SITTING_OUT,
            target=self.target,
            max_buy_in=self.settings.max_buy_in_bb if self.settings is not None else None,
            policy=self.rebuy_policy,
            hands_played=self.hands_played,
            max_hands=self.max_hands,
            stopping=self.stop_requested,
        )

    def _send_hand_end(self, seq: int, reply: HandEndChoice) -> None:
        response = self._request(
            hand_end_request(self.name, seq, reply.rebuy_bb, reply.sit_in, reply.leave)
        )
        if response.get("ok"):
            self.hand_ends_answered += 1
        else:
            self.rejected_hand_ends += 1
            self.log.warning("hand_end reply %s rejected: %s", seq, response.get("error"))

    # --- between hands ----------------------------------------------------------------------

    def _on_lobby(self, msg: dict) -> None:
        """No hand is running. A busted or sitting-out bot may get no further
        `hand_end` to retry in (e.g. heads-up the table cannot start without
        it), so it asks directly, once per lobby status change."""
        try:
            lobby = decode_lobby(msg)
        except _DECODE_ERRORS as e:
            self.log.debug("lobby message skipped: %s", e)
            return
        if self.status not in (SEATED, SITTING_OUT) or self.stop_requested:
            return
        stack = self.stack if self.stack is not None else 0.0
        busted = is_busted(stack)
        if not (self.status == SITTING_OUT or (busted and self.rebuy_policy != "never")):
            return
        key = (lobby.state, lobby.active, lobby.needed)
        if key == self._retry_key:
            return
        self._retry_key = key
        reply = self._reply_for(stack)
        if reply.leave:
            return
        if reply.rebuy_bb:
            response = self._request(rebuy_request(self.name, reply.rebuy_bb))
            if response.get("ok"):
                self.log.info("rebuy of %s BB between hands", format_bb(reply.rebuy_bb))
                return  # a rebuy also brings a sitting-out player back
            self.log.warning(
                "rebuy of %s BB between hands rejected: %s",
                format_bb(reply.rebuy_bb),
                response.get("error"),
            )
            return
        if reply.sit_in and not busted:
            response = self._request(sit_in_request(self.name))
            if response.get("ok"):
                self.log.info("asked to sit in between hands")
            else:
                self.log.warning("sit-in between hands rejected: %s", response.get("error"))

    # --- user stop ------------------------------------------------------------------------------

    def _on_stop(self) -> None:
        self._stops_handled += 1
        if self.stop_requested:
            self.log.warning("second stop: exiting now")
            raise _Done(0)
        self.stop_requested = True
        if self.status == LEFT:
            return  # already gone: `close` / the end of the channel follows
        self.log.info("leaving: stopping")
        if self.status != LEAVING:
            self.status = LEAVING
        response = self._request(leave_request(self.name))  # ConnectionLost -> exit 1
        if response.get("ok"):
            self._leave_sent = True
        else:
            self.log.warning("leave request rejected: %s", response.get("error"))
        self._leave_deadline = self.clock() + self.leave_timeout

    _HANDLERS: dict[str, Callable[["BotClient", dict], None]] = {
        "seated": _on_seated,
        "hole_cards": _on_hole_cards,
        "event": _on_event,
        "turn": _on_turn,
        "hand_end": _on_hand_end,
        "sat_out": _on_sat_out,
        "returned": _on_returned,
        "acting": _on_acting,
        "lobby": _on_lobby,
        "left": _on_left,
        "close": _on_close,
    }
