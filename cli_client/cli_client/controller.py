"""Client: message handling, menus, replies and presets. No curses code, so tests
can drive it headless through `drain()`, `handle_key(key)` and `tick(now)`.

Keys are strings: printable characters as themselves, plus "up", "down",
"enter", "esc", "backspace", "pgup", "pgdn", "F2", "F3" and "ctrl-c".
"""

from __future__ import annotations

import logging
import math
import queue
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from engine import HandStarted, LegalActions, PlayerActed, TableView
from engine.events import FOLD_TIMEOUT

from cli_server.protocol import (
    LobbyStatus,
    ProtocolError,
    TableSettings,
    action_request,
    decode_acting,
    decode_event,
    decode_hand,
    decode_hand_result,
    decode_legal,
    decode_lobby,
    decode_table_view,
    hand_end_request,
    leave_request,
    rebuy_request,
    sit_in_request,
)

from .cards import fmt_amount
from .net import ConnectionLost
from .state import TableState

__all__ = [
    "ClientSettings",
    "MenuItem",
    "UIView",
    "Client",
    "PRESET_CHECK_FOLD",
    "PRESET_CALL_ANY",
    "PRESET_LABELS",
    "preset_action",
]

_log = logging.getLogger("cli_client")

PRESET_CHECK_FOLD = "check_fold"
PRESET_CALL_ANY = "call_any"
PRESET_LABELS = {PRESET_CHECK_FOLD: "Check/Fold", PRESET_CALL_ANY: "Call any"}

LOG_PAGE = 10
_EPS = 1e-9

HELP_LINES = (
    "1-9 / ↑↓ Enter   choose a menu option",
    "f c r            fold, check/call, bet/raise",
    "x or F2          preset Check/Fold",
    "a or F3          preset Call any",
    "t                top up (after this hand)",
    "r / s            rebuy / sit in (sitting out)",
    "l q Ctrl-C       leave the table",
    "PgUp PgDn        scroll the table log",
    "?                close this help",
)


@dataclass
class ClientSettings:
    stack_bb: float = 100.0  # buy-in, and the default rebuy / top-up target
    hand_end_timeout: float = 20.0  # rebuy / sit-in menu: default answer after this
    ascii: bool = False
    exit_delay: float = 1.5  # seconds the screen stays up after `left`


@dataclass
class MenuItem:
    label: str
    kind: str  # action: fold/check/call/raise/leave; hand end: rebuy/sit_out/sit_in/keep_out/leave_table
    hotkey: Optional[str] = None
    amount: Optional[float] = None


@dataclass
class UIView:
    """What the renderer needs besides the TableState."""

    now: float = 0.0
    ascii: bool = False
    menu_title: str = ""
    menu: list[MenuItem] = field(default_factory=list)
    selected: int = 0
    countdown: Optional[float] = None  # seconds left for my turn / hand-end menu
    preset: Optional[str] = None  # label of the active preset
    prompt: Optional[str] = None  # confirmation or input line
    status: str = ""
    error: str = ""
    lobby_text: Optional[str] = None
    show_help: bool = False
    help_lines: tuple[str, ...] = HELP_LINES
    log_scroll: int = 0


@dataclass
class _Input:
    prompt: str
    text: str
    on_submit: Callable[[float], None]


def preset_action(preset: str, legal: LegalActions) -> str:
    """The action a preset sends at the next turn."""
    if preset == PRESET_CHECK_FOLD:
        return "check" if legal.can_check else "fold"
    return "check" if legal.can_check else "call"


def _is_stale_turn(error: str) -> bool:
    return "pending turn" in error


def _is_during_hand(error: str) -> bool:
    return "during" in error


class Client:
    def __init__(
        self,
        name: str,
        conn: Any,
        inbox: "queue.Queue[Any]",
        clock: Callable[[], float] = time.monotonic,
        settings: Optional[ClientSettings] = None,
        table: Optional[TableSettings] = None,
        queued: bool = False,
    ) -> None:
        self.name = name
        self.conn = conn
        self.inbox = inbox
        self.clock = clock
        self.settings = settings or ClientSettings()
        self.table = table
        self.state = TableState(name, table, ascii=self.settings.ascii)
        self.state.queued = queued
        self.now = clock()
        # my turn
        self.pending_turn: Optional[tuple[int, LegalActions]] = None
        self.turn_deadline: Optional[float] = None
        # hand end
        self.pending_hand_end: Optional[int] = None
        self.hand_end_deadline: Optional[float] = None
        self.hand_end_default: dict = {}
        self.rebuy_amount: Optional[float] = None
        self.last_result: Any = None
        self.model_stacks_at_hand_end: dict[str, float] = {}
        # menus and prompts
        self.menu: list[MenuItem] = []
        self.menu_kind: Optional[str] = None  # "action" | "hand_end"
        self.menu_title = ""
        self.selected = 0
        self.confirm: Optional[tuple[str, Callable[[], None]]] = None
        self.input: Optional[_Input] = None
        # queued choices
        self.preset: Optional[str] = None
        self.queued_topup: Optional[float] = None
        self.queued_leave = False
        self.deferred: Optional[str] = None  # "rebuy" | "sit_in", rejected during a hand
        self.leaving = False
        # display
        self.lobby: Optional[tuple[LobbyStatus, float]] = None
        self.status = ""
        self.error = ""
        self.show_help = False
        self.log_scroll = 0
        self.dirty = True
        self.bell = False
        # lifecycle
        self.done = False
        self.exit_code = 0
        self.exit_at: Optional[float] = None
        self.final_stack: Optional[float] = None
        self.hands_played = 0
        self.server_closed = False
        self._ctrl_c = 0
        if queued:
            self.status = "A hand is running: you will be seated when it ends"

    # --- UI loop API -------------------------------------------------------------

    def drain(self) -> int:
        """Handle every message waiting in the inbox. Returns how many."""
        n = 0
        while True:
            try:
                msg = self.inbox.get_nowait()
            except queue.Empty:
                break
            self.handle_message(msg)
            n += 1
        if n:
            self.dirty = True
        return n

    def tick(self, now: Optional[float] = None) -> None:
        """Timers: the hand-end default answer, the exit delay, countdown redraws."""
        now = self.clock() if now is None else now
        previous, self.now = self.now, now
        if (
            self.pending_hand_end is not None
            and self.hand_end_deadline is not None
            and now >= self.hand_end_deadline
        ):
            self.input = None
            self.confirm = None
            default = dict(self.hand_end_default)
            self._send_hand_end(**default)
            self.status = "No answer in time: " + (
                "leaving" if default.get("leave") else "sitting out"
            )
            self.dirty = True
        if self.exit_at is not None and now >= self.exit_at:
            self.done = True
        if math.floor(previous) != math.floor(now):
            self.dirty = True

    # --- messages ------------------------------------------------------------------

    def handle_message(self, msg: Any) -> None:
        self.now = self.clock()
        self.dirty = True
        if isinstance(msg, tuple):
            if msg and msg[0] == "disconnected":
                self._on_closed()
            return
        if not isinstance(msg, dict):
            return
        kind = msg.get("type")
        handler = _HANDLERS.get(kind) if isinstance(kind, str) else None
        if handler is None:
            return  # unknown message type: forward compatible
        try:
            handler(self, msg)
        except (ProtocolError, KeyError) as e:
            _log.warning("cannot handle %r message: %s", kind, e)

    def _msg_seated(self, msg: dict) -> None:
        view = decode_table_view(msg["view"])
        self.state.apply_view(view, hand_running=False)
        self.state.queued = False
        if self.status.startswith("A hand is running"):
            self.status = ""
        self.state.add_log(f"You sit down at seat {view.seat + 1}", mine=True)

    def _msg_hole_cards(self, msg: dict) -> None:
        view = decode_table_view(msg["view"])
        self.state.apply_view(view, hand_running=True)
        self.state.my_cards = decode_hand(msg["cards"])
        self.hands_played += 1

    def _msg_event(self, msg: dict) -> None:
        try:
            event = decode_event(msg["event"])
        except ProtocolError:
            return  # unknown event type: ignored
        self.state.apply_event(event)
        if isinstance(event, HandStarted):
            self.lobby = None
            self.preset = None
            self.error = ""
            self.log_scroll = 0 if self.log_scroll == 0 else self.log_scroll
        elif isinstance(event, PlayerActed) and event.name == self.name:
            if self.pending_turn is not None:
                # The dealer moved on without us (timeout).
                self.pending_turn = None
                self.turn_deadline = None
                if self.menu_kind == "action":
                    self._clear_menu()
            if event.reason == FOLD_TIMEOUT:
                self.status = "You did not act in time and will sit out"

    def _msg_turn(self, msg: dict) -> None:
        seq = msg["seq"]
        view = decode_table_view(msg["view"])
        legal = decode_legal(msg["legal"])
        timeout = msg.get("timeout")
        self.state.apply_view(view, hand_running=True)
        self.pending_turn = (seq, legal)
        self.turn_deadline = (
            self.now + float(timeout) if isinstance(timeout, (int, float)) and timeout > 0 else None
        )
        self.state.acting = (view.seat, self.turn_deadline)
        self.error = ""
        if self.preset is not None:
            action = preset_action(self.preset, legal)
            self.preset = None
            self._send_action(action)
            return
        self._build_action_menu(view, legal)
        self.bell = True

    def _msg_hand_end(self, msg: dict) -> None:
        seq = msg["seq"]
        seat = msg.get("seat")
        result = decode_hand_result(msg["result"])
        self.model_stacks_at_hand_end = {s.name: s.stack for s in self.state.seats}
        self.state.apply_result(result)
        self.last_result = result
        self.preset = None
        self.pending_turn = None
        self.turn_deadline = None
        if self.menu_kind == "action":
            self._clear_menu()
        self.pending_hand_end = seq
        self.hand_end_deadline = None
        stack = result.stacks.get(seat) if isinstance(seat, int) else None
        if stack is None:
            me = self.state.me
            stack = me.stack if me is not None else None
        if self.queued_leave:
            self.queued_leave = False
            self._send_hand_end(leave=True)
            return
        if stack is not None and stack <= _EPS:
            self.rebuy_amount = self._default_rebuy(0.0)
            self._build_busted_menu()
        elif self.state.sitting_out:
            self._build_sitting_out_menu()
        else:
            topup, self.queued_topup = self.queued_topup, None
            self._send_hand_end(rebuy_bb=topup)
            return
        self.hand_end_deadline = self.now + self.settings.hand_end_timeout
        self.hand_end_default = {}
        self.bell = True

    def _msg_sat_out(self, msg: dict) -> None:
        self.state.sitting_out = True
        me = self.state.me
        if me is not None:
            from engine import PlayerStatus

            me.status = PlayerStatus.SITTING_OUT
        self.status = "You are sitting out"

    def _msg_returned(self, msg: dict) -> None:
        self.state.sitting_out = False
        me = self.state.me
        if me is not None:
            from engine import PlayerStatus

            me.status = PlayerStatus.ACTIVE
        self.status = "You are back in the game"
        self.deferred = None

    def _msg_left(self, msg: dict) -> None:
        self.state.left = True
        final = msg.get("final_stack_bb")
        if isinstance(final, (int, float)):
            self.final_stack = float(final)
        self._clear_pending()
        self.state.add_log(f"You left the table with {fmt_amount(self.final_stack)} BB", mine=True)
        self.status = "You left the table"
        self.exit_code = 0
        self.exit_at = self.now + self.settings.exit_delay

    def _msg_close(self, msg: dict) -> None:
        self._on_closed()

    def _msg_acting(self, msg: dict) -> None:
        acting = decode_acting(msg)
        seat = self.state.seat_of(acting.name)
        if seat is None:
            seat = acting.seat
        deadline = None if acting.timeout is None else self.now + acting.timeout
        self.state.acting = (seat, deadline)

    def _msg_lobby(self, msg: dict) -> None:
        self.lobby = (decode_lobby(msg), self.now)

    def _on_closed(self) -> None:
        if self.state.left:
            if self.exit_at is None:
                self.exit_at = self.now + self.settings.exit_delay
            return
        if self.server_closed:
            return
        self.server_closed = True
        self.exit_code = 1
        self._clear_pending()
        self.status = "Server closed the table — press any key to exit"
        self.state.add_log(("Server closed the table", "error"))

    def _connection_lost(self) -> None:
        if self.server_closed:
            return
        self.server_closed = True
        self.exit_code = 1
        self._clear_pending()
        self.status = "Connection to server lost — press any key to exit"
        self.state.add_log(("Connection to server lost", "error"))

    def _clear_pending(self) -> None:
        self.pending_turn = None
        self.turn_deadline = None
        self.pending_hand_end = None
        self.hand_end_deadline = None
        self.confirm = None
        self.input = None
        self._clear_menu()

    # --- requests ---------------------------------------------------------------

    def _request(self, msg: dict) -> Optional[dict]:
        try:
            return self.conn.request(msg)
        except ConnectionLost:
            self._connection_lost()
            return None

    def _send_action(self, action: str) -> None:
        if self.pending_turn is None:
            return
        seq, _legal = self.pending_turn
        response = self._request(action_request(self.name, seq, action))
        if response is None:
            return
        if response.get("ok"):
            self.pending_turn = None
            self.turn_deadline = None
            self.error = ""
            if self.menu_kind == "action":
                self._clear_menu()
            if action == "leave":
                self.leaving = True
                self.status = "Leaving after this hand…"
            return
        error = str(response.get("error") or "action rejected")
        if _is_stale_turn(error):
            self.pending_turn = None
            self.turn_deadline = None
            if self.menu_kind == "action":
                self._clear_menu()
            self.error = f"Too late: {error}"
        else:
            self.error = error  # the turn is still pending: the menu stays

    def _send_hand_end(
        self, rebuy_bb: Optional[float] = None, sit_in: bool = False, leave: bool = False
    ) -> None:
        seq = self.pending_hand_end
        if seq is None:
            return
        self.pending_hand_end = None
        self.hand_end_deadline = None
        if self.menu_kind == "hand_end":
            self._clear_menu()
        response = self._request(hand_end_request(self.name, seq, rebuy_bb, sit_in, leave))
        if response is None:
            return
        if not response.get("ok"):
            self.error = str(response.get("error") or "reply rejected")
            return
        if rebuy_bb or sit_in:
            self.deferred = None
        if rebuy_bb:
            self.status = f"Rebuy of {fmt_amount(rebuy_bb)} BB requested"
        elif sit_in:
            self.status = "Sitting in for the next hand"
        if leave:
            self.leaving = True
            self.status = "Leaving the table…"

    # --- menus ---------------------------------------------------------------------

    def _set_menu(self, kind: str, title: str, items: list[MenuItem], selected: int = 0) -> None:
        self.menu_kind = kind
        self.menu_title = title
        self.menu = items
        self.selected = max(0, min(selected, len(items) - 1))

    def _clear_menu(self) -> None:
        self.menu = []
        self.menu_kind = None
        self.menu_title = ""
        self.selected = 0

    def _build_action_menu(self, view: TableView, legal: LegalActions) -> None:
        me = view.seats[view.seat] if 0 <= view.seat < len(view.seats) else None
        stack = me.stack if me is not None else math.inf
        street_bet = me.street_bet if me is not None else 0.0
        items = [MenuItem("Fold", "fold", "f")]
        if legal.can_check:
            items.append(MenuItem("Check", "check", "c"))
        elif legal.call_amount > 0:
            label = f"Call {fmt_amount(legal.call_amount)}"
            if legal.call_amount >= stack - _EPS:
                label += " (all-in)"
            items.append(MenuItem(label, "call", "c", legal.call_amount))
        if legal.can_raise:
            verb = "Bet" if view.current_bet <= _EPS else "Raise to"
            label = f"{verb} {fmt_amount(legal.raise_to)}"
            if legal.raise_to - street_bet >= stack - _EPS:
                label += " (all-in)"
            items.append(MenuItem(label, "raise", "r", legal.raise_to))
        items.append(MenuItem("Leave table", "leave", "l"))
        self._set_menu("action", "Your turn", items, selected=1)

    def _build_busted_menu(self) -> None:
        amount = self.rebuy_amount
        items = []
        if amount is not None and amount > 0:
            items.append(MenuItem(f"Rebuy {fmt_amount(amount)} BB", "rebuy", "r", amount))
        items.append(MenuItem("Sit out", "sit_out"))
        items.append(MenuItem("Leave table", "leave_table", "l"))
        title = "You are out of chips"
        if amount is not None:
            title += " ([e] edits the amount)"
        self._set_menu("hand_end", title, items, selected=0)

    def _build_sitting_out_menu(self) -> None:
        items = [
            MenuItem("Sit in", "sit_in", "s"),
            MenuItem("Keep sitting out", "keep_out"),
            MenuItem("Leave table", "leave_table", "l"),
        ]
        selected = 0 if self.deferred == "sit_in" else 1
        self._set_menu("hand_end", "You are sitting out", items, selected=selected)

    def _max_add(self, current: float) -> Optional[float]:
        cap = self.table.max_buy_in_bb if self.table is not None else None
        return None if cap is None else round(cap - current, 2)

    def _default_rebuy(self, current: float) -> Optional[float]:
        """Back to --stack, capped by the max buy-in when known."""
        amount = round(self.settings.stack_bb - current, 2)
        cap = self._max_add(current)
        if cap is not None:
            amount = min(amount, cap)
        return amount if amount > 0 else None

    def _choose(self, item: MenuItem) -> None:
        kind = item.kind
        if self.menu_kind == "action":
            if kind == "fold":
                legal = self.pending_turn[1] if self.pending_turn is not None else None
                if legal is not None and legal.can_check:
                    self.confirm = ("Fold instead of checking? [y/N]", lambda: self._send_action("fold"))
                else:
                    self._send_action("fold")
            elif kind in ("check", "call", "raise"):
                self._send_action(kind)
            elif kind == "leave":
                self.confirm = (
                    "Leave the table? Your hand is folded. [y/N]",
                    lambda: self._send_action("leave"),
                )
        elif self.menu_kind == "hand_end":
            if kind == "rebuy":
                self._send_hand_end(rebuy_bb=item.amount)
            elif kind == "sit_in":
                self._send_hand_end(sit_in=True)
            elif kind in ("sit_out", "keep_out"):
                self._send_hand_end()
            elif kind == "leave_table":
                self._send_hand_end(leave=True)

    # --- keys -----------------------------------------------------------------------

    def handle_key(self, key: str) -> None:
        self.now = self.clock()
        self.dirty = True
        if key == "ctrl-c":
            self._ctrl_c_key()
            return
        if self.server_closed or self.state.left:
            self.done = True
            return
        if self.show_help:
            self.show_help = False
            return
        if self.input is not None:
            self._input_key(key)
            return
        if self.confirm is not None:
            prompt, action = self.confirm
            self.confirm = None
            if key in ("y", "Y"):
                action()
            return
        if key == "pgup":
            self.log_scroll = min(self.log_scroll + LOG_PAGE, max(0, len(self.state.log) - 1))
            return
        if key == "pgdn":
            self.log_scroll = max(0, self.log_scroll - LOG_PAGE)
            return
        if key == "?":
            self.show_help = True
            return
        if key == "esc":
            self.error = ""
            return
        if self.menu and self._menu_key(key):
            return
        if key in ("x", "F2"):
            self._toggle_preset(PRESET_CHECK_FOLD)
        elif key in ("a", "F3"):
            self._toggle_preset(PRESET_CALL_ANY)
        elif key == "t":
            self._top_up_key()
        elif key == "r":
            self._rebuy_key()
        elif key == "s":
            self._sit_in_key()
        elif key in ("l", "q"):
            self._leave_key()

    def _menu_key(self, key: str) -> bool:
        if len(key) == 1 and key.isdigit() and key != "0":
            index = int(key) - 1
            if index < len(self.menu):
                self.selected = index
                self._choose(self.menu[index])
                return True
            return False
        if key == "up":
            self.selected = (self.selected - 1) % len(self.menu)
            return True
        if key == "down":
            self.selected = (self.selected + 1) % len(self.menu)
            return True
        if key == "enter":
            self._choose(self.menu[self.selected])
            return True
        if key == "e" and self.menu_kind == "hand_end" and any(i.kind == "rebuy" for i in self.menu):
            self.input = _Input(
                "Rebuy amount (BB): ", fmt_amount(self.rebuy_amount), self._submit_rebuy_amount
            )
            return True
        for i, item in enumerate(self.menu):
            if item.hotkey is not None and item.hotkey == key:
                self.selected = i
                self._choose(item)
                return True
        return False

    def _input_key(self, key: str) -> None:
        inp = self.input
        assert inp is not None
        if key == "esc":
            self.input = None
        elif key == "backspace":
            inp.text = inp.text[:-1]
        elif key == "enter":
            self.input = None
            try:
                amount = float(inp.text)
            except ValueError:
                self.error = f"Not an amount: {inp.text!r}"
                return
            if not math.isfinite(amount) or amount <= 0:
                self.error = "The amount must be positive"
                return
            inp.on_submit(round(amount, 2))
        elif len(key) == 1 and (key.isdigit() or key == "."):
            inp.text += key

    def _ctrl_c_key(self) -> None:
        self._ctrl_c += 1
        if self._ctrl_c >= 2 or self.server_closed or self.state.left:
            if not self.leaving and not self.state.left and not self.server_closed:
                try:
                    self.conn.request(leave_request(self.name))
                except ConnectionLost:
                    pass
            self.done = True
            return
        self.input = None
        self.confirm = None
        self.show_help = False
        self._leave_key()

    def _toggle_preset(self, preset: str) -> None:
        me = self.state.me
        if self.pending_turn is not None:
            return  # it is my turn: use the menu
        if not self.state.in_hand or me is None or me.all_in:
            self.error = "Presets are available while you are in a hand"
            return
        self.preset = None if self.preset == preset else preset
        self.error = ""

    def _top_up_key(self) -> None:
        me = self.state.me
        if me is None:
            self.error = "You are not seated yet"
            return
        default = self._default_rebuy(me.stack)
        self.input = _Input("Top up (BB): ", fmt_amount(default) if default else "", self._submit_topup)

    def _submit_topup(self, amount: float) -> None:
        me = self.state.me
        cap = self._max_add(me.stack if me is not None else 0.0)
        if cap is not None and amount > cap + _EPS:
            self.error = f"The max buy-in allows at most {fmt_amount(cap)} BB more"
            return
        if self.state.hand_running:
            self.queued_topup = amount
            self.status = f"Top-up of {fmt_amount(amount)} BB will be added after this hand"
            return
        response = self._request(rebuy_request(self.name, amount))
        if response is None:
            return
        if response.get("ok"):
            self.status = f"Added {fmt_amount(amount)} BB"
            self.error = ""
            return
        error = str(response.get("error") or "rebuy rejected")
        if _is_during_hand(error):
            self.queued_topup = amount
            self.status = f"Top-up of {fmt_amount(amount)} BB will be added after this hand"
        else:
            self.error = error

    def _submit_rebuy_amount(self, amount: float) -> None:
        cap = self._max_add(0.0)
        if cap is not None and amount > cap + _EPS:
            amount = cap
            self.status = f"Capped by the max buy-in: {fmt_amount(cap)} BB"
        self.rebuy_amount = amount
        if self.menu_kind == "hand_end":
            self._build_busted_menu()

    def _rebuy_key(self) -> None:
        me = self.state.me
        if me is None:
            self.error = "You are not seated yet"
            return
        if me.stack > _EPS:
            self.error = "You still have chips: press [t] to top up"
            return
        amount = self._default_rebuy(0.0) or self.settings.stack_bb
        response = self._request(rebuy_request(self.name, amount))
        if response is None:
            return
        if response.get("ok"):
            self.status = f"Rebought {fmt_amount(amount)} BB"
            self.error = ""
            self.deferred = None
            return
        error = str(response.get("error") or "rebuy rejected")
        if _is_during_hand(error):
            self.deferred = "rebuy"
            self.status = "A hand is running: you can rebuy when it ends"
        else:
            self.error = error

    def _sit_in_key(self) -> None:
        me = self.state.me
        if me is None:
            self.error = "You are not seated yet"
            return
        if not self.state.sitting_out:
            self.error = "You are not sitting out"
            return
        if me.stack <= _EPS:
            self.error = "You have no chips: press [r] to rebuy"
            return
        response = self._request(sit_in_request(self.name))
        if response is None:
            return
        if response.get("ok"):
            self.status = "Sitting in…"
            self.error = ""
            return
        error = str(response.get("error") or "sit-in rejected")
        if _is_during_hand(error):
            self.deferred = "sit_in"
            self.status = "A hand is running: you can sit in when it ends"
        else:
            self.error = error

    def _leave_key(self) -> None:
        if self.leaving:
            self.status = "Leaving after this hand…"
            return
        if self.pending_hand_end is not None:
            self._send_hand_end(leave=True)
            return
        if self.state.in_hand:
            self.confirm = ("Leave the table? Your hand is folded. [y/N]", self._leave)
        else:
            self._leave()

    def _leave(self) -> None:
        if self.pending_turn is not None:
            # The dealer is waiting for this turn: answer it with "leave".
            self._send_action("leave")
            return
        response = self._request(leave_request(self.name))
        if response is None:
            return
        if response.get("ok"):
            self.leaving = True
            self.preset = None
            self.status = "Leaving after this hand…" if self.state.hand_running else "Leaving…"
        else:
            self.error = str(response.get("error") or "leave rejected")

    # --- display ---------------------------------------------------------------------

    def lobby_text(self) -> Optional[str]:
        if self.state.hand_running:
            return None
        if self.state.queued:
            return "Waiting for the current hand to finish"
        if self.lobby is not None:
            status, received = self.lobby
            remaining = None
            if status.starts_in is not None:
                remaining = max(0, math.ceil(status.starts_in - (self.now - received)))
            if status.state == "waiting":
                return f"Waiting for players ({status.active}/{status.needed})"
            if status.state == "starting":
                return "Game starts soon" if remaining is None else f"Game starts in {remaining} s"
            if status.state == "next_hand":
                return "Next hand soon" if remaining is None else f"Next hand in {remaining} s"
        active = sum(1 for s in self.state.seats if not s.sitting_out and s.stack > _EPS)
        if active < 2 and self.state.hand_number == 0 or active < 2:
            return "Waiting for players"
        return None

    def countdown(self) -> Optional[float]:
        deadline = None
        if self.pending_turn is not None:
            deadline = self.turn_deadline
        elif self.pending_hand_end is not None:
            deadline = self.hand_end_deadline
        return None if deadline is None else max(0.0, deadline - self.now)

    def ui_view(self) -> UIView:
        prompt = None
        if self.input is not None:
            prompt = f"{self.input.prompt}{self.input.text}_  (Enter: OK, Esc: cancel)"
        elif self.confirm is not None:
            prompt = self.confirm[0]
        status = self.status
        if self.leaving and not self.state.left and not status:
            status = "Leaving after this hand…"
        if self.queued_topup and not status:
            status = f"Top-up of {fmt_amount(self.queued_topup)} BB after this hand"
        return UIView(
            now=self.now,
            ascii=self.settings.ascii,
            menu_title=self.menu_title,
            menu=list(self.menu),
            selected=self.selected,
            countdown=self.countdown(),
            preset=PRESET_LABELS.get(self.preset) if self.preset else None,
            prompt=prompt,
            status=status,
            error=self.error,
            lobby_text=self.lobby_text(),
            show_help=self.show_help,
            log_scroll=self.log_scroll,
        )

    def summary(self) -> str:
        stack = self.final_stack
        if stack is None:
            me = self.state.me
            stack = me.stack if me is not None else None
        hands = "1 hand" if self.hands_played == 1 else f"{self.hands_played} hands"
        return f"{self.name}: {hands} played, final stack {fmt_amount(stack)} BB"


_HANDLERS: dict[str, Callable[[Client, dict], None]] = {
    "seated": Client._msg_seated,
    "hole_cards": Client._msg_hole_cards,
    "event": Client._msg_event,
    "turn": Client._msg_turn,
    "hand_end": Client._msg_hand_end,
    "sat_out": Client._msg_sat_out,
    "returned": Client._msg_returned,
    "left": Client._msg_left,
    "close": Client._msg_close,
    "acting": Client._msg_acting,
    "lobby": Client._msg_lobby,
}
