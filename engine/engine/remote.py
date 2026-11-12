"""Adapter for agents running in separate processes.

The host process owns the Environment and Dealer. Each remote seat gets a
`RemoteAgent` proxy that talks to `agent_loop` running in the agent's process
over a `multiprocessing` connection.

Protocol (all messages are picklable):
  host -> agent: ("turn", seq, TableView, LegalActions)
                 ("hand_end", seq, seat, HandResult)   # also sent while sitting out
                 ("hole_cards", Hand, TableView)       # one-way: own cards, before any turn
                 ("event", TableEvent)                 # one-way: every public table event
                 ("seated", TableView)                 # one-way notification
                 ("sat_out",) / ("returned",)          # one-way status notifications
                 ("left", final_stack_bb)              # removed; "close" follows
                 ("joined", JoinReply)                 # answer to a Join request
                 ("close",)
  agent -> host: Join(name, stack_bb, seat)            # first message on a join connection
                 ("action", seq, Action | Leave | None)
                 ("hand_end", seq, HandEndReply)
                 Leave()                               # unsolicited, outside a turn

Replies carry the request's sequence number (the turn id), so a late reply to a
request that already timed out is discarded instead of being taken as the next
answer. One-way messages are sent in order on the same connection and never
wait for a reply, so a slow observer does not delay the hand, and the agent
process sees its hole cards and all preceding events before the turn request.

On the host, each `RemoteAgent` owns a reader thread for its connection: it
queues replies for the dealer and turns an unsolicited `Leave` (or a connection
closed by the agent) into `env.leave(player)`.

Timeout: `RemoteAgent(timeout=...)`, or else the environment's `action_timeout`.
No valid reply in time counts as not acting: the dealer folds the hand and the
player sits out from the next hand.

Joining from another process: the agent connects (e.g. to a `JoinListener`),
sends a `Join`, and the host answers with a `JoinReply` (accepted / queued /
error). If accepted, the same connection is then used for play.
"""

from __future__ import annotations

import logging
import multiprocessing
import queue
import threading
import time
from dataclasses import dataclass
from multiprocessing.connection import Client, Listener
from typing import Any, Iterable, Optional

from entities import Hand

from .actions import Action, ActionType, IllegalActionError, LegalActions
from .events import TableEvent
from .player import Player, PlayerStatus
from .views import HandResult, TableView

__all__ = [
    "RemoteAgent",
    "RemotePlayer",
    "Join",
    "JoinReply",
    "Leave",
    "HandEndReply",
    "JoinListener",
    "agent_loop",
    "handle_join",
    "join_table",
    "connect_and_join",
    "start_agent_process",
]

_log = logging.getLogger("engine.remote")


@dataclass(frozen=True)
class Join:
    """Request from an agent process to join the table."""

    name: Optional[str]
    stack_bb: float
    seat: Optional[int] = None  # offset from the button; None = join as big blind


@dataclass(frozen=True)
class JoinReply:
    """Host answer to a `Join`. `seat` is set when the player was seated at once."""

    accepted: bool
    queued: bool = False
    name: Optional[str] = None
    seat: Optional[int] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class Leave:
    """Agent -> host: leave the table (as a turn reply, or unsolicited)."""


@dataclass(frozen=True)
class HandEndReply:
    """Agent -> host answer to "hand_end": what to do in the rebuy window."""

    rebuy_bb: Optional[float] = None
    sit_in: bool = False
    leave: bool = False


_CLOSED = object()  # reader sentinel: the connection is gone


class RemoteAgent:
    """Host-side `Agent` that forwards callbacks to an agent in another process."""

    def __init__(self, conn: Any, timeout: Optional[float] = None) -> None:
        self.conn = conn
        self.timeout = timeout
        self.player: Optional[Player] = None
        self.closed = False  # nothing more can be sent / received
        self.discarded = 0  # stale (late) replies thrown away
        self._closing = False  # the host closed the connection on purpose
        self._seq = 0
        self._send_lock = threading.Lock()
        self._replies: "queue.Queue[Any]" = queue.Queue()
        self._reader: Optional[threading.Thread] = None
        self._reader_lock = threading.Lock()
        self._leave_pending = False

    # --- plumbing ------------------------------------------------------------

    def bind(self, player: Player) -> None:
        """Called by the environment when the player is attached to the table."""
        self.player = player
        self._ensure_reader()
        if self._leave_pending:
            self._leave_pending = False
            self._leave()

    def _ensure_reader(self) -> None:
        with self._reader_lock:
            if self._reader is None and not self.closed:
                self._reader = threading.Thread(target=self._read_loop, daemon=True)
                self._reader.start()

    def _read_loop(self) -> None:
        while True:
            try:
                msg = self.conn.recv()
            except (EOFError, OSError):
                lost = not self._closing
                self.closed = True
                self._replies.put(_CLOSED)
                if lost:
                    self._leave()  # the agent disconnected
                return
            except Exception:  # noqa: BLE001 - malformed message: ignore it
                continue
            if isinstance(msg, Leave):
                self._leave()
            else:
                self._replies.put(msg)

    def _leave(self) -> None:
        player = self.player
        if player is None:
            self._leave_pending = True
            return
        try:
            player.leave()
        except (ValueError, IllegalActionError):
            pass

    def _send(self, msg: Any) -> bool:
        """Send one message atomically (the reader / listener threads and the
        dealer may share the connection). Returns False if the connection is gone."""
        if self.closed:
            return False
        try:
            with self._send_lock:
                self.conn.send(msg)
        except (OSError, EOFError, ValueError):
            self.closed = True
            return False
        return True

    def _timeout_for(self, player: Optional[Player]) -> Optional[float]:
        if self.timeout is not None:
            return self.timeout
        env = getattr(player, "_env", None)
        return getattr(env, "action_timeout", None)

    def _request(self, kind: str, player: Player, *payload: Any) -> Optional[tuple]:
        """Send a request and wait for the reply with the same sequence number.
        Returns None on timeout or when the connection is gone."""
        if self.closed:
            return None
        self._ensure_reader()
        self._seq += 1
        seq = self._seq
        if not self._send((kind, seq, *payload)):
            return None
        timeout = self._timeout_for(player)
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            try:
                if deadline is None:
                    reply = self._replies.get()
                else:
                    reply = self._replies.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                return None  # timed out; a late reply will be discarded
            if reply is _CLOSED:
                self._replies.put(_CLOSED)  # keep later requests from blocking
                return None
            if isinstance(reply, tuple) and len(reply) == 3 and reply[1] == seq:
                return reply
            self.discarded += 1  # stale or malformed reply

    # --- Agent callbacks -----------------------------------------------------

    def on_turn(self, player: Player, view: TableView, legal: LegalActions) -> None:
        if self.player is None:
            self.bind(player)
        reply = self._request("turn", player, view, legal)
        if reply is None or reply[0] != "action":
            return  # no action: the dealer folds and the player sits out
        action = reply[2]
        if isinstance(action, Leave):
            player.leave()  # counts as a fold
        elif isinstance(action, Action) and legal.allows(action.type):
            player.act(action)

    def on_hand_end(self, player: Player, result: HandResult) -> None:
        if self.player is None:
            self.bind(player)
        reply = self._request("hand_end", player, player.seat, result)
        if reply is None or reply[0] != "hand_end" or not isinstance(reply[2], HandEndReply):
            return
        answer: HandEndReply = reply[2]
        if answer.rebuy_bb:
            try:
                player.rebuy(answer.rebuy_bb)
            except (ValueError, TypeError, IllegalActionError):
                pass  # invalid rebuy requests are ignored
        if answer.sit_in:
            try:
                player.sit_in()
            except IllegalActionError:
                pass
        if answer.leave:
            player.leave()

    def on_hole_cards(self, player: Player, hand: Hand, view: TableView) -> None:
        if self.player is None:
            self.bind(player)
        self._send(("hole_cards", hand, view))

    def on_event(self, player: Player, event: TableEvent) -> None:
        self._send(("event", event))

    def on_seated(self, player: Player, view: TableView) -> None:
        if self.player is None:
            self.bind(player)
        self._send(("seated", view))

    def on_sit_out(self, player: Player) -> None:
        self._send(("sat_out",))

    def on_return(self, player: Player) -> None:
        self._send(("returned",))

    def on_left(self, player: Player, final_stack_bb: float) -> None:
        self._send(("left", final_stack_bb))
        self.close()

    def close(self) -> None:
        if not self.closed:
            self._closing = True
            self._send(("close",))
            self.closed = True


class RemotePlayer:
    """Agent-side stand-in for `Player`, with the same action/rebuy/leave API.

    It records the agent's decision, which `agent_loop` sends back to the host;
    the host re-validates it against the real rules.
    """

    def __init__(self, conn: Any = None) -> None:
        self.seat: Optional[int] = None
        self.name: Optional[str] = None
        self.stack = 0.0
        self.hole_cards: Optional[Hand] = None
        self.status = PlayerStatus.WAITING
        self.final_stack: Optional[float] = None
        self._conn = conn
        self._legal: Optional[LegalActions] = None
        self._pending: Optional[Any] = None  # Action or Leave
        self._hand_end_open = False
        self._rebuy = 0.0
        self._sit_in = False
        self._leave = False
        self._left = False

    @property
    def sitting_out(self) -> bool:
        return self.status is PlayerStatus.SITTING_OUT

    def _update_from_view(self, view: TableView) -> None:
        self.seat = view.seat
        me = view.seats[view.seat]
        self.name = me.name
        self.stack = me.stack
        self.status = me.status
        self.hole_cards = view.hole_cards

    def _update_from_event(self, event: TableEvent) -> None:
        """Keep `stack` current from public events about this player."""
        if self.name is None or getattr(event, "name", None) != self.name:
            return
        stack = getattr(event, "stack_bb", None)
        if stack is not None:
            self.stack = stack
        seat = getattr(event, "seat", None)
        if seat is not None and event.hand_no is None:
            self.seat = seat  # table events between hands use the current numbering

    def fold(self) -> None:
        self._record(ActionType.FOLD)

    def check(self) -> None:
        self._record(ActionType.CHECK)

    def call(self) -> None:
        self._record(ActionType.CALL)

    def raise_(self) -> None:
        self._record(ActionType.RAISE)

    def act(self, action: Action) -> None:
        if not isinstance(action, Action):
            raise IllegalActionError(f"not an Action: {action!r}")
        self._record(action.type)

    def _record(self, action_type: ActionType) -> None:
        if self._legal is None:
            raise IllegalActionError("it is not this player's turn")
        if self._pending is not None:
            raise IllegalActionError("already acted this turn")
        if not self._legal.allows(action_type):
            raise IllegalActionError(f"{action_type.value} is not legal now")
        self._pending = Action(action_type)

    def rebuy(self, amount_bb: float) -> None:
        if not self._hand_end_open:
            raise IllegalActionError("remote players rebuy from on_hand_end")
        if isinstance(amount_bb, bool) or not isinstance(amount_bb, (int, float)) or amount_bb <= 0:
            raise ValueError(f"rebuy amount must be positive, got {amount_bb!r}")
        self._rebuy += amount_bb
        self.stack += amount_bb

    def sit_in(self) -> None:
        if not self._hand_end_open:
            raise IllegalActionError("remote players sit in from on_hand_end")
        self._sit_in = True

    def leave(self) -> None:
        """During a turn: answer with Leave (a fold); in on_hand_end: leave flag in
        the reply; otherwise an unsolicited Leave message. Idempotent."""
        if self._left:
            return
        self._left = True
        if self._legal is not None:
            self._pending = Leave()
        elif self._hand_end_open:
            self._leave = True
        elif self._conn is not None:
            try:
                self._conn.send(Leave())
            except (OSError, EOFError, ValueError):
                pass

    def show_cards(self) -> Optional[Hand]:
        return self.hole_cards


def _call(agent: Any, name: str, *args: Any) -> None:
    callback = getattr(agent, name, None)
    if callable(callback):
        callback(*args)


def _notify(agent: Any, name: str, *args: Any) -> None:
    """Like `_call`, for one-way notifications: an exception in the agent is
    logged instead of ending the loop."""
    try:
        _call(agent, name, *args)
    except Exception:  # noqa: BLE001
        _log.exception("agent callback %s raised", name)


def agent_loop(conn: Any, agent: Any, _initial: Iterable[Any] = ()) -> None:
    """Run in the agent process: dispatch host messages to a local agent.

    `agent` has the same callbacks as an in-process agent (`on_turn`,
    `on_hand_end`, optional `on_hole_cards`, `on_event`, `on_seated`,
    `on_sit_out`, `on_return`, `on_left`) and receives a `RemotePlayer`.
    Returns when the host sends "close" / "left" or the connection is closed.
    """
    player = RemotePlayer(conn)
    initial = list(_initial)
    try:
        while True:
            if initial:
                msg = initial.pop(0)
            else:
                msg = conn.recv()
            kind = msg[0]
            if kind == "close":
                return
            if kind == "event":
                event = msg[1]
                player._update_from_event(event)
                _notify(agent, "on_event", player, event)
            elif kind == "hole_cards":
                _, hand, view = msg
                player._update_from_view(view)
                player.hole_cards = hand
                _notify(agent, "on_hole_cards", player, hand, view)
            elif kind == "seated":
                view = msg[1]
                player._update_from_view(view)
                _call(agent, "on_seated", player, view)
            elif kind == "sat_out":
                player.status = PlayerStatus.SITTING_OUT
                _call(agent, "on_sit_out", player)
            elif kind == "returned":
                player.status = PlayerStatus.ACTIVE
                _call(agent, "on_return", player)
            elif kind == "left":
                player.status = PlayerStatus.LEFT
                player.final_stack = msg[1]
                player.seat = None
                _call(agent, "on_left", player, msg[1])
                return
            elif kind == "turn":
                _, seq, view, legal = msg
                player._update_from_view(view)
                player._legal = legal
                player._pending = None
                try:
                    agent.on_turn(player, view, legal)
                except IllegalActionError:
                    pass
                action = player._pending
                player._legal = None
                player._pending = None
                conn.send(("action", seq, action))
            elif kind == "hand_end":
                _, seq, seat, result = msg
                player.seat = seat
                player.stack = result.stacks.get(seat, player.stack)
                player._rebuy = 0.0
                player._sit_in = False
                player._leave = False
                player._hand_end_open = True
                try:
                    agent.on_hand_end(player, result)
                finally:
                    player._hand_end_open = False
                    player.hole_cards = None
                conn.send(
                    (
                        "hand_end",
                        seq,
                        HandEndReply(player._rebuy or None, player._sit_in, player._leave),
                    )
                )
    except (EOFError, OSError):
        return


# --- joining from another process ---------------------------------------------


def handle_join(env: Any, conn: Any, timeout: Optional[float] = None) -> Optional[Player]:
    """Host side: read a `Join` from `conn`, join `env` with a `RemoteAgent`
    on that connection and send back a `JoinReply`.

    Returns the new player, or None if the request was rejected (the connection
    is then closed).
    """
    try:
        msg = conn.recv()
    except (EOFError, OSError):
        return None
    remote = RemoteAgent(conn, timeout=timeout)
    if not isinstance(msg, Join):
        remote._send(("joined", JoinReply(False, error=f"expected a Join message, got {msg!r}")))
        remote._closing = remote.closed = True
        conn.close()
        return None
    # Hold the table lock so the player cannot be seated (end of a hand) between
    # joining and reading his status.
    with env._lock:
        try:
            player = env._join(msg.stack_bb, seat=msg.seat, name=msg.name, agent=remote)
        except (ValueError, TypeError) as e:  # includes TableFullError
            remote._send(("joined", JoinReply(False, name=msg.name, error=str(e))))
            remote._closing = remote.closed = True
            conn.close()
            return None
        queued = player.status is PlayerStatus.WAITING
        remote._send(
            ("joined", JoinReply(True, queued=queued, name=player.name, seat=player.seat))
        )
    # Deliver what seating queued under the lock (on_seated, PlayerJoined).
    flush = getattr(env, "_flush", None)
    if callable(flush):
        flush()
    return player


class JoinListener:
    """Host-side thread that accepts agent connections and lets them join `env`.

    `address`/`family`/`authkey` are passed to `multiprocessing.connection.Listener`;
    agents connect with `connect_and_join(listener.address, ...)`. `timeout`
    defaults to the environment's `action_timeout`.
    """

    def __init__(
        self,
        env: Any,
        address: Any = None,
        family: Optional[str] = None,
        authkey: Optional[bytes] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.env = env
        self.timeout = timeout
        self.authkey = authkey
        self.listener = Listener(address, family, authkey=authkey)
        self.address = self.listener.address
        self.players: list[Player] = []  # accepted joiners
        self.rejected = 0
        self._closed = False
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self) -> "JoinListener":
        self._thread.start()
        return self

    def _serve(self) -> None:
        while not self._closed:
            try:
                conn = self.listener.accept()
            except Exception:  # noqa: BLE001 - closed listener / failed handshake
                if self._closed:
                    return
                continue
            if self._closed:
                conn.close()
                return
            player = handle_join(self.env, conn, timeout=self.timeout)
            if player is None:
                self.rejected += 1
            else:
                self.players.append(player)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:  # wake up a blocking accept()
            Client(self.address, authkey=self.authkey).close()
        except Exception:  # noqa: BLE001
            pass
        self.listener.close()
        self._thread.join(5)

    def __enter__(self) -> "JoinListener":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()


def join_table(conn: Any, join: Join, agent: Any) -> JoinReply:
    """Agent side: send `join` over `conn`, wait for the host's reply and, if
    accepted, play with `agent` until the host closes the connection.

    If the agent has an `on_join_reply(reply)` method it is called as soon as
    the reply arrives. Returns the reply.
    """
    conn.send(join)
    early: list[Any] = []  # e.g. "seated" sent before the reply
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return JoinReply(False, name=join.name, error="connection closed")
        if isinstance(msg, tuple) and msg and msg[0] == "joined":
            reply = msg[1]
            break
        early.append(msg)
    _call(agent, "on_join_reply", reply)
    if reply.accepted:
        agent_loop(conn, agent, _initial=early)
    return reply


def connect_and_join(
    address: Any, join: Join, agent: Any, authkey: Optional[bytes] = None
) -> JoinReply:
    """Agent side: connect to a `JoinListener` and play via `join_table`."""
    conn = Client(address, authkey=authkey)
    try:
        return join_table(conn, join, agent)
    finally:
        conn.close()


def start_agent_process(
    agent: Any, timeout: Optional[float] = None, context: Any = None
) -> tuple[RemoteAgent, Any]:
    """Start `agent` in a child process; return (RemoteAgent for the host, Process).

    With the "spawn" start method (macOS/Windows default) the agent must be
    picklable and importable in the child.
    """
    ctx = context if context is not None else multiprocessing.get_context()
    host_conn, agent_conn = ctx.Pipe()
    proc = ctx.Process(target=agent_loop, args=(agent_conn, agent), daemon=True)
    proc.start()
    agent_conn.close()  # the child owns its end now
    return RemoteAgent(host_conn, timeout=timeout), proc
