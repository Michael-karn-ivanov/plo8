"""ClientSession: one joined client, used as the `conn` of the engine's `RemoteAgent`.

`RemoteAgent` only needs `send(msg)`, `recv()` and `close()`, so it keeps doing the
sequence numbers, the action timeout, dropping late replies and handling
disconnects. The session does the rest:

- `send` (called on the dealer thread) turns an engine message tuple into a JSON
  line and puts it on an outbound queue. A writer thread writes it to the client's
  callback port, so a slow client never blocks the dealer.
- Replies do not arrive on the callback channel. The server's request handlers
  check them against the pending `turn` / `hand_end` and put them into `inbox`,
  which `recv` reads.
- When the callback channel breaks (a write fails, or the client closes it),
  `recv` raises EOFError. `RemoteAgent` takes that as a disconnect and makes the
  player leave.
"""

from __future__ import annotations

import logging
import queue
import socket
import threading
from typing import Any, Callable, Optional, Union

from engine import (
    Action,
    HandEndReply,
    HandStarted,
    JoinReply,
    LegalActions,
    Leave,
    Player,
    PlayerActed,
)

from .protocol import encode_line, encode_server_message

__all__ = ["ClientSession", "SessionError"]

_log = logging.getLogger("cli_server.session")

_EOF = object()  # inbox sentinel: the channel is gone
_STOP = object()  # writer sentinel: flush, then close the socket


class SessionError(Exception):
    """A client request that does not fit the session's state (no such pending
    turn, illegal action, closed session...)."""


class ClientSession:
    """One client's server -> client channel plus the inbox of its replies."""

    def __init__(
        self,
        name: str,
        sock: socket.socket,
        on_close: Optional[Callable[["ClientSession"], None]] = None,
    ) -> None:
        self.name = name
        self.player: Optional[Player] = None  # set by the server once joined
        self.inbox: "queue.Queue[Any]" = queue.Queue()
        self.pending_turn: Optional[tuple[int, LegalActions]] = None
        self.pending_hand_end: Optional[int] = None
        self.join_reply: Optional[JoinReply] = None
        self.closed = False  # no more messages are accepted
        self.broken = False  # the callback channel failed (not closed by us)
        self._sock = sock
        self._on_close = on_close
        self._out: "queue.Queue[Any]" = queue.Queue()
        self._lock = threading.Lock()
        self._started = False
        self._released = False
        self._finished = threading.Event()
        self._writer = threading.Thread(
            target=self._write_loop, name=f"session-writer-{name}", daemon=True
        )
        self._monitor = threading.Thread(
            target=self._monitor_loop, name=f"session-monitor-{name}", daemon=True
        )

    @classmethod
    def connect(
        cls,
        name: str,
        port: int,
        host: str = "127.0.0.1",
        timeout: float = 5.0,
        on_close: Optional[Callable[["ClientSession"], None]] = None,
    ) -> "ClientSession":
        """Open the callback connection to `host:port` (raises OSError on failure).
        The session is not started yet: register it first, then call `start()`."""
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.settimeout(None)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        return cls(name, sock, on_close)

    def start(self) -> "ClientSession":
        """Start the writer and the disconnect monitor."""
        with self._lock:
            if self._started:
                return self
            self._started = True
            finished = self._finished.is_set()
        if not finished:
            self._writer.start()
            self._monitor.start()
        return self

    # --- the RemoteAgent connection API --------------------------------------

    def send(self, msg: Any) -> None:
        """Forward an engine message to the client without blocking.

        ("joined", JoinReply) is kept in `join_reply` (it becomes the response
        to the `join` request) rather than forwarded. Raises OSError once the
        channel is closed or broken.
        """
        if not isinstance(msg, tuple) or not msg:
            raise ValueError(f"not an engine message: {msg!r}")
        kind = msg[0]
        if kind == "joined":
            self.join_reply = msg[1]
            return
        with self._lock:
            if self.closed:
                raise OSError(f"the callback channel of {self.name} is closed")
            data = encode_line(encode_server_message(msg, self.name))
            if kind == "turn":
                self.pending_turn = (msg[1], msg[3])
            elif kind == "hand_end":
                self.pending_hand_end = msg[1]
            elif kind == "event":
                self._track_event(msg[1])
            self._out.put(data)
            if kind == "close":
                self.closed = True
                self._out.put(_STOP)
        if kind == "close":
            # Unregister now, so the name is free by the time the client reads "close".
            self._release()

    def recv(self) -> Any:
        """Next reply for the RemoteAgent. Raises EOFError once the channel is gone."""
        item = self.inbox.get()
        if item is _EOF:
            self.inbox.put(_EOF)  # keep later calls from blocking
            raise EOFError(f"the session of {self.name} is closed")
        return item

    def close(self) -> None:
        """Flush what is queued, then close the callback connection."""
        with self._lock:
            already = self.closed
            self.closed = True
            if not already:
                self._out.put(_STOP)
            started = self._started
        if not started:
            self._finish(broken=False)

    # --- requests from the client (server port) -------------------------------

    def feed(self, item: Any) -> None:
        """Put a message into the inbox as-is (e.g. the `Join` for `handle_join`)."""
        self.inbox.put(item)

    def submit_action(self, seq: int, action: Union[Action, Leave]) -> None:
        """Answer to the pending `turn` with that `seq`. A rejected action does
        not use up the turn."""
        with self._lock:
            pending = self.pending_turn
            if pending is None:
                raise SessionError(f"{self.name} has no pending turn")
            if pending[0] != seq:
                raise SessionError(f"turn {seq} is not the pending turn of {self.name} ({pending[0]})")
            legal = pending[1]
            if isinstance(action, Action) and not legal.allows(action.type):
                allowed = ", ".join(t.value for t in legal.actions)
                raise SessionError(f"{action.type.value} is not legal now (legal: {allowed}, leave)")
            if not isinstance(action, (Action, Leave)):
                raise SessionError(f"not an action: {action!r}")
            self.pending_turn = None
            self.inbox.put(("action", seq, action))

    def submit_hand_end(self, seq: int, reply: HandEndReply) -> None:
        """Answer to the pending `hand_end` (rebuy / sit-in / leave window)."""
        with self._lock:
            if self.pending_hand_end is None:
                raise SessionError(f"{self.name} has no pending hand_end")
            if self.pending_hand_end != seq:
                raise SessionError(
                    f"hand_end {seq} is not the pending hand_end of {self.name} "
                    f"({self.pending_hand_end})"
                )
            self.pending_hand_end = None
            self.inbox.put(("hand_end", seq, reply))

    def submit_leave(self) -> None:
        """Leave the table at any time (`RemoteAgent` calls `player.leave()`)."""
        self.inbox.put(Leave())

    # --- lifecycle -------------------------------------------------------------

    def wait_closed(self, timeout: Optional[float] = None) -> bool:
        return self._finished.wait(timeout)

    def abort(self) -> None:
        """Close the callback connection at once, dropping anything not yet written."""
        self._finish(broken=False)

    def _track_event(self, event: Any) -> None:
        """Clear pending requests the dealer has moved past (timed out), so a late
        reply is refused instead of accepted and then dropped. Called under the lock."""
        if isinstance(event, HandStarted):
            self.pending_turn = None
            self.pending_hand_end = None
        elif isinstance(event, PlayerActed) and event.name == self.name:
            self.pending_turn = None

    def _write_loop(self) -> None:
        while True:
            item = self._out.get()
            if item is _STOP:
                break
            try:
                self._sock.sendall(item)
            except OSError as e:
                if not self._finished.is_set():
                    _log.info("callback channel of %s failed: %s", self.name, e)
                self._finish(broken=True)
                return
        self._finish(broken=False)

    def _monitor_loop(self) -> None:
        """The client never writes on the callback channel; reading only detects
        that it closed it."""
        while not self._finished.is_set():
            try:
                data = self._sock.recv(4096)
            except OSError:
                data = b""
            if not data:
                break
        if not self._finished.is_set():
            _log.info("callback channel of %s closed by the client", self.name)
        self._finish(broken=True)

    def _finish(self, broken: bool) -> None:
        with self._lock:
            if self._finished.is_set():
                return
            self._finished.set()
            if broken and not self.closed:
                self.broken = True
            self.closed = True
        self._release()
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass
        self._out.put(_STOP)  # wake the writer if it is idle
        self.inbox.put(_EOF)

    def _release(self) -> None:
        """Tell the owner (once) that this session is over."""
        with self._lock:
            if self._released:
                return
            self._released = True
        if self._on_close is not None:
            try:
                self._on_close(self)
            except Exception:  # noqa: BLE001
                _log.exception("session close callback failed")

    def __repr__(self) -> str:
        state = "broken" if self.broken else "closed" if self.closed else "open"
        return f"ClientSession({self.name!r}, {state})"
