"""Test helper: a scripted cli_server client with its own callback port.

A `FakeClient` listens on a free localhost port for the server's callback
connection and records every message it receives. Requests go to the server
port over one persistent connection. With `auto=True` it answers `turn` with
`policy(msg)` (default: check, else call) and `hand_end` with
`hand_end_reply(msg)` (default: no rebuy, no sit-in, no leave).
"""

from __future__ import annotations

import json
import queue
import socket
import threading
import time
from typing import Any, Callable, Optional

from cli_server.protocol import (
    action_request,
    hand_end_request,
    join_request,
    leave_request,
    rebuy_request,
    sit_in_request,
)

HOST = "127.0.0.1"


def passive_policy(msg: dict) -> str:
    """Check if possible, else call."""
    return "check" if msg["legal"]["can_check"] else "call"


def wait_until(
    predicate: Callable[[], Any],
    timeout: float = 10.0,
    interval: float = 0.01,
    message: str = "condition",
) -> Any:
    """Poll `predicate` until it returns something truthy; return that value."""
    deadline = time.monotonic() + timeout
    while True:
        value = predicate()
        if value:
            return value
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out waiting for {message}")
        time.sleep(interval)


def unused_port() -> int:
    """A localhost port nobody listens on (bound, then released)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, 0))
    port = s.getsockname()[1]
    s.close()
    return port


def is_type(kind: str) -> Callable[[dict], bool]:
    return lambda m: m.get("type") == kind


def is_event(kind: str, **fields: Any) -> Callable[[dict], bool]:
    def pred(m: dict) -> bool:
        if m.get("type") != "event" or m["event"].get("type") != kind:
            return False
        return all(m["event"].get(k) == v for k, v in fields.items())

    return pred


class FakeClient:
    def __init__(
        self,
        server_port: int,
        name: str,
        auto: bool = False,
        policy: Callable[[dict], str] = passive_policy,
        hand_end_reply: Optional[Callable[[dict], dict]] = None,
        timeout: float = 10.0,
    ) -> None:
        self.name = name
        self.server_port = server_port
        self.policy = policy
        self.hand_end_reply = hand_end_reply or (lambda msg: {})
        self.messages: list[dict] = []  # everything received on the callback port
        self.errors: list[dict] = []  # failed automatic answers
        self.channel_eofs = 0  # callback connections closed by the server
        self._auto = auto
        self._unanswered: dict[int, dict] = {}  # seq -> turn / hand_end message
        self._cond = threading.Condition()
        self._todo: "queue.Queue[Optional[dict]]" = queue.Queue()
        self._closed = False
        self._conns: list[socket.socket] = []

        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.bind((HOST, 0))
        self._listener.listen(8)
        self._listener.settimeout(0.1)  # so close() can stop the accept loop
        self.callback_port: int = self._listener.getsockname()[1]

        self._req = socket.create_connection((HOST, server_port), timeout=timeout)
        self._rfile = self._req.makefile("rb")
        self._req_lock = threading.Lock()

        threading.Thread(target=self._accept_loop, daemon=True).start()
        threading.Thread(target=self._answer_loop, daemon=True).start()

    # --- requests ---------------------------------------------------------------

    def request(self, msg: dict) -> dict:
        return self.raw(json.dumps(msg).encode("utf-8") + b"\n")

    def raw(self, data: bytes) -> dict:
        """Send raw bytes (one request line) and read one response line."""
        with self._req_lock:
            try:
                self._req.sendall(data)
                line = self._rfile.readline()
            except OSError as e:
                return {"type": "response", "ok": False, "error": f"connection error: {e}"}
        if not line:
            return {"type": "response", "ok": False, "error": "connection closed"}
        return json.loads(line)

    def join(
        self,
        stack_bb: Any = 40.0,
        seat: Optional[int] = None,
        callback_port: Optional[int] = None,
    ) -> dict:
        port = self.callback_port if callback_port is None else callback_port
        return self.request(join_request(self.name, stack_bb, port, seat))

    def act(self, seq: int, action: str) -> dict:
        response = self.request(action_request(self.name, seq, action))
        if response.get("ok"):
            with self._cond:
                self._unanswered.pop(seq, None)
        return response

    def reply_hand_end(
        self,
        seq: int,
        rebuy_bb: Optional[float] = None,
        sit_in: bool = False,
        leave: bool = False,
    ) -> dict:
        response = self.request(hand_end_request(self.name, seq, rebuy_bb, sit_in, leave))
        if response.get("ok"):
            with self._cond:
                self._unanswered.pop(seq, None)
        return response

    def rebuy(self, amount_bb: float) -> dict:
        return self.request(rebuy_request(self.name, amount_bb))

    def sit_in(self) -> dict:
        return self.request(sit_in_request(self.name))

    def leave(self) -> dict:
        return self.request(leave_request(self.name))

    # --- automatic play ----------------------------------------------------------

    def set_auto(self, auto: bool, answer_pending: bool = True) -> None:
        """Turn automatic answers on or off. When turning on, also answer the
        turn / hand_end already waiting (unless `answer_pending` is False, in
        which case they are forgotten)."""
        with self._cond:
            self._auto = auto
            if auto:
                if answer_pending:
                    for seq in sorted(self._unanswered):
                        self._todo.put(self._unanswered[seq])
                else:
                    self._unanswered.clear()

    def _answer_loop(self) -> None:
        while True:
            msg = self._todo.get()
            if msg is None:
                return
            with self._cond:
                if msg["seq"] not in self._unanswered:
                    continue  # answered already
            if msg["type"] == "turn":
                response = self.act(msg["seq"], self.policy(msg))
            else:
                response = self.reply_hand_end(msg["seq"], **self.hand_end_reply(msg))
            if not response.get("ok"):
                self.errors.append(response)

    # --- callback channel ---------------------------------------------------------

    def _accept_loop(self) -> None:
        while not self._closed:
            try:
                conn, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            conn.settimeout(None)
            with self._cond:
                if self._closed:
                    conn.close()
                    return
                self._conns.append(conn)
            threading.Thread(target=self._read_loop, args=(conn,), daemon=True).start()

    def _read_loop(self, conn: socket.socket) -> None:
        f = conn.makefile("rb")
        while True:
            try:
                line = f.readline()
            except (OSError, ValueError):
                line = b""
            if not line:
                break
            try:
                msg = json.loads(line)
            except ValueError:
                msg = {"type": "_invalid", "raw": line.decode("utf-8", "replace")}
            with self._cond:
                self.messages.append(msg)
                if msg.get("type") in ("turn", "hand_end"):
                    self._unanswered[msg["seq"]] = msg
                    if self._auto:
                        self._todo.put(msg)
                self._cond.notify_all()
        with self._cond:
            self.channel_eofs += 1
            self._cond.notify_all()

    def close_callback(self) -> None:
        """Close the callback port and connections (the server should notice)."""
        with self._cond:
            conns = list(self._conns)
            self._conns.clear()
        for conn in conns:
            _close(conn)
        try:
            self._listener.close()
        except OSError:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._todo.put(None)
        self.close_callback()
        _close(self._req)

    # --- inspection ----------------------------------------------------------------

    def snapshot(self) -> list[dict]:
        with self._cond:
            return list(self.messages)

    def of_type(self, kind: str) -> list[dict]:
        return [m for m in self.snapshot() if m.get("type") == kind]

    def wait_for(
        self, pred: Callable[[dict], bool], timeout: float = 10.0, start: int = 0
    ) -> dict:
        """First message (from index `start`) satisfying `pred`, waiting for it."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                for m in self.messages[start:]:
                    if pred(m):
                        return m
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    kinds = [
                        m["event"]["type"] if m.get("type") == "event" else m.get("type")
                        for m in self.messages
                    ]
                    raise AssertionError(f"{self.name}: timed out; received {kinds}")
                self._cond.wait(remaining)

    def wait_type(self, kind: str, timeout: float = 10.0) -> dict:
        return self.wait_for(is_type(kind), timeout)

    def wait_eof(self, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        with self._cond:
            while self.channel_eofs == 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise AssertionError(f"{self.name}: callback channel still open")
                self._cond.wait(remaining)


def _close(sock: socket.socket) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass
