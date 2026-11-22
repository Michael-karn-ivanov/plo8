"""Test helpers: an in-process server, a scripted check/call client, bot
clients on threads, and polling."""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable, Optional

from cli_client.net import DISCONNECTED, CallbackListener, ConnectionLost, ServerConnection
from cli_server.protocol import (
    action_request,
    decode_server_message,
    hand_end_request,
    join_request,
    leave_request,
)
from engine import HandStarted

from bot_client.main import main

GRACE = 0.05
SEED = 11


def server(**env: Any):
    """A started in-process GameServer (call `shutdown()` when done)."""
    from cli_server.server import GameServer

    kwargs: dict[str, Any] = {"seed": SEED, "action_timeout": 5.0}
    kwargs.update(env)
    return GameServer(port=0, grace_period=GRACE, hand_pause=0.0, env_kwargs=kwargs).start()


def wait_for(cond: Callable[[], Any], timeout: float = 30.0, what: str = "condition") -> Any:
    deadline = time.monotonic() + timeout
    while True:
        value = cond()
        if value:
            return value
        if time.monotonic() > deadline:
            raise AssertionError(f"timed out after {timeout:g}s waiting for {what}")
        time.sleep(0.01)


def run_bot(argv: list, leave_timeout: float = 5.0):
    """`main(argv)` on a thread. Returns (thread, box, client): `box["code"]` is
    the exit status once the thread ends; `client` is None if the startup failed."""
    box: dict[str, Any] = {}
    started = threading.Event()

    def on_start(client):
        box["client"] = client
        started.set()

    def target():
        try:
            box["code"] = main(argv, on_start=on_start, leave_timeout=leave_timeout)
        except SystemExit as e:  # argparse
            box["code"] = e.code
        except BaseException as e:  # noqa: BLE001
            box["error"] = e
            box["code"] = -1
        finally:
            started.set()

    thread = threading.Thread(target=target, name=f"bot-{argv}", daemon=True)
    thread.start()
    started.wait(30.0)
    return thread, box, box.get("client")


def bot_argv(port: int, name: str, *extra: str) -> list:
    return ["--server-port", str(port), "--port", "0", "--name", name, "--log-level", "warning", *extra]


def participants(result) -> dict:
    """name -> starting stack of the hand (its HandStarted event)."""
    for e in result.history:
        if isinstance(e, HandStarted):
            return {name: stack for _seat, name, stack in e.participants}
    return {}


def seat_of(result, name: str) -> Optional[int]:
    for s, n in result.names.items():
        if n == name:
            return s
    return None


class ScriptedClient:
    """A minimal check/call client (the spec's "scripted client"), built on the
    networking library only: answers every `turn` with check or call and every
    `hand_end` with a rebuy back to `rebuy_to` (and a sit-in when sat out).

    With `auto=False` turns are kept in `pending_turn` until `release()`."""

    def __init__(self, port: int, name: str, stack: float = 100.0, rebuy_to: Optional[float] = None, auto: bool = True):
        self.name = name
        self.stack = stack
        self.rebuy_to = stack if rebuy_to is None else rebuy_to
        self.auto = auto
        self.inbox: "queue.Queue[Any]" = queue.Queue()
        self.listener = CallbackListener(name, self.inbox, port=0).start()
        self.conn = ServerConnection(port)
        self.lock = threading.Lock()
        self.pending_turn = None  # (seq, legal)
        self.turns = 0
        self.hand_ends = 0
        self.sat_out = False
        self.closed = False
        self.left = False
        self.errors: list = []
        self.response = self.conn.request(join_request(name, stack, self.listener.port, None))
        self._stop = False
        self._thread = threading.Thread(target=self._loop, name=f"scripted-{name}", daemon=True)
        if self.response.get("ok"):
            self._thread.start()

    @property
    def ok(self) -> bool:
        return bool(self.response.get("ok"))

    def _answer(self, seq, legal) -> dict:
        action = "check" if legal.can_check else "call"
        self.turns += 1
        return self.conn.request(action_request(self.name, seq, action))

    def release(self) -> None:
        """Answer turns automatically from now on (also the one kept)."""
        with self.lock:
            self.auto = True
            pending, self.pending_turn = self.pending_turn, None
        if pending is not None:
            self._answer(*pending)

    def _loop(self) -> None:
        while not self._stop:
            try:
                msg = self.inbox.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if msg == DISCONNECTED:
                    self.closed = True
                    return
                kind = msg.get("type")
                if kind == "turn":
                    _, seq, _view, legal = decode_server_message(msg)
                    with self.lock:
                        if not self.auto:
                            self.pending_turn = (seq, legal)
                            continue
                    response = self._answer(seq, legal)
                    if not response.get("ok"):
                        self.errors.append(response.get("error"))
                elif kind == "hand_end":
                    _, seq, seat, result = decode_server_message(msg)
                    stack = result.stacks.get(seat, 0.0)
                    missing = round(self.rebuy_to - stack, 2)
                    rebuy = missing if missing > 0 else None
                    self.hand_ends += 1
                    response = self.conn.request(
                        hand_end_request(self.name, seq, rebuy, self.sat_out, False)
                    )
                    if not response.get("ok"):
                        self.errors.append(response.get("error"))
                elif kind == "sat_out":
                    self.sat_out = True
                elif kind == "returned":
                    self.sat_out = False
                elif kind == "left":
                    self.left = True
                elif kind == "close":
                    self.closed = True
                    return
            except ConnectionLost:
                self.closed = True
                return

    def leave(self) -> None:
        try:
            self.conn.request(leave_request(self.name))
        except ConnectionLost:
            pass

    def close(self) -> None:
        self._stop = True
        self.listener.close()
        self.conn.close()
