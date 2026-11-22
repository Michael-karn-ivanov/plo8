"""Networking library (`//cli_client:net`): requests to the server port and the
server's callback channel.

- `ServerConnection` (client -> server): one persistent connection to the
  server's port, opened on the first request; each request line gets exactly
  one response line.
- `CallbackListener` (server -> client): listens on the client's own port. The
  server opens one connection per joined player and sends notifications on it.
  Every message goes, as a raw dict, on the caller's inbox (any `queue.Queue`);
  the end of the channel puts `DISCONNECTED` there.

This module is also used by other projects (`//bot_client`). It imports only the
standard library and `cli_server.protocol` — never `curses` or any other
cli_client module — writes nothing to stdout / stderr, and logs to the
`cli_client.net` logger without installing handlers. Building requests, sending
them in order, checking `ok` and decoding messages are left to the caller.
"""

from __future__ import annotations

import logging
import queue
import socket
import threading
from typing import Any, Optional

from cli_server.protocol import LineTooLongError, ProtocolError, read_message, write_message

__all__ = ["HOST", "DISCONNECTED", "ConnectionLost", "ServerConnection", "CallbackListener"]

_log = logging.getLogger("cli_client.net")

HOST = "127.0.0.1"
DISCONNECTED = ("disconnected",)  # inbox marker: the callback channel is gone
REQUEST_TIMEOUT = 10.0


class ConnectionLost(Exception):
    """The server port cannot be reached (socket error, timeout or an over-long
    line, even after one reconnect), or the connection was closed."""


def _close(sock: Optional[socket.socket]) -> None:
    if sock is None:
        return
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    try:
        sock.close()
    except OSError:
        pass


class ServerConnection:
    """Client -> server requests over one persistent TCP connection.

    `request(msg)` returns the response (also `ok: false` ones; a malformed
    response line comes back as `ok: false` with "bad response: ..."). A socket
    error, timeout or over-long line drops the connection, reconnects and
    resends once; if that fails too, `ConnectionLost` is raised. The server
    tells clients apart by `name`, so a new connection is fine.

    `request` is safe from any thread (a lock serialises calls). `close()` may
    be called more than once; a `request` after it raises `ConnectionLost`.
    """

    def __init__(self, port: int, host: str = HOST, timeout: float = REQUEST_TIMEOUT) -> None:
        self.port = port
        self.host = host
        self.timeout = timeout
        self._lock = threading.Lock()
        self._sock: Optional[socket.socket] = None
        self._rfile: Any = None
        self._wfile: Any = None
        self._closed = False

    def _open(self) -> None:
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        sock.settimeout(self.timeout)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        self._sock = sock
        self._rfile = sock.makefile("rb")
        self._wfile = sock.makefile("wb")

    def _drop(self) -> None:
        for f in (self._rfile, self._wfile):
            try:
                if f is not None:
                    f.close()
            except (OSError, ValueError):
                pass
        _close(self._sock)
        self._sock = self._rfile = self._wfile = None

    def request(self, msg: dict) -> dict:
        with self._lock:
            if self._closed:
                raise ConnectionLost("the connection is closed")
            last: Optional[BaseException] = None
            for _attempt in range(2):
                try:
                    if self._sock is None:
                        self._open()
                    write_message(self._wfile, msg)
                    response = read_message(self._rfile)
                    if response is None:
                        raise ConnectionError("the server closed the connection")
                    _log.debug("request %s -> %s", msg, response)
                    return response
                except LineTooLongError as e:
                    last = e
                    self._drop()
                except ProtocolError as e:
                    # One bad response line; the stream is still in sync.
                    _log.warning("bad response to %s: %s", msg, e)
                    return {"type": "response", "ok": False, "error": f"bad response: {e}"}
                except (OSError, ValueError) as e:
                    last = e
                    self._drop()
            raise ConnectionLost(str(last))

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._drop()


class CallbackListener:
    """Listens on `host:port` (0 = a free port; see `self.port`) for the
    server's callback connection. Binding happens in the constructor, so bind
    errors are raised there; `start()` launches the accept thread.

    A new connection replaces the old one (which ends without `DISCONNECTED`).
    Messages addressed to another player (`to`) are dropped and counted in
    `dropped`. A malformed line is logged, counted in `malformed` and skipped;
    the reader keeps going. Unknown message types are passed through. Nothing
    is put on the inbox after `close()`. All threads are daemon threads.
    """

    def __init__(
        self, name: str, inbox: "queue.Queue[Any]", host: str = HOST, port: int = 0
    ) -> None:
        self.name = name
        self.inbox = inbox
        self.dropped = 0
        self.malformed = 0
        self._lock = threading.Lock()
        self._conn: Optional[socket.socket] = None
        self._generation = 0
        self._closed = False
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind((host, port))
            self._sock.listen(4)
        except OSError:
            self._sock.close()
            raise
        self._sock.settimeout(0.2)  # so close() can stop the accept loop
        self.port: int = self._sock.getsockname()[1]
        self._thread = threading.Thread(target=self._accept_loop, name="callback-accept", daemon=True)

    def start(self) -> "CallbackListener":
        self._thread.start()
        return self

    def _accept_loop(self) -> None:
        while True:
            with self._lock:
                if self._closed:
                    return
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            conn.settimeout(None)
            with self._lock:
                if self._closed:
                    _close(conn)
                    return
                old, self._conn = self._conn, conn
                self._generation += 1
                generation = self._generation
            _close(old)
            threading.Thread(
                target=self._read_loop, args=(conn, generation), name="callback-reader", daemon=True
            ).start()

    def _read_loop(self, conn: socket.socket, generation: int) -> None:
        f = conn.makefile("rb")
        try:
            while True:
                try:
                    msg = read_message(f)
                except LineTooLongError as e:
                    _log.warning("callback line too long: %s", e)
                    break
                except ProtocolError as e:
                    self.malformed += 1
                    _log.warning("malformed callback message skipped: %s", e)
                    continue
                except (OSError, ValueError):
                    break
                if msg is None:
                    break
                to = msg.get("to")
                if to is not None and to != self.name:
                    self.dropped += 1
                    continue
                with self._lock:
                    closed = self._closed
                if closed:
                    break  # nothing goes on the inbox after close()
                _log.debug("received %s", msg)
                self.inbox.put(msg)
        finally:
            try:
                f.close()
            except (OSError, ValueError):
                pass
            with self._lock:
                current = generation == self._generation and not self._closed
            if current:
                self.inbox.put(DISCONNECTED)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            conn, self._conn = self._conn, None
        _close(conn)
        try:
            self._sock.close()
        except OSError:
            pass
