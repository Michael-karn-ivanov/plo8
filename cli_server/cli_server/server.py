"""GameServer: request listener, session registry, lobby and game loop.

The server holds no poker rules: seating, the waiting queue, blinds, betting,
showdown, the rebuy window, sitting out, leaving and the action timeout are the
engine's `Environment` / `Dealer`. Remote play goes through the engine's
`RemoteAgent` and `handle_join`, with a `ClientSession` as the connection.
"""

from __future__ import annotations

import collections
import logging
import socket
import socketserver
import threading
import time
from typing import Any, Callable, Optional

from engine import (
    Environment,
    HandEndReply,
    HandResult,
    IllegalActionError,
    Join,
    NotEnoughPlayersError,
    Player,
    RemoteAgent,
    handle_join,
)

from .protocol import (
    LineTooLongError,
    ProtocolError,
    decode_action,
    encode_join_response,
    error_response,
    ok_response,
    read_message,
    validate_request,
    write_message,
)
from .session import ClientSession, SessionError

__all__ = ["GameServer", "HOST", "LOBBY", "PLAYING", "STOPPED", "describe_result"]

_log = logging.getLogger("cli_server")

HOST = "127.0.0.1"  # server and clients share a host
CONNECT_TIMEOUT = 5.0  # seconds to reach a client's callback port
HISTORY_LIMIT = 1000  # hand results / start times kept for inspection

LOBBY = "lobby"  # no hand is running; waiting for players / the grace period
PLAYING = "playing"  # hands follow each other
STOPPED = "stopped"

_RESERVED = object()  # registry placeholder while a join is in progress


class _RequestServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False

    def __init__(self, address: tuple[str, int], game: "GameServer") -> None:
        self.game = game
        super().__init__(address, _RequestHandler)


class _RequestHandler(socketserver.StreamRequestHandler):
    """One client -> server connection: request lines in, one response line each."""

    def setup(self) -> None:
        super().setup()
        self.server.game._track_connection(self.connection, True)  # type: ignore[attr-defined]

    def finish(self) -> None:
        self.server.game._track_connection(self.connection, False)  # type: ignore[attr-defined]
        try:
            super().finish()
        except OSError:
            pass

    def handle(self) -> None:
        game: GameServer = self.server.game  # type: ignore[attr-defined]
        while True:
            try:
                msg = read_message(self.rfile)
            except LineTooLongError as e:
                self._reply(error_response(str(e)))
                return  # cannot find the next line boundary reliably
            except ProtocolError as e:
                if not self._reply(error_response(str(e))):
                    return
                continue
            except OSError:
                return
            if msg is None:
                return
            if not self._reply(game.handle_request(msg)):
                return

    def _reply(self, response: dict) -> bool:
        try:
            write_message(self.wfile, response)
            return True
        except (OSError, ValueError):
            return False


class GameServer:
    """Hosts one table on `127.0.0.1:port`.

    `env_kwargs` are passed to `Environment` (`action_timeout`, `max_seats`,
    `max_buy_in`, `seed`). Use `serve_forever()` to run the game loop in the
    calling thread (the binary does this, so SIGINT aborts a running hand), or
    `start()` to run it in a background thread (tests). `shutdown()` stops it.
    """

    def __init__(
        self,
        port: int = 8008,
        grace_period: float = 15.0,
        env_kwargs: Optional[dict[str, Any]] = None,
        host: str = HOST,
        connect_timeout: float = CONNECT_TIMEOUT,
    ) -> None:
        if grace_period < 0:
            raise ValueError(f"grace_period cannot be negative, got {grace_period!r}")
        self.env = Environment(**(env_kwargs or {}))
        self.grace_period = grace_period
        self.host = host
        self.connect_timeout = connect_timeout
        self.state = LOBBY
        # Inspection (logging, tests): recent hand results and hand start times
        # (time.monotonic(), taken just before each hand starts).
        self.results: collections.deque[HandResult] = collections.deque(maxlen=HISTORY_LIMIT)
        self.hand_start_times: collections.deque[float] = collections.deque(maxlen=HISTORY_LIMIT)

        self._sessions: dict[str, Any] = {}  # name -> ClientSession | _RESERVED
        self._connections: set[socket.socket] = set()
        self._lock = threading.Lock()  # guards the registry and connections
        self._cond = threading.Condition()  # lobby wake-ups; guards the fields below
        self._stopping = False
        self._shut_down = False
        self._last_change = time.monotonic()

        self._tcp = _RequestServer((host, port), self)
        self.port: int = self._tcp.server_address[1]
        self._listener_thread: Optional[threading.Thread] = None
        self._game_thread: Optional[threading.Thread] = None

    # --- running ---------------------------------------------------------------

    def start_listener(self) -> None:
        """Accept client requests in a background thread."""
        if self._listener_thread is None:
            self._listener_thread = threading.Thread(
                target=self._tcp.serve_forever,
                kwargs={"poll_interval": 0.1},
                name="request-listener",
                daemon=True,
            )
            self._listener_thread.start()

    def start(self) -> "GameServer":
        """Start the listener and the game loop in background threads."""
        self.start_listener()
        if self._game_thread is None:
            self._game_thread = threading.Thread(target=self.run, name="game-loop", daemon=True)
            self._game_thread.start()
        return self

    def serve_forever(self) -> None:
        """Start the listener and run the game loop in this thread until `stop()`."""
        self.start_listener()
        self.run()

    def run(self) -> None:
        """The lobby / game loop. Returns after `stop()` / `shutdown()`."""
        try:
            while True:
                with self._cond:
                    self._set_state(LOBBY)
                    if not self._wait_for_start():
                        break
                    self._set_state(PLAYING)
                _log.info("game starts")
                while not self._stopping and self.env.can_start():
                    if not self._play_hand():
                        break
                if not self._stopping:
                    _log.info("waiting for players")
        finally:
            with self._cond:
                self._set_state(STOPPED)

    def _wait_for_start(self) -> bool:
        """LOBBY: wait (holding the condition) until at least two players can play
        and the grace period after the last arrival is over. False when stopping."""
        while not self._stopping:
            if self.env.can_start():
                remaining = self._last_change + self.grace_period - time.monotonic()
                if remaining <= 0:
                    return True
                self._cond.wait(remaining)
            else:
                self._cond.wait()
        return False

    def _set_state(self, state: str) -> None:
        self.state = state
        self._cond.notify_all()

    def _play_hand(self) -> bool:
        players = self.env.active_players
        _log.info(
            "hand %d starts: %s",
            self.env.dealer.hand_number + 1,
            ", ".join(f"{p.name} ({p.stack:g} BB)" for p in players),
        )
        self.hand_start_times.append(time.monotonic())
        try:
            result = self.env.start_round()
        except NotEnoughPlayersError:
            self.hand_start_times.pop()
            return False  # players left between the check and the start
        self.results.append(result)
        _log.info("%s", describe_result(result))
        return True

    def stop(self) -> None:
        """Ask the game loop to return (after the running hand, if any)."""
        with self._cond:
            self._stopping = True
            self._cond.notify_all()

    def shutdown(self, timeout: float = 10.0) -> None:
        """Stop accepting requests, send `close` to every session and stop the loop."""
        with self._cond:
            if self._shut_down:
                return
            self._shut_down = True
            self._stopping = True
            self._cond.notify_all()
        if self._listener_thread is not None:
            self._tcp.shutdown()
            self._listener_thread.join(timeout)
        self._tcp.server_close()
        with self._lock:
            connections = list(self._connections)
            sessions = [s for s in self._sessions.values() if isinstance(s, ClientSession)]
        for conn in connections:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        for session in sessions:
            agent = getattr(session.player, "agent", None)
            if isinstance(agent, RemoteAgent):
                agent.close()  # sends "close"; not taken for a disconnect
            session.close()
        for session in sessions:
            if not session.wait_closed(1.0):
                session.abort()
        game = self._game_thread
        if game is not None and game is not threading.current_thread():
            game.join(timeout)
        _log.info("server stopped")

    # --- registry ---------------------------------------------------------------

    def session_names(self) -> list[str]:
        with self._lock:
            return [n for n, s in self._sessions.items() if isinstance(s, ClientSession)]

    def _session(self, name: str) -> ClientSession:
        with self._lock:
            session = self._sessions.get(name)
        if session is None:
            raise SessionError(f"unknown player {name!r}")
        if session is _RESERVED:
            raise SessionError(f"{name} is still joining")
        return session

    def _drop(self, name: str, expected: Any) -> None:
        with self._lock:
            if self._sessions.get(name) is expected:
                del self._sessions[name]

    def _unregister(self, session: ClientSession) -> None:
        """Session callback: its channel is over; the name can be used again."""
        self._drop(session.name, session)
        if session.broken:
            _log.info("%s disconnected", session.name)
        else:
            _log.info("session of %s closed", session.name)
        with self._cond:
            self._cond.notify_all()

    def _track_connection(self, conn: socket.socket, add: bool) -> None:
        with self._lock:
            if add:
                self._connections.add(conn)
            else:
                self._connections.discard(conn)

    # --- requests ---------------------------------------------------------------

    def handle_request(self, msg: dict) -> dict:
        """Process one decoded request and return its response."""
        try:
            req = validate_request(msg)
            kind = req["type"]
            if kind == "join":
                return self._join(req)
            session = self._session(req["name"])
            if kind == "action":
                session.submit_action(req["seq"], decode_action(req["action"]))
            elif kind == "hand_end":
                session.submit_hand_end(
                    req["seq"], HandEndReply(req["rebuy_bb"], req["sit_in"], req["leave"])
                )
            elif kind == "rebuy":
                amount = req["amount_bb"]
                return self._seat_change(session, "rebuy", lambda p: p.rebuy(amount))
            elif kind == "sit_in":
                return self._seat_change(session, "sit_in", lambda p: p.sit_in())
            elif kind == "leave":
                _log.info("%s asks to leave", session.name)
                session.submit_leave()
                with self._cond:
                    self._cond.notify_all()
            return ok_response()
        except (ProtocolError, SessionError) as e:
            return error_response(str(e))
        except Exception:  # noqa: BLE001 - never let one request kill the connection
            _log.exception("request %r failed", msg)
            return error_response("internal server error")

    def _seat_change(
        self, session: ClientSession, what: str, fn: Callable[[Player], None]
    ) -> dict:
        """rebuy / sit_in from the handler thread: the engine guards the bookkeeping
        with its lock and rejects it during betting."""
        player = session.player
        if player is None:
            raise SessionError(f"{session.name} has not joined yet")
        try:
            fn(player)
        except (IllegalActionError, ValueError, TypeError) as e:
            return error_response(str(e))
        _log.info("%s: %s ok (stack %g BB)", session.name, what, player.stack)
        with self._cond:
            if self.state == LOBBY:
                # A player came back: the grace period restarts.
                self._last_change = time.monotonic()
            self._cond.notify_all()
        return ok_response()

    def _join(self, req: dict) -> dict:
        name = req["name"]
        port = req["callback_port"]
        with self._lock:
            if self._stopping:
                return error_response("the server is shutting down")
            if name in self._sessions:
                return error_response(f"name {name!r} is already taken at this table")
            self._sessions[name] = _RESERVED  # concurrent joins with the same name
        session: Optional[ClientSession] = None
        try:
            try:
                session = ClientSession.connect(
                    name, port, host=self.host, timeout=self.connect_timeout, on_close=self._unregister
                )
            except OSError as e:
                self._drop(name, _RESERVED)
                return error_response(f"cannot reach callback port {port}: {e}")
            # Register before joining: a turn could reach the client, and its
            # action arrive, before handle_join returns.
            with self._lock:
                self._sessions[name] = session
            session.start()
            session.feed(Join(name, req["stack_bb"], req["seat"]))
            player = handle_join(self.env, session, timeout=self.env.action_timeout)
        except BaseException:
            self._drop(name, _RESERVED)
            if session is not None:
                session.close()
                self._drop(name, session)
            raise
        reply = session.join_reply
        if player is None or reply is None or not reply.accepted:
            session.close()  # handle_join closed it already; idempotent
            self._drop(name, session)
            error = reply.error if reply is not None and reply.error else "join rejected"
            _log.info("join of %s rejected: %s", name, error)
            return error_response(error)
        session.player = player
        with self._cond:
            # Every new arrival restarts the grace period.
            self._last_change = time.monotonic()
            self._cond.notify_all()
        _log.info(
            "%s joined with %g BB (%s)",
            name,
            req["stack_bb"],
            "queued until the hand is over" if reply.queued else f"seat {reply.seat}",
        )
        return encode_join_response(reply)


def describe_result(result: HandResult) -> str:
    """One log line: button, board, winners per pot, stacks."""
    names = result.names
    awards = []
    for i, pot in enumerate(result.pots):
        for half, shares in (("high", pot.high), ("low", pot.low)):
            for seat, amount in shares:
                share = "whole" if pot.uncontested else half
                awards.append(f"{names.get(seat, seat)} wins {amount:g} BB (pot {i}, {share})")
    stacks = ", ".join(f"{names.get(s, s)} {st:g}" for s, st in sorted(result.stacks.items()))
    return (
        f"hand {result.hand_number} over (button {names.get(result.button, result.button)}): "
        f"board [{result.board}], {'showdown' if result.showdown else 'no showdown'}; "
        f"{'; '.join(awards) or 'no awards'}; stacks: {stacks}"
    )
