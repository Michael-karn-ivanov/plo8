"""GameServer: request listener, session registry, lobby and game loop.

The server holds no poker rules: seating, the waiting queue, blinds, betting,
showdown, the rebuy window, sitting out, leaving and the action timeout are the
engine's `Environment` / `Dealer`. Remote play goes through the engine's
`RemoteAgent` and `handle_join`, with a `ClientSession` as the connection.

On top of the engine the server sends the table settings with every accepted
join, announces each turn to the other seated players (`acting`), keeps clients
informed about the lobby (`lobby`) and pauses between hands.

Concurrency: the main thread never holds the lobby condition while calling into
the engine, and session hooks (which may run while the engine's table lock is
held) never call into the engine; they only queue messages and poke the lobby.
So the table lock and the lobby condition are never taken in opposite orders.
"""

from __future__ import annotations

import collections
import logging
import socket
import socketserver
import threading
import time
from typing import Any, Callable, NamedTuple, Optional

from engine import (
    MAX_PLAYERS,
    MIN_PLAYERS,
    Environment,
    HandEndReply,
    HandResult,
    IllegalActionError,
    Join,
    NotEnoughPlayersError,
    Player,
    PlayerStatus,
    RemoteAgent,
    TableView,
    handle_join,
)

from .protocol import (
    Acting,
    LineTooLongError,
    LobbyStatus,
    ProtocolError,
    TableSettings,
    decode_action,
    encode_acting,
    encode_join_response,
    encode_lobby,
    error_response,
    ok_response,
    read_message,
    validate_request,
    write_message,
)
from .session import ClientSession, SessionError

__all__ = [
    "GameServer",
    "HOST",
    "LOBBY",
    "PLAYING",
    "STOPPED",
    "DEFAULT_GRACE_PERIOD",
    "DEFAULT_HAND_PAUSE",
    "describe_result",
]

_log = logging.getLogger("cli_server")

HOST = "127.0.0.1"  # server and clients share a host
CONNECT_TIMEOUT = 5.0  # seconds to reach a client's callback port
HISTORY_LIMIT = 1000  # hand results / start times kept for inspection
DEFAULT_GRACE_PERIOD = 15.0
DEFAULT_HAND_PAUSE = 3.0
REQUEST_DRAIN_TIMEOUT = 1.0  # shutdown: seconds to let requests in flight be answered

LOBBY = "lobby"  # no hand is running; waiting for players / the grace period
PLAYING = "playing"  # hands follow each other (including the pauses between them)
STOPPED = "stopped"

_RESERVED = object()  # registry placeholder while a join is in progress
_SEATED = (PlayerStatus.ACTIVE, PlayerStatus.SITTING_OUT)


class _Lobby(NamedTuple):
    """Lobby status as the main thread sees it: `deadline` is a monotonic time."""

    state: str  # "waiting" / "starting" / "next_hand"
    active: int
    needed: int
    deadline: Optional[float]


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
            game._request_begins()
            try:
                answered = self._reply(game.handle_request(msg))
            finally:
                game._request_ends()
            if not answered:
                return

    def _reply(self, response: dict) -> bool:
        try:
            write_message(self.wfile, response)
            return True
        except (OSError, ValueError):
            return False


class GameServer:
    """Hosts one table on `127.0.0.1:port`.

    `env_kwargs` are passed to `Environment` as they are (`action_timeout`,
    `max_seats`, `max_buy_in`, `seed`, and e.g. `chips_per_bb` / `deck` for the
    e2e driver). `grace_period` is the wait after the last arrival before the
    first hand; `hand_pause` the wait between hands (0 = none).
    `required_players` (>= MIN_PLAYERS) active players are needed before the
    grace period starts. With `max_hands` set, the game loop returns after that
    many hands (the sessions stay open until `shutdown()`).

    Use `serve_forever()` to run the game loop in the calling thread (the binary
    does this, so SIGINT aborts a running hand), or `start()` to run it in a
    background thread (tests). `shutdown()` stops it.
    """

    def __init__(
        self,
        port: int = 8008,
        grace_period: float = DEFAULT_GRACE_PERIOD,
        hand_pause: float = DEFAULT_HAND_PAUSE,
        env_kwargs: Optional[dict[str, Any]] = None,
        host: str = HOST,
        connect_timeout: float = CONNECT_TIMEOUT,
        required_players: int = MIN_PLAYERS,
        max_hands: Optional[int] = None,
    ) -> None:
        if grace_period < 0:
            raise ValueError(f"grace_period cannot be negative, got {grace_period!r}")
        if hand_pause < 0:
            raise ValueError(f"hand_pause cannot be negative, got {hand_pause!r}")
        if (
            isinstance(required_players, bool)
            or not isinstance(required_players, int)
            or not MIN_PLAYERS <= required_players <= MAX_PLAYERS
        ):
            raise ValueError(
                f"required_players must be {MIN_PLAYERS}-{MAX_PLAYERS}, got {required_players!r}"
            )
        if max_hands is not None and (
            isinstance(max_hands, bool) or not isinstance(max_hands, int) or max_hands < 1
        ):
            raise ValueError(f"max_hands must be a positive int or None, got {max_hands!r}")
        self.env = Environment(**(env_kwargs or {}))
        self.grace_period = grace_period
        self.hand_pause = hand_pause
        self.required_players = required_players
        self.max_hands = max_hands
        self.hands_played = 0
        self.host = host
        self.connect_timeout = connect_timeout
        self.settings = self._table_settings()  # fixed: sent with every accepted join
        self.state = LOBBY
        # Inspection (logging, tests): recent hand results, hand start times
        # (time.monotonic(), taken just before each hand starts) and the lobby
        # statuses published to clients.
        self.results: collections.deque[HandResult] = collections.deque(maxlen=HISTORY_LIMIT)
        self.hand_start_times: collections.deque[float] = collections.deque(maxlen=HISTORY_LIMIT)
        self.lobby_history: collections.deque[LobbyStatus] = collections.deque(maxlen=HISTORY_LIMIT)

        # Registry, guarded by registry_lock (never held while sending or calling
        # into the engine).
        self.registry_lock = threading.Lock()
        self._sessions: dict[str, Any] = {}  # name -> ClientSession | _RESERVED
        self._connections: set[socket.socket] = set()
        self._last_lobby: Optional[tuple] = None  # last published (state, active, needed, deadline)
        self._lobby_fresh: set[ClientSession] = set()  # get the lobby status even if unchanged
        self._joining = 0  # joins being processed; no hand starts meanwhile

        # Requests being processed (shutdown lets them be answered first).
        self._requests = threading.Condition()
        self._in_flight = 0

        # Lobby condition: guards the fields below; never held while calling the engine.
        self._cond = threading.Condition()
        self._generation = 0  # incremented by every poke
        self._stopping = False
        self._shut_down = False
        self._last_arrival = time.monotonic()

        self._tcp = _RequestServer((host, port), self)
        self.port: int = self._tcp.server_address[1]
        self._listener_thread: Optional[threading.Thread] = None
        self._game_thread: Optional[threading.Thread] = None

    def _table_settings(self) -> TableSettings:
        dealer = self.env.dealer
        big_blind = dealer.to_bb(dealer.big_blind)
        return TableSettings(
            small_blind_bb=dealer.to_bb(dealer.small_blind),
            big_blind_bb=big_blind,
            small_bet_bb=big_blind,  # preflop and flop: 1 BB
            big_bet_bb=2 * big_blind,  # turn and river: 2 BB
            max_seats=self.env.max_seats,
            max_buy_in_bb=self.env.max_buy_in,
            action_timeout=self.env.action_timeout,
            grace_period=self.grace_period,
            hand_pause=self.hand_pause,
        )

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
        """Start the listener and run the game loop in this thread until `stop()`
        (or until `max_hands` hands were played). May run on any thread."""
        self.start_listener()
        self.run()

    def run(self) -> None:
        """The lobby / game loop. Returns after `stop()` / `shutdown()`, or once
        `max_hands` hands were played."""
        try:
            while True:
                self._set_state(LOBBY)
                if not self._wait_for_start():
                    break
                self._set_state(PLAYING)
                _log.info("game starts")
                while not self._stopping and self.env.can_start():
                    if not self._play_hand():
                        break
                    self.hands_played += 1
                    if self.max_hands is not None and self.hands_played >= self.max_hands:
                        _log.info("%d hand(s) played, the game loop ends", self.hands_played)
                        return
                    if self.hand_pause > 0 and not self._stopping and self.env.can_start():
                        self._pause_between_hands()
                if self._stopping:
                    break
                _log.info("waiting for players")
        finally:
            self._set_state(STOPPED)

    def _wait_for_start(self) -> bool:
        """LOBBY: wait until `required_players` players can play and the grace
        period after the last arrival is over, publishing the lobby status as it
        changes. A join still being processed (its client not answered yet)
        holds the start back. False when stopping."""
        while True:
            with self._cond:
                if self._stopping:
                    return False
                gen = self._generation  # read before env: a later poke is not lost
            status = self._lobby_status()
            self._publish_lobby(status)
            with self.registry_lock:
                joining = self._joining > 0
            now = time.monotonic()
            if (
                status.state == "starting"
                and status.deadline is not None
                and status.deadline <= now
                and not joining
                and self.env.can_start()
            ):
                return True
            # A finishing join pokes, so no timeout is needed while one runs.
            timeout = None if status.deadline is None or joining else status.deadline - now
            self._wait_for_poke(gen, timeout)

    def _pause_between_hands(self) -> None:
        """Wait `hand_pause` seconds so the showdown stays on the clients' screens.
        Joins are seated at once meanwhile. Ends early when stopping or when fewer
        than two players are left (the loop then goes back to LOBBY)."""
        until = time.monotonic() + self.hand_pause
        while True:
            with self._cond:
                if self._stopping:
                    return
                gen = self._generation
            if not self.env.can_start():
                return
            self._publish_lobby(self._lobby_status(pause_until=until))
            remaining = until - time.monotonic()
            if remaining <= 0:
                return
            self._wait_for_poke(gen, remaining)

    def _set_state(self, state: str) -> None:
        with self._cond:
            self.state = state
            if state == LOBBY:
                # The grace period never counts from before play stopped.
                self._last_arrival = time.monotonic()
            self._generation += 1
            self._cond.notify_all()

    def _play_hand(self) -> bool:
        players = self.env.active_players
        _log.info(
            "hand %d starts: %s",
            self.env.dealer.hand_number + 1,
            ", ".join(f"{p.name} ({p.stack:g} BB)" for p in players),
        )
        with self.registry_lock:
            # Clients drop the lobby display on hand_started: after the hand the
            # status is sent again, even if it equals the last one.
            self._last_lobby = None
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
        """Ask the game loop to return (after the running hand, if any). A lobby
        wait or a pause between hands ends at once."""
        with self._cond:
            self._stopping = True
            self._generation += 1
            self._cond.notify_all()

    def shutdown(self, timeout: float = 10.0) -> None:
        """Stop accepting requests, answer the ones in flight, send `close` to
        every session and stop the loop."""
        with self._cond:
            if self._shut_down:
                return
            self._shut_down = True
            self._stopping = True
            self._generation += 1
            self._cond.notify_all()
        if self._listener_thread is not None:
            self._tcp.shutdown()
            self._listener_thread.join(timeout)
        self._tcp.server_close()
        with self._requests:
            # e.g. the response to the last hand_end reply of a hand that just ended
            self._requests.wait_for(lambda: self._in_flight == 0, REQUEST_DRAIN_TIMEOUT)
        with self.registry_lock:
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

    def _request_begins(self) -> None:
        with self._requests:
            self._in_flight += 1

    def _request_ends(self) -> None:
        with self._requests:
            self._in_flight -= 1
            self._requests.notify_all()

    # --- lobby -------------------------------------------------------------------

    def _poke(self, arrival: bool = False) -> None:
        """Wake the lobby / pause. `arrival`: a player arrived (seated, or back
        after a rebuy / sit-in); in LOBBY the grace period restarts."""
        with self._cond:
            if arrival and self.state == LOBBY:
                self._last_arrival = time.monotonic()
            self._generation += 1
            self._cond.notify_all()

    def _wait_for_poke(self, gen: int, timeout: Optional[float]) -> None:
        """Wait until something pokes the lobby after generation `gen`, the server
        stops, or `timeout` seconds pass (None: no timeout)."""
        with self._cond:
            self._cond.wait_for(
                lambda: self._generation != gen or self._stopping,
                timeout=None if timeout is None else max(0.0, timeout),
            )

    def _lobby_status(self, pause_until: Optional[float] = None) -> _Lobby:
        """The lobby status, computed from `env` (outside the lobby condition)."""
        active = len(self.env.active_players)
        needed = self.required_players
        if pause_until is not None:
            return _Lobby("next_hand", active, needed, pause_until)
        if active < needed:
            return _Lobby("waiting", active, needed, None)
        with self._cond:
            last_arrival = self._last_arrival
        return _Lobby("starting", active, needed, last_arrival + self.grace_period)

    def _publish_lobby(self, status: _Lobby) -> None:
        """Send `lobby` to every joined session if the status changed since the
        last one published, and to newly joined sessions in any case."""
        key = tuple(status)
        with self.registry_lock:
            sessions = [
                s
                for s in self._sessions.values()
                if isinstance(s, ClientSession) and s.player is not None
            ]
            fresh, self._lobby_fresh = self._lobby_fresh, set()
            changed = key != self._last_lobby
            self._last_lobby = key
        targets = sessions if changed else [s for s in sessions if s in fresh]
        if not changed and not targets:
            return
        starts_in = None if status.deadline is None else max(0.0, status.deadline - time.monotonic())
        wire = LobbyStatus(status.state, status.active, status.needed, starts_in)
        if changed:
            self.lobby_history.append(wire)
            if starts_in is None:
                _log.info("lobby: %s (%d/%d active)", status.state, status.active, status.needed)
            else:
                _log.info(
                    "lobby: %s (%d/%d active), next hand in %.1fs",
                    status.state,
                    status.active,
                    status.needed,
                    starts_in,
                )
        for session in targets:
            session.notify(encode_lobby(wire, session.name))

    # --- session hooks (any thread; may hold the table lock) ----------------------

    def _announce_turn(self, session: ClientSession, view: TableView) -> None:
        """`acting` to every other seated player, right after the actor's `turn`."""
        try:
            name = view.me.name
        except (IndexError, TypeError):
            return
        acting = Acting(view.hand_number, view.seat, name, self.env.action_timeout)
        with self.registry_lock:
            others = [
                s
                for s in self._sessions.values()
                if isinstance(s, ClientSession) and s is not session
            ]
        for other in others:
            player = other.player
            if player is not None and player.status in _SEATED:
                other.notify(encode_acting(acting, other.name))

    def _session_status(self, session: ClientSession, kind: str) -> None:
        """`seated` / `sat_out` / `returned` / `left`: the number of active players
        changed (or the session ends). Being seated (a join) and returning (a
        rebuy / sit-in that brings a player back) count as arrivals. These run
        inside `handle_join` / `rebuy` / `sit_in` before they return, so the grace
        period restarts before the request is answered."""
        self._poke(arrival=kind in ("seated", "returned"))

    # --- registry ---------------------------------------------------------------

    def session_names(self) -> list[str]:
        with self.registry_lock:
            return [n for n, s in self._sessions.items() if isinstance(s, ClientSession)]

    def _session(self, name: str) -> ClientSession:
        with self.registry_lock:
            session = self._sessions.get(name)
        if session is None:
            raise SessionError(f"unknown player {name!r}")
        if session is _RESERVED:
            raise SessionError(f"{name} is still joining")
        return session

    def _drop(self, name: str, expected: Any) -> None:
        with self.registry_lock:
            if self._sessions.get(name) is expected:
                del self._sessions[name]
            if isinstance(expected, ClientSession):
                self._lobby_fresh.discard(expected)

    def _unregister(self, session: ClientSession) -> None:
        """Session callback: its channel is over; the name can be used again."""
        self._drop(session.name, session)
        if session.broken:
            _log.info("%s disconnected", session.name)
        else:
            _log.info("session of %s closed", session.name)
        self._poke()

    def _track_connection(self, conn: socket.socket, add: bool) -> None:
        with self.registry_lock:
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
                self._poke()
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
        # The game may now be able to start. A player brought back has already
        # restarted the grace period through the `returned` hook.
        self._poke()
        return ok_response()

    def _join(self, req: dict) -> dict:
        name = req["name"]
        with self.registry_lock:
            if self._stopping:
                return error_response("the server is shutting down")
            if name in self._sessions:
                return error_response(f"name {name!r} is already taken at this table")
            self._sessions[name] = _RESERVED  # concurrent joins with the same name
            # Until the session knows its player, the lobby does not start a
            # hand: the joiner would miss the `acting` announcements.
            self._joining += 1
        try:
            return self._join_reserved(name, req)
        finally:
            with self.registry_lock:
                self._joining -= 1
            self._poke()

    def _join_reserved(self, name: str, req: dict) -> dict:
        port = req["callback_port"]
        session: Optional[ClientSession] = None
        try:
            try:
                session = ClientSession.connect(
                    name,
                    port,
                    host=self.host,
                    timeout=self.connect_timeout,
                    on_close=self._unregister,
                    action_timeout=self.env.action_timeout,
                    on_turn=self._announce_turn,
                    on_status=self._session_status,
                )
            except OSError as e:
                self._drop(name, _RESERVED)
                return error_response(f"cannot reach callback port {port}: {e}")
            # Register before joining: a turn could reach the client, and its
            # action arrive, before handle_join returns.
            with self.registry_lock:
                self._sessions[name] = session
            session.start()
            session.feed(Join(name, req["stack_bb"], req["seat"]))
            # Seating (at once, between hands) restarts the grace period via the
            # `seated` hook before handle_join returns.
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
        with self.registry_lock:
            # Together, so the lobby publisher sees the session either as joined
            # and fresh, or not yet (no duplicate status).
            session.player = player
            if self._sessions.get(name) is session:
                self._lobby_fresh.add(session)  # gets the lobby status at once
        _log.info(
            "%s joined with %g BB (%s)",
            name,
            req["stack_bb"],
            "queued until the hand is over" if reply.queued else f"seat {reply.seat}",
        )
        return encode_join_response(reply, self.settings)


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
