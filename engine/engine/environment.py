"""Environment: a table of players with a dealer; players joining and leaving it."""

from __future__ import annotations

import functools
import random
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

from entities import Deck

from .dealer import Dealer, NotEnoughPlayersError
from .events import PlayerJoined, PlayerLeft, TableEvent
from .player import CHIPS_PER_BB, Player, PlayerStatus, call_safely, to_chips
from .views import HandResult

__all__ = [
    "Environment",
    "TableFullError",
    "NotEnoughPlayersError",
    "MIN_PLAYERS",
    "MAX_PLAYERS",
    "DEFAULT_ACTION_TIMEOUT",
]

MIN_PLAYERS = 2
MAX_PLAYERS = 10
DEFAULT_ACTION_TIMEOUT = 600.0  # seconds; only meant to catch hung / disconnected agents


class TableFullError(ValueError):
    """No seat is free for a new player (seated + waiting would exceed max_seats)."""


@dataclass
class _JoinRequest:
    player: Player
    offset: Optional[int]  # seat relative to the button; None = join as big blind


def _bind_agent(player: Player) -> None:
    """Let an agent that wants it (e.g. RemoteAgent) know its player."""
    bind = getattr(player.agent, "bind", None)
    if callable(bind):
        bind(player)


class Environment:
    """Seated players (clockwise; seat = index), a dealer, the button and a join queue.

    The environment owns event broadcasting: every public event gets the next
    `seq` of the table and is delivered to all seated players in seat order.
    Agent notifications (events, `on_seated`, `on_left`, ...) are queued under
    the table lock and delivered outside it, in order, by `_flush()`.
    """

    def __init__(
        self,
        players: Iterable[Player] = (),
        button: int = 0,
        chips_per_bb: int = CHIPS_PER_BB,
        seed: Optional[int] = None,
        max_buy_in: Optional[float] = None,
        max_seats: int = MAX_PLAYERS,
        action_timeout: Optional[float] = DEFAULT_ACTION_TIMEOUT,
        deck: Optional[Deck] = None,
    ) -> None:
        if (
            isinstance(max_seats, bool)
            or not isinstance(max_seats, int)
            or not MIN_PLAYERS <= max_seats <= MAX_PLAYERS
        ):
            raise ValueError(f"max_seats must be {MIN_PLAYERS}-{MAX_PLAYERS}, got {max_seats!r}")
        if action_timeout is not None and (
            isinstance(action_timeout, bool)
            or not isinstance(action_timeout, (int, float))
            or action_timeout <= 0
        ):
            raise ValueError(f"action_timeout must be a positive number, got {action_timeout!r}")
        players = list(players)
        if len(players) > max_seats:
            raise TableFullError(f"a table seats at most {max_seats} players, got {len(players)}")
        names: set[str] = set()
        for p in players:
            if p.chips <= 0:
                raise ValueError(f"player {p.name!r} must have a positive stack to be seated")
            if p.name is not None:
                if p.name in names:
                    raise ValueError(f"duplicate player name {p.name!r}")
                names.add(p.name)
        for p in players:
            if p.name is None:
                p.name = _default_name(names)
                names.add(p.name)

        self.max_seats = max_seats
        self.action_timeout = action_timeout
        self.pending_joins: deque[_JoinRequest] = deque()
        self._lock = threading.RLock()
        # Notifications waiting to be delivered (outside the lock), and the lock
        # that makes a single thread deliver them at a time, in order.
        self._outbox: deque[Callable[[], None]] = deque()
        self._deliver_lock = threading.RLock()
        self._seq = 0
        self.rng = random.Random(seed)
        self.dealer = Dealer(
            players,
            rng=self.rng,
            button=button if players else 0,  # no players: no button yet
            chips_per_bb=chips_per_bb,
            max_buy_in=max_buy_in,
            deck=deck,
            lock=self._lock,
        )
        self.dealer.after_hand = self._after_hand
        self.dealer.waiting_names = self._waiting_names
        self.dealer.broadcast = self._broadcast
        self.dealer.schedule = self._schedule
        self.dealer.flush = self._flush
        for p in players:
            p._env = self
            _bind_agent(p)

    # --- table state ---------------------------------------------------------

    @property
    def players(self) -> list[Player]:
        """Seated players (active and sitting out) in clockwise order (a copy)."""
        return list(self.dealer.players)

    @property
    def active_players(self) -> list[Player]:
        """Seated players who will be dealt into the next hand."""
        return self.dealer.ready_players()

    @property
    def waiting(self) -> list[Player]:
        """Players who joined during a hand and wait to be seated (FIFO)."""
        with self._lock:
            return [r.player for r in self.pending_joins]

    @property
    def button(self) -> Optional[int]:
        """Seat of the button; None at an empty table."""
        return self.dealer.button

    @property
    def chips_per_bb(self) -> int:
        return self.dealer.chips_per_bb

    @property
    def max_buy_in(self) -> Optional[float]:
        return self.dealer.max_buy_in

    def can_start(self) -> bool:
        return len(self.dealer.ready_players()) >= MIN_PLAYERS

    def _waiting_names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(r.player.name for r in self.pending_joins)  # type: ignore[misc]

    def _taken_names(self) -> set[str]:
        names = {p.name for p in self.dealer.players}
        names.update(r.player.name for r in self.pending_joins)
        return names  # type: ignore[return-value]

    def _active_seats(self) -> list[int]:
        """Seats of active players that are not leaving (join anchors / button)."""
        return [
            i
            for i, p in enumerate(self.dealer.players)
            if p.status is PlayerStatus.ACTIVE and not p.leave_requested
        ]

    # --- events --------------------------------------------------------------

    def _broadcast(self, event: TableEvent) -> None:
        """Stamp `event` with the next `seq` (and the hand number), record it in
        the hand's history and deliver it to every seated player."""
        with self._lock:
            self._seq += 1
            event = self.dealer.stamp(event, self._seq)
            recipients = self.dealer.recipients()
            self._outbox.append(functools.partial(Dealer.deliver, event, recipients))
        self._flush()

    def _schedule(self, fn: Callable[[], None]) -> None:
        """Queue an agent notification behind the events already queued."""
        with self._lock:
            self._outbox.append(fn)
        self._flush()

    def _flush(self) -> None:
        """Deliver queued notifications, in order, outside the table lock.

        A no-op while this thread holds the lock: whoever releases it flushes
        (join / leave / rebuy / the end of a hand do).
        """
        if self._lock._is_owned():  # type: ignore[attr-defined]
            return
        with self._deliver_lock:
            while True:
                with self._lock:
                    if not self._outbox:
                        return
                    fn = self._outbox.popleft()
                fn()

    # --- hands ---------------------------------------------------------------

    def start_round(self) -> HandResult:
        """Start and play a new hand.

        The deck is reset and shuffled, hole cards, per-hand player state and the
        hand's event history are cleared, the board and pots are emptied, the SB
        and BB players post the blinds, and the dealer runs the hand with the
        active players. Afterwards players who asked to leave are removed and
        queued joiners are seated. Raises NotEnoughPlayersError with fewer than
        two active players.
        """
        return self.dealer.play_hand()

    def run(self, n_hands: int) -> list[HandResult]:
        """Play up to `n_hands`; stops early when fewer than two players are active."""
        results = []
        for _ in range(n_hands):
            if not self.can_start():
                break
            results.append(self.start_round())
        return results

    # --- joining -------------------------------------------------------------

    def join(
        self,
        stack_bb: float,
        seat: Optional[int] = None,
        name: Optional[str] = None,
        agent: Optional[Any] = None,
    ) -> None:
        """A new player joins the table with `stack_bb` big blinds.

        `seat` is the position relative to the button (1 = first seat left of the
        button, ..., n = right of the button, n counting all seated players);
        None joins as big blind. At an empty / one-active-player table `seat` is
        ignored: the first joiner takes the button (SB heads-up) and the next one
        sits after them as BB. Between hands the player is seated at once; during a
        hand he is queued and seated when the hand is over (relative to the new
        button). Either way, if this does not raise the player is dealt into the
        next hand. The agent receives its `Player` through `on_seated` and the
        later callbacks. Thread-safe.
        """
        self._join(stack_bb, seat=seat, name=name, agent=agent)

    def _join(
        self,
        stack_bb: float,
        seat: Optional[int] = None,
        name: Optional[str] = None,
        agent: Optional[Any] = None,
    ) -> Player:
        """`join`, returning the new player (used by the remote join handler)."""
        chips = to_chips(stack_bb, self.chips_per_bb)
        if chips <= 0:
            raise ValueError(f"buy-in must be positive, got {stack_bb!r}")
        max_buy_in = self.max_buy_in
        if max_buy_in is not None and chips > to_chips(max_buy_in, self.chips_per_bb):
            raise ValueError(f"buy-in of {stack_bb} BB exceeds the max buy-in of {max_buy_in} BB")
        if seat is not None and (isinstance(seat, bool) or not isinstance(seat, int)):
            raise ValueError(f"seat must be an int offset from the button, got {seat!r}")
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError(f"name must be a non-empty string, got {name!r}")

        with self._lock:
            if len(self.dealer.players) + len(self.pending_joins) >= self.max_seats:
                raise TableFullError(f"the table is full ({self.max_seats} seats)")
            taken = self._taken_names()
            if name is None:
                name = _default_name(taken)
            elif name in taken:
                raise ValueError(f"name {name!r} is already taken at this table")
            n = len(self.dealer.players)
            # With fewer than two active players `seat` is ignored (see _seat).
            if seat is not None and len(self._active_seats()) >= 2 and not 1 <= seat <= n:
                raise ValueError(f"seat offset must be 1..{n}, got {seat}")

            player = Player(name, 0, agent)
            player._attach(None, self.dealer, self)
            player._chips = chips
            _bind_agent(player)
            if self.dealer.in_hand:
                self.pending_joins.append(_JoinRequest(player, seat))
            else:
                self._seat(player, seat)
        self._flush()
        return player

    def _after_hand(self) -> None:
        """Called by the dealer after a hand: remove leavers, then seat queued
        joiners in FIFO order (relative to the new button)."""
        with self._lock:
            for p in [p for p in self.dealer.players if p.leave_requested]:
                self._remove(p)
            while self.pending_joins and not self.dealer.in_hand:
                req = self.pending_joins.popleft()
                if req.player.status is PlayerStatus.WAITING:
                    self._seat(req.player, req.offset)

    def _seat(self, player: Player, offset: Optional[int]) -> None:
        """Insert `player` at `offset` clockwise from the button and renumber seats.
        Called under the lock; `on_seated` and `PlayerJoined` are queued."""
        dealer = self.dealer
        players = dealer.players
        n = len(players)
        active = self._active_seats()
        if n == 0:
            players.append(player)
            dealer.button = 0  # the first player takes the button
        elif len(active) <= 1:
            # No game running: the (only) active player holds the button and posts
            # SB heads-up; the joiner sits right after them and posts BB. With no
            # active player at all, the joiner takes the button.
            if active:
                anchor = active[0]
            else:
                anchor = dealer.button if dealer.button is not None else n - 1
            idx = anchor + 1
            players.insert(idx, player)
            dealer.button = anchor if active else idx
        else:
            b = dealer.button
            if b is None:
                b = active[0]
            elif b not in active:  # keep the button on an active player
                b = next((b + k) % n for k in range(1, n + 1) if (b + k) % n in active)
            dealer.button = b
            if offset is None:
                # Big blind: directly after the first active player clockwise from
                # the button (the next hand's small blind).
                k = next(k for k in range(1, n) if (b + k) % n in active)
                offset = k + 1
            offset = max(1, min(offset, n))
            # Insert just before the player now at `offset` (offset n: before the button).
            j = (b + offset) % n
            idx = n if j == 0 else j
            players.insert(idx, player)
            if idx <= b:
                dealer.button = b + 1
        for i, p in enumerate(players):
            p.seat = i
        player.status = PlayerStatus.ACTIVE
        dealer.sb_seat = dealer.bb_seat = None  # refer to the old numbering

        callback = getattr(player.agent, "on_seated", None)
        if callable(callback):
            view = dealer.table_view(player)
            self._schedule(functools.partial(call_safely, callback, player, view))
        # Delivered to the joiner too (after on_seated).
        self._broadcast(PlayerJoined(player.seat, player.name, player.stack))  # type: ignore[arg-type]

    # --- leaving -------------------------------------------------------------

    def leave(self, player: Player) -> None:
        """`player` leaves the table (also `Player.leave()`).

        A waiting player is dropped from the join queue. Between hands a seated
        player is removed at once (a leaving button passes to the next active
        player). During a hand the removal is deferred to its end: a participant
        still in the hand is folded at his next turn (an all-in one stays in and is
        paid normally). On removal `on_left(player, final_stack_bb)` is called and
        `PlayerLeft` is broadcast to the remaining players. Leaving twice is a
        no-op. Thread-safe.
        """
        with self._lock:
            if player.status is PlayerStatus.LEFT:
                return
            if player.status is PlayerStatus.WAITING:
                for req in self.pending_joins:
                    if req.player is player:
                        self.pending_joins.remove(req)
                        break
                else:
                    raise ValueError(f"{player.name} is not waiting at this table")
                player.status = PlayerStatus.LEFT
                player.seat = None
                self._schedule_left(player)
            elif not any(p is player for p in self.dealer.players):
                raise ValueError(f"{player.name} is not seated at this table")
            elif self.dealer.in_hand:
                player.leave_requested = True
            else:
                self._remove(player)
        self._flush()

    def _remove(self, player: Player) -> None:
        """Take `player` out of the seat list, renumber seats and fix the button."""
        dealer = self.dealer
        players = dealer.players
        idx = player.seat
        assert idx is not None and players[idx] is player
        b = dealer.button
        new_button: Optional[Player] = None
        if b == idx:
            # The button passes to the next active player clockwise (or, failing
            # that, any remaining non-leaving player).
            for pred in (
                lambda q: q.status is PlayerStatus.ACTIVE and not q.leave_requested,
                lambda q: not q.leave_requested,
            ):
                s = dealer._next_seat(idx, lambda q, pred=pred: q is not player and pred(q))
                if s is not None:
                    new_button = players[s]
                    break
        elif b is not None:
            new_button = players[b]
        players.pop(idx)
        for i, p in enumerate(players):
            p.seat = i
        if not players:
            dealer.button = None
        elif new_button is not None:
            dealer.button = new_button.seat
        else:
            dealer.button = min(idx, len(players) - 1)
        dealer.sb_seat = dealer.bb_seat = None  # refer to the old numbering

        player.status = PlayerStatus.LEFT
        player.seat = None
        player.leave_requested = False
        player.in_hand = False
        self._schedule_left(player)
        # Only the remaining seated players get it.
        self._broadcast(PlayerLeft(idx, player.name, player.stack))  # type: ignore[arg-type]

    def _schedule_left(self, player: Player) -> None:
        callback = getattr(player.agent, "on_left", None)
        if callable(callback):
            self._schedule(functools.partial(call_safely, callback, player, player.stack))


def _default_name(taken: Iterable[Optional[str]]) -> str:
    taken = set(taken)
    k = 1
    while f"Player{k}" in taken:
        k += 1
    return f"Player{k}"
