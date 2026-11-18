"""HandTracker: the public state of one hand, folded from the engine's events.

The tracker reads only public events: `HandStarted`, `BlindPosted`,
`PlayerActed` and `BoardDealt`. Everything after the last decision
(`HandShown`, `BetReturned`, `PotAwarded`) and table events are ignored. All
amounts are taken from the events (`amount_bb`, `to_bb`, `stack_bb`), never
recomputed by betting rules.

State is kept per participant index (participants in seat order, from
`HandStarted`). Events can be fed one by one, so the tracker can be kept up to
date incrementally (training) or built from a whole `TableView.history` (a bot
at a real table); both give the same state.

This module never sees hidden cards: it gets no hole cards at all (the acting
player's own cards are passed to the encoder separately).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable

from entities import Board
from engine import ActionType, BlindPosted, BoardDealt, HandStarted, PlayerActed

from .actions import BET_RAISE, CHECK_CALL, FOLD

__all__ = ["HandTracker", "N_RECENT"]

N_RECENT = 8  # recent actions kept for the observation

_FOLD = ActionType.FOLD
_RAISE = ActionType.RAISE
_KIND = {
    ActionType.FOLD: FOLD,
    ActionType.CHECK: CHECK_CALL,
    ActionType.CALL: CHECK_CALL,
    ActionType.RAISE: BET_RAISE,
}


class HandTracker:
    """Public state of one hand. Amounts in big blinds (as in the events).

    Per participant index `i` (0..n-1, seat order): `stack[i]`, `total[i]`
    (chips in the pot this hand), `street_bet[i]`, `folded[i]`, `all_in[i]`,
    `bets[i][street]` (bets / raises made per street), `acted[i]` (a voluntary
    action on this street; blinds do not count).

    For the hand: `street` (0..3), `board_cards[street]` (card indices dealt on
    that street), `board` (the engine `Board`), `pot`, `current_bet`,
    `street_bets` (bets made this street; the big blind counts preflop, every
    RAISE event counts), `aggressor` (last aggressor on this street, -1 if none),
    `prev_aggressor` (last aggressor before this street, -1 if none) and
    `recent` (the last 8 actions as (participant index, action kind, street)).

    `cache` is a per-hand scratch dict for the encoder (static per-seat parts and
    hand-strength hints per (participant, street)).
    """

    __slots__ = (
        "started",
        "n",
        "seats",
        "index",
        "button",
        "sb",
        "bb",
        "stack",
        "total",
        "street_bet",
        "folded",
        "all_in",
        "bets",
        "acted",
        "street",
        "board_cards",
        "board",
        "pot",
        "current_bet",
        "street_bets",
        "aggressor",
        "prev_aggressor",
        "recent",
        "n_events",
        "cache",
    )

    def __init__(self) -> None:
        self.started = False
        self.n = 0
        self.seats: tuple[int, ...] = ()
        self.index: dict[int, int] = {}
        self.button = -1
        self.sb = -1
        self.bb = -1
        self.stack: list[float] = []
        self.total: list[float] = []
        self.street_bet: list[float] = []
        self.folded: list[bool] = []
        self.all_in: list[bool] = []
        self.bets: list[list[int]] = []
        self.acted: list[bool] = []
        self.street = 0
        self.board_cards: list[tuple[int, ...]] = [(), (), (), ()]
        self.board = Board.empty()
        self.pot = 0.0
        self.current_bet = 0.0
        self.street_bets = 0
        self.aggressor = -1
        self.prev_aggressor = -1
        self.recent: deque[tuple[int, int, int]] = deque(maxlen=N_RECENT)
        self.n_events = 0
        self.cache: dict[Any, Any] = {}

    @classmethod
    def from_events(cls, events: Iterable[Any]) -> "HandTracker":
        tracker = cls()
        tracker.feed_all(events)
        return tracker

    # --- feeding ---------------------------------------------------------------

    def feed_all(self, events: Iterable[Any]) -> None:
        feed = self.feed
        for ev in events:
            feed(ev)

    def feed(self, ev: Any) -> None:
        """Fold one public event into the state (unknown event types are ignored)."""
        self.n_events += 1
        t = type(ev)
        if t is PlayerActed:
            self._player_acted(ev)
        elif t is BlindPosted:
            self._blind_posted(ev)
        elif t is BoardDealt:
            self._board_dealt(ev)
        elif t is HandStarted:
            self._hand_started(ev)

    def _hand_started(self, ev: HandStarted) -> None:
        if self.started:
            raise ValueError("a second HandStarted in one hand's events")
        participants = sorted(ev.participants, key=lambda p: p[0])
        n = len(participants)
        self.started = True
        self.n = n
        self.seats = tuple(p[0] for p in participants)
        self.index = {seat: i for i, seat in enumerate(self.seats)}
        self.button = self.index[ev.button]
        self.sb = self.index[ev.sb_seat]
        self.bb = self.index[ev.bb_seat]
        self.stack = [float(p[2]) for p in participants]
        self.total = [0.0] * n
        self.street_bet = [0.0] * n
        self.folded = [False] * n
        self.all_in = [False] * n
        self.bets = [[0, 0, 0, 0] for _ in range(n)]
        self.acted = [False] * n

    def _blind_posted(self, ev: BlindPosted) -> None:
        i = self.index[ev.seat]
        amount = ev.amount_bb
        self.stack[i] = ev.stack_bb
        self.street_bet[i] += amount
        self.total[i] += amount
        self.pot += amount
        if ev.all_in:
            self.all_in[i] = True
        if ev.blind == "BB":
            # The big blind is the opening bet preflop; callers must match a full
            # big blind (1 BB, the unit of every amount) even if it is short.
            self.street_bets = 1
            if self.current_bet < 1.0:
                self.current_bet = 1.0

    def _player_acted(self, ev: PlayerActed) -> None:
        i = self.index[ev.seat]
        amount = ev.amount_bb
        street = self.street
        self.stack[i] = ev.stack_bb
        self.street_bet[i] = ev.to_bb
        self.total[i] += amount
        self.pot += amount
        if ev.all_in:
            self.all_in[i] = True
        action = ev.action
        if action is _FOLD:
            self.folded[i] = True
        elif action is _RAISE:
            self.bets[i][street] += 1
            self.street_bets += 1
            self.aggressor = i
            if ev.to_bb > self.current_bet:
                self.current_bet = ev.to_bb
        self.acted[i] = True
        self.recent.append((i, _KIND[action], street))

    def _board_dealt(self, ev: BoardDealt) -> None:
        street = ev.street.value
        self.board_cards[street] = tuple(c.index for c in ev.cards)
        self.board = ev.board
        self.street = street
        n = self.n
        self.street_bet = [0.0] * n
        self.acted = [False] * n
        self.current_bet = 0.0
        self.street_bets = 0
        if self.aggressor >= 0:
            self.prev_aggressor = self.aggressor
        self.aggressor = -1

    # --- derived ----------------------------------------------------------------

    def live_count(self) -> int:
        """Participants still in the hand (not folded)."""
        return self.folded.count(False)

    def can_act_count(self) -> int:
        """Participants still able to act (not folded, not all-in)."""
        folded, all_in = self.folded, self.all_in
        return sum(1 for i in range(self.n) if not folded[i] and not all_in[i])

    def order_position(self, i: int) -> int:
        """Position of participant `i` in this street's action order: from the
        left of the big blind preflop, from the left of the button after."""
        anchor = self.bb if self.street == 0 else self.button
        return (i - anchor - 1) % self.n

    def snapshot(self) -> tuple:
        """The whole public state as a comparable tuple (tests)."""
        return (
            self.started,
            self.n,
            self.seats,
            self.button,
            self.sb,
            self.bb,
            tuple(self.stack),
            tuple(self.total),
            tuple(self.street_bet),
            tuple(self.folded),
            tuple(self.all_in),
            tuple(tuple(b) for b in self.bets),
            tuple(self.acted),
            self.street,
            tuple(self.board_cards),
            self.board,
            self.pot,
            self.current_bet,
            self.street_bets,
            self.aggressor,
            self.prev_aggressor,
            tuple(self.recent),
        )
