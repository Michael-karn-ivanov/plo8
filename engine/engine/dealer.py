"""Dealer: dealing, fixed-limit betting rounds, showdown and button movement."""

from __future__ import annotations

import random
import threading
from typing import Callable, Iterable, Optional

from entities import Board, Card, Deck, Hand, Street

from .actions import Action, ActionType, IllegalActionError, LegalActions
from .evaluator import HighRank, LowRank, best_omaha
from .player import CHIPS_PER_BB, DefaultAgent, Player, PlayerStatus, to_bb
from .pots import build_pots, split_pot
from .views import (
    Event,
    EventType,
    HandResult,
    PotResult,
    PotView,
    SeatView,
    TableView,
)

__all__ = ["Dealer", "MAX_BETS_PER_STREET", "NotEnoughPlayersError"]

# One bet plus three raises per street (the big blind is the opening bet preflop).
MAX_BETS_PER_STREET = 4

_DEFAULT_AGENT = DefaultAgent()


class NotEnoughPlayersError(RuntimeError):
    """A hand needs at least two active players with chips."""


def _callback(p: Player, name: str) -> Optional[Callable]:
    callback = getattr(p.agent, name, None)
    return callback if callable(callback) else None


class Dealer:
    """Runs hands of fixed-limit Omaha Hi-Lo 8-or-better. Internally all amounts are chips.

    The dealer sees all seated players (active and sitting out) but deals in only
    the active ones; the participants of a hand are fixed when it starts. `in_hand`
    is true from the start of a hand until its end-of-hand processing (rebuy
    callbacks, status updates, button move) is done; rebuys / sit-ins are allowed
    inside that processing through `rebuy_window`. When the hand is over,
    `after_hand` (if set) is called under `lock` — the environment uses it to
    remove leavers and seat queued joiners.
    """

    def __init__(
        self,
        players: Iterable[Player],
        rng: Optional[random.Random] = None,
        button: int = 0,
        chips_per_bb: int = CHIPS_PER_BB,
        max_buy_in: Optional[float] = None,
        deck: Optional[Deck] = None,
        lock: Optional[threading.RLock] = None,
    ) -> None:
        players = list(players)
        if len({id(p) for p in players}) != len(players):
            raise ValueError("the same player is seated twice")
        if isinstance(chips_per_bb, bool) or not isinstance(chips_per_bb, int) or chips_per_bb < 2:
            raise ValueError(f"chips_per_bb must be an int >= 2, got {chips_per_bb!r}")
        if players and not 0 <= button < len(players):
            raise ValueError(f"button seat {button} is not at the table")
        if not players and button != 0:
            raise ValueError("there is no button at an empty table")
        if max_buy_in is not None and max_buy_in <= 0:
            raise ValueError(f"max_buy_in must be positive, got {max_buy_in!r}")

        self.players = players
        self.chips_per_bb = chips_per_bb
        self.small_blind = chips_per_bb // 2
        self.big_blind = chips_per_bb
        self.max_buy_in = max_buy_in
        self.deck = deck if deck is not None else Deck(rng)
        self.board = Board.empty()
        self.button: Optional[int] = button if players else None
        self.sb_seat: Optional[int] = None
        self.bb_seat: Optional[int] = None
        self.in_hand = False
        self.rebuy_window = False
        self.hand_number = 0
        self.history: list[Event] = []
        self.last_result: Optional[HandResult] = None
        self.lock = lock if lock is not None else threading.RLock()
        self.after_hand: Optional[Callable[[], None]] = None
        # Names of players waiting to be seated (provided by the environment).
        self.waiting_names: Callable[[], tuple[str, ...]] = tuple

        self._active: list[Player] = []
        self._current_bet = 0
        self._bets = 0
        self._start_chips: dict[int, int] = {}

        for seat, p in enumerate(players):
            p._attach(seat, self)

    # --- helpers -----------------------------------------------------------

    def to_bb(self, chips: int) -> float:
        return to_bb(chips, self.chips_per_bb)

    @property
    def active_players(self) -> list[Player]:
        """Players dealt into the current (or last) hand."""
        return list(self._active)

    @property
    def hand_players(self) -> list[Player]:
        """Participants of the current (or last) hand (same as `active_players`)."""
        return list(self._active)

    @staticmethod
    def _ready(p: Player) -> bool:
        """Can be dealt into the next hand."""
        return p.status is PlayerStatus.ACTIVE and p.chips > 0 and not p.leave_requested

    def ready_players(self) -> list[Player]:
        """Seated players who would be dealt into the next hand."""
        return [p for p in self.players if self._ready(p)]

    def players_with_chips(self) -> list[Player]:
        return [p for p in self.players if p.chips > 0]

    def _order_after(self, seat: int) -> list[Player]:
        """Hand participants in seat order, starting with the first one left of `seat`."""
        n = len(self.players)
        return sorted(self._active, key=lambda p: (p.seat - seat - 1) % n)

    def _next_seat(self, seat: int, pred: Callable[[Player], bool]) -> Optional[int]:
        """First seat clockwise after `seat` whose player satisfies `pred`."""
        n = len(self.players)
        for k in range(1, n + 1):
            s = (seat + k) % n
            if pred(self.players[s]):
                return s
        return None

    def _live(self) -> list[Player]:
        return [p for p in self._active if not p.folded]

    def _bet_unit(self) -> int:
        """Small bet (1 BB) preflop and on the flop, big bet (2 BB) on turn and river."""
        if self.board.street in (Street.PREFLOP, Street.FLOP):
            return self.big_blind
        return 2 * self.big_blind

    def _emit(self, event: Event) -> None:
        self.history.append(event)
        self.notify(event)

    def notify(self, event: Event) -> None:
        """Send `event` to every seated player's agent without recording it."""
        for p in list(self.players):
            callback = _callback(p, "on_event")
            if callback is not None:
                callback(event)

    def _put(self, p: Player, chips: int) -> None:
        """Move chips from a player's stack into his bets."""
        p._chips -= chips
        p.street_chips += chips
        p.total_chips += chips
        if p._chips == 0:
            p.all_in = True

    # --- status changes ------------------------------------------------------

    def _set_sitting_out(self, p: Player) -> None:
        p.status = PlayerStatus.SITTING_OUT
        callback = _callback(p, "on_sit_out")
        if callback is not None:
            callback(p)
        self.notify(Event(EventType.PLAYER_SAT_OUT, Street.PREFLOP, seat=p.seat, name=p.name))

    def _set_active(self, p: Player) -> None:
        p.status = PlayerStatus.ACTIVE
        callback = _callback(p, "on_return")
        if callback is not None:
            callback(p)
        self.notify(Event(EventType.PLAYER_RETURNED, Street.PREFLOP, seat=p.seat, name=p.name))

    def _request_return(self, p: Player) -> None:
        """A sitting-out player rebought or called sit_in(). Inside the end-of-hand
        window the status changes with the other status updates; between hands at
        once."""
        if self.in_hand:
            p._return_requested = True
        elif p.chips > 0:
            self._set_active(p)

    # --- hand flow ---------------------------------------------------------

    def play_hand(self) -> HandResult:
        """Play one complete hand and return its result."""
        self.start_hand()
        try:
            self.post_blinds()
            self.deal_hole_cards()
            self._play_streets()
            result = self._settle()
        except BaseException:
            self._abort_hand()
            raise
        self._end_hand(result)
        return result

    def start_hand(self) -> None:
        """Reset the deck, board, pots and per-hand player state."""
        with self.lock:
            if self.in_hand:
                raise RuntimeError("a hand is already in progress")
            if len(self.ready_players()) < 2:
                raise NotEnoughPlayersError(
                    "at least two active players with chips are needed to start a hand"
                )
            if self.button is None or not self._ready(self.players[self.button]):
                start = self.button if self.button is not None else len(self.players) - 1
                self.button = self._next_seat(start, self._ready)
                assert self.button is not None

            self.deck.reset()
            self.deck.shuffle()
            self.board = Board.empty()
            self.history = []
            self._current_bet = 0
            self._bets = 0
            self._start_chips = {p.seat: p.chips for p in self.players}  # type: ignore[misc]
            for p in self.players:
                p._reset_for_hand(dealt=self._ready(p))
            self._active = [p for p in self.players if p.in_hand]
            self.sb_seat = self.bb_seat = None
            self.hand_number += 1
            self.in_hand = True
        self._emit(Event(EventType.HAND_START, Street.PREFLOP, seat=self.button))

    def post_blinds(self) -> None:
        """Small and big blind are posted; short stacks post all-in.

        SB/BB are the first and second participants clockwise from the button
        (heads-up: the button is the small blind); sitting-out seats are skipped.
        """
        assert self.button is not None
        if len(self._active) == 2:
            sb = self.players[self.button]
        else:
            sb = self._order_after(self.button)[0]
        bb = self._order_after(sb.seat)[0]  # type: ignore[arg-type]
        self.sb_seat, self.bb_seat = sb.seat, bb.seat
        self._post(sb, self.small_blind, EventType.POST_SMALL_BLIND)
        self._post(bb, self.big_blind, EventType.POST_BIG_BLIND)
        # Callers must match the full big blind even if the BB is all-in for less.
        self._current_bet = self.big_blind
        self._bets = 1

    def _post(self, p: Player, amount: int, event_type: EventType) -> None:
        chips = min(amount, p.chips)
        self._put(p, chips)
        self._emit(
            Event(
                event_type,
                Street.PREFLOP,
                seat=p.seat,
                amount=self.to_bb(chips),
                to=self.to_bb(p.street_chips),
                all_in=p.all_in,
            )
        )

    def deal_hole_cards(self) -> None:
        """Deal 4 cards one at a time to each participant, starting left of the button."""
        order = self._order_after(self.button)  # type: ignore[arg-type]
        dealt: dict[int, list[Card]] = {p.seat: [] for p in order}  # type: ignore[misc]
        for _ in range(4):
            for p in order:
                dealt[p.seat].append(self.deck.deal_one())  # type: ignore[index]
        for p in order:
            p.hole_cards = Hand(dealt[p.seat])  # type: ignore[index]

    def _play_streets(self) -> None:
        # Preflop action starts left of the big blind (heads-up: the button/SB).
        self._betting_round(self._order_after(self.bb_seat))  # type: ignore[arg-type]
        for street in (Street.FLOP, Street.TURN, Street.RIVER):
            if len(self._live()) < 2:
                return  # hand is over, remaining streets are skipped
            self._deal_board(street)
            self._start_street()
            self._betting_round(self._order_after(self.button))  # type: ignore[arg-type]

    def _deal_board(self, street: Street) -> None:
        if street is Street.FLOP:
            cards = tuple(self.deck.deal_one() for _ in range(3))
            self.board = self.board.with_flop(*cards)
        elif street is Street.TURN:
            cards = (self.deck.deal_one(),)
            self.board = self.board.with_turn(cards[0])
        else:
            cards = (self.deck.deal_one(),)
            self.board = self.board.with_river(cards[0])
        self._emit(Event(EventType.BOARD, street, cards=cards))

    def _start_street(self) -> None:
        for p in self._active:
            p.street_chips = 0
            p.has_acted = False
        self._current_bet = 0
        self._bets = 0

    # --- betting -----------------------------------------------------------

    def _needs_to_act(self, p: Player) -> bool:
        if p.folded or p.all_in:
            return False
        return not p.has_acted or p.street_chips < self._current_bet

    def _betting_round(self, order: list[Player]) -> None:
        n = len(order)
        i = 0
        while True:
            live = self._live()
            if len(live) <= 1:
                return  # everyone else folded
            can_act = [p for p in live if not p.all_in]
            if not can_act:
                return
            if len(can_act) == 1 and can_act[0].street_chips >= self._current_bet:
                return  # nobody left to bet against
            for k in range(n):
                p = order[(i + k) % n]
                if self._needs_to_act(p):
                    break
            else:
                return  # everyone acted and matched the current bet
            self._take_turn(p)
            i = (i + k + 1) % n

    def legal_actions(self, p: Player) -> LegalActions:
        """Legal actions for `p` in the current betting state.

        A player may raise only if the betting is open to him: he has not acted
        since the last full raise. An all-in raise of less than half a bet makes
        players who already acted call or fold, without the option to re-raise.
        """
        to_call = max(0, self._current_bet - p.street_chips)
        unit = self._bet_unit()
        capped = len(self._live()) > 2
        raises_left = max(0, MAX_BETS_PER_STREET - self._bets) if capped else None
        others_can_act = any(q is not p and not q.folded and not q.all_in for q in self._active)
        can_raise = (
            not p.has_acted
            and p.chips > to_call
            and others_can_act
            and (raises_left is None or raises_left > 0)
        )
        raise_to = min(self._current_bet + unit, p.street_chips + p.chips)
        return LegalActions(
            can_check=to_call == 0,
            call_amount=self.to_bb(min(to_call, p.chips)),
            can_raise=can_raise,
            raise_to=self.to_bb(raise_to),
            bet_unit=self.to_bb(unit),
            raises_left=raises_left,
        )

    def _take_turn(self, p: Player) -> None:
        """Ask `p`'s agent for an action.

        A player who asked to leave is folded without asking the agent (a leave()
        from inside on_turn is also a fold). A turn that ends without a valid
        action folds the hand, and the player sits out from the next hand.
        """
        if p.leave_requested:
            self._apply(p, Action.fold())
            return
        legal = self.legal_actions(p)
        view = self.table_view(p)
        agent = p.agent if p.agent is not None else _DEFAULT_AGENT
        p._begin_turn(legal)
        try:
            agent.on_turn(p, view, legal)
        except IllegalActionError:
            pass  # an uncaught illegal action counts as no action
        finally:
            action = p._end_turn()
        if p.leave_requested:
            action = Action.fold()
        elif action is None or not legal.allows(action.type):
            p.missed_turn = True
            action = Action.fold()
        self._apply(p, action)

    def _apply(self, p: Player, action: Action) -> None:
        t = action.type
        put = 0
        if t is ActionType.FOLD:
            p.folded = True
        elif t is ActionType.CALL:
            put = min(self._current_bet - p.street_chips, p.chips)
            self._put(p, put)
        elif t is ActionType.RAISE:
            unit = self._bet_unit()
            target = min(self._current_bet + unit, p.street_chips + p.chips)
            put = target - p.street_chips
            self._put(p, put)
            # A raise of at least half a bet (e.g. all-in for less) counts as a full
            # raise: it reopens the betting and counts toward the cap.
            if (target - self._current_bet) * 2 >= unit:
                self._bets += 1
                for q in self._active:
                    if q is not p:
                        q.has_acted = False
            self._current_bet = target
        p.has_acted = True
        self._emit(
            Event(
                EventType[t.name],
                self.board.street,
                seat=p.seat,
                amount=self.to_bb(put),
                to=self.to_bb(p.street_chips),
                all_in=p.all_in,
            )
        )

    # --- views -------------------------------------------------------------

    def table_view(self, p: Player) -> TableView:
        """Snapshot for `p`: public table state plus only p's own hole cards."""
        pots = build_pots(self._active) if self.in_hand else []
        pot = sum(q.total_chips for q in self._active) if self.in_hand else 0
        return TableView(
            hand_number=self.hand_number,
            seat=p.seat,  # type: ignore[arg-type]
            hole_cards=p.hole_cards,
            street=self.board.street,
            board=self.board,
            pot=self.to_bb(pot),
            side_pots=tuple(PotView(self.to_bb(pot.amount), pot.eligible) for pot in pots),
            current_bet=self.to_bb(self._current_bet),
            bet_unit=self.to_bb(self._bet_unit()),
            raises=self._bets,
            button=self.button,
            sb_seat=self.sb_seat,
            bb_seat=self.bb_seat,
            seats=tuple(
                SeatView(
                    seat=q.seat,  # type: ignore[arg-type]
                    name=q.name,  # type: ignore[arg-type]
                    stack=q.stack,
                    street_bet=q.street_bet,
                    total_bet=q.total_bet,
                    folded=q.folded,
                    all_in=q.all_in,
                    in_hand=q.in_hand,
                    status=q.status,
                    leaving=q.leave_requested,
                )
                for q in self.players
            ),
            history=tuple(self.history),
            waiting=tuple(self.waiting_names()),
        )

    # --- showdown and settlement --------------------------------------------

    def _settle(self) -> HandResult:
        order = self._order_after(self.button)  # type: ignore[arg-type]
        seat_order = [p.seat for p in order]
        live = [p for p in order if not p.folded]
        showdown = len(live) >= 2

        shown: dict[int, Hand] = {}
        ranks: dict[int, tuple[HighRank, Optional[LowRank]]] = {}
        if showdown:
            for p in live:
                cards = p.show_cards()
                shown[p.seat] = cards  # type: ignore[index]
                ranks[p.seat] = best_omaha(cards, self.board)  # type: ignore[index]
                self._emit(
                    Event(EventType.SHOW, self.board.street, seat=p.seat, cards=tuple(cards))
                )

        won = {p.seat: 0 for p in self._active}
        pot_results = []
        for pot in build_pots(self._active):
            if pot.uncontested:
                high = {pot.eligible[0]: pot.amount}
                low: dict[int, int] = {}
            else:
                high, low = split_pot(pot, ranks, seat_order)  # type: ignore[arg-type]
            for shares in (high, low):
                for seat, amount in shares.items():
                    won[seat] += amount
            pot_results.append(
                PotResult(
                    amount=self.to_bb(pot.amount),
                    eligible=pot.eligible,
                    high=tuple((s, self.to_bb(a)) for s, a in high.items()),
                    low=tuple((s, self.to_bb(a)) for s, a in low.items()),
                    uncontested=pot.uncontested,
                )
            )

        for seat in seat_order:
            amount = won[seat]
            if amount:
                self.players[seat]._chips += amount  # type: ignore[index]
                self._emit(
                    Event(EventType.AWARD, self.board.street, seat=seat, amount=self.to_bb(amount))
                )

        return HandResult(
            hand_number=self.hand_number,
            button=self.button,  # type: ignore[arg-type]
            sb_seat=self.sb_seat,
            bb_seat=self.bb_seat,
            board=self.board,
            showdown=showdown,
            shown=shown,
            pots=tuple(pot_results),
            net={
                p.seat: self.to_bb(p.chips - self._start_chips[p.seat])  # type: ignore[index,misc]
                for p in self.players
            },
            stacks={p.seat: p.stack for p in self.players},  # type: ignore[misc]
            history=tuple(self.history),
            names={p.seat: p.name for p in self.players},  # type: ignore[misc]
            dealt_in={p.seat: p.in_hand for p in self.players},  # type: ignore[misc]
        )

    def _end_hand(self, result: HandResult) -> None:
        """End-of-hand processing, in order:

        1. `on_hand_end` for every seated player that is not leaving — participants
           and sitting-out players (rebuy / sit-in / leave window). Players who
           joined during the hand are not seated yet, so they get none.
        2. Status updates: busted or non-acting active players sit out; sitting-out
           players who rebought / sat in (and have chips) return.
        3. The button moves to the next active, non-leaving player.
        4. `in_hand` is cleared and the environment removes leavers and seats joiners.
        """
        self.last_result = result
        self.rebuy_window = True
        try:
            for p in list(self.players):
                if p.status is PlayerStatus.LEFT or p.leave_requested:
                    continue
                agent = p.agent if p.agent is not None else _DEFAULT_AGENT
                agent.on_hand_end(p, result)
        finally:
            self.rebuy_window = False
            try:
                self._update_statuses()
                self._move_button()
            finally:
                self._finish_hand()

    def _update_statuses(self) -> None:
        for p in list(self.players):
            if not p.leave_requested:
                if p.status is PlayerStatus.ACTIVE:
                    if p.chips <= 0 or p.missed_turn:
                        self._set_sitting_out(p)
                elif p.status is PlayerStatus.SITTING_OUT:
                    if p._return_requested and p.chips > 0:
                        self._set_active(p)
            p._return_requested = False
            p.missed_turn = False

    def _move_button(self) -> None:
        if self.button is None:
            return
        nxt = self._next_seat(self.button, self._ready)
        if nxt is not None:
            self.button = nxt

    def _abort_hand(self) -> None:
        """Undo a hand that failed midway (e.g. an agent raised): refund all bets."""
        for p in self._active:
            p._chips += p.total_chips
            p.total_chips = 0
            p.street_chips = 0
            p._legal = None
            p._pending = None
        for p in self.players:
            p.missed_turn = False
            p._return_requested = False
        self.rebuy_window = False
        self._finish_hand()

    def _finish_hand(self) -> None:
        """Leave the hand and let the environment remove leavers / seat joiners."""
        with self.lock:
            self.in_hand = False
            if self.after_hand is not None:
                self.after_hand()
