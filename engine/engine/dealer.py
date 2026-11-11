"""Dealer: dealing, fixed-limit betting rounds, showdown and button movement."""

from __future__ import annotations

import random
from typing import Iterable, Optional

from entities import Board, Card, Deck, Hand, Street

from .actions import Action, ActionType, IllegalActionError, LegalActions
from .evaluator import HighRank, LowRank, best_omaha
from .player import CHIPS_PER_BB, DefaultAgent, Player, fallback_action, to_bb
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

__all__ = ["Dealer", "MAX_BETS_PER_STREET"]

# One bet plus three raises per street (the big blind is the opening bet preflop).
MAX_BETS_PER_STREET = 4

_DEFAULT_AGENT = DefaultAgent()


class Dealer:
    """Runs hands of fixed-limit Omaha Hi-Lo 8-or-better. Internally all amounts are chips."""

    def __init__(
        self,
        players: Iterable[Player],
        rng: Optional[random.Random] = None,
        button: int = 0,
        chips_per_bb: int = CHIPS_PER_BB,
        max_buy_in: Optional[float] = None,
        deck: Optional[Deck] = None,
    ) -> None:
        players = list(players)
        if len({id(p) for p in players}) != len(players):
            raise ValueError("the same player is seated twice")
        if isinstance(chips_per_bb, bool) or not isinstance(chips_per_bb, int) or chips_per_bb < 2:
            raise ValueError(f"chips_per_bb must be an int >= 2, got {chips_per_bb!r}")
        if not 0 <= button < len(players):
            raise ValueError(f"button seat {button} is not at the table")
        if max_buy_in is not None and max_buy_in <= 0:
            raise ValueError(f"max_buy_in must be positive, got {max_buy_in!r}")

        self.players = players
        self.chips_per_bb = chips_per_bb
        self.small_blind = chips_per_bb // 2
        self.big_blind = chips_per_bb
        self.max_buy_in = max_buy_in
        self.deck = deck if deck is not None else Deck(rng)
        self.board = Board.empty()
        self.button = button
        self.sb_seat: Optional[int] = None
        self.bb_seat: Optional[int] = None
        self.in_hand = False
        self.hand_number = 0
        self.history: list[Event] = []
        self.last_result: Optional[HandResult] = None

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

    def players_with_chips(self) -> list[Player]:
        return [p for p in self.players if p.chips > 0]

    def _order_after(self, seat: int) -> list[Player]:
        """Active players in seat order, starting with the first one left of `seat`."""
        n = len(self.players)
        return sorted(self._active, key=lambda p: (p.seat - seat - 1) % n)

    def _next_seat_with_chips(self, seat: int) -> Optional[int]:
        n = len(self.players)
        for k in range(1, n + 1):
            s = (seat + k) % n
            if self.players[s].chips > 0:
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
        for p in self.players:
            callback = getattr(p.agent, "on_event", None)
            if callable(callback):
                callback(event)

    def _put(self, p: Player, chips: int) -> None:
        """Move chips from a player's stack into his bets."""
        p._chips -= chips
        p.street_chips += chips
        p.total_chips += chips
        if p._chips == 0:
            p.all_in = True

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
        if self.in_hand:
            raise RuntimeError("a hand is already in progress")
        if len(self.players_with_chips()) < 2:
            raise RuntimeError("at least two players with chips are needed to start a hand")
        if self.players[self.button].chips <= 0:
            nxt = self._next_seat_with_chips(self.button)
            assert nxt is not None
            self.button = nxt

        self.deck.reset()
        self.deck.shuffle()
        self.board = Board.empty()
        self.history = []
        self._current_bet = 0
        self._bets = 0
        self._start_chips = {p.seat: p.chips for p in self.players}  # type: ignore[misc]
        for p in self.players:
            p._reset_for_hand(dealt=p.chips > 0)
        self._active = [p for p in self.players if p.in_hand]
        self.sb_seat = self.bb_seat = None
        self.hand_number += 1
        self.in_hand = True
        self._emit(Event(EventType.HAND_START, Street.PREFLOP, seat=self.button))

    def post_blinds(self) -> None:
        """Small and big blind are posted; short stacks post all-in."""
        if len(self._active) == 2:
            sb = self.players[self.button]  # heads-up: the button is the small blind
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
        """Deal 4 cards one at a time to each player, starting left of the button."""
        order = self._order_after(self.button)
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
            self._betting_round(self._order_after(self.button))

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
        if action is None or not legal.allows(action.type):
            action = fallback_action(legal)
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
        return TableView(
            hand_number=self.hand_number,
            seat=p.seat,  # type: ignore[arg-type]
            hole_cards=p.hole_cards,
            street=self.board.street,
            board=self.board,
            pot=self.to_bb(sum(q.total_chips for q in self._active)),
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
                    name=q.name,
                    stack=q.stack,
                    street_bet=q.street_bet,
                    total_bet=q.total_bet,
                    folded=q.folded,
                    all_in=q.all_in,
                    in_hand=q.in_hand,
                )
                for q in self.players
            ),
            history=tuple(self.history),
        )

    # --- showdown and settlement --------------------------------------------

    def _settle(self) -> HandResult:
        order = self._order_after(self.button)
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
            button=self.button,
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
        )

    def _end_hand(self, result: HandResult) -> None:
        """Rebuy window for every seated player, then move the button."""
        self.in_hand = False
        self.last_result = result
        try:
            for p in self.players:
                agent = p.agent if p.agent is not None else _DEFAULT_AGENT
                agent.on_hand_end(p, result)
        finally:
            nxt = self._next_seat_with_chips(self.button)
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
        self.in_hand = False
