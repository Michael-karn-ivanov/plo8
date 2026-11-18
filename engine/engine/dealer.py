"""Dealer: dealing, fixed-limit betting rounds, showdown and button movement.

Deck use and deal order (public contract)
-----------------------------------------
How the dealer consumes its deck is part of the engine's contract, so that a
caller-supplied deck (e.g. a stacked `Deck` subclass replaying a recorded hand)
produces exactly the intended hole cards and board:

- Once per hand, before any card is dealt, the dealer calls `deck.reset()` and
  then `deck.shuffle()` - exactly once each, in that order, on the deck object
  itself (never a copy). A subclass overriding `shuffle()` (e.g.
  `super().shuffle()` then `self.put_on_top(script)`) controls the top of the
  deck for that hand.
- During the hand the dealer only calls `deck.deal_one()`, one card at a time
  from the top. It never calls `deal(n)`, `remove`, `put_on_top`, `shuffle` or
  `reset` mid-hand, and never looks at cards without dealing them.
- Hole cards: four rounds; in each round one card to every participant,
  clockwise, starting with the first participant left of the button and ending
  with the button. Sitting-out seats are skipped. Heads-up the BB (left of the
  button) gets the first card and the button/SB the second. A player's `Hand`
  lists the cards in round order.
- Board: then the flop (next 3 cards, in that order), the turn (next card) and
  the river (next card). No burn cards. The same order is used for an all-in
  run-out. Streets that are not reached are not dealt.

With `n` participants counted from the left of the button (participant 0 =
first left of the button, n-1 = the button), deck card `i` (0-based, top first)
for `i < 4n` goes to participant `i % n` as hole card `i // n`; cards `4n`,
`4n+1`, `4n+2` are the flop, `4n+3` the turn and `4n+4` the river.

The same in callback mode and in step mode. Changing this order is a breaking
change for callers that stack decks.

One hand flow, two drivers
--------------------------
The whole hand is one generator, `_hand_steps()`: it runs until a player has
to decide, yields `(player, legal)` and waits for the answer `(action,
reason)`; when the hand is settled it returns the `HandResult`. The callback
driver (`play_hand()`) answers through the agent's `on_turn`; the step driver
(`begin_step_hand()` / `submit()`) answers with the caller's action. Both run
the same end-of-hand processing, so they produce identical hands.
"""

from __future__ import annotations

import functools
import random
import threading
from typing import Callable, Generator, Iterable, Optional, Union

from entities import Board, Card, Deck, Hand, Street

from .actions import Action, ActionType, IllegalActionError, LegalActions
from .evaluator import HighRank, LowRank, best_omaha_many
from .events import (
    FOLD_LEAVE,
    FOLD_TIMEOUT,
    BetReturned,
    BlindPosted,
    BoardDealt,
    HandShown,
    HandStarted,
    PlayerActed,
    PlayerReturned,
    PlayerSatOut,
    PotAwarded,
    TableEvent,
)
from .player import (
    CHIPS_PER_BB,
    Player,
    PlayerStatus,
    call_safely,
    fallback_action,
    to_bb,
    to_chips,
)
from .pots import build_pots, split_pot
from .step import HandInProgressError, NoDecisionError
from .views import HandResult, PotResult, PotView, SeatView, TableView

__all__ = ["Dealer", "MAX_BETS_PER_STREET", "NotEnoughPlayersError"]

# One bet plus three raises per street (the big blind is the opening bet preflop).
MAX_BETS_PER_STREET = 4

_FOLD = Action.fold()
_FOLD_TYPE = ActionType.FOLD
_CALL_TYPE = ActionType.CALL
_RAISE_TYPE = ActionType.RAISE
_LEFT = PlayerStatus.LEFT

# What the hand generator yields (a decision) and is sent back (the answer).
_Request = tuple[Player, LegalActions]
_Answer = tuple[Action, Optional[str]]
_HandSteps = Generator[_Request, _Answer, HandResult]


class NotEnoughPlayersError(RuntimeError):
    """A hand needs at least two active players with chips."""


def _run_now(fn: Callable[[], None]) -> None:
    fn()


class Dealer:
    """Runs hands of fixed-limit Omaha Hi-Lo 8-or-better. Internally all amounts are chips.

    The chip unit is `chips_per_bb` (an even int >= 2): the big blind is
    `chips_per_bb` chips, the small blind half of it.

    The dealer sees all seated players (active and sitting out) but deals in only
    the active ones; the participants of a hand are fixed when it starts. `in_hand`
    is true from the start of a hand until its end-of-hand processing (rebuy
    callbacks, status updates, button move) is done - in step mode also while a
    decision is pending; rebuys / sit-ins are allowed inside that processing
    through `rebuy_window`. When the hand is over, `after_hand` (if set) is called
    under `lock` — the environment uses it to remove leavers and seat queued
    joiners.

    Public events go through `broadcast(event)` and other agent notifications
    through `schedule(fn)`; both are replaced by the environment (which owns the
    `seq` counter and delivers outside its lock). `flush()` delivers anything
    queued while the lock was held. The defaults deliver at once.
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
        if (
            isinstance(chips_per_bb, bool)
            or not isinstance(chips_per_bb, int)
            or chips_per_bb < 2
            or chips_per_bb % 2
        ):
            raise ValueError(f"chips_per_bb must be an even int >= 2, got {chips_per_bb!r}")
        if deck is not None and not isinstance(deck, Deck):
            raise TypeError(f"deck must be an entities.Deck, got {type(deck).__name__}")
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
        self.history: list[TableEvent] = []  # public events of the current hand
        self.last_result: Optional[HandResult] = None
        self.lock = lock if lock is not None else threading.RLock()
        self.after_hand: Optional[Callable[[], None]] = None
        # Names of players waiting to be seated (provided by the environment).
        self.waiting_names: Callable[[], tuple[str, ...]] = tuple
        # Event / notification delivery (replaced by the environment).
        self.broadcast: Callable[[TableEvent], None] = self._broadcast_now
        self.schedule: Callable[[Callable[[], None]], None] = _run_now
        self.flush: Callable[[], None] = lambda: None
        self._seq = 0

        # Per-hand state.
        self._active: list[Player] = []  # participants, in seat order
        self._order: list[Player] = []  # participants from the left of the button
        self._preflop_order: list[Player] = []  # participants from the left of the BB
        self._current_bet = 0
        self._bets = 0
        self._unit = self.big_blind  # bet unit of the current street
        self._n_live = 0  # participants that have not folded
        self._n_can_act = 0  # participants that have not folded and are not all-in
        self._start_chips: dict[int, int] = {}

        # Step mode: the running hand generator and the pending decision.
        self._steps: Optional[_HandSteps] = None
        self.pending: Optional[_Request] = None

        for seat, p in enumerate(players):
            p._attach(seat, self)

    # --- helpers -----------------------------------------------------------

    def to_bb(self, chips: int) -> float:
        """Chips of this table to big blinds."""
        return to_bb(chips, self.chips_per_bb)

    def to_chips(self, amount_bb: float) -> int:
        """Big blinds to whole chips of this table (rounded)."""
        return to_chips(amount_bb, self.chips_per_bb)

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
        """Small bet (1 BB) preflop and on the flop, big bet (2 BB) on turn and
        river. Set once per street (when the hand starts and when a street is dealt)."""
        return self._unit

    def _put(self, p: Player, chips: int) -> None:
        """Move chips from a player's stack into his bets."""
        p._chips -= chips
        p.street_chips += chips
        p.total_chips += chips
        if p._chips == 0 and not p.all_in:
            p.all_in = True
            if not p.folded:
                self._n_can_act -= 1

    # --- events and notifications --------------------------------------------

    def recipients(self) -> list[Player]:
        """Players who receive public events: everyone seated (active or sitting
        out, including those about to leave)."""
        return [p for p in self.players if p.seated]

    def stamp(self, event: TableEvent, seq: int) -> TableEvent:
        """Set `seq` / `hand_no` on the freshly built `event` (in place, before
        anyone else has seen it) and record it in the hand's history."""
        in_hand = self.in_hand
        object.__setattr__(event, "hand_no", self.hand_number if in_hand else None)
        object.__setattr__(event, "seq", seq)
        if in_hand:
            self.history.append(event)
        return event

    @staticmethod
    def deliver(event: TableEvent, recipients: Iterable[Player]) -> None:
        """Call `on_event(player, event)` for each recipient, in seat order."""
        for p in recipients:
            if p.status is _LEFT:
                continue
            callback = p._callback("on_event")
            if callback is not None:
                call_safely(callback, p, event)

    def _broadcast_now(self, event: TableEvent) -> None:
        """Default broadcast for a dealer used without an environment."""
        with self.lock:
            self._seq += 1
            event = self.stamp(event, self._seq)
            recipients = self.recipients()
        self.deliver(event, recipients)

    def _emit(self, event: TableEvent) -> None:
        self.broadcast(event)

    def _notify(self, p: Player, name: str, *args: object) -> None:
        """Schedule an optional notification callback `name(p, *args)`."""
        callback = p._callback(name)
        if callback is not None:
            self.schedule(functools.partial(call_safely, callback, p, *args))

    # --- status changes ------------------------------------------------------

    def _set_sitting_out(self, p: Player) -> None:
        p.status = PlayerStatus.SITTING_OUT
        self._notify(p, "on_sit_out")
        self._emit(PlayerSatOut(p.seat, p.name, p.stack))  # type: ignore[arg-type]

    def _set_active(self, p: Player) -> None:
        p.status = PlayerStatus.ACTIVE
        self._notify(p, "on_return")
        self._emit(PlayerReturned(p.seat, p.name, p.stack))  # type: ignore[arg-type]

    def _request_return(self, p: Player) -> None:
        """A sitting-out player rebought or called sit_in(). Inside the end-of-hand
        window the status changes with the other status updates; between hands at
        once."""
        if self.in_hand:
            p._return_requested = True
        elif p.chips > 0:
            self._set_active(p)

    # --- callback driver ---------------------------------------------------

    def play_hand(self) -> HandResult:
        """Play one complete hand in callback mode and return its result."""
        self.start_hand()
        steps = self._hand_steps()
        try:
            answer: Optional[_Answer] = None
            while True:
                try:
                    p, legal = steps.send(answer)  # type: ignore[arg-type]
                except StopIteration as stop:
                    result: HandResult = stop.value
                    break
                answer = self._callback_turn(p, legal)
        except BaseException:
            steps.close()
            self._abort_hand()
            raise
        self._end_hand(result)
        return result

    def _callback_turn(self, p: Player, legal: LegalActions) -> _Answer:
        """Ask `p`'s agent for an action (callback mode).

        The default agent (no agent) is answered with `fallback_action` without
        building a view. A leave() from inside on_turn is a fold. A turn that ends
        without a valid action folds the hand, and the player sits out from the
        next hand.
        """
        agent = p.agent
        if agent is None:
            if p.leave_requested:
                return _FOLD, FOLD_LEAVE
            return fallback_action(legal), None
        view = self.table_view(p)
        p._begin_turn(legal)
        try:
            agent.on_turn(p, view, legal)
        except IllegalActionError:
            pass  # an uncaught illegal action counts as no action
        finally:
            action = p._end_turn()
        if p.leave_requested:
            return _FOLD, FOLD_LEAVE
        if action is None or not legal.allows(action.type):
            p.missed_turn = True
            return _FOLD, FOLD_TIMEOUT
        return action, None

    # --- step driver -------------------------------------------------------

    def begin_step_hand(self) -> Optional[_Request]:
        """Start a hand in step mode and run it to its first decision.

        Returns the pending `(player, legal)`, or None if the hand needed no
        decision at all (it has then been played to the end, including the
        end-of-hand processing, and `last_result` holds its result).
        """
        self.start_hand()
        self._steps = self._hand_steps()
        return self._advance(None)

    def submit(self, action: Union[Action, ActionType]) -> Optional[_Request]:
        """Answer the pending step-mode decision.

        Everything is validated before anything changes: a rejected action
        (`NoDecisionError` / `IllegalActionError`) leaves the hand exactly as it
        was. A player who asked to leave while the decision was pending is folded
        (`reason="leave"`), whatever legal action is submitted. Returns the next
        decision, or None when the hand is over.
        """
        pending = self.pending
        if pending is None:
            raise NoDecisionError("no decision is pending")
        if isinstance(action, ActionType):
            action_type = action
        elif isinstance(action, Action):
            action_type = action.type
        else:
            raise IllegalActionError(f"not an Action or ActionType: {action!r}")
        p, legal = pending
        if not legal.allows(action_type):
            raise IllegalActionError(
                f"{getattr(action_type, 'value', action_type)} is not legal now (legal: "
                f"{', '.join(t.value for t in legal.actions)})"
            )
        if p.leave_requested:
            answer: _Answer = (_FOLD, FOLD_LEAVE)
        else:
            answer = (Action.of(action_type), None)
        return self._advance(answer)

    def _advance(self, answer: Optional[_Answer]) -> Optional[_Request]:
        """Run the step-mode hand to its next decision (stored in `pending`) or to
        its end (end-of-hand processing done). Any exception aborts the hand."""
        steps = self._steps
        assert steps is not None
        self.pending = None
        try:
            request = steps.send(answer)  # type: ignore[arg-type]
        except StopIteration as stop:
            result: HandResult = stop.value
        except BaseException:
            self._steps = None
            self._abort_hand()
            raise
        else:
            self.pending = request
            return request
        self._steps = None
        self._end_hand(result)
        return None

    # --- hand flow ---------------------------------------------------------

    def start_hand(self) -> None:
        """Prepare the deck (one `reset()` then one `shuffle()`, see the module
        docstring), reset the board, pots, the event history and per-hand player
        state; fix the participants, the button and the blind seats."""
        with self.lock:
            if self.in_hand:
                raise HandInProgressError("a hand is already in progress")
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
            self._unit = self.big_blind
            self._start_chips = {p.seat: p.chips for p in self.players}  # type: ignore[misc]
            ready = self._ready
            for p in self.players:
                p._reset_for_hand(dealt=ready(p))
            self._active = [p for p in self.players if p.in_hand]
            self._order = self._order_after(self.button)
            sb, bb = self._blind_players()
            self.sb_seat, self.bb_seat = sb.seat, bb.seat
            self._preflop_order = self._order_after(bb.seat)  # type: ignore[arg-type]
            self._n_live = self._n_can_act = len(self._active)
            self.pending = None
            self.hand_number += 1
            self.in_hand = True

    def _hand_steps(self) -> _HandSteps:
        """The whole hand, from `HandStarted` to the settlement: yields each
        decision `(player, legal)` and takes the answer `(action, reason)`;
        returns the `HandResult`. Shared by both drivers."""
        self._emit(self._hand_started())
        self.post_blinds()
        self.deal_hole_cards()
        # Preflop action starts left of the big blind (heads-up: the button/SB).
        yield from self._betting_round(self._preflop_order)
        for street in (Street.FLOP, Street.TURN, Street.RIVER):
            if self._n_live < 2:
                break  # hand is over, remaining streets are skipped
            self._deal_board(street)
            self._start_street()
            yield from self._betting_round(self._order)
        return self._settle()

    def _blind_players(self) -> tuple[Player, Player]:
        """SB/BB are the first and second participants clockwise from the button
        (heads-up: the button is the small blind); sitting-out seats are skipped."""
        assert self.button is not None
        if len(self._active) == 2:
            sb = self.players[self.button]
        else:
            sb = self._order_after(self.button)[0]
        bb = self._order_after(sb.seat)[0]  # type: ignore[arg-type]
        return sb, bb

    def _hand_started(self) -> HandStarted:
        return HandStarted(
            button=self.button,  # type: ignore[arg-type]
            sb_seat=self.sb_seat,  # type: ignore[arg-type]
            bb_seat=self.bb_seat,  # type: ignore[arg-type]
            participants=tuple((p.seat, p.name, p.stack) for p in self._active),  # type: ignore[misc]
        )

    def post_blinds(self) -> None:
        """Small and big blind are posted; short stacks post all-in."""
        if self.sb_seat is None or self.bb_seat is None:
            sb, bb = self._blind_players()
            self.sb_seat, self.bb_seat = sb.seat, bb.seat
        self._post(self.players[self.sb_seat], self.small_blind, "SB")
        self._post(self.players[self.bb_seat], self.big_blind, "BB")
        # Callers must match the full big blind even if the BB is all-in for less.
        self._current_bet = self.big_blind
        self._bets = 1

    def _post(self, p: Player, amount: int, blind: str) -> None:
        chips = min(amount, p.chips)
        self._put(p, chips)
        self._emit(
            BlindPosted(
                p.seat,  # type: ignore[arg-type]
                p.name,  # type: ignore[arg-type]
                blind,
                self.to_bb(chips),
                p.all_in,
                p.stack,
            )
        )

    def deal_hole_cards(self) -> None:
        """Deal 4 rounds of one card to each participant, starting left of the
        button and ending with the button (see the module docstring); then tell
        each participant (same order) its own hand. A participant's view is built
        only if its agent has `on_hole_cards`."""
        order = self._order
        deal_one = self.deck.deal_one
        dealt: list[list[Card]] = [[] for _ in order]
        for _ in range(4):
            for cards in dealt:
                cards.append(deal_one())
        for p, cards in zip(order, dealt):
            p.hole_cards = Hand(cards)
        for p in order:
            callback = p._callback("on_hole_cards")
            if callback is not None:
                self.schedule(
                    functools.partial(call_safely, callback, p, p.hole_cards, self.table_view(p))
                )

    def _deal_board(self, street: Street) -> None:
        """Deal the next street from the top of the deck, one card at a time (no
        burn), and set the street's bet unit."""
        deal_one = self.deck.deal_one
        if street is Street.FLOP:
            cards: tuple[Card, ...] = (deal_one(), deal_one(), deal_one())
            self.board = self.board.with_flop(*cards)
            self._unit = self.big_blind
        elif street is Street.TURN:
            cards = (deal_one(),)
            self.board = self.board.with_turn(cards[0])
            self._unit = 2 * self.big_blind
        else:
            cards = (deal_one(),)
            self.board = self.board.with_river(cards[0])
            self._unit = 2 * self.big_blind
        self._emit(BoardDealt(street, cards, self.board))

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

    def _betting_round(self, order: list[Player]) -> Generator[_Request, _Answer, None]:
        """One betting round (generator): yields each decision and applies the
        answer. A player who asked to leave is folded without a decision."""
        n = len(order)
        i = 0
        while True:
            if self._n_live <= 1 or self._n_can_act == 0:
                return  # everyone else folded / nobody can bet
            current = self._current_bet
            for k in range(n):
                p = order[(i + k) % n]
                if not p.folded and not p.all_in and (not p.has_acted or p.street_chips < current):
                    break
            else:
                return  # everyone acted and matched the current bet
            if self._n_can_act == 1 and p.street_chips >= current:
                return  # nobody left to bet against
            i = (i + k + 1) % n
            if p.leave_requested:
                self._apply(p, _FOLD, FOLD_LEAVE)
                continue
            action, reason = yield p, self.legal_actions(p)
            self._apply(p, action, reason)

    def legal_actions(self, p: Player) -> LegalActions:
        """Legal actions for `p` in the current betting state.

        A player may raise only if the betting is open to him: he has not acted
        since the last full raise. An all-in raise of less than half a bet makes
        players who already acted call or fold, without the option to re-raise.
        """
        current = self._current_bet
        street = p.street_chips
        chips = p._chips
        to_call = current - street
        if to_call < 0:
            to_call = 0
        unit = self._unit
        if self._n_live > 2:
            raises_left: Optional[int] = MAX_BETS_PER_STREET - self._bets
            if raises_left < 0:  # type: ignore[operator]
                raises_left = 0
        else:
            raises_left = None
        own = 1 if (p.in_hand and not p.folded and not p.all_in) else 0
        others_can_act = self._n_can_act - own > 0
        can_raise = (
            not p.has_acted
            and chips > to_call
            and others_can_act
            and (raises_left is None or raises_left > 0)
        )
        raise_to = current + unit
        if raise_to > street + chips:
            raise_to = street + chips
        cpb = self.chips_per_bb
        return LegalActions(
            can_check=to_call == 0,
            call_amount=(to_call if to_call < chips else chips) / cpb,
            can_raise=can_raise,
            raise_to=raise_to / cpb,
            bet_unit=unit / cpb,
            raises_left=raises_left,
        )

    def _apply(self, p: Player, action: Action, reason: Optional[str] = None) -> None:
        t = action.type
        put = 0
        if t is _FOLD_TYPE:
            p.folded = True
            self._n_live -= 1
            if not p.all_in:
                self._n_can_act -= 1
        elif t is _CALL_TYPE:
            put = min(self._current_bet - p.street_chips, p._chips)
            self._put(p, put)
        elif t is _RAISE_TYPE:
            unit = self._unit
            target = min(self._current_bet + unit, p.street_chips + p._chips)
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
        cpb = self.chips_per_bb
        self._emit(
            PlayerActed(
                p.seat,  # type: ignore[arg-type]
                p.name,  # type: ignore[arg-type]
                self.board.street,
                t,
                put / cpb,
                p.street_chips / cpb,
                p.all_in,
                p._chips / cpb,
                reason,
            )
        )

    # --- views -------------------------------------------------------------

    def table_view(self, p: Player) -> TableView:
        """Snapshot for `p`: public table state plus only p's own hole cards.

        The only place views are built; it is called only when a view is consumed
        (an agent callback that receives it, or `Environment.view()`)."""
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
            bet_unit=self.to_bb(self._unit),
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

    def _return_uncalled(self) -> dict[int, int]:
        """Give the uncalled part of the biggest bet back to its (non-folded)
        owner, emitting `BetReturned`. Returns {seat: chips}."""
        by_size = sorted(self._active, key=lambda p: p.total_chips, reverse=True)
        if not by_size:
            return {}
        top = by_size[0]
        second = by_size[1].total_chips if len(by_size) > 1 else 0
        excess = top.total_chips - second
        if excess <= 0 or top.folded:
            return {}
        top.total_chips -= excess
        top.street_chips = max(0, top.street_chips - excess)
        top._chips += excess
        self._emit(
            BetReturned(
                top.seat,  # type: ignore[arg-type]
                top.name,  # type: ignore[arg-type]
                self.to_bb(excess),
                top.stack,
            )
        )
        return {top.seat: excess}  # type: ignore[dict-item]

    def _settle(self) -> HandResult:
        """Showdown (if two or more players are left), uncalled bet, pots.

        Hands are shown in order starting left of the button (`HandShown`) and
        evaluated together against one set of board triples; then the uncalled
        bet is returned (`BetReturned`) and each pot is credited share by share
        (`PotAwarded`).
        """
        order = self._order
        seat_order = [p.seat for p in order]
        live = [p for p in order if not p.folded]
        showdown = len(live) >= 2

        shown: dict[int, Hand] = {}
        ranks: dict[int, tuple[HighRank, Optional[LowRank]]] = {}
        if showdown:
            hands = []
            for p in live:
                cards = p.show_cards()
                shown[p.seat] = cards  # type: ignore[index]
                hands.append(cards)
                self._emit(HandShown(p.seat, p.name, cards, p.stack))  # type: ignore[arg-type]
            for p, rank in zip(live, best_omaha_many(hands, self.board)):
                ranks[p.seat] = rank  # type: ignore[index]

        returned = self._return_uncalled()

        pot_results = []
        for index, pot in enumerate(build_pots(self._active)):
            if pot.uncontested:
                high = {pot.eligible[0]: pot.amount}
                low: dict[int, int] = {}
                shares: tuple[tuple[int, str, int], ...] = ((pot.eligible[0], "whole", pot.amount),)
            else:
                split = split_pot(pot, ranks, seat_order)  # type: ignore[arg-type]
                high, low = split.high, split.low
                shares = split.shares
            for seat, share, amount in shares:
                if amount <= 0:
                    continue
                winner = self.players[seat]
                winner._chips += amount
                self._emit(
                    PotAwarded(
                        seat,
                        winner.name,  # type: ignore[arg-type]
                        index,
                        share,
                        self.to_bb(amount),
                        winner.stack,
                    )
                )
            pot_results.append(
                PotResult(
                    amount=self.to_bb(pot.amount),
                    eligible=pot.eligible,
                    high=tuple((s, self.to_bb(a)) for s, a in high.items()),
                    low=tuple((s, self.to_bb(a)) for s, a in low.items()),
                    uncontested=pot.uncontested,
                )
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
            returned={s: self.to_bb(c) for s, c in returned.items()},
        )

    def _end_hand(self, result: HandResult) -> None:
        """End-of-hand processing (identical in both modes), in order:

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
                if p.status is _LEFT or p.leave_requested:
                    continue
                agent = p.agent
                if agent is not None:  # the default agent does nothing here
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
        self.pending = None
        self._steps = None
        with self.lock:
            self.in_hand = False
            if self.after_hand is not None:
                self.after_hand()
        self.flush()
