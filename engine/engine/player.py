"""Player: a seat at the table, driven by an agent through callbacks."""

from __future__ import annotations

import contextlib
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Protocol

from entities import Hand

from .actions import Action, ActionType, IllegalActionError, LegalActions

if TYPE_CHECKING:  # pragma: no cover
    from .dealer import Dealer
    from .environment import Environment
    from .views import Event, HandResult, TableView

__all__ = [
    "CHIPS_PER_BB",
    "to_chips",
    "to_bb",
    "Agent",
    "DefaultAgent",
    "fallback_action",
    "Player",
    "PlayerStatus",
]

CHIPS_PER_BB = 100


def to_chips(amount_bb: float, chips_per_bb: int = CHIPS_PER_BB) -> int:
    """Convert big blinds to whole chips (rounded)."""
    if isinstance(amount_bb, bool) or not isinstance(amount_bb, (int, float)):
        raise TypeError(f"amount must be a number of big blinds, got {amount_bb!r}")
    return int(round(amount_bb * chips_per_bb))


def to_bb(chips: int, chips_per_bb: int = CHIPS_PER_BB) -> float:
    """Convert chips to big blinds."""
    return chips / chips_per_bb


class PlayerStatus(Enum):
    WAITING = "waiting"  # joined, queued for a seat until the current hand is over
    ACTIVE = "active"  # seated with chips; dealt into the next hand
    SITTING_OUT = "sitting_out"  # seated, but busted or did not act on a turn
    LEFT = "left"  # removed from the table (terminal)


class Agent(Protocol):
    """Callbacks an agent implements.

    Optional: `on_seated(player, view)`, `on_sit_out(player)`, `on_return(player)`,
    `on_left(player, final_stack_bb)` and `on_event(event)`.
    """

    def on_turn(self, player: "Player", view: "TableView", legal: LegalActions) -> None:
        """Call exactly one of player.fold/check/call/raise_ (or player.leave())
        before returning; returning without a valid action folds the hand and the
        player sits out from the next hand."""

    def on_hand_end(self, player: "Player", result: "HandResult") -> None:
        """Rebuy window: the agent may call player.rebuy(...), sit_in() or leave()."""


def fallback_action(legal: LegalActions) -> Action:
    """The default agent's play: check if possible, else fold."""
    return Action.check() if legal.can_check else Action.fold()


class DefaultAgent:
    """Used when a player has no agent: check if possible, else fold; never rebuys."""

    def on_turn(self, player: "Player", view: "TableView", legal: LegalActions) -> None:
        player.act(fallback_action(legal))

    def on_hand_end(self, player: "Player", result: "HandResult") -> None:
        pass


class Player:
    """A seat at the table. Amounts in the public API are in big blinds.

    `name` is the player's nick; if None, the environment assigns "Player<N>" when
    the player is seated / joins. `seat` is None unless the player is seated.
    """

    def __init__(
        self, name: Optional[str] = None, stack: float = 0.0, agent: Optional[Any] = None
    ) -> None:
        if name is not None and (not isinstance(name, str) or not name):
            raise ValueError(f"name must be a non-empty string, got {name!r}")
        chips = to_chips(stack, CHIPS_PER_BB)
        if chips < 0:
            raise ValueError(f"stack cannot be negative, got {stack!r}")
        self.name: Optional[str] = name
        self.seat: Optional[int] = None
        self.status = PlayerStatus.WAITING
        self.agent = agent
        self._chips_per_bb = CHIPS_PER_BB
        self._chips = chips
        self._dealer: Optional["Dealer"] = None
        self._env: Optional["Environment"] = None
        self.rebuy_chips = 0  # total chips added through rebuy()
        self.leave_requested = False  # leave() during a hand: removed when it ends

        # Per-hand state (managed by the dealer).
        self.hole_cards: Optional[Hand] = None
        self.street_chips = 0
        self.total_chips = 0
        self.folded = False
        self.all_in = False
        self.has_acted = False
        self.in_hand = False  # dealt into the current/last hand
        self.missed_turn = False  # a turn ended without a valid action this hand
        self._return_requested = False  # rebuy()/sit_in() while sitting out, in the window

        # Turn state.
        self._legal: Optional[LegalActions] = None
        self._pending: Optional[Action] = None

    # --- amounts -----------------------------------------------------------

    @property
    def chips(self) -> int:
        return self._chips

    @property
    def stack(self) -> float:
        return to_bb(self._chips, self._chips_per_bb)

    @property
    def street_bet(self) -> float:
        return to_bb(self.street_chips, self._chips_per_bb)

    @property
    def total_bet(self) -> float:
        return to_bb(self.total_chips, self._chips_per_bb)

    @property
    def rebought(self) -> float:
        return to_bb(self.rebuy_chips, self._chips_per_bb)

    @property
    def seated(self) -> bool:
        return self.status in (PlayerStatus.ACTIVE, PlayerStatus.SITTING_OUT)

    @property
    def sitting_out(self) -> bool:
        return self.status is PlayerStatus.SITTING_OUT

    def set_agent(self, agent: Any) -> None:
        self.agent = agent

    # --- actions (during this player's turn) -------------------------------

    def fold(self) -> None:
        self._record(ActionType.FOLD)

    def check(self) -> None:
        self._record(ActionType.CHECK)

    def call(self) -> None:
        self._record(ActionType.CALL)

    def raise_(self) -> None:
        self._record(ActionType.RAISE)

    def act(self, action: Action) -> None:
        if not isinstance(action, Action):
            raise IllegalActionError(f"not an Action: {action!r}")
        self._record(action.type)

    def _record(self, action_type: ActionType) -> None:
        if self._legal is None:
            raise IllegalActionError(f"it is not {self.name}'s turn")
        if self.leave_requested:
            raise IllegalActionError(f"{self.name} is leaving the table")
        if self._pending is not None:
            raise IllegalActionError(f"{self.name} already acted this turn")
        if not self._legal.allows(action_type):
            raise IllegalActionError(
                f"{action_type.value} is not legal now (legal: "
                f"{', '.join(t.value for t in self._legal.actions)})"
            )
        self._pending = Action(action_type)

    # --- outside betting ---------------------------------------------------

    def _lock(self) -> Any:
        dealer = self._dealer
        return dealer.lock if dealer is not None else contextlib.nullcontext()

    def _check_can_manage_seat(self, what: str) -> None:
        """rebuy()/sit_in() are only for seated players, outside betting."""
        if self.status is PlayerStatus.WAITING:
            raise IllegalActionError(f"cannot {what} while waiting to be seated")
        if self.status is PlayerStatus.LEFT:
            raise IllegalActionError(f"cannot {what}: {self.name} has left the table")
        if self.leave_requested:
            raise IllegalActionError(f"cannot {what}: {self.name} is leaving the table")
        dealer = self._dealer
        if dealer is not None and dealer.in_hand and not dealer.rebuy_window:
            raise IllegalActionError(f"cannot {what} during a hand")

    def rebuy(self, amount_bb: float) -> None:
        """Add `amount_bb` big blinds to the stack.

        Allowed inside `on_hand_end` or between hands. A sitting-out player who
        rebuys becomes active and is dealt into the next hand.
        """
        with self._lock():
            self._check_can_manage_seat("rebuy")
            dealer = self._dealer
            chips = to_chips(amount_bb, self._chips_per_bb)
            if chips <= 0:
                raise ValueError(f"rebuy amount must be positive, got {amount_bb!r}")
            max_buy_in = dealer.max_buy_in if dealer is not None else None
            if max_buy_in is not None and self._chips + chips > to_chips(
                max_buy_in, self._chips_per_bb
            ):
                raise ValueError(
                    f"rebuy of {amount_bb} BB would exceed the max buy-in of {max_buy_in} BB"
                )
            self._chips += chips
            self.rebuy_chips += chips
            if self.status is PlayerStatus.SITTING_OUT and dealer is not None:
                dealer._request_return(self)

    def sit_in(self) -> None:
        """Return to play after sitting out (the player must have chips; a busted
        player returns by rebuying). Allowed inside `on_hand_end` or between hands.
        No-op for an active player."""
        with self._lock():
            self._check_can_manage_seat("sit in")
            if self.status is PlayerStatus.ACTIVE:
                return
            if self._chips <= 0:
                raise IllegalActionError(f"{self.name} has no chips: rebuy to return")
            if self._dealer is not None:
                self._dealer._request_return(self)

    def leave(self) -> None:
        """Leave the table, emptying the seat (see `Environment.leave`). Idempotent."""
        if self.status is PlayerStatus.LEFT:
            return
        if self._env is not None:
            self._env.leave(self)
        elif self._dealer is None:
            self.status = PlayerStatus.LEFT  # never was at a table
        else:
            raise IllegalActionError("leaving needs a table managed by an Environment")

    # --- showdown ----------------------------------------------------------

    def show_cards(self) -> Hand:
        if self.hole_cards is None:
            raise IllegalActionError(f"{self.name} has no cards to show")
        return self.hole_cards

    # --- dealer hooks ------------------------------------------------------

    def _attach(
        self, seat: Optional[int], dealer: "Dealer", env: Optional["Environment"] = None
    ) -> None:
        """Bind the player to a table; `seat` None means waiting to be seated."""
        if self._dealer is not None and self._dealer is not dealer:
            raise ValueError(f"{self.name} is already seated at another table")
        if self.status is PlayerStatus.LEFT:
            raise ValueError(f"{self.name} has left the table")
        if dealer.chips_per_bb != self._chips_per_bb:
            self._chips = to_chips(self._chips / self._chips_per_bb, dealer.chips_per_bb)
            self.rebuy_chips = to_chips(self.rebuy_chips / self._chips_per_bb, dealer.chips_per_bb)
            self._chips_per_bb = dealer.chips_per_bb
        self.seat = seat
        if seat is None:
            self.status = PlayerStatus.WAITING
        else:
            self.status = PlayerStatus.ACTIVE if self._chips > 0 else PlayerStatus.SITTING_OUT
        self._dealer = dealer
        if env is not None:
            self._env = env

    def _reset_for_hand(self, dealt: bool) -> None:
        self.hole_cards = None
        self.street_chips = 0
        self.total_chips = 0
        self.folded = False
        self.all_in = False
        self.has_acted = False
        self.in_hand = dealt
        self.missed_turn = False
        self._return_requested = False
        self._legal = None
        self._pending = None

    def _begin_turn(self, legal: LegalActions) -> None:
        self._legal = legal
        self._pending = None

    def _end_turn(self) -> Optional[Action]:
        action = self._pending
        self._legal = None
        self._pending = None
        return action

    def __repr__(self) -> str:
        return (
            f"Player({self.name!r}, seat={self.seat}, stack={self.stack:g}, "
            f"{self.status.value})"
        )
