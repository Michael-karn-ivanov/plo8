"""Player: a seat at the table, driven by an agent through callbacks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Protocol

from entities import Hand

from .actions import Action, ActionType, IllegalActionError, LegalActions

if TYPE_CHECKING:  # pragma: no cover
    from .dealer import Dealer
    from .views import Event, HandResult, TableView

__all__ = [
    "CHIPS_PER_BB",
    "to_chips",
    "to_bb",
    "Agent",
    "DefaultAgent",
    "fallback_action",
    "Player",
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


class Agent(Protocol):
    """Callbacks an agent implements. `on_event(event)` is optional."""

    def on_turn(self, player: "Player", view: "TableView", legal: LegalActions) -> None:
        """Call exactly one of player.fold/check/call/raise_ before returning."""

    def on_hand_end(self, player: "Player", result: "HandResult") -> None:
        """Rebuy window: the agent may call player.rebuy(...)."""


def fallback_action(legal: LegalActions) -> Action:
    """What happens when an agent takes no (valid) action: check if possible, else fold."""
    return Action.check() if legal.can_check else Action.fold()


class DefaultAgent:
    """Used when a player has no agent: check if possible, else fold."""

    def on_turn(self, player: "Player", view: "TableView", legal: LegalActions) -> None:
        player.act(fallback_action(legal))

    def on_hand_end(self, player: "Player", result: "HandResult") -> None:
        pass


class Player:
    """A seat at the table. Amounts in the public API are in big blinds."""

    def __init__(self, name: str, stack: float = 0.0, agent: Optional[Any] = None) -> None:
        chips = to_chips(stack, CHIPS_PER_BB)
        if chips < 0:
            raise ValueError(f"stack cannot be negative, got {stack!r}")
        self.name = name
        self.seat: Optional[int] = None
        self.agent = agent
        self._chips_per_bb = CHIPS_PER_BB
        self._chips = chips
        self._dealer: Optional["Dealer"] = None
        self.rebuy_chips = 0  # total chips added through rebuy()

        # Per-hand state (managed by the dealer).
        self.hole_cards: Optional[Hand] = None
        self.street_chips = 0
        self.total_chips = 0
        self.folded = False
        self.all_in = False
        self.has_acted = False
        self.in_hand = False  # dealt into the current/last hand

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
        if self._pending is not None:
            raise IllegalActionError(f"{self.name} already acted this turn")
        if not self._legal.allows(action_type):
            raise IllegalActionError(
                f"{action_type.value} is not legal now (legal: "
                f"{', '.join(t.value for t in self._legal.actions)})"
            )
        self._pending = Action(action_type)

    # --- between hands -----------------------------------------------------

    def rebuy(self, amount_bb: float) -> None:
        """Add `amount_bb` big blinds to the stack. Only allowed between hands."""
        if self._dealer is not None and self._dealer.in_hand:
            raise IllegalActionError("cannot rebuy during a hand")
        chips = to_chips(amount_bb, self._chips_per_bb)
        if chips <= 0:
            raise ValueError(f"rebuy amount must be positive, got {amount_bb!r}")
        max_buy_in = self._dealer.max_buy_in if self._dealer is not None else None
        if max_buy_in is not None and self._chips + chips > to_chips(max_buy_in, self._chips_per_bb):
            raise ValueError(
                f"rebuy of {amount_bb} BB would exceed the max buy-in of {max_buy_in} BB"
            )
        self._chips += chips
        self.rebuy_chips += chips

    # --- showdown ----------------------------------------------------------

    def show_cards(self) -> Hand:
        if self.hole_cards is None:
            raise IllegalActionError(f"{self.name} has no cards to show")
        return self.hole_cards

    # --- dealer hooks ------------------------------------------------------

    def _attach(self, seat: int, dealer: "Dealer") -> None:
        if self._dealer is not None and self._dealer is not dealer:
            raise ValueError(f"{self.name} is already seated at another table")
        if dealer.chips_per_bb != self._chips_per_bb:
            self._chips = to_chips(self._chips / self._chips_per_bb, dealer.chips_per_bb)
            self.rebuy_chips = to_chips(self.rebuy_chips / self._chips_per_bb, dealer.chips_per_bb)
            self._chips_per_bb = dealer.chips_per_bb
        self.seat = seat
        self._dealer = dealer

    def _reset_for_hand(self, dealt: bool) -> None:
        self.hole_cards = None
        self.street_chips = 0
        self.total_chips = 0
        self.folded = False
        self.all_in = False
        self.has_acted = False
        self.in_hand = dealt
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
        return f"Player({self.name!r}, seat={self.seat}, stack={self.stack:g})"
