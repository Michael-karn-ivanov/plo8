"""Step mode types: the pending decision, the hand history window and step-mode
errors.

In step mode (`Environment.start_hand()` / `Environment.step()`) the caller,
not the player's agent, answers each decision of the hand.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterator, Union, overload

from .actions import LegalActions

if TYPE_CHECKING:  # pragma: no cover
    from .events import TableEvent
    from .player import Player

__all__ = ["Decision", "HandHistory", "HandInProgressError", "NoDecisionError"]


class HandInProgressError(RuntimeError):
    """A hand was started (either mode) while another one is in progress."""


class NoDecisionError(RuntimeError):
    """`step()` / `view()` / `hole_cards()` was called while no decision is pending."""


@dataclass(frozen=True)
class Decision:
    """The pending step-mode decision: who is to act and what they may do.

    It carries no view and no cards; `Environment.view()` gives the acting
    player's table view on request. `player` is the `Player` object, for
    convenience (not part of equality or repr, so decisions of two tables
    playing the same hand compare equal).
    """

    hand_number: int
    seat: int
    name: str
    player: "Player" = field(compare=False, repr=False)
    legal: LegalActions


class HandHistory(Sequence):
    """Read-only, live window over the public events of one hand.

    Not a copy: events stamped later in the same hand show up in it. An int
    index gives the event, a slice gives a tuple of those events. There are no
    mutating methods. A caller reads only the new events with a cursor:
    `new = env.history[seen:]; seen = len(env.history)`.
    """

    __slots__ = ("_events",)

    def __init__(self, events: list["TableEvent"]) -> None:
        self._events = events

    def __len__(self) -> int:
        return len(self._events)

    @overload
    def __getitem__(self, index: int) -> "TableEvent": ...

    @overload
    def __getitem__(self, index: slice) -> tuple["TableEvent", ...]: ...

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union["TableEvent", tuple["TableEvent", ...]]:
        if isinstance(index, slice):
            return tuple(self._events[index])
        return self._events[index]

    def __iter__(self) -> Iterator["TableEvent"]:
        return iter(self._events)

    def __repr__(self) -> str:
        return f"HandHistory({len(self._events)} events)"
