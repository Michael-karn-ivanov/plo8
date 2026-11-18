"""Step mode types: the pending decision and step-mode errors.

In step mode (`Environment.start_hand()` / `Environment.step()`) the caller,
not the player's agent, answers each decision of the hand.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .actions import LegalActions

if TYPE_CHECKING:  # pragma: no cover
    from .player import Player

__all__ = ["Decision", "HandInProgressError", "NoDecisionError"]


class HandInProgressError(RuntimeError):
    """A hand was started (either mode) while another one is in progress."""


class NoDecisionError(RuntimeError):
    """`step()` / `view()` was called while no decision is pending."""


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
