"""Players: who decides a turn.

The networking, the turn handling and the hand-end handling only see a
`Player` (`label`, `decide(view, legal) -> ActionType`). `--bot NAME` makes a
`BotPlayer`; a later `--model PATH` adds another branch in `make_player` (an
`OnnxPlayer`) and nothing else.
"""

from __future__ import annotations

from typing import Any, Protocol

import arena
from engine import ActionType, LegalActions, TableView

__all__ = ["Player", "BotPlayer", "make_player"]


class Player(Protocol):
    label: str  # for logs: "rule", later e.g. "model x.onnx"

    def decide(self, view: TableView, legal: LegalActions) -> ActionType: ...


class BotPlayer:
    """An arena bot, deciding exactly as behind arena's `BotAgent`
    (`DecisionBatch.single` -> `bot.act` -> `to_engine`). One instance lives for
    the whole process, so a seeded RandomBot makes the same choices for the same
    sequence of turns. `BotAgent`'s own rebuy is not used (`rebuy_to_bb=None`):
    rebuys go through the `hand_end` reply."""

    def __init__(self, bot_name: str, seed: int = 0) -> None:
        self.label = bot_name
        self.agent = arena.BotAgent(arena.make_bot(bot_name, seed))

    def decide(self, view: TableView, legal: LegalActions) -> ActionType:
        return self.agent.decide(view, legal)

    def __repr__(self) -> str:
        return f"BotPlayer({self.label!r}, {self.agent.bot!r})"


def make_player(args: Any) -> Player:
    """The player chosen on the command line (`--bot` now; `--model` later)."""
    return BotPlayer(args.bot, args.seed)
