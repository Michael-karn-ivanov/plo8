"""BotAgent: any arena bot (any rl_env `Controller`) as an engine `Agent`.

On each turn it builds a one-row batch (`DecisionBatch.single`: the same
view, legal actions and mask a `TableBatch` row would have), asks the bot,
maps the action to the engine (`to_engine`, which raises on a masked action)
and submits it to the player. At the end of a hand it may rebuy up to a
configured stack (off by default). It has no `on_hole_cards` / `on_event`, so
the engine builds no views for them.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from engine import ActionType, HandResult, LegalActions, Player, TableView
from rl_env import DecisionBatch, to_engine

__all__ = ["BotAgent"]

_log = logging.getLogger("arena.agent")


class BotAgent:
    """Engine agent around a bot (see the module docstring)."""

    def __init__(self, bot: Any, *, rebuy_to_bb: Optional[float] = None) -> None:
        if rebuy_to_bb is not None and not rebuy_to_bb > 0:
            raise ValueError(f"rebuy_to_bb must be positive, got {rebuy_to_bb!r}")
        self.bot = bot
        self.rebuy_to_bb = rebuy_to_bb

    def decide(self, view: TableView, legal: LegalActions) -> ActionType:
        """The engine action the bot chooses for this view and legal actions."""
        batch = DecisionBatch.single(view, legal)
        out = self.bot.act(batch)
        actions = out[0] if isinstance(out, tuple) else out
        return to_engine(int(actions[0]), legal)

    def on_turn(self, player: Player, view: TableView, legal: LegalActions) -> None:
        t = self.decide(view, legal)
        if t is ActionType.FOLD:
            player.fold()
        elif t is ActionType.CHECK:
            player.check()
        elif t is ActionType.CALL:
            player.call()
        else:
            player.raise_()

    def on_hand_end(self, player: Player, result: HandResult) -> None:
        target = self.rebuy_to_bb
        if target is None:
            return
        missing = round(target - player.stack, 2)
        if missing <= 0:
            return
        try:
            player.rebuy(missing)
        except Exception:  # noqa: BLE001 - e.g. above the table's max buy-in
            _log.warning("rebuy of %s BB for %s failed", missing, player.name, exc_info=True)
