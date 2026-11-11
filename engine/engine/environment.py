"""Environment: a table of players with a dealer."""

from __future__ import annotations

import random
from typing import Iterable, Optional

from entities import Deck

from .dealer import Dealer
from .player import CHIPS_PER_BB, Player
from .views import HandResult

__all__ = ["Environment", "MIN_PLAYERS", "MAX_PLAYERS"]

MIN_PLAYERS = 2
MAX_PLAYERS = 10


class Environment:
    """Players (seat = index in `players`), a dealer and the starting button seat."""

    def __init__(
        self,
        players: Iterable[Player],
        button: int = 0,
        chips_per_bb: int = CHIPS_PER_BB,
        seed: Optional[int] = None,
        max_buy_in: Optional[float] = None,
        deck: Optional[Deck] = None,
    ) -> None:
        players = list(players)
        if not MIN_PLAYERS <= len(players) <= MAX_PLAYERS:
            raise ValueError(
                f"a table seats {MIN_PLAYERS}-{MAX_PLAYERS} players, got {len(players)}"
            )
        self.players = players
        self.rng = random.Random(seed)
        self.dealer = Dealer(
            players,
            rng=self.rng,
            button=button,
            chips_per_bb=chips_per_bb,
            max_buy_in=max_buy_in,
            deck=deck,
        )

    @property
    def button(self) -> int:
        return self.dealer.button

    @property
    def chips_per_bb(self) -> int:
        return self.dealer.chips_per_bb

    def can_start(self) -> bool:
        return len(self.dealer.players_with_chips()) >= MIN_PLAYERS

    def start_round(self) -> HandResult:
        """Start and play a new hand.

        The deck is reset and shuffled, hole cards and per-hand player state are
        cleared, the board and pots are emptied, the SB and BB players post the
        blinds, and the dealer runs the hand to the end.
        """
        return self.dealer.play_hand()

    def run(self, n_hands: int) -> list[HandResult]:
        """Play up to `n_hands`; stops early if fewer than two players have chips."""
        results = []
        for _ in range(n_hands):
            if not self.can_start():
                break
            results.append(self.start_round())
        return results
