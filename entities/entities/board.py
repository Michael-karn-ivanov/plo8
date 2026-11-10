"""Board: the shared community cards (flop, turn, river)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Optional

from .card import Card, parse_cards

__all__ = ["Board", "Street"]


class Street(Enum):
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3


@dataclass(frozen=True, slots=True)
class Board:
    flop: Optional[tuple[Card, Card, Card]] = None
    turn: Optional[Card] = None
    river: Optional[Card] = None

    def __post_init__(self) -> None:
        if self.flop is not None:
            if isinstance(self.flop, (str, bytes)):
                raise ValueError("flop must be Card objects; use Board.parse() for strings")
            try:
                flop = tuple(self.flop)
            except TypeError:
                raise ValueError(f"flop must be an iterable of cards, got {self.flop!r}") from None
            if len(flop) != 3:
                raise ValueError(f"flop needs exactly 3 cards, got {len(flop)}")
            for c in flop:
                if not isinstance(c, Card):
                    raise ValueError(f"not a Card: {c!r}")
            object.__setattr__(self, "flop", flop)
        if self.turn is not None:
            if not isinstance(self.turn, Card):
                raise ValueError(f"turn is not a Card: {self.turn!r}")
            if self.flop is None:
                raise ValueError("a turn requires a flop")
        if self.river is not None:
            if not isinstance(self.river, Card):
                raise ValueError(f"river is not a Card: {self.river!r}")
            if self.turn is None:
                raise ValueError("a river requires a turn")
        cards = self.cards
        if len(set(cards)) != len(cards):
            raise ValueError(f"duplicate cards on board: {''.join(map(str, cards))}")

    # --- stage transitions -------------------------------------------------

    @classmethod
    def empty(cls) -> "Board":
        return cls()

    def with_flop(self, c1: Card, c2: Card, c3: Card) -> "Board":
        if self.street is not Street.PREFLOP:
            raise ValueError(f"cannot deal the flop on {self.street.name}")
        return Board(flop=(c1, c2, c3))

    def with_turn(self, c: Card) -> "Board":
        if self.street is not Street.FLOP:
            raise ValueError(f"cannot deal the turn on {self.street.name}")
        return Board(flop=self.flop, turn=c)

    def with_river(self, c: Card) -> "Board":
        if self.street is not Street.TURN:
            raise ValueError(f"cannot deal the river on {self.street.name}")
        return Board(flop=self.flop, turn=self.turn, river=c)

    # --- views -------------------------------------------------------------

    @property
    def cards(self) -> tuple[Card, ...]:
        out: tuple[Card, ...] = self.flop or ()
        if self.turn is not None:
            out += (self.turn,)
        if self.river is not None:
            out += (self.river,)
        return out

    @property
    def street(self) -> Street:
        if self.river is not None:
            return Street.RIVER
        if self.turn is not None:
            return Street.TURN
        if self.flop is not None:
            return Street.FLOP
        return Street.PREFLOP

    @classmethod
    def parse(cls, s: str) -> "Board":
        """Parse e.g. "Ah7d2c 9s Td", "Ah7d2c" or ""."""
        cards = parse_cards(s)
        n = len(cards)
        if n == 0:
            return cls()
        if n not in (3, 4, 5):
            raise ValueError(f"a board has 0, 3, 4 or 5 cards, got {n}: {s!r}")
        return cls(
            flop=cards[:3],
            turn=cards[3] if n >= 4 else None,
            river=cards[4] if n == 5 else None,
        )

    def __str__(self) -> str:
        if self.flop is None:
            return ""
        parts = ["".join(str(c) for c in self.flop)]
        if self.turn is not None:
            parts.append(str(self.turn))
        if self.river is not None:
            parts.append(str(self.river))
        return " ".join(parts)

    def __repr__(self) -> str:
        return f"Board('{self}')"

    def __iter__(self) -> Iterator[Card]:
        return iter(self.cards)

    def __len__(self) -> int:
        return len(self.cards)

    def __contains__(self, card: object) -> bool:
        return card in self.cards
