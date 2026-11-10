"""Hand: a player's four private hole cards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

from .card import Card, parse_cards

__all__ = ["Hand"]

HAND_SIZE = 4


@dataclass(frozen=True, slots=True)
class Hand:
    """Exactly 4 distinct cards. Input order is preserved."""

    cards: tuple[Card, Card, Card, Card]

    def __init__(self, cards: Iterable[Card]) -> None:
        if isinstance(cards, (str, bytes)):
            raise ValueError("Hand expects Card objects; use Hand.parse() for strings")
        try:
            items = tuple(cards)
        except TypeError:
            raise ValueError(f"Hand expects an iterable of cards, got {cards!r}") from None
        object.__setattr__(self, "cards", items)
        self.__post_init__()

    def __post_init__(self) -> None:
        cards = self.cards
        for c in cards:
            if not isinstance(c, Card):
                raise ValueError(f"not a Card: {c!r}")
        if len(cards) != HAND_SIZE:
            raise ValueError(f"a hand needs exactly {HAND_SIZE} cards, got {len(cards)}")
        if len(set(cards)) != len(cards):
            raise ValueError(f"duplicate cards in hand: {''.join(map(str, cards))}")

    @classmethod
    def parse(cls, s: str) -> "Hand":
        return cls(parse_cards(s))

    def __str__(self) -> str:
        return "".join(str(c) for c in self.cards)

    def __repr__(self) -> str:
        return f"Hand('{self}')"

    def __iter__(self) -> Iterator[Card]:
        return iter(self.cards)

    def __len__(self) -> int:
        return len(self.cards)

    def __contains__(self, card: object) -> bool:
        return card in self.cards
