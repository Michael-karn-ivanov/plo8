"""Rank, Suit and Card value types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum

__all__ = ["Rank", "Suit", "Card", "ALL_CARDS", "parse_cards"]


class Rank(IntEnum):
    """Card rank. Integer order is high-hand order (deuce lowest, ace highest)."""

    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    @property
    def symbol(self) -> str:
        return _RANK_TO_SYMBOL[self]

    @property
    def low_value(self) -> int:
        """Value for low hands: ace is 1, every other rank is its integer value."""
        return 1 if self is Rank.ACE else int(self)

    @classmethod
    def from_symbol(cls, s: str) -> "Rank":
        if not isinstance(s, str):
            raise ValueError(f"rank symbol must be a string, got {s!r}")
        try:
            return _SYMBOL_TO_RANK[s.upper()]
        except KeyError:
            raise ValueError(f"invalid rank symbol: {s!r}") from None


_RANK_TO_SYMBOL: dict[Rank, str] = {
    **{Rank(v): str(v) for v in range(2, 10)},
    Rank.TEN: "T",
    Rank.JACK: "J",
    Rank.QUEEN: "Q",
    Rank.KING: "K",
    Rank.ACE: "A",
}
_SYMBOL_TO_RANK: dict[str, Rank] = {sym: r for r, sym in _RANK_TO_SYMBOL.items()}
_SYMBOL_TO_RANK["10"] = Rank.TEN


class Suit(Enum):
    """Card suit. Suits have no poker ordering; `index` only fixes the encoding."""

    CLUBS = "c"
    DIAMONDS = "d"
    HEARTS = "h"
    SPADES = "s"

    @property
    def symbol(self) -> str:
        return self.value

    @property
    def index(self) -> int:
        return _SUIT_TO_INDEX[self]

    @classmethod
    def from_symbol(cls, s: str) -> "Suit":
        if not isinstance(s, str):
            raise ValueError(f"suit symbol must be a string, got {s!r}")
        try:
            return cls(s.lower())
        except ValueError:
            raise ValueError(f"invalid suit symbol: {s!r}") from None


_SUIT_ORDER: tuple[Suit, ...] = (Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES)
_SUIT_TO_INDEX: dict[Suit, int] = {s: i for i, s in enumerate(_SUIT_ORDER)}


@dataclass(frozen=True, slots=True, order=True)
class Card:
    """An immutable playing card.

    Equality, hashing and ordering use the `(rank, suit_index)` key. Ordering is
    only for deterministic display and sorting, not a poker comparison.
    """

    rank: Rank = field(compare=False)
    suit: Suit = field(compare=False)
    _key: tuple[int, int] = field(init=False, repr=False, compare=True)

    def __post_init__(self) -> None:
        rank = self.rank
        if not isinstance(rank, Rank):
            if isinstance(rank, int) and not isinstance(rank, bool):
                try:
                    rank = Rank(rank)
                except ValueError:
                    raise ValueError(f"invalid rank: {self.rank!r}") from None
            else:
                raise ValueError(f"invalid rank: {self.rank!r}")
            object.__setattr__(self, "rank", rank)
        if not isinstance(self.suit, Suit):
            raise ValueError(f"invalid suit: {self.suit!r}")
        object.__setattr__(self, "_key", (int(rank), self.suit.index))

    @property
    def index(self) -> int:
        """Stable integer encoding in 0..51."""
        return (int(self.rank) - 2) * 4 + self.suit.index

    @classmethod
    def from_index(cls, i: int) -> "Card":
        if isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < 52:
            raise ValueError(f"card index must be an int in 0..51, got {i!r}")
        return ALL_CARDS[i]

    @classmethod
    def parse(cls, s: str) -> "Card":
        """Parse a card such as "As", "td" or "10s"."""
        if not isinstance(s, str):
            raise ValueError(f"card string must be a string, got {s!r}")
        text = s.strip()
        if len(text) not in (2, 3):
            raise ValueError(f"invalid card string: {s!r}")
        rank = Rank.from_symbol(text[:-1])
        suit = Suit.from_symbol(text[-1])
        return ALL_CARDS[(int(rank) - 2) * 4 + suit.index]

    from_str = parse

    def __str__(self) -> str:
        return self.rank.symbol + self.suit.symbol

    def __repr__(self) -> str:
        return f"Card('{self}')"


ALL_CARDS: tuple[Card, ...] = tuple(
    Card(Rank(r), s) for r in range(2, 15) for s in _SUIT_ORDER
)


def parse_cards(s: str) -> tuple[Card, ...]:
    """Parse a run of cards such as "AsKd7h2c" or "Ah 7d 10c". Whitespace is ignored."""
    if not isinstance(s, str):
        raise ValueError(f"cards string must be a string, got {s!r}")
    text = "".join(s.split())
    cards: list[Card] = []
    i = 0
    while i < len(text):
        width = 3 if text.startswith("10", i) else 2
        token = text[i : i + width]
        if len(token) != width:
            raise ValueError(f"invalid cards string: {s!r}")
        cards.append(Card.parse(token))
        i += width
    return tuple(cards)
