"""Deck: a mutable 52-card deck."""

from __future__ import annotations

import random
from typing import Iterable, Optional, Union

from .card import ALL_CARDS, Card
from .hand import HAND_SIZE, Hand

__all__ = ["Deck"]


class Deck:
    """A mutable deck. Cards are dealt from the top (front) of the deck."""

    __slots__ = ("_rng", "_cards")

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self._rng = rng if rng is not None else random.Random()
        self._cards: list[Card] = list(ALL_CARDS)

    # Explicit pickle state (a __slots__ class has no __dict__), so every pickle
    # protocol works. The RNG state travels with the deck, so an unpickled deck
    # shuffles exactly like the original would have.
    def __getstate__(self) -> tuple[random.Random, list[Card]]:
        return (self._rng, list(self._cards))

    def __setstate__(self, state: tuple[random.Random, list[Card]]) -> None:
        rng, cards = state
        self._rng = rng
        self._cards = list(cards)

    def shuffle(self) -> None:
        self._rng.shuffle(self._cards)

    def deal(self, n: int = 1) -> list[Card]:
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError(f"n must be a non-negative int, got {n!r}")
        if n > len(self._cards):
            raise ValueError(f"cannot deal {n} cards, only {len(self._cards)} remain")
        dealt = self._cards[:n]
        del self._cards[:n]
        return dealt

    def deal_one(self) -> Card:
        return self.deal(1)[0]

    def deal_hand(self) -> Hand:
        return Hand(self.deal(HAND_SIZE))

    def remove(self, cards: Union[Card, Iterable[Card]]) -> None:
        """Remove known cards from the deck. All-or-nothing: on error the deck is unchanged."""
        if isinstance(cards, Card):
            cards = (cards,)
        to_remove = list(cards)
        seen: set[Card] = set()
        for c in to_remove:
            if c in seen:
                raise ValueError(f"card listed twice: {c}")
            if c not in self._cards:
                raise ValueError(f"card not in deck: {c}")
            seen.add(c)
        self._cards = [c for c in self._cards if c not in seen]

    def put_on_top(self, cards: Union[Card, Iterable[Card]]) -> None:
        """Stack the deck: move ``cards`` to the top, in the given order.

        The first given card is dealt first; the other cards keep their relative
        order. Every card must still be in the deck and be listed once.
        All-or-nothing: on error the deck is unchanged. ``shuffle()`` and
        ``reset()`` discard the stack.
        """
        if isinstance(cards, Card):
            cards = (cards,)
        stacked = list(cards)
        stacked_set: set[Card] = set()
        present = set(self._cards)
        for c in stacked:
            if not isinstance(c, Card):
                raise TypeError(f"expected a Card, got {type(c).__name__}: {c!r}")
            if c in stacked_set:
                raise ValueError(f"card listed twice: {c}")
            if c not in present:
                raise ValueError(f"card not in deck: {c}")
            stacked_set.add(c)
        if not stacked:
            return
        self._cards = stacked + [c for c in self._cards if c not in stacked_set]

    def remaining(self) -> tuple[Card, ...]:
        """The cards left, in deal order (top first)."""
        return tuple(self._cards)

    def reset(self) -> None:
        """Restore all 52 cards in canonical order. Does not shuffle."""
        self._cards = list(ALL_CARDS)

    def __len__(self) -> int:
        return len(self._cards)

    def __contains__(self, card: object) -> bool:
        return card in self._cards

    def __repr__(self) -> str:
        return f"Deck({len(self._cards)} cards)"
