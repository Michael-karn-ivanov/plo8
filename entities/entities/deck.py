"""Deck: a mutable 52-card deck."""

from __future__ import annotations

import random
from typing import Iterable, Optional, Union

from .card import ALL_CARDS, Card
from .hand import HAND_SIZE, Hand

__all__ = ["Deck"]

# Canonical order, bottom first: the top of a fresh deck (ALL_CARDS[0]) is last.
_REVERSED_ALL_CARDS: tuple[Card, ...] = ALL_CARDS[::-1]


class Deck:
    """A mutable deck. Cards are dealt from the top of the deck.

    Storage: ``_pile`` is kept **bottom first**, so the top card (the next one
    dealt) is ``_pile[-1]`` and dealing one card is an O(1) ``list.pop()``. All
    public methods use top-first order.
    """

    __slots__ = ("_rng", "_pile")

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self._rng = rng if rng is not None else random.Random()
        self._pile: list[Card] = list(_REVERSED_ALL_CARDS)

    # Explicit pickle state (a __slots__ class has no __dict__), so every pickle
    # protocol works. The RNG state travels with the deck, so an unpickled deck
    # shuffles exactly like the original would have. The card list in the state
    # is top first, the format used before storage became bottom first.
    def __getstate__(self) -> tuple[random.Random, list[Card]]:
        return (self._rng, self._pile[::-1])

    def __setstate__(self, state: tuple[random.Random, list[Card]]) -> None:
        rng, cards = state
        self._rng = rng
        pile = list(cards)
        pile.reverse()
        self._pile = pile

    def shuffle(self) -> None:
        # Shuffle the top-first list (same RNG calls and permutation as a
        # top-first deck), then store it bottom first.
        cards = self._pile[::-1]
        self._rng.shuffle(cards)
        cards.reverse()
        self._pile = cards

    def deal(self, n: int = 1) -> list[Card]:
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError(f"n must be a non-negative int, got {n!r}")
        pile = self._pile
        if n > len(pile):
            raise ValueError(f"cannot deal {n} cards, only {len(pile)} remain")
        if n == 0:
            return []  # pile[-0:] would be the whole pile
        dealt = pile[-n:]
        del pile[-n:]
        dealt.reverse()
        return dealt

    def deal_one(self) -> Card:
        try:
            return self._pile.pop()
        except IndexError:
            raise ValueError("cannot deal 1 cards, only 0 remain") from None

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
            if c not in self._pile:
                raise ValueError(f"card not in deck: {c}")
            seen.add(c)
        self._pile = [c for c in self._pile if c not in seen]

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
        present = set(self._pile)
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
        rest = [c for c in self._pile if c not in stacked_set]
        rest.extend(reversed(stacked))
        self._pile = rest

    def remaining(self) -> tuple[Card, ...]:
        """The cards left, in deal order (top first)."""
        return tuple(reversed(self._pile))

    def reset(self) -> None:
        """Restore all 52 cards in canonical order. Does not shuffle."""
        self._pile = list(_REVERSED_ALL_CARDS)

    def __len__(self) -> int:
        return len(self._pile)

    def __contains__(self, card: object) -> bool:
        return card in self._pile

    def __repr__(self) -> str:
        return f"Deck({len(self._pile)} cards)"
