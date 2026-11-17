"""Engine seating of the case players, and a deck stacked with the case's cards.

Seating: the engine numbers seats clockwise from 0, without gaps. The case
players are ordered clockwise by their case seat, starting with the button
player, and the k-th of them (k = 0 is the button) gets engine seat k. The
driver makes the joins land there: the button player joins an empty table and
takes the button, the second sits right after them, and the k-th (k >= 2) joins
with offset k ("right of the button" with k players seated).

Deal order (engine contract, see `engine.dealer`): four rounds of one hole card
to each participant, starting left of the button (engine seats 1, 2, ..., n-1,
0), then flop (3), turn (1), river (1); no burn cards.
"""

from __future__ import annotations

import random
from typing import Optional, Sequence

from entities import Card, Deck

from .case import Case, CasePlayer, clockwise_ids

__all__ = ["StackedDeck", "seating", "engine_seats", "join_offset", "deal_script"]


def seating(case: Case) -> list[CasePlayer]:
    """Case players in engine seat order (index = engine seat, 0 = the button)."""
    return [case.player(pid) for pid in clockwise_ids(case)]


def engine_seats(case: Case) -> dict[str, int]:
    """Player id -> engine seat."""
    return {p.id: k for k, p in enumerate(seating(case))}


def join_offset(k: int) -> Optional[int]:
    """The `seat` offset the k-th joiner (0-based) sends: none for the first two
    (the engine ignores it at a table with fewer than two active players), k
    afterwards (right of the button, i.e. the end of the clockwise order)."""
    return None if k < 2 else k


def deal_script(case: Case) -> list[Optional[Card]]:
    """The cards in the order the dealer takes them from the deck; None where
    the case does not fix the card (unknown hole cards, streets not reached)."""
    order = seating(case)
    n = len(order)
    deal_order = order[1:] + order[:1]  # left of the button first, the button last
    holes = {p.id: case.hole_of(p.id) for p in order}
    script: list[Optional[Card]] = []
    for r in range(4):
        for p in deal_order:
            hole = holes[p.id]
            script.append(hole[r] if hole is not None else None)
    for street, size in (("flop", 3), ("turn", 1), ("river", 1)):
        board = case.board_of(street)
        if board is None:
            script.extend([None] * size)
        else:
            script.extend(board)
    assert len(script) == 4 * n + 5
    return script


class StackedDeck(Deck):
    """A deck whose top, after every `shuffle()`, is `script`.

    `None` slots of the script are filled with the next shuffled cards the
    script does not use, so unknown cards are random (seeded through `rng`) but
    never duplicate a known card. Every shuffle (one per hand) restacks.
    """

    def __init__(self, script: Sequence[Optional[Card]], rng: Optional[random.Random] = None) -> None:
        super().__init__(rng)
        script = list(script)
        if len(script) > 52:
            raise ValueError(f"a script has at most 52 cards, got {len(script)}")
        known = [c for c in script if c is not None]
        for c in known:
            if not isinstance(c, Card):
                raise TypeError(f"expected a Card or None, got {c!r}")
        if len(set(known)) != len(known):
            raise ValueError("the script lists a card twice")
        self.script: tuple[Optional[Card], ...] = tuple(script)
        self.shuffles = 0  # how many times the deck was shuffled (one per hand)

    def shuffle(self) -> None:
        super().shuffle()
        known = {c for c in self.script if c is not None}
        filler = iter([c for c in self.remaining() if c not in known])
        top = [c if c is not None else next(filler) for c in self.script]
        self.put_on_top(top)
        self.shuffles += 1
