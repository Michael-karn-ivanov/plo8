"""Test helpers: card parsing, stacked decks, a table stopped at a decision,
and a mask-checking controller wrapper."""

from __future__ import annotations

import random
from typing import Optional, Sequence

import numpy as np

from entities import Board, Card, Deck, Hand
from entities.card import parse_cards
from rl_env import Table


def cards(s: str) -> tuple[Card, ...]:
    """cards("As 2d 7h") -> (Card('As'), Card('2d'), Card('7h'))."""
    return parse_cards(s)


def hand(s: str) -> Hand:
    return Hand(cards(s))


def board(s: str) -> Board:
    return Board.parse(s)


class StackedDeck(Deck):
    """A deck whose top, after every `shuffle()`, is `top` (the engine calls
    reset() + shuffle() once per hand and then only deal_one())."""

    def __init__(self, top: Sequence[Card], rng: Optional[random.Random] = None) -> None:
        super().__init__(rng if rng is not None else random.Random(0))
        self.top = list(top)

    def shuffle(self) -> None:
        super().shuffle()
        self.put_on_top(self.top)


def deal_script(holes_by_seat: Sequence[str], board_cards: str, button: int) -> list[Card]:
    """Deck top that deals `holes_by_seat[s]` to seat `s` (engine deal order:
    four rounds from the left of the button, the button last), then the board."""
    n = len(holes_by_seat)
    holes = [cards(h) for h in holes_by_seat]
    for s, h in enumerate(holes):
        assert len(h) == 4, f"seat {s} needs 4 hole cards"
    script = []
    for r in range(4):
        for i in range(n):
            script.append(holes[(button + 1 + i) % n][r])
    script.extend(cards(board_cards))
    return script


def table_at_decision(
    holes_by_seat: Sequence[str],
    board_cards: str,
    actions: Sequence[int] = (),
    *,
    button: int = 0,
    stacks: Optional[Sequence[float]] = None,
) -> Table:
    """An rl_env `Table` with a stacked deck, after the scripted masked
    `actions`; a decision is pending."""
    n = len(holes_by_seat)
    stacks = tuple(stacks) if stacks is not None else (100.0,) * n
    deck = StackedDeck(deal_script(holes_by_seat, board_cards, button))
    table = Table()
    table.start_hand(n, stacks, button, 0, deck=deck)
    for a in actions:
        assert table.decision_pending, "the hand ended before the actions did"
        table.act(a)
    assert table.decision_pending, "no decision pending after the actions"
    return table


class MaskChecking:
    """Wraps a controller; asserts every returned action is allowed by its
    row's mask (before `submit`, which checks again) and counts decisions,
    actions and views."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.wants_views = getattr(inner, "wants_views", False)
        self.wants_obs = getattr(inner, "wants_obs", True)
        self.decisions = 0
        self.views_seen = 0
        self.counts = np.zeros(3, dtype=np.int64)
        self.log: list[tuple[tuple[bool, bool, bool], int]] = []

    def act(self, batch):
        out = self.inner.act(batch)
        actions = np.asarray(out[0] if isinstance(out, tuple) else out)
        assert actions.shape == (len(batch),), actions.shape
        for r in range(len(batch)):
            a = int(actions[r])
            row = tuple(bool(x) for x in batch.mask[r])
            assert 0 <= a <= 2 and row[a], f"masked action {a} for mask {row}"
            self.counts[a] += 1
            self.log.append((row, a))
        self.decisions += len(batch)
        if batch.views is not None:
            self.views_seen += len(batch.views)
        return out
