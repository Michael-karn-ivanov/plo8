"""Test helpers: stacked decks, random specs and play, a view-reading controller."""

from __future__ import annotations

import random
from typing import Callable, Optional, Sequence

import numpy as np

from entities import Card, Deck
from rl_env import BET_RAISE, CHECK_CALL, FOLD, HandSpec, Table

ALL_CARDS = [Card.from_index(i) for i in range(52)]


class StackedDeck(Deck):
    """A deck whose top, after every `shuffle()`, is `top` (entities
    `put_on_top`; the engine calls reset() + shuffle() once per hand and then
    only deal_one(), see the engine's deal-order contract)."""

    def __init__(self, top: Sequence[Card], rng: Optional[random.Random] = None) -> None:
        super().__init__(rng if rng is not None else random.Random(0))
        self.top = list(top)

    def shuffle(self) -> None:
        super().shuffle()
        self.put_on_top(self.top)


def deal_script(holes: Sequence[Sequence[Card]], board: Sequence[Card]) -> list[Card]:
    """Deck top for `holes[i]` = hole cards of participant i counted from the
    left of the button (i = n-1 is the button), then the board (up to 5)."""
    n = len(holes)
    script = [holes[i][r] for r in range(4) for i in range(n)]
    script.extend(board)
    return script


def seat_of(n: int, button: int, i: int) -> int:
    """Seat of participant i counted from the left of the button."""
    return (button + 1 + i) % n


def random_stacks(rng: random.Random, n: int) -> tuple[float, ...]:
    """Deep, mixed and very short stacks (all-ins on the blinds, side pots)."""
    r = rng.random()
    if r < 0.35:
        return (100.0,) * n
    out = []
    for _ in range(n):
        u = rng.random()
        if u < 0.2:
            out.append(rng.choice([0.5, 0.75, 1.0, 1.5, 2.0, 3.25]))
        elif u < 0.5:
            out.append(round(rng.uniform(2.0, 15.0), 2))
        else:
            out.append(round(rng.uniform(15.0, 150.0), 2))
    return tuple(out)


def random_spec(rng: random.Random, n: Optional[int] = None) -> HandSpec:
    n = n if n is not None else rng.randint(2, 10)
    return HandSpec(n, random_stacks(rng, n), rng.randrange(n), rng.getrandbits(63))


def random_action(rng: random.Random, mask) -> int:
    return rng.choice([a for a in (FOLD, CHECK_CALL, BET_RAISE) if mask[a]])


def play_random(
    table: Table,
    spec: HandSpec,
    rng: random.Random,
    on_decision: Optional[Callable[[Table], None]] = None,
    deck: Optional[Deck] = None,
) -> list[int]:
    """Play one hand with random masked actions; returns the actions."""
    actions = []
    table.start_hand(spec.n_seats, spec.stacks_bb, spec.button, spec.deck_seed, deck=deck)
    while table.decision_pending:
        if on_decision is not None:
            on_decision(table)
        a = random_action(rng, table.mask)
        actions.append(a)
        table.act(a)
    return actions


def replay(
    table: Table,
    spec: HandSpec,
    actions: Sequence[int],
    on_decision: Optional[Callable[[Table], None]] = None,
    deck: Optional[Deck] = None,
) -> None:
    """Play one hand with the given masked actions."""
    table.start_hand(spec.n_seats, spec.stacks_bb, spec.button, spec.deck_seed, deck=deck)
    for a in actions:
        assert table.decision_pending, "the hand ended before the actions did"
        if on_decision is not None:
            on_decision(table)
        table.act(a)
    assert not table.decision_pending, "the actions ended before the hand did"


def passive(table: Table) -> None:
    """Check / call to the end of the hand."""
    while table.decision_pending:
        table.act(CHECK_CALL)


class CheckCallController:
    """A tiny controller that reads `views` and `legal` (proves the view path of
    the interface): always check / call, after checking the view it got."""

    wants_views = True

    def __init__(self) -> None:
        self.views_seen = 0

    def act(self, batch) -> np.ndarray:
        assert batch.views is not None and len(batch.views) == len(batch.legal) == len(batch)
        out = []
        for r, (view, legal) in enumerate(zip(batch.views, batch.legal)):
            assert view.seat == int(batch.seats[r])
            assert view.hole_cards is not None
            assert view.to_call == view.to_call  # readable
            self.views_seen += 1
            assert legal.can_check or legal.call_amount > 0
            out.append(CHECK_CALL)
        return np.array(out, dtype=np.int64)
