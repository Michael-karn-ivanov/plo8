"""The observation vector (`ENCODER_VERSION = 1`, `OBS_DIM = 554`).

The observation is a function of the acting player's public information only:
the public events of the hand so far (folded into a `HandTracker`), the
player's own hole cards and its `LegalActions`. Two entry points give the same
bytes:

- `encode_view(view, legal)` - from an engine `TableView` (a bot at a real
  table, e.g. a view decoded from the cli_server protocol): a fresh tracker over
  `view.history`, `view.seat` and `view.hole_cards`; nothing else of the view;
- `Table.observation()` - training: the table's incremental tracker, the acting
  seat's own cards from the engine and `legal` from the decision.

Both call `collect` / `encode` below. This module never imports the critic
input (`critic.py`), and nothing here can see other players' hole cards.

Every amount is in big bets (1 big bet = 2 BB) and written as `min(x / S, 2)`;
counts are `min(c, cap) / cap`; one-hots are 0/1. The exact layout is in
plan.md (Observation -> Layout) and in `OBS_LAYOUT`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from engine import LegalActions, TableView

from .hints import hole_hints, street_hints
from .tracker import HandTracker

__all__ = [
    "ENCODER_VERSION",
    "OBS_DIM",
    "OBS_LAYOUT",
    "OBS_BLOCKS",
    "MAX_SLOTS",
    "SLOT_SIZE",
    "RECENT_SIZE",
    "collect",
    "encode",
    "encode_view",
]

ENCODER_VERSION = 1

# (block, offset, size), contiguous, in vector order.
OBS_LAYOUT: tuple[tuple[str, int, int], ...] = (
    ("hole_cards", 0, 52),
    ("flop", 52, 52),
    ("turn", 104, 52),
    ("river", 156, 52),
    ("street", 208, 4),
    ("players_dealt_in", 212, 9),
    ("position", 221, 10),
    ("betting_state", 231, 17),
    ("own_action_history", 248, 6),
    ("other_seats", 254, 135),
    ("recent_actions", 389, 144),
    ("hints", 533, 21),
)
OBS_BLOCKS: dict[str, tuple[int, int]] = {name: (off, size) for name, off, size in OBS_LAYOUT}
OBS_DIM = 554

MAX_SLOTS = 9  # other seats (slots 1..9)
SLOT_SIZE = 15
RECENT_SIZE = 18

_HOLE = 0
_FLOP = 52
_TURN = 104
_RIVER = 156
_STREET = 208
_DEALT = 212
_POSITION = 221
# Betting state.
_POT = 231
_TO_CALL = 232
_POT_ODDS = 233
_STREET_BETS = 234
_OWN_TOTAL = 239
_OWN_STREET = 240
_LIVE = 241
_CAN_ACT = 242
_OWN_STACK = 243
_CAN_RAISE = 244
_CAN_CHECK = 245
_CAP_LIFTED = 246
_RAISES_LEFT = 247
# Own action history.
_OWN_BETS = 248
_OWN_AGGRESSOR = 252
_OWN_PREV_AGGRESSOR = 253
_OTHERS = 254
_RECENT = 389
_HINTS = 533

# Scales, applied to amounts in big bets.
_BB = 0.5  # big blinds -> big bets
_POT_SCALE = _BB / 25.0  # pot, chips in the pot (total): 2.0 = 50 big bets
_STACK_SCALE = _BB / 50.0  # stacks: 150 BB = 1.5
_STREET_SCALE = _BB / 2.0  # street bets, amount to call
_CLIP = 2.0


def collect(
    tr: HandTracker,
    me_seat: int,
    hole_cards,
    legal: LegalActions,
    cols: list,
    vals: list,
) -> None:
    """Append the non-zero (column, value) pairs of the acting player's
    observation to `cols` / `vals` (no column is written twice)."""
    me = tr.index[me_seat]
    n = tr.n
    street = tr.street
    cache = tr.cache
    col = cols.append
    val = vals.append

    # Static per hand and seat: hole cards, players dealt in, position, hole hints.
    static = cache.get((me, -1))
    if static is None:
        static = [(_HOLE + c.index, 1.0) for c in hole_cards]
        static.append((_DEALT + n - 2, 1.0))
        static.append((_POSITION + (me - tr.button) % n, 1.0))
        static.extend((_HINTS + off, v) for off, v in hole_hints(hole_cards))
        cache[(me, -1)] = static
    for c, v in static:
        col(c)
        val(v)

    # Board, street.
    board_cards = tr.board_cards
    for base, s in ((_FLOP, 1), (_TURN, 2), (_RIVER, 3)):
        for ci in board_cards[s]:
            col(base + ci)
            val(1.0)
    col(_STREET + street)
    val(1.0)

    # Betting state.
    pot = tr.pot
    x = pot * _POT_SCALE
    if x > 0.0:
        col(_POT)
        val(x if x < _CLIP else _CLIP)
    to_call = legal.call_amount
    if to_call > 0:
        x = to_call * _STREET_SCALE
        col(_TO_CALL)
        val(x if x < _CLIP else _CLIP)
        col(_POT_ODDS)
        val(to_call / (pot + to_call))
    nb = tr.street_bets
    col(_STREET_BETS + (nb if nb < 4 else 4))
    val(1.0)
    x = tr.total[me] * _POT_SCALE
    if x > 0.0:
        col(_OWN_TOTAL)
        val(x if x < _CLIP else _CLIP)
    x = tr.street_bet[me] * _STREET_SCALE
    if x > 0.0:
        col(_OWN_STREET)
        val(x if x < _CLIP else _CLIP)
    folded = tr.folded
    all_in = tr.all_in
    live = 0
    can_act = 0
    for i in range(n):
        if not folded[i]:
            live += 1
            if not all_in[i]:
                can_act += 1
    if live:
        col(_LIVE)
        val(live / 10)
    if can_act:
        col(_CAN_ACT)
        val(can_act / 10)
    x = tr.stack[me] * _STACK_SCALE
    if x > 0.0:
        col(_OWN_STACK)
        val(x if x < _CLIP else _CLIP)
    if legal.can_raise:
        col(_CAN_RAISE)
        val(1.0)
    if legal.can_check:
        col(_CAN_CHECK)
        val(1.0)
    raises_left = legal.raises_left
    if raises_left is None:
        col(_CAP_LIFTED)
        val(1.0)
    elif raises_left > 0:
        col(_RAISES_LEFT)
        val((raises_left if raises_left < 4 else 4) / 4)

    # Own action history.
    bets = tr.bets
    own = bets[me]
    for s in range(4):
        b = own[s]
        if b:
            col(_OWN_BETS + s)
            val((b if b < 4 else 4) / 4)
    aggressor = tr.aggressor
    prev_aggressor = tr.prev_aggressor
    if aggressor == me:
        col(_OWN_AGGRESSOR)
        val(1.0)
    if prev_aggressor == me:
        col(_OWN_PREV_AGGRESSOR)
        val(1.0)

    # Other seats, slot k = the k-th participant clockwise after the acting one.
    anchor = tr.bb if street == 0 else tr.button
    my_pos = (me - anchor - 1) % n
    total = tr.total
    street_bet = tr.street_bet
    stack = tr.stack
    acted = tr.acted
    button = tr.button
    for k in range(1, n):
        j = (me + k) % n
        base = _OTHERS + SLOT_SIZE * (k - 1)
        col(base)
        val(1.0)
        if not folded[j]:
            col(base + 1)
            val(1.0)
        if all_in[j]:
            col(base + 2)
            val(1.0)
        x = total[j] * _POT_SCALE
        if x > 0.0:
            col(base + 3)
            val(x if x < _CLIP else _CLIP)
        x = street_bet[j] * _STREET_SCALE
        if x > 0.0:
            col(base + 4)
            val(x if x < _CLIP else _CLIP)
        other = bets[j]
        for s in range(4):
            b = other[s]
            if b:
                col(base + 5 + s)
                val((b if b < 4 else 4) / 4)
        if aggressor == j:
            col(base + 9)
            val(1.0)
        if prev_aggressor == j:
            col(base + 10)
            val(1.0)
        if not folded[j] and not all_in[j] and (j - anchor - 1) % n > my_pos:
            col(base + 11)
            val(1.0)
        if acted[j]:
            col(base + 12)
            val(1.0)
        x = stack[j] * _STACK_SCALE
        if x > 0.0:
            col(base + 13)
            val(x if x < _CLIP else _CLIP)
        if j == button:
            col(base + 14)
            val(1.0)

    # Recent actions, most recent first.
    base = _RECENT
    for i, kind, s in reversed(tr.recent):
        col(base)
        val(1.0)
        col(base + 1 + (i - me) % n)
        val(1.0)
        col(base + 11 + kind)
        val(1.0)
        col(base + 14 + s)
        val(1.0)
        base += RECENT_SIZE

    # Hand-strength hints of the street (cached per participant and street).
    if street > 0:
        key = (me, street)
        hints = cache.get(key)
        if hints is None:
            hints = cache[key] = [(_HINTS + off, v) for off, v in street_hints(hole_cards, tr.board)]
        for c, v in hints:
            col(c)
            val(v)


def encode(
    tr: HandTracker,
    me_seat: int,
    hole_cards,
    legal: LegalActions,
    out_row: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Write the observation into `out_row` (zeroed first; a new float32 row of
    `OBS_DIM` when None) and return it."""
    if out_row is None:
        out_row = np.zeros(OBS_DIM, dtype=np.float32)
    else:
        out_row[:] = 0.0
    cols: list[int] = []
    vals: list[float] = []
    collect(tr, me_seat, hole_cards, legal, cols, vals)
    out_row[cols] = vals
    return out_row


def encode_view(view: TableView, legal: LegalActions, out: Optional[np.ndarray] = None) -> np.ndarray:
    """The observation from an engine `TableView` and `LegalActions`: uses only
    `view.history`, `view.seat` and `view.hole_cards`."""
    if view.hole_cards is None:
        raise ValueError("the view has no hole cards: the player is not dealt in")
    tr = HandTracker()
    tr.feed_all(view.history)
    return encode(tr, view.seat, view.hole_cards, legal, out)
