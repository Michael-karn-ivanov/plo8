"""Card / suit text and colours, amounts and hand descriptions.

The spec's four-colour deck: hearts red, diamonds blue, spades yellow, clubs
green. Styles are names (see `render.STYLES`); `ui.py` maps them to curses
colours.
"""

from __future__ import annotations

from typing import Iterable, Optional

from entities import Board, Card, Hand, Rank, Suit
from engine.evaluator import (
    FLUSH,
    FULL_HOUSE,
    HIGH_CARD,
    PAIR,
    QUADS,
    STRAIGHT,
    STRAIGHT_FLUSH,
    TRIPS,
    TWO_PAIR,
    best_omaha,
)

__all__ = [
    "Segment",
    "SUIT_SYMBOLS",
    "SUIT_LETTERS",
    "SUIT_STYLES",
    "SUIT_COLORS",
    "card_text",
    "suit_style",
    "suit_color",
    "card_segments",
    "back_segments",
    "empty_slot_segments",
    "segments_text",
    "fmt_amount",
    "describe_high",
    "describe_low",
    "describe_hand",
]

# (text, style); style None means "the surrounding style".
Segment = tuple[str, Optional[str]]

SUIT_SYMBOLS = {Suit.HEARTS: "♥", Suit.DIAMONDS: "♦", Suit.SPADES: "♠", Suit.CLUBS: "♣"}
SUIT_LETTERS = {Suit.HEARTS: "h", Suit.DIAMONDS: "d", Suit.SPADES: "s", Suit.CLUBS: "c"}
SUIT_STYLES = {
    Suit.HEARTS: "hearts",
    Suit.DIAMONDS: "diamonds",
    Suit.SPADES: "spades",
    Suit.CLUBS: "clubs",
}
# The colour of each suit style (used by ui.py).
SUIT_COLORS = {"hearts": "red", "diamonds": "blue", "spades": "yellow", "clubs": "green"}

CARD_BACK = "▒▒"
CARD_BACK_ASCII = "##"
EMPTY_SLOT = "  "


def card_text(card: Card, ascii: bool = False) -> str:
    """'A♥' (or 'Ah' with `ascii`)."""
    suits = SUIT_LETTERS if ascii else SUIT_SYMBOLS
    return card.rank.symbol + suits[card.suit]


def suit_style(suit: Suit) -> str:
    return SUIT_STYLES[suit]


def suit_color(suit: Suit) -> str:
    return SUIT_COLORS[SUIT_STYLES[suit]]


def card_segments(
    cards: Iterable[Card], ascii: bool = False, brackets: bool = True, sep: str = ""
) -> list[Segment]:
    """`[A♥][K♦]`: one segment per card, in its suit style."""
    out: list[Segment] = []
    for i, card in enumerate(cards):
        if i and sep:
            out.append((sep, None))
        text = card_text(card, ascii)
        out.append((f"[{text}]" if brackets else text, suit_style(card.suit)))
    return out


def back_segments(n: int, ascii: bool = False, brackets: bool = True, sep: str = "") -> list[Segment]:
    """`n` face-down cards: `[▒▒]` (or `[##]`)."""
    back = CARD_BACK_ASCII if ascii else CARD_BACK
    out: list[Segment] = []
    for i in range(n):
        if i and sep:
            out.append((sep, None))
        out.append((f"[{back}]" if brackets else back, "card_back"))
    return out


def empty_slot_segments(n: int) -> list[Segment]:
    """`n` empty board slots `[  ]`."""
    return [(f"[{EMPTY_SLOT}]", "empty_slot") for _ in range(n)]


def segments_text(segments: Iterable[Segment]) -> str:
    return "".join(text for text, _ in segments)


def fmt_amount(amount: Optional[float]) -> str:
    """Big blinds without trailing zeros: 2.0 -> '2', 1.5 -> '1.5'."""
    if amount is None:
        return "?"
    value = round(float(amount), 2)
    if value == int(value):
        return str(int(value))
    return f"{value:g}"


# --- hand descriptions ------------------------------------------------------------


def _rank(value: int) -> str:
    return Rank(value).symbol


def _plural(value: int) -> str:
    return _rank(value) + "s"


def describe_high(rank: tuple[int, ...]) -> str:
    """'Flush, A high', 'Two pair, Ks and 7s', ... from a `rank_high` tuple."""
    category, rest = rank[0], rank[1:]
    if category == STRAIGHT_FLUSH:
        return "Royal flush" if rest[0] == int(Rank.ACE) else f"Straight flush, {_rank(rest[0])} high"
    if category == QUADS:
        return f"Four of a kind, {_plural(rest[0])}"
    if category == FULL_HOUSE:
        return f"Full house, {_plural(rest[0])} full of {_plural(rest[1])}"
    if category == FLUSH:
        return f"Flush, {_rank(rest[0])} high"
    if category == STRAIGHT:
        return f"Straight, {_rank(rest[0])} high"
    if category == TRIPS:
        return f"Three of a kind, {_plural(rest[0])}"
    if category == TWO_PAIR:
        return f"Two pair, {_plural(rest[0])} and {_plural(rest[1])}"
    if category == PAIR:
        return f"Pair of {_plural(rest[0])}"
    if category == HIGH_CARD:
        return f"High card, {_rank(rest[0])}"
    return "Unknown hand"


def describe_low(low: Optional[tuple[int, ...]]) -> str:
    """'7-5-4-2-A low', or 'no low'."""
    if low is None:
        return "no low"
    return "-".join("A" if v == 1 else _rank(v) for v in low) + " low"


def describe_hand(hand: Hand, board: Board) -> str:
    """'Flush, A high; 7-5-4-2-A low'. Uses the engine's evaluator, so the Omaha
    exactly-two-hole-cards rule applies. Empty before the flop."""
    if len(board.cards) < 3:
        return ""
    high, low = best_omaha(hand, board)
    return f"{describe_high(high)}; {describe_low(low)}"
