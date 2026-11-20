"""Readable hand histories from the engine's `HandResult` and its events.

Amounts are in BB (the engine's unit, stated in the header); the final nets
are also given in big bets. Cards and boards use their entities text form.
Seats are shown with their report names (the engine's `s0`... names are not
used). Hole cards of every seat are given by the caller: the evaluator is
not a player and may show them all.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from engine import (
    ActionType,
    BetReturned,
    BlindPosted,
    BoardDealt,
    HandResult,
    HandShown,
    PlayerActed,
)
from entities import Board, Hand, Street
from rl_env import HandSpec

__all__ = ["format_hand", "cards_text"]

_STREET_NAME = {
    Street.PREFLOP: "Preflop",
    Street.FLOP: "Flop",
    Street.TURN: "Turn",
    Street.RIVER: "River",
}


def cards_text(cards: Optional[Hand | Board | Sequence]) -> str:
    """Cards separated by spaces ("As 2s Ah 3h"); "" for None / no cards."""
    if cards is None:
        return ""
    return " ".join(str(c) for c in cards)


def _amount(x: float) -> str:
    return f"{x:.2f}"


def _signed(x: float) -> str:
    return f"{x:+.2f}"


def _action_text(e: PlayerActed) -> str:
    t = e.action
    if t is ActionType.FOLD:
        text = "folds"
    elif t is ActionType.CHECK:
        text = "checks"
    elif t is ActionType.CALL:
        text = f"calls {_amount(e.amount_bb)}"
    else:
        text = f"raises to {_amount(e.to_bb)}"
    if e.all_in:
        text += " all-in"
    if e.reason:
        text += f" ({e.reason})"
    return text


def format_hand(
    deal: int,
    rotation: int,
    spec: HandSpec,
    names: Sequence[str],
    hole_cards: Mapping[int, Hand],
    result: HandResult,
) -> str:
    """The hand as text (see the module docstring)."""
    n = spec.n_seats
    if len(names) != n:
        raise ValueError(f"need {n} names, got {len(names)}")

    def who(seat: int) -> str:
        return f"seat {seat} {names[seat]}"

    width = max(len(s) for s in names)
    lines = [
        f"Deal {deal}, rotation {rotation} - {n} players, button seat {spec.button} "
        f"(amounts in BB)"
    ]
    for s in range(n):
        lines.append(
            f"Seat {s}  {names[s]:<{width}}  {spec.stacks_bb[s]:>7.2f}  "
            f"{cards_text(hole_cards.get(s))}".rstrip()
        )

    blinds: list[str] = []
    streets: list[tuple[str, list[str]]] = []  # (header, actions)
    shows: list[str] = []
    returned: list[str] = []
    for e in result.history:
        if isinstance(e, BlindPosted):
            text = f"{who(e.seat)} {e.blind} {_amount(e.amount_bb)}"
            if e.all_in:
                text += " all-in"
            blinds.append(text)
        elif isinstance(e, PlayerActed):
            if not streets:
                streets.append((_STREET_NAME[Street.PREFLOP], []))
            streets[-1][1].append(f"{who(e.seat)} {_action_text(e)}")
        elif isinstance(e, BoardDealt):
            if not streets:
                streets.append((_STREET_NAME[Street.PREFLOP], []))
            header = f"{_STREET_NAME[e.street]} {cards_text(e.cards)}"
            if e.street is not Street.FLOP:
                header += f" ({cards_text(e.board)})"
            streets.append((header, []))
        elif isinstance(e, HandShown):
            shows.append(f"{who(e.seat)} {cards_text(e.hand)}")
        elif isinstance(e, BetReturned):
            returned.append(f"{who(e.seat)} {_amount(e.amount_bb)}")

    if blinds:
        lines.append("Blinds: " + "; ".join(blinds))
    for header, actions in streets:
        lines.append(header + (": " + "; ".join(actions) if actions else ""))
    if shows:
        lines.append("Shows: " + "; ".join(shows))
    if returned:
        lines.append("Returned: " + "; ".join(returned))
    for i, pot in enumerate(result.pots, start=1):
        head = f"Pot {i} {_amount(pot.amount)}: "
        if pot.uncontested:
            parts = [f"{who(s)} wins uncontested {_amount(a)}" for s, a in pot.high]
        else:
            parts = [f"high {who(s)} {_amount(a)}" for s, a in pot.high]
            if pot.low:
                parts.extend(f"low {who(s)} {_amount(a)}" for s, a in pot.low)
            else:
                parts.append("no low")
        lines.append(head + "; ".join(parts))
    nets = []
    for s in range(n):
        net = result.net.get(s, 0.0)
        nets.append(f"{who(s)} {_signed(net)} ({_signed(net / 2.0)} big bets)")
    lines.append("Net: " + "; ".join(nets))
    return "\n".join(lines)
