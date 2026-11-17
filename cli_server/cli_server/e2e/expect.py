"""What every client is expected to receive, derived from a case.

Everything is in chips (the engine runs with `chips_per_bb = case bb`, and the
checker converts the BB floats on the wire back with `round(x * bb)`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .case import Case, returned_chips
from .deck import engine_seats, seating

__all__ = ["ExpectedTurn", "ExpectedPot", "Expected", "build_expected", "client_scripts"]


@dataclass(frozen=True)
class ExpectedTurn:
    """One voluntary action of the hand (one `turn` to the actor)."""

    player: str
    seat: int
    street: str
    action: str  # engine action: fold / check / call / raise
    stacks: dict[str, int]  # every player's stack when the turn is asked
    event_index: int  # index of its `player_acted` in `Expected.events`


@dataclass(frozen=True)
class ExpectedPot:
    amount: int
    eligible: tuple[int, ...]  # engine seats, sorted
    high_winners: frozenset[int]
    high_amount: int
    low_winners: frozenset[int]
    low_amount: int
    awards: dict[int, int]  # seat -> chips
    uncontested: bool


@dataclass(frozen=True)
class Expected:
    bb: int
    seats: dict[str, int]  # id -> engine seat
    names: dict[int, str]  # engine seat -> id
    start_stacks: dict[str, int]
    sb_seat: int
    bb_seat: int
    # Public events from hand_started up to the settlement, in order (chips).
    # Only the keys listed are compared.
    events: tuple[dict, ...]
    turns: tuple[ExpectedTurn, ...]
    holes: dict[str, Optional[tuple[str, ...]]]  # None: not fixed by the case
    known_cards: frozenset[str]
    shown: dict[str, tuple[str, ...]]  # hands shown at showdown (empty after a fold)
    stacks_after_betting: dict[str, int]
    returned: dict[str, int]
    pots: tuple[ExpectedPot, ...]
    showdown: bool
    final_stacks: dict[str, int]
    net: dict[str, int]
    scripts: dict[str, list[dict]] = field(default_factory=dict)

    def turns_of(self, pid: str) -> list[ExpectedTurn]:
        return [t for t in self.turns if t.player == pid]


def client_scripts(case: Case) -> dict[str, list[dict]]:
    """Each player's actions in order, as `{street, action}` with the engine
    action (`bet` -> `raise`); the blinds are posted by the engine."""
    out: dict[str, list[dict]] = {p.id: [] for p in case.players}
    for street, a in case.actions:
        if not a.is_blind:
            out[a.player].append({"street": street, "action": a.engine_action})
    return out


def build_expected(case: Case) -> Expected:
    bb = case.game.bb
    seats = engine_seats(case)
    names = {s: pid for pid, s in seats.items()}
    order = seating(case)
    n = len(order)
    sb_seat, bb_seat = (0, 1) if n == 2 else (1, 2)

    stack = {p.id: p.stack for p in case.players}
    start = dict(stack)
    total = {p.id: 0 for p in case.players}
    folded: set[str] = set()
    events: list[dict] = [
        {
            "type": "hand_started",
            "button": 0,
            "sb_seat": sb_seat,
            "bb_seat": bb_seat,
            "participants": [[k, p.id, p.stack] for k, p in enumerate(order)],
        }
    ]
    turns: list[ExpectedTurn] = []
    for st in case.streets:
        street_bet = {p.id: 0 for p in case.players}
        if st.street != "preflop":
            events.append(
                {"type": "board_dealt", "street": st.street, "cards": [str(c) for c in st.board]}
            )
        for a in st.actions:
            before = dict(stack)
            stack[a.player] -= a.amount
            street_bet[a.player] += a.amount
            total[a.player] += a.amount
            if a.action == "fold":
                folded.add(a.player)
            if a.is_blind:
                events.append(
                    {
                        "type": "blind_posted",
                        "seat": seats[a.player],
                        "name": a.player,
                        "blind": "SB" if a.action == "post_sb" else "BB",
                        "amount": a.amount,
                        "all_in": stack[a.player] == 0,
                        "stack": stack[a.player],
                    }
                )
                continue
            ev = {
                "type": "player_acted",
                "seat": seats[a.player],
                "name": a.player,
                "street": st.street,
                "action": a.engine_action,
                "amount": a.amount,
                "all_in": stack[a.player] == 0,
                "stack": stack[a.player],
                "reason": None,
            }
            if a.to is not None:
                ev["to"] = a.to
            turns.append(
                ExpectedTurn(
                    player=a.player,
                    seat=seats[a.player],
                    street=st.street,
                    action=a.engine_action,
                    stacks=before,
                    event_index=len(events),
                )
            )
            events.append(ev)

    holes: dict[str, Optional[tuple[str, ...]]] = {}
    known: set[str] = set()
    for p in case.players:
        hole = case.hole_of(p.id)
        holes[p.id] = None if hole is None else tuple(str(c) for c in hole)
        if hole is not None:
            known.update(str(c) for c in hole)
    for st in case.streets:
        known.update(str(c) for c in st.board)

    showdown = case.result.ended_by == "showdown"
    shown = {s.player: holes[s.player] for s in case.showdown} if showdown else {}

    pots = []
    for pot in case.pots:
        lo_winners = frozenset(seats[w] for w in pot.lo.winners) if pot.lo else frozenset()
        pots.append(
            ExpectedPot(
                amount=pot.amount,
                eligible=tuple(sorted(seats[e] for e in pot.eligible)),
                high_winners=frozenset(seats[w] for w in pot.hi.winners),
                high_amount=pot.hi.amount,
                low_winners=lo_winners,
                low_amount=pot.lo.amount if pot.lo else 0,
                awards={seats[pid]: chips for pid, chips in pot.awards.items() if chips},
                uncontested=len(pot.eligible) == 1,
            )
        )

    return Expected(
        bb=bb,
        seats=seats,
        names=names,
        start_stacks=start,
        sb_seat=sb_seat,
        bb_seat=bb_seat,
        events=tuple(events),
        turns=tuple(turns),
        holes=holes,
        known_cards=frozenset(known),
        shown=shown,  # type: ignore[arg-type]
        stacks_after_betting=dict(stack),
        returned=returned_chips(total, folded),
        pots=tuple(pots),
        showdown=showdown,
        final_stacks=dict(case.result.final_stacks),
        net=dict(case.result.net),
        scripts=client_scripts(case),
    )
