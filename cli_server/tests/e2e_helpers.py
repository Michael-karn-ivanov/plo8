"""Test helpers for the e2e mode: small valid cases and synthetic transcripts.

`fold_case_data()` is a three-player hand with seat gaps and a button that is
not the lowest seat, ended by folds preflop. `correct_runs(case)` builds what a
correct server would send every client for a case (the mirror image of
`cli_server.e2e.check`), so tests can break one thing at a time.
"""

from __future__ import annotations

import itertools
import json

from cli_server.e2e.check import ClientRun
from cli_server.e2e.driver import E2E_DIR
from cli_server.e2e.expect import build_expected

SAMPLE = "12_heads_up_three_raises_preflop_bb_scoops"


def sample_data() -> dict:
    return json.loads((E2E_DIR / f"{SAMPLE}.json").read_text(encoding="utf-8"))


def game(bb: int = 2) -> dict:
    return {
        "variant": "omaha_hi_lo_8",
        "betting": "fixed_limit",
        "unit": "chips",
        "blinds": {"sb": bb // 2, "bb": bb},
        "small_bet": bb,
        "big_bet": 2 * bb,
        "cap_raises": 4,
        "rake": 0,
    }


def fold_case_data() -> dict:
    """Seats 1 (A), 4 (B, button), 6 (C): clockwise from the button B, C, A.
    C posts the SB, A the BB; B and C fold; A gets 1 chip back and wins 2."""
    return {
        "hand_id": "test-fold",
        "description": "three players, everybody folds to the big blind",
        "game": game(2),
        "players": [
            {"id": "A", "seat": 1, "stack": 40, "hole": ["As", "2s", "3s", "4s"]},
            {"id": "B", "seat": 4, "stack": 50, "hole": None},
            {"id": "C", "seat": 6, "stack": 60, "hole": ["Kd", "Qd", "Jd", "Td"]},
        ],
        "button": "B",
        "hero": "A",
        "streets": [
            {
                "street": "preflop",
                "board": [],
                "actions": [
                    {"player": "C", "action": "post_sb", "amount": 1},
                    {"player": "A", "action": "post_bb", "amount": 2},
                    {"player": "B", "action": "fold"},
                    {"player": "C", "action": "fold"},
                ],
                "pot_after": 3,
            }
        ],
        "showdown": [],
        "pots": [
            {
                "type": "main",
                "amount": 2,
                "eligible": ["A"],
                "hi": {"winners": ["A"], "amount": 2},
                "lo": None,
                "odd_chips": [],
                "awards": {"A": 2},
            }
        ],
        "result": {
            "ended_by": "fold",
            "net": {"A": 1, "B": 0, "C": -1},
            "final_stacks": {"A": 41, "B": 50, "C": 59},
        },
    }


def _to_wire(event: dict, seq: int, bb: int) -> dict:
    out = {"type": event["type"], "hand_no": 1, "seq": seq}
    for key, value in event.items():
        if key == "type":
            continue
        if key in ("amount", "to", "stack"):
            out[f"{key}_bb"] = value / bb
        elif key == "participants":
            out[key] = [[s, n, st / bb] for s, n, st in value]
        else:
            out[key] = value
    return out


def correct_runs(case) -> dict:
    """Per player id, the `ClientRun` a correct server and client would produce."""
    x = build_expected(case)
    bb = x.bb
    seq = itertools.count(1)
    ids = list(x.seats)
    holes = {}
    spare = (f"{r}{s}" for r in "23456789TJQKA" for s in "cdhs")
    for pid in ids:
        if x.holes[pid] is not None:
            holes[pid] = list(x.holes[pid])
        else:
            holes[pid] = []
            while len(holes[pid]) < 4:
                c = next(spare)
                if c not in x.known_cards and all(c not in h for h in holes.values()):
                    holes[pid].append(c)
    msgs = {
        pid: [
            {"type": "seated", "to": pid, "view": {"seat": x.seats[pid], "hole_cards": None}},
            {"type": "lobby", "to": pid, "state": "starting", "active": len(ids), "needed": len(ids), "starts_in": 0.0},
        ]
        for pid in ids
    }

    def broadcast(event: dict) -> None:
        for pid in ids:
            msgs[pid].append({"type": "event", "to": pid, "event": dict(event)})

    turns = {t.event_index: t for t in x.turns}
    for k, event in enumerate(x.events):
        t = turns.get(k)
        if t is not None:
            for pid in ids:
                if pid == t.player:
                    msgs[pid].append(
                        {
                            "type": "turn",
                            "to": pid,
                            "seq": k,
                            "view": {
                                "street": t.street,
                                "seat": t.seat,
                                "hole_cards": holes[pid],
                                "seats": [
                                    {"seat": x.seats[n], "name": n, "stack": t.stacks[n] / bb}
                                    for n in ids
                                ],
                            },
                            "legal": {},
                            "timeout": 10.0,
                        }
                    )
                else:
                    msgs[pid].append(
                        {"type": "acting", "to": pid, "hand_number": 1, "seat": t.seat, "name": t.player, "timeout": 10.0}
                    )
        broadcast(_to_wire(event, next(seq), bb))
        if k == 2:  # after both blinds
            for pid in ids:
                msgs[pid].append(
                    {
                        "type": "hole_cards",
                        "to": pid,
                        "hand_number": 1,
                        "cards": holes[pid],
                        "view": {"seat": x.seats[pid], "hole_cards": holes[pid]},
                    }
                )

    running = dict(x.stacks_after_betting)
    for pid, hole in x.shown.items():
        broadcast(
            {"type": "hand_shown", "hand_no": 1, "seq": next(seq), "seat": x.seats[pid], "name": pid,
             "hand": list(hole), "stack_bb": running[pid] / bb}
        )
    for pid, chips in x.returned.items():
        running[pid] += chips
        broadcast(
            {"type": "bet_returned", "hand_no": 1, "seq": next(seq), "seat": x.seats[pid], "name": pid,
             "amount_bb": chips / bb, "stack_bb": running[pid] / bb}
        )
    result_pots = []
    for k, pot in enumerate(x.pots):
        high, low = [], []
        for share, winners, amount, out in (
            ("high", pot.high_winners, pot.high_amount, high),
            ("low", pot.low_winners, pot.low_amount, low),
        ):
            for s in sorted(winners):
                chips = amount // len(winners)
                name = x.names[s]
                running[name] += chips
                out.append([s, chips / bb])
                broadcast(
                    {"type": "pot_awarded", "hand_no": 1, "seq": next(seq), "seat": s, "name": name,
                     "pot_index": k, "share": "whole" if pot.uncontested else share,
                     "amount_bb": chips / bb, "stack_bb": running[name] / bb}
                )
        result_pots.append(
            {"amount": pot.amount / bb, "eligible": list(pot.eligible), "high": high, "low": low,
             "uncontested": pot.uncontested}
        )
    result = {
        "hand_number": 1,
        "showdown": x.showdown,
        "pots": result_pots,
        "seats": [
            {"seat": x.seats[n], "name": n, "stack": x.final_stacks[n] / bb, "net": x.net[n] / bb}
            for n in ids
        ],
    }
    runs = {}
    for pid in ids:
        msgs[pid].append({"type": "hand_end", "to": pid, "seq": 99, "seat": x.seats[pid], "result": result})
        msgs[pid].append({"type": "close", "to": pid})
        transcript = {
            "type": "transcript",
            "name": pid,
            "messages": msgs[pid],
            "responses": [{"type": "response", "ok": True}],
            "problems": [],
            "unused_script": [],
        }
        joined = {
            "type": "response",
            "ok": True,
            "accepted": True,
            "queued": False,
            "name": pid,
            "seat": x.seats[pid],
            "table": {"small_blind_bb": 0.5, "big_blind_bb": 1.0},
        }
        runs[pid] = ClientRun(joined=joined, transcript=transcript, exit_code=0)
    return runs
