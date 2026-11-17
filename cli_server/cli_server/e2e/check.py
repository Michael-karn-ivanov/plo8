"""Compare client transcripts with what the case says every client must see.

`check(expected, clients)` takes, per player id, a `ClientRun` (the `join`
response, the transcript printed by the scripted client and its exit status)
and returns every mismatch found, not just the first one. Amounts on the wire
are BB floats; they are converted back to chips with `round(x * bb)`.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

from .expect import Expected

__all__ = ["Mismatch", "ClientRun", "check", "normalize_event"]

SETTLEMENT = ("hand_shown", "bet_returned", "pot_awarded")


@dataclass(frozen=True)
class Mismatch:
    client: str  # player id, or "-" for the case as a whole
    index: Optional[int]  # message index in the client's transcript
    what: str
    expected: Any = None
    actual: Any = None

    def __str__(self) -> str:
        where = self.client if self.index is None else f"{self.client} #{self.index}"
        return f"{where}: {self.what}: expected {self.expected!r}, got {self.actual!r}"


@dataclass
class ClientRun:
    """What the driver collected for one client process."""

    joined: Optional[dict] = None  # the join response
    transcript: Optional[dict] = None  # the transcript line
    exit_code: Optional[int] = None
    stderr: str = ""
    messages: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.transcript is not None and not self.messages:
            self.messages = list(self.transcript.get("messages") or [])


def _chips(x: Any, bb: int) -> Any:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return x
    return int(round(x * bb))


def normalize_event(ev: dict, bb: int) -> dict:
    """A wire event with amounts in chips and the stack under `stack`."""
    t = ev.get("type")
    out: dict[str, Any] = {"type": t}
    for k, v in ev.items():
        if k in ("type", "hand_no", "seq"):
            continue
        if k == "participants":
            out[k] = [[s, n, _chips(st, bb)] for s, n, st in v]
        elif k == "hand":
            out["cards"] = list(v)
        elif k == "board":
            continue
        elif k.endswith("_bb"):
            out[k[:-3] if k != "stack_bb" else "stack"] = _chips(v, bb)
        else:
            out[k] = v
    return out


class _Checker:
    def __init__(self, expected: Expected, clients: dict[str, ClientRun]) -> None:
        self.x = expected
        self.clients = clients
        self.out: list[Mismatch] = []

    def add(self, client: str, index: Optional[int], what: str, expected: Any = None, actual: Any = None) -> None:
        self.out.append(Mismatch(client, index, what, expected, actual))

    def chips(self, v: Any) -> Any:
        return _chips(v, self.x.bb)

    # --- entry ------------------------------------------------------------------

    def run(self) -> list[Mismatch]:
        streams: dict[str, list] = {}
        for pid in self.x.seats:
            run = self.clients.get(pid)
            if run is None or run.transcript is None:
                self.add(pid, None, "client transcript", "a transcript", None)
                continue
            self.check_join(pid, run)
            self.check_script(pid, run)
            streams[pid] = self.check_client(pid, run.messages)
        self.check_same_stream(streams)
        return self.out

    # --- joins and script -------------------------------------------------------

    def check_join(self, pid: str, run: ClientRun) -> None:
        r = run.joined or {}
        seat = self.x.seats[pid]
        got = (r.get("ok"), r.get("accepted"), r.get("queued"), r.get("seat"))
        if got != (True, True, False, seat):
            self.add(pid, None, "join (ok, accepted, queued, seat)", (True, True, False, seat), got)
        table = r.get("table") or {}
        blinds = (table.get("small_blind_bb"), table.get("big_blind_bb"))
        if blinds != (0.5, 1.0):
            self.add(pid, None, "table blinds in BB", (0.5, 1.0), blinds)
        if not any(m.get("type") == "seated" for m in run.messages):
            self.add(pid, None, "seated message", "seated", None)

    def check_script(self, pid: str, run: ClientRun) -> None:
        t = run.transcript or {}
        for p in t.get("problems") or []:
            self.add(pid, None, "client problem", None, p)
        unused = t.get("unused_script") or []
        if unused:
            self.add(pid, None, "unused script", [], unused)
        for i, r in enumerate(t.get("responses") or []):
            if not r.get("ok"):
                self.add(pid, None, f"request {i} rejected", "ok", r.get("error"))
        if run.exit_code != 0:
            self.add(pid, None, "client exit status", 0, run.exit_code)

    # --- one client ---------------------------------------------------------------

    def check_client(self, pid: str, msgs: list[dict]) -> list:
        """All per-client checks; returns the raw hand events (for comparison
        between clients)."""
        x = self.x
        events = [(i, m["event"]) for i, m in enumerate(msgs) if m.get("type") == "event"]
        start = next((k for k, (_, e) in enumerate(events) if e.get("type") == "hand_started"), None)
        if start is None:
            self.add(pid, None, "hand_started event", "hand_started", None)
            return []
        last_award = max(
            (k for k, (_, e) in enumerate(events) if e.get("type") == "pot_awarded"), default=None
        )
        if last_award is None:
            self.add(pid, None, "pot_awarded events", "at least one", None)
            last_award = len(events) - 1
        hand = events[start : last_award + 1]
        hand_start_index = hand[0][0]

        # Ordered part: up to the first settlement event.
        split = next(
            (k for k, (_, e) in enumerate(hand) if e.get("type") in SETTLEMENT), len(hand)
        )
        ordered, settlement = hand[:split], hand[split:]
        positions: Optional[list[int]] = None
        if len(ordered) != len(x.events):
            self.add(
                pid,
                ordered[-1][0] if ordered else None,
                "number of events before the settlement",
                [e["type"] for e in x.events],
                [e.get("type") for _, e in ordered],
            )
        else:
            positions = [i for i, _ in ordered]
        for k, (exp, (i, ev)) in enumerate(zip(x.events, ordered)):
            got = normalize_event(ev, x.bb)
            got = {key: got.get(key) for key in exp}
            if got != exp:
                self.add(pid, i, f"event {k} ({exp['type']})", exp, got)

        self.check_settlement(pid, settlement)
        hand_end = self.check_hand_end(pid, msgs)
        self.check_holes(pid, msgs)
        self.check_turns(pid, msgs, positions)
        self.check_acting(pid, msgs, positions)
        end_index = hand_end if hand_end is not None else len(msgs)
        for i in range(hand_start_index, end_index):
            if msgs[i].get("type") == "lobby":
                self.add(pid, i, "lobby message during the hand", None, msgs[i].get("state"))
        return [e for _, e in hand]

    def check_settlement(self, pid: str, settlement: list) -> None:
        x = self.x
        running = dict(x.stacks_after_betting)
        shown: dict[str, list] = {}
        returned: dict[str, int] = defaultdict(int)
        awards: dict[tuple[int, int], int] = defaultdict(int)  # (pot, seat) -> chips
        shares: dict[tuple[int, str], int] = defaultdict(int)  # (pot, hi/lo) -> chips
        winners: dict[tuple[int, str], set] = defaultdict(set)
        for i, ev in settlement:
            t = ev.get("type")
            name = ev.get("name")
            if t not in SETTLEMENT:
                self.add(pid, i, "settlement event", " / ".join(SETTLEMENT), t)
                continue
            if name not in running:
                self.add(pid, i, f"{t} player", sorted(running), name)
                continue
            if t == "hand_shown":
                if name in shown:
                    self.add(pid, i, f"hand of {name} shown twice", 1, 2)
                shown[name] = list(ev.get("hand") or [])
            elif t == "bet_returned":
                amount = self.chips(ev.get("amount_bb"))
                returned[name] += amount
                running[name] += amount
            else:
                amount = self.chips(ev.get("amount_bb"))
                pot = ev.get("pot_index")
                share = "lo" if ev.get("share") == "low" else "hi"  # "whole" counts as hi
                awards[(pot, ev.get("seat"))] += amount
                shares[(pot, share)] += amount
                winners[(pot, share)].add(ev.get("seat"))
                running[name] += amount
            stack = self.chips(ev.get("stack_bb"))
            if stack != running[name]:
                self.add(pid, i, f"stack of {name} after {t}", running[name], stack)

        want_shown = {k: list(v) for k, v in x.shown.items()}
        if shown != want_shown:
            self.add(pid, None, "hands shown", want_shown, shown)
        if dict(returned) != {k: v for k, v in x.returned.items() if v}:
            self.add(pid, None, "uncalled bets returned", x.returned, dict(returned))
        for k, pot in enumerate(x.pots):
            got_awards = {s: a for (p, s), a in awards.items() if p == k}
            if got_awards != pot.awards:
                self.add(pid, None, f"pot {k} awards per seat", pot.awards, got_awards)
            got = (shares.get((k, "hi"), 0), shares.get((k, "lo"), 0))
            if got != (pot.high_amount, pot.low_amount):
                self.add(pid, None, f"pot {k} (hi, lo) amounts", (pot.high_amount, pot.low_amount), got)
            got_w = (winners.get((k, "hi"), set()), winners.get((k, "lo"), set()))
            if got_w != (set(pot.high_winners), set(pot.low_winners)):
                self.add(
                    pid,
                    None,
                    f"pot {k} (hi, lo) winner seats",
                    (sorted(pot.high_winners), sorted(pot.low_winners)),
                    (sorted(got_w[0]), sorted(got_w[1])),
                )
        extra = sorted({p for (p, _) in awards} - set(range(len(x.pots))))
        if extra:
            self.add(pid, None, "awards from unknown pots", list(range(len(x.pots))), extra)
        if running != x.final_stacks:
            self.add(pid, None, "stacks after the settlement events", x.final_stacks, running)

    def check_hand_end(self, pid: str, msgs: list[dict]) -> Optional[int]:
        x = self.x
        ends = [i for i, m in enumerate(msgs) if m.get("type") == "hand_end"]
        if len(ends) != 1:
            self.add(pid, ends[1] if len(ends) > 1 else None, "hand_end messages", 1, len(ends))
            if not ends:
                return None
        i = ends[0]
        result = msgs[i].get("result") or {}
        if msgs[i].get("seat") != x.seats[pid]:
            self.add(pid, i, "hand_end seat", x.seats[pid], msgs[i].get("seat"))
        if result.get("showdown") != x.showdown:
            self.add(pid, i, "result showdown", x.showdown, result.get("showdown"))
        seats = {s.get("name"): s for s in result.get("seats") or []}
        stacks = {n: self.chips(s.get("stack")) for n, s in seats.items()}
        nets = {n: self.chips(s.get("net")) for n, s in seats.items()}
        if stacks != x.final_stacks:
            self.add(pid, i, "result stacks", x.final_stacks, stacks)
        if nets != x.net:
            self.add(pid, i, "result net", x.net, nets)
        total = sum(v for v in stacks.values() if isinstance(v, int))
        if total != sum(x.start_stacks.values()):
            self.add(pid, i, "sum of the final stacks", sum(x.start_stacks.values()), total)
        pots = result.get("pots") or []
        if len(pots) != len(x.pots):
            self.add(pid, i, "number of pots", len(x.pots), len(pots))
        for k, (exp, got) in enumerate(zip(x.pots, pots)):
            high = {s: self.chips(a) for s, a in got.get("high") or [] if self.chips(a)}
            low = {s: self.chips(a) for s, a in got.get("low") or [] if self.chips(a)}
            per_winner: dict[int, int] = defaultdict(int)
            for s, a in list(high.items()) + list(low.items()):
                per_winner[s] += a
            want = (
                exp.amount,
                list(exp.eligible),
                sorted(exp.high_winners),
                exp.high_amount,
                sorted(exp.low_winners),
                exp.low_amount,
                exp.awards,
                exp.uncontested,
            )
            actual = (
                self.chips(got.get("amount")),
                sorted(got.get("eligible") or []),
                sorted(high),
                sum(high.values()),
                sorted(low),
                sum(low.values()),
                dict(per_winner),
                got.get("uncontested"),
            )
            if want != actual:
                self.add(
                    pid,
                    i,
                    f"result pot {k} (amount, eligible, high winners, high, low winners, "
                    "low, per winner, uncontested)",
                    want,
                    actual,
                )
        return i

    def check_holes(self, pid: str, msgs: list[dict]) -> None:
        x = self.x
        holes = [i for i, m in enumerate(msgs) if m.get("type") == "hole_cards"]
        if len(holes) != 1:
            self.add(pid, holes[1] if len(holes) > 1 else None, "hole_cards messages", 1, len(holes))
        if not holes:
            return
        i = holes[0]
        own = list(msgs[i].get("cards") or [])
        first_turn = next((j for j, m in enumerate(msgs) if m.get("type") == "turn"), None)
        if first_turn is not None and first_turn < i:
            self.add(pid, i, "hole_cards before the first turn", f"before #{first_turn}", f"#{i}")
        want = x.holes.get(pid)
        if want is not None:
            if own != list(want):
                self.add(pid, i, "own hole cards", list(want), own)
        elif len(own) != 4 or len(set(own)) != 4 or set(own) & x.known_cards:
            self.add(pid, i, "random hole cards (4 cards not fixed by the case)", "4 free cards", own)
        # Nobody else's cards before the showdown, in any view.
        for j, m in enumerate(msgs):
            view = m.get("view")
            if not isinstance(view, dict):
                continue
            seen = view.get("hole_cards")
            if seen is not None and list(seen) != own:
                self.add(pid, j, f"hole cards in the {m.get('type')} view", own, seen)

    def check_turns(self, pid: str, msgs: list[dict], positions: Optional[list[int]]) -> None:
        x = self.x
        want = x.turns_of(pid)
        turns = [(i, m) for i, m in enumerate(msgs) if m.get("type") == "turn"]
        if len(turns) != len(want):
            self.add(
                pid,
                None,
                "turns",
                [(t.street, t.action) for t in want],
                [m.get("view", {}).get("street") for _, m in turns],
            )
        for t, (i, m) in zip(want, turns):
            view = m.get("view") or {}
            if view.get("street") != t.street:
                self.add(pid, i, "turn street", t.street, view.get("street"))
            if view.get("seat") != t.seat:
                self.add(pid, i, "turn seat", t.seat, view.get("seat"))
            stacks = {s.get("name"): self.chips(s.get("stack")) for s in view.get("seats") or []}
            if stacks != t.stacks:
                self.add(pid, i, "stacks in the turn view", t.stacks, stacks)
            if positions is not None:
                prev, acted = positions[t.event_index - 1], positions[t.event_index]
                if not prev < i < acted:
                    self.add(pid, i, "turn position", f"between #{prev} and #{acted}", f"#{i}")

    def check_acting(self, pid: str, msgs: list[dict], positions: Optional[list[int]]) -> None:
        x = self.x
        want = [t for t in x.turns if t.player != pid]
        acting = [(i, m) for i, m in enumerate(msgs) if m.get("type") == "acting"]
        if len(acting) != len(want):
            self.add(
                pid,
                None,
                "acting messages (seat, name)",
                [(t.seat, t.player) for t in want],
                [(m.get("seat"), m.get("name")) for _, m in acting],
            )
        for t, (i, m) in zip(want, acting):
            got = (m.get("seat"), m.get("name"))
            if got != (t.seat, t.player):
                self.add(pid, i, "acting (seat, name)", (t.seat, t.player), got)
            if positions is not None:
                prev, acted = positions[t.event_index - 1], positions[t.event_index]
                if not prev < i < acted:
                    self.add(
                        pid,
                        i,
                        f"acting for {t.player} position",
                        f"between #{prev} and #{acted}",
                        f"#{i}",
                    )

    # --- between clients ---------------------------------------------------------

    def check_same_stream(self, streams: dict[str, list]) -> None:
        names = [n for n in self.x.seats if n in streams]
        if not names:
            return
        ref_name = names[0]
        ref = streams[ref_name]
        for name in names[1:]:
            got = streams[name]
            if got == ref:
                continue
            k = next((k for k, (a, b) in enumerate(zip(ref, got)) if a != b), min(len(ref), len(got)))
            self.add(
                name,
                None,
                f"hand event {k} compared with {ref_name}'s",
                ref[k] if k < len(ref) else None,
                got[k] if k < len(got) else None,
            )


def check(expected: Expected, clients: dict[str, ClientRun]) -> list[Mismatch]:
    """Every mismatch between the client runs and the expectations."""
    return _Checker(expected, clients).run()
