"""Case: one recorded hand from `e2e_test_sets/*.json`, loaded and validated.

All amounts in a case are chips. A file that fails validation raises
`CaseError` with the reason; the driver reports it as an ERROR (a bad test, not
a failed one).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from entities import Card

__all__ = [
    "CaseError",
    "CaseGame",
    "CasePlayer",
    "CaseAction",
    "CaseStreet",
    "CaseShowdown",
    "CaseShare",
    "CasePot",
    "CaseResult",
    "Case",
    "STREETS",
    "BOARD_SIZES",
    "load_case",
    "parse_case",
    "clockwise_ids",
    "blind_ids",
    "returned_chips",
]

STREETS = ("preflop", "flop", "turn", "river")
BOARD_SIZES = {"preflop": 0, "flop": 3, "turn": 1, "river": 1}
ACTIONS = ("post_sb", "post_bb", "fold", "check", "call", "bet", "raise")
BLIND_ACTIONS = ("post_sb", "post_bb")
ENGINE_CAP = 4  # the engine's MAX_BETS_PER_STREET: a bet and three raises
MAX_CASE_PLAYERS = 10


class CaseError(ValueError):
    """The case file is not a valid test (bad JSON, inconsistent numbers...)."""


@dataclass(frozen=True)
class CaseGame:
    variant: str
    betting: str
    unit: str
    sb: int
    bb: int
    small_bet: int
    big_bet: int
    cap_raises: int
    rake: int


@dataclass(frozen=True)
class CasePlayer:
    id: str
    seat: int
    stack: int
    hole: Optional[tuple[Card, ...]]  # None when unknown


@dataclass(frozen=True)
class CaseAction:
    player: str
    action: str  # post_sb / post_bb / fold / check / call / bet / raise
    amount: int  # chips put in by this action (0 for fold / check)
    to: Optional[int]  # the player's street bet after it, when the case gives it

    @property
    def is_blind(self) -> bool:
        return self.action in BLIND_ACTIONS

    @property
    def engine_action(self) -> str:
        """The engine action: a bet is a raise from 0."""
        return "raise" if self.action == "bet" else self.action


@dataclass(frozen=True)
class CaseStreet:
    street: str
    board: tuple[Card, ...]
    actions: tuple[CaseAction, ...]
    pot_after: Optional[int]


@dataclass(frozen=True)
class CaseShowdown:
    player: str
    hole: Optional[tuple[Card, ...]]  # None: the hole listed under `players` is used
    mucked: bool


@dataclass(frozen=True)
class CaseShare:
    winners: tuple[str, ...]
    amount: int


@dataclass(frozen=True)
class CasePot:
    type: str  # "main" / "side"
    amount: int
    eligible: tuple[str, ...]
    hi: CaseShare  # an uncontested ("whole") pot counts as hi
    lo: Optional[CaseShare]
    odd_chips: tuple[Any, ...]
    awards: Mapping[str, int]


@dataclass(frozen=True)
class CaseResult:
    ended_by: str  # "showdown" / "fold"
    net: Mapping[str, int]
    final_stacks: Mapping[str, int]
    # The uncalled bet given back, when the case lists it. Cases that leave it
    # out may keep it inside the pot (hand-history style); see _validate.
    uncalled_returned: Optional[Mapping[str, int]] = None


@dataclass(frozen=True)
class Case:
    name: str  # the file name without .json
    hand_id: str
    description: str
    game: CaseGame
    players: tuple[CasePlayer, ...]
    button: str
    streets: tuple[CaseStreet, ...]
    showdown: tuple[CaseShowdown, ...]
    pots: tuple[CasePot, ...]
    result: CaseResult

    def player(self, pid: str) -> CasePlayer:
        for p in self.players:
            if p.id == pid:
                return p
        raise KeyError(pid)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(p.id for p in self.players)

    def hole_of(self, pid: str) -> Optional[tuple[Card, ...]]:
        """The player's hole cards, from `players` or `showdown` (None if unknown)."""
        hole = self.player(pid).hole
        if hole is not None:
            return hole
        for s in self.showdown:
            if s.player == pid and s.hole is not None:
                return s.hole
        return None

    def board_of(self, street: str) -> Optional[tuple[Card, ...]]:
        """The cards dealt on `street`; None if the hand did not reach it."""
        for s in self.streets:
            if s.street == street:
                return s.board
        return None

    @property
    def actions(self) -> tuple[tuple[str, CaseAction], ...]:
        """Every action of the hand, in order, as (street, action)."""
        return tuple((s.street, a) for s in self.streets for a in s.actions)


# --- parsing helpers ------------------------------------------------------------------


def _obj(x: Any, what: str) -> dict:
    if not isinstance(x, dict):
        raise CaseError(f"{what} must be an object, got {x!r}")
    return x


def _list(x: Any, what: str) -> list:
    if not isinstance(x, list):
        raise CaseError(f"{what} must be a list, got {x!r}")
    return x


def _get(d: dict, key: str, what: str) -> Any:
    if key not in d:
        raise CaseError(f"{what} is missing {key!r}")
    return d[key]


def _int(x: Any, what: str) -> int:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or int(x) != x:
        raise CaseError(f"{what} must be a whole number of chips, got {x!r}")
    return int(x)


def _str(x: Any, what: str) -> str:
    if not isinstance(x, str) or not x:
        raise CaseError(f"{what} must be a non-empty string, got {x!r}")
    return x


def _card(x: Any, what: str) -> Card:
    try:
        return Card.parse(x)
    except ValueError as e:
        raise CaseError(f"{what}: {e}") from None


def _cards(x: Any, what: str) -> tuple[Card, ...]:
    return tuple(_card(c, what) for c in _list(x, what))


def _hole(x: Any, what: str) -> Optional[tuple[Card, ...]]:
    if x is None:
        return None
    cards = _cards(x, what)
    if len(cards) != 4:
        raise CaseError(f"{what} must have 4 cards, got {len(cards)}")
    if len(set(cards)) != 4:
        raise CaseError(f"{what} has a duplicate card")
    return cards


def _chip_map(x: Any, what: str) -> dict[str, int]:
    d = _obj(x, what)
    return {_str(k, what): _int(v, f"{what}[{k}]") for k, v in d.items()}


def _fmt(cards: Any) -> str:
    return " ".join(str(c) for c in cards)


# --- loading ----------------------------------------------------------------------------


def load_case(path: Union[str, Path]) -> Case:
    """Read and validate one case file. Raises CaseError."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as e:
        raise CaseError(f"cannot read {path}: {e}") from None
    except json.JSONDecodeError as e:
        raise CaseError(f"invalid JSON in {path.name}: {e}") from None
    return parse_case(data, path.stem)


def parse_case(data: Any, name: str = "case") -> Case:
    """Parse and validate a decoded case object. Raises CaseError."""
    d = _obj(data, "case")
    game = _parse_game(_obj(_get(d, "game", "case"), "game"))
    players = _parse_players(_list(_get(d, "players", "case"), "players"))
    ids = [p.id for p in players]
    button = _str(_get(d, "button", "case"), "button")
    if button not in ids:
        raise CaseError(f"button {button!r} is not one of the players")
    streets = _parse_streets(_list(_get(d, "streets", "case"), "streets"), ids)
    showdown = _parse_showdown(_list(d.get("showdown") or [], "showdown"), ids)
    pots = _parse_pots(_list(_get(d, "pots", "case"), "pots"), ids)
    result = _parse_result(_obj(_get(d, "result", "case"), "result"))
    case = Case(
        name=name,
        hand_id=str(d.get("hand_id", "")),
        description=str(d.get("description", "")),
        game=game,
        players=tuple(players),
        button=button,
        streets=tuple(streets),
        showdown=tuple(showdown),
        pots=tuple(pots),
        result=result,
    )
    return _validate(case)


def _parse_game(g: dict) -> CaseGame:
    blinds = _obj(_get(g, "blinds", "game"), "game.blinds")
    game = CaseGame(
        variant=str(_get(g, "variant", "game")),
        betting=str(_get(g, "betting", "game")),
        unit=str(_get(g, "unit", "game")),
        sb=_int(_get(blinds, "sb", "game.blinds"), "game.blinds.sb"),
        bb=_int(_get(blinds, "bb", "game.blinds"), "game.blinds.bb"),
        small_bet=_int(_get(g, "small_bet", "game"), "game.small_bet"),
        big_bet=_int(_get(g, "big_bet", "game"), "game.big_bet"),
        cap_raises=_int(_get(g, "cap_raises", "game"), "game.cap_raises"),
        rake=_int(g.get("rake", 0), "game.rake"),
    )
    if game.variant != "omaha_hi_lo_8":
        raise CaseError(f"variant must be 'omaha_hi_lo_8', got {game.variant!r}")
    if game.betting != "fixed_limit":
        raise CaseError(f"betting must be 'fixed_limit', got {game.betting!r}")
    if game.unit != "chips":
        raise CaseError(f"unit must be 'chips', got {game.unit!r}")
    if game.rake != 0:
        raise CaseError(f"the engine plays without rake, got rake {game.rake}")
    if game.sb <= 0 or game.sb * 2 != game.bb:
        raise CaseError(f"blinds must be sb = bb / 2 > 0, got {game.sb}/{game.bb}")
    if game.small_bet != game.bb:
        raise CaseError(f"small_bet must equal the big blind ({game.bb}), got {game.small_bet}")
    if game.big_bet != 2 * game.bb:
        raise CaseError(f"big_bet must be twice the big blind ({2 * game.bb}), got {game.big_bet}")
    if game.cap_raises != ENGINE_CAP:
        raise CaseError(
            f"cap_raises must be {ENGINE_CAP} (a bet and three raises per street), "
            f"got {game.cap_raises}"
        )
    return game


def _parse_players(items: list) -> list[CasePlayer]:
    if not 2 <= len(items) <= MAX_CASE_PLAYERS:
        raise CaseError(f"a case needs 2-{MAX_CASE_PLAYERS} players, got {len(items)}")
    players = []
    for i, item in enumerate(items):
        what = f"players[{i}]"
        p = _obj(item, what)
        pid = _str(_get(p, "id", what), f"{what}.id")
        seat = _int(_get(p, "seat", what), f"{what}.seat")
        stack = _int(_get(p, "stack", what), f"{what}.stack")
        if stack <= 0:
            raise CaseError(f"player {pid!r} must have a positive stack, got {stack}")
        hole = _hole(p.get("hole"), f"hole of {pid!r}")
        players.append(CasePlayer(pid, seat, stack, hole))
    ids = [p.id for p in players]
    if len(set(ids)) != len(ids):
        raise CaseError(f"player ids must be unique, got {ids}")
    seats = [p.seat for p in players]
    if len(set(seats)) != len(seats):
        raise CaseError(f"player seats must be unique, got {seats}")
    return players


def _parse_streets(items: list, ids: list[str]) -> list[CaseStreet]:
    if not items:
        raise CaseError("a case needs at least the preflop street")
    streets = []
    for i, item in enumerate(items):
        s = _obj(item, f"streets[{i}]")
        street = _get(s, "street", f"streets[{i}]")
        if i >= len(STREETS) or street != STREETS[i]:
            expected = STREETS[i] if i < len(STREETS) else "no more streets"
            raise CaseError(f"streets[{i}] must be {expected!r}, got {street!r}")
        board = _cards(s.get("board") or [], f"{street} board")
        if len(board) != BOARD_SIZES[street]:
            raise CaseError(
                f"the {street} board must have {BOARD_SIZES[street]} card(s), got {len(board)}"
            )
        actions = []
        for j, a in enumerate(_list(s.get("actions") or [], f"{street} actions")):
            what = f"{street} action {j}"
            a = _obj(a, what)
            player = _get(a, "player", what)
            if player not in ids:
                raise CaseError(f"{what}: unknown player {player!r}")
            action = _get(a, "action", what)
            if action not in ACTIONS:
                raise CaseError(f"{what}: unknown action {action!r}")
            amount = a.get("amount")
            if amount is None:
                if action not in ("fold", "check"):
                    raise CaseError(f"{what}: {action} needs an 'amount'")
                amount = 0
            amount = _int(amount, f"{what} amount")
            to = a.get("to")
            to = None if to is None else _int(to, f"{what} to")
            actions.append(CaseAction(player, action, amount, to))
        pot_after = s.get("pot_after")
        pot_after = None if pot_after is None else _int(pot_after, f"{street} pot_after")
        streets.append(CaseStreet(street, board, tuple(actions), pot_after))
    return streets


def _parse_showdown(items: list, ids: list[str]) -> list[CaseShowdown]:
    out = []
    for i, item in enumerate(items):
        s = _obj(item, f"showdown[{i}]")
        player = _get(s, "player", f"showdown[{i}]")
        if player not in ids:
            raise CaseError(f"showdown[{i}]: unknown player {player!r}")
        hole = _hole(s.get("hole"), f"showdown hole of {player!r}")
        out.append(CaseShowdown(player, hole, bool(s.get("mucked", False))))
    return out


def _parse_share(x: Any, what: str, ids: list[str]) -> Optional[CaseShare]:
    if x is None:
        return None
    s = _obj(x, what)
    winners = tuple(_list(_get(s, "winners", what), f"{what}.winners"))
    for w in winners:
        if w not in ids:
            raise CaseError(f"{what}: unknown winner {w!r}")
    return CaseShare(winners, _int(_get(s, "amount", what), f"{what}.amount"))


def _parse_pots(items: list, ids: list[str]) -> list[CasePot]:
    if not items:
        raise CaseError("a case needs at least one pot")
    pots = []
    for i, item in enumerate(items):
        what = f"pots[{i}]"
        p = _obj(item, what)
        kind = _get(p, "type", what)
        want = "main" if i == 0 else "side"
        # Side pots may be numbered ("side_1", "side_2", ...).
        if not (kind == want or (want == "side" and re.fullmatch(r"side_\d+", str(kind)))):
            raise CaseError(f"{what} must be a {want} pot (main first), got {kind!r}")
        eligible = tuple(_list(_get(p, "eligible", what), f"{what}.eligible"))
        for e in eligible:
            if e not in ids:
                raise CaseError(f"{what}: unknown eligible player {e!r}")
        hi = _parse_share(_get(p, "hi", what), f"{what}.hi", ids)
        if hi is None:
            raise CaseError(f"{what} needs a 'hi' share")
        lo = _parse_share(p.get("lo"), f"{what}.lo", ids)
        awards = _chip_map(_get(p, "awards", what), f"{what}.awards")
        for k in awards:
            if k not in ids:
                raise CaseError(f"{what}: award to unknown player {k!r}")
        pots.append(
            CasePot(
                type=want,
                amount=_int(_get(p, "amount", what), f"{what}.amount"),
                eligible=eligible,
                hi=hi,
                lo=lo,
                odd_chips=tuple(p.get("odd_chips") or ()),
                awards=awards,
            )
        )
    return pots


def _parse_result(r: dict) -> CaseResult:
    ended_by = _get(r, "ended_by", "result")
    if ended_by not in ("showdown", "fold"):
        raise CaseError(f"result.ended_by must be 'showdown' or 'fold', got {ended_by!r}")
    return CaseResult(
        ended_by=ended_by,
        net=_chip_map(_get(r, "net", "result"), "result.net"),
        final_stacks=_chip_map(_get(r, "final_stacks", "result"), "result.final_stacks"),
        uncalled_returned=(
            _chip_map(r["uncalled_returned"], "result.uncalled_returned")
            if r.get("uncalled_returned") is not None
            else None
        ),
    )


# --- seating helpers -------------------------------------------------------------------


def clockwise_ids(case: Case) -> list[str]:
    """Player ids clockwise by case seat, starting with the button."""
    ordered = sorted(case.players, key=lambda p: p.seat)
    b = next(i for i, p in enumerate(ordered) if p.id == case.button)
    return [p.id for p in ordered[b:] + ordered[:b]]


def blind_ids(case: Case) -> tuple[str, str]:
    """(SB, BB) as the engine picks them: heads-up the button is the SB,
    otherwise the two players after the button."""
    order = clockwise_ids(case)
    if len(order) == 2:
        return order[0], order[1]
    return order[1], order[2]


def returned_chips(total: Mapping[str, int], folded: set[str]) -> dict[str, int]:
    """The uncalled bet given back: the excess of the biggest contribution over
    the second biggest, if its owner did not fold (the engine's rule)."""
    if not total:
        return {}
    by_size = sorted(total.items(), key=lambda kv: kv[1], reverse=True)
    top_id, top = by_size[0]
    second = by_size[1][1] if len(by_size) > 1 else 0
    excess = top - second
    if excess <= 0 or top_id in folded:
        return {}
    return {top_id: excess}


# --- validation ------------------------------------------------------------------------


def _move_uncalled_out_of_pots(
    pots: tuple[CasePot, ...], returned: Mapping[str, int]
) -> tuple[CasePot, ...]:
    """Take the uncalled bet out of the pot it was awarded from.

    Hand histories often keep an uncalled bet (e.g. the BB's extra chip in a walk,
    or an open raise everybody folds to) inside the pot the bettor wins. The engine
    gives it back first (`bet_returned`) and awards the rest. Net results are the
    same; this rewrites the case's pots to the engine's convention.
    """
    out = list(pots)
    for pid, chips in returned.items():
        for i in range(len(out) - 1, -1, -1):
            pot = out[i]
            if pot.awards.get(pid, 0) < chips:
                continue
            share = pot.hi if pid in pot.hi.winners else pot.lo
            if share is None or share.amount < chips:
                continue
            new_share = replace(share, amount=share.amount - chips)
            awards = dict(pot.awards)
            awards[pid] -= chips
            if awards[pid] == 0:
                del awards[pid]
            out[i] = replace(
                pot,
                amount=pot.amount - chips,
                hi=new_share if share is pot.hi else pot.hi,
                lo=new_share if share is pot.lo else pot.lo,
                awards=awards,
            )
            break
        else:
            raise CaseError(
                f"the uncalled bet of {chips} to {pid!r} is not in any pot awarded to them"
            )
    return tuple(p for p in out if p.amount > 0)


def _validate(case: Case) -> Case:
    """Validate the case; returns it with the pots in the engine's convention."""
    ids = case.ids
    _validate_cards(case)
    shown_ids = {s.player for s in case.showdown}

    # Betting: replay the actions with the running sums.
    sb_id, bb_id = blind_ids(case)
    stack = {p.id: p.stack for p in case.players}
    total = {pid: 0 for pid in ids}
    folded: set[str] = set()
    first = case.streets[0].actions
    if len(first) < 2 or first[0].action != "post_sb" or first[1].action != "post_bb":
        raise CaseError("preflop must start with post_sb and post_bb")
    for st in case.streets:
        street_bet = {pid: 0 for pid in ids}
        current = 0
        for j, a in enumerate(st.actions):
            what = f"{st.street} action {j} ({a.player} {a.action})"
            if a.is_blind:
                if st.street != "preflop" or j > 1:
                    raise CaseError(f"{what}: the blinds are posted first, preflop")
                want = sb_id if a.action == "post_sb" else bb_id
                if a.player != want:
                    raise CaseError(
                        f"{what}: with this button the engine puts {want!r} on the "
                        f"{'small' if a.action == 'post_sb' else 'big'} blind"
                    )
                blind = case.game.sb if a.action == "post_sb" else case.game.bb
                if a.amount != min(blind, stack[a.player]):
                    raise CaseError(f"{what}: amount must be {min(blind, stack[a.player])}")
            if a.player in folded:
                raise CaseError(f"{what}: the player has folded")
            if not a.is_blind and stack[a.player] == 0:
                raise CaseError(f"{what}: the player is all-in")
            if a.action in ("fold", "check") and a.amount != 0:
                raise CaseError(f"{what}: amount must be 0")
            if a.action == "check" and street_bet[a.player] != current:
                raise CaseError(f"{what}: cannot check facing a bet")
            if a.action == "bet" and current != 0:
                raise CaseError(f"{what}: a bet only when nobody has bet on the street yet")
            if a.action == "raise" and current == 0:
                raise CaseError(f"{what}: nobody has bet yet, use 'bet'")
            if a.action in ("call", "bet", "raise") and a.amount <= 0:
                raise CaseError(f"{what}: amount must be positive")
            if a.amount > stack[a.player]:
                raise CaseError(f"{what}: amount {a.amount} exceeds the stack {stack[a.player]}")
            stack[a.player] -= a.amount
            street_bet[a.player] += a.amount
            total[a.player] += a.amount
            if a.action == "fold":
                folded.add(a.player)
            if a.action == "call" and street_bet[a.player] > current:
                raise CaseError(f"{what}: a call cannot exceed the current bet {current}")
            if a.action in ("bet", "raise") and street_bet[a.player] <= current:
                raise CaseError(f"{what}: a {a.action} must exceed the current bet {current}")
            if a.to is not None and a.to != street_bet[a.player]:
                raise CaseError(
                    f"{what}: 'to' is {a.to} but the street bet is {street_bet[a.player]}"
                )
            current = max(current, street_bet[a.player])
        if st.pot_after is not None and st.pot_after != sum(total.values()):
            raise CaseError(
                f"{st.street} pot_after is {st.pot_after} but the bets sum to {sum(total.values())}"
            )

    live = [pid for pid in ids if pid not in folded]
    ended_by_fold = len(live) == 1
    if case.result.ended_by != ("fold" if ended_by_fold else "showdown"):
        raise CaseError(
            f"result.ended_by is {case.result.ended_by!r} but "
            f"{len(live)} player(s) are left at the end"
        )
    if ended_by_fold and case.showdown:
        raise CaseError("a hand that ended by a fold has no showdown")
    if not ended_by_fold and shown_ids != set(live):
        raise CaseError(
            f"the showdown lists {sorted(shown_ids)} but the players left are {sorted(live)}"
        )

    returned = returned_chips(total, folded)
    listed = case.result.uncalled_returned
    if listed is not None and {k: v for k, v in listed.items() if v} != returned:
        raise CaseError(
            f"result.uncalled_returned is {dict(listed)} but the uncalled bet is {returned}"
        )
    if (
        listed is None
        and returned
        and sum(p.amount for p in case.pots) == sum(total.values())
    ):
        case = replace(case, pots=_move_uncalled_out_of_pots(case.pots, returned))
    awards_total = {pid: 0 for pid in ids}
    for i, pot in enumerate(case.pots):
        if sum(pot.awards.values()) != pot.amount:
            raise CaseError(
                f"pots[{i}]: awards sum to {sum(pot.awards.values())}, not the amount {pot.amount}"
            )
        lo_amount = pot.lo.amount if pot.lo is not None else 0
        if pot.hi.amount + lo_amount != pot.amount:
            raise CaseError(
                f"pots[{i}]: hi + lo amounts ({pot.hi.amount} + {lo_amount}) != {pot.amount}"
            )
        for pid, chips in pot.awards.items():
            awards_total[pid] += chips
    pots_sum = sum(p.amount for p in case.pots)
    bets = sum(total.values()) - sum(returned.values())
    if pots_sum != bets:
        raise CaseError(
            f"the pots sum to {pots_sum} but {bets} chips were bet (after the uncalled bet)"
        )

    net, final = case.result.net, case.result.final_stacks
    for what, m in (("result.net", net), ("result.final_stacks", final)):
        if set(m) != set(ids):
            raise CaseError(f"{what} must list every player, got {sorted(m)}")
    if sum(net.values()) != 0:
        raise CaseError(f"result.net must sum to 0, got {sum(net.values())}")
    for p in case.players:
        if final[p.id] != p.stack + net[p.id]:
            raise CaseError(
                f"final_stacks of {p.id!r} is {final[p.id]}, but stack + net = {p.stack + net[p.id]}"
            )
        computed = awards_total[p.id] + returned.get(p.id, 0) - total[p.id]
        if net[p.id] != computed:
            raise CaseError(
                f"net of {p.id!r} is {net[p.id]}, but awards + returned - bets = {computed}"
            )
    return case


def _validate_cards(case: Case) -> None:
    """Holes from `players` and `showdown` agree, showdown holes are known, and
    no card appears twice in the case."""
    holes: dict[str, tuple[Card, ...]] = {
        p.id: p.hole for p in case.players if p.hole is not None
    }
    shown: set[str] = set()
    for s in case.showdown:
        if s.player in shown:
            raise CaseError(f"{s.player!r} is listed twice in the showdown")
        shown.add(s.player)
        known = holes.get(s.player)
        if s.hole is None:
            if known is None:
                raise CaseError(
                    f"showdown player {s.player!r} has unknown hole cards (a random hand "
                    "could change the result)"
                )
            continue
        if known is not None and known != s.hole:
            raise CaseError(
                f"hole cards of {s.player!r} differ between players ({_fmt(known)}) "
                f"and showdown ({_fmt(s.hole)})"
            )
        holes[s.player] = s.hole
    seen: dict[Card, str] = {}
    for pid, hole in holes.items():
        for c in hole:
            if c in seen:
                raise CaseError(f"duplicate card {c} (hole of {pid!r} and {seen[c]})")
            seen[c] = f"hole of {pid!r}"
    for st in case.streets:
        for c in st.board:
            if c in seen:
                raise CaseError(f"duplicate card {c} ({st.street} board and {seen[c]})")
            seen[c] = f"{st.street} board"
