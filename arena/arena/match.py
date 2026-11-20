"""Duplicate matches: every deal is played `table_size` times, rotating the lineup.

A deal is one deck seed, one button and one set of per-seat stacks. Rotation
`r = 0..n-1` of deal `d` plays the same `HandSpec` with lineup position `p` at
seat `(p + r) % n` (seat `s` is played by position `(s - r) % n`). By the
engine's deal-order contract seat `s` gets the same hole cards and the board is
the same as far as it is dealt, whatever the actions, so over the rotation
group every position holds every seat's cards (and stack and position to the
button) exactly once.

Accounting is in integer chips, keyed by (deal, rotation, seat -> position),
and converted to big bets only for the report (`stats.summarize`). Zero sum is
checked per hand and for the totals; a violation raises `AccountingError`.

Only this module reads every seat's hole cards (`TableBatch.finished_hole_cards`,
for hand histories and deal-order checks), after the hand is over and before
its table is restarted. Nothing read there reaches a controller.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from rl_env import FinishedHand, HandSpec, StackConfig, TableBatch, draw_stacks

from .bots import make_bot
from .history import format_hand
from .stats import CHIPS_PER_BB, CHIPS_PER_BIG_BET, AccountingError, summarize, to_chips

__all__ = [
    "Entrant",
    "NameStats",
    "HandRecord",
    "MatchResult",
    "AccountingError",
    "play_match",
    "combine",
]

_SEED_HIGH = 2**63 - 1  # deck seeds are drawn in [0, 2**63 - 1), as rl_env does

BotSpec = Union[str, Callable[[int], Any], Any]


@dataclass(frozen=True)
class Entrant:
    name: str  # report name; equal names are pooled
    bot: BotSpec  # bot name for make_bot, a factory taking a seed, or a ready controller


@dataclass(frozen=True)
class NameStats:
    name: str
    positions: tuple[int, ...]
    hands: int
    deals: int
    total_chips: int
    total_bb: float  # big bets
    bb_per_100: float  # big bets per 100 hands
    ci95: float  # half-width of the 95% interval in bb/100 (nan with one deal)
    samples: np.ndarray  # float64 (D,), big bets per hand per rotation group


@dataclass(frozen=True)
class HandRecord:
    deal: int
    rotation: int
    spec: HandSpec
    names: tuple[str, ...]  # per seat
    positions: tuple[int, ...]  # per seat
    chips: tuple[int, ...]  # net per seat
    hole_cards: tuple[str, ...]  # per seat, text form
    board: str


@dataclass(frozen=True)
class MatchResult:
    table_size: int
    n_deals: int
    seed: int
    lineup: tuple[str, ...]  # report name per lineup position
    stats: dict  # dict[str, NameStats], in first-appearance order of the lineup
    position_chips: np.ndarray  # int64 (n, D): per lineup position, per deal
    histories: tuple[str, ...]  # sample hand histories, job order
    hands: Optional[tuple[HandRecord, ...]]
    wall_time: float
    hands_per_second: float


# --- inputs -------------------------------------------------------------------------


def _entrants(lineup: Sequence[Union[Entrant, str]]) -> list[Entrant]:
    out = []
    for e in lineup:
        if isinstance(e, Entrant):
            out.append(e)
        elif isinstance(e, str):
            out.append(Entrant(name=e, bot=e))
        else:
            raise TypeError(f"a lineup entry must be an Entrant or a bot name, got {e!r}")
    for e in out:
        if not isinstance(e.name, str) or not e.name:
            raise ValueError(f"report names must be non-empty strings, got {e.name!r}")
    return out


def _is_controller(bot: Any) -> bool:
    return not isinstance(bot, (str, type)) and callable(getattr(bot, "act", None))


def _controllers(
    entrants: list[Entrant], bot_ss: np.random.SeedSequence
) -> tuple[dict[int, Any], tuple[int, ...]]:
    """Controllers keyed by small ints, and the key of each lineup position.

    Named / factory bots: one controller per distinct (name, bot) in lineup
    order, the i-th of them seeded from the i-th child of `bot_ss`. Ready
    controllers: used as given; the same object at several positions is one
    controller."""
    keys: list[int] = []
    seen: dict[tuple, int] = {}
    built: list[tuple[int, Entrant]] = []  # (key, entrant) still to build
    controllers: dict[int, Any] = {}
    for e in entrants:
        bot = e.bot
        if isinstance(bot, str):
            ident: tuple = ("name", e.name, bot)
        elif _is_controller(bot):
            ident = ("ready", id(bot))
        elif callable(bot):
            ident = ("factory", e.name, id(bot))
        else:
            raise TypeError(
                f"Entrant.bot must be a bot name, a factory or a controller, got {bot!r}"
            )
        key = seen.get(ident)
        if key is None:
            key = len(seen)
            seen[ident] = key
            if ident[0] == "ready":
                controllers[key] = bot
            else:
                built.append((key, e))
        keys.append(key)
    children = bot_ss.spawn(len(built)) if built else []
    for i, (key, e) in enumerate(built):
        seed = int(children[i].generate_state(1)[0])
        controllers[key] = make_bot(e.bot, seed) if isinstance(e.bot, str) else e.bot(seed)
    return controllers, tuple(keys)


def _deals(
    n: int,
    n_deals: int,
    stacks: Union[float, Sequence[float], StackConfig],
    deal_ss: np.random.SeedSequence,
) -> list[HandSpec]:
    rng = np.random.Generator(np.random.PCG64(deal_ss))
    fixed: Optional[tuple[float, ...]] = None
    cfg: Optional[StackConfig] = None
    if isinstance(stacks, StackConfig):
        cfg = stacks
    elif isinstance(stacks, (int, float)) and not isinstance(stacks, bool):
        fixed = (float(stacks),) * n
    else:
        fixed = tuple(float(s) for s in stacks)  # type: ignore[union-attr]
        if len(fixed) != n:
            raise ValueError(f"need {n} stacks, got {len(fixed)}")
    if fixed is not None and any(not s > 0 for s in fixed):
        raise ValueError(f"stacks must be positive, got {fixed}")
    specs = []
    for _ in range(n_deals):
        st = draw_stacks(rng, n, cfg) if cfg is not None else fixed
        button = int(rng.integers(n))
        deck_seed = int(rng.integers(0, _SEED_HIGH, dtype=np.int64))
        specs.append(HandSpec(n, st, button, deck_seed))  # type: ignore[arg-type]
    return specs


# --- stats --------------------------------------------------------------------------


def _name_stats(lineup: tuple[str, ...], position_chips: np.ndarray) -> dict:
    n = len(lineup)
    order: list[str] = []
    for name in lineup:
        if name not in order:
            order.append(name)
    stats: dict[str, NameStats] = {}
    total = 0
    for name in order:
        positions = tuple(p for p in range(n) if lineup[p] == name)
        chips = position_chips[list(positions), :].sum(axis=0).astype(np.int64)
        s = summarize(chips, len(positions), n, CHIPS_PER_BIG_BET)
        total += s.total_chips
        stats[name] = NameStats(
            name=name,
            positions=positions,
            hands=s.hands,
            deals=s.deals,
            total_chips=s.total_chips,
            total_bb=s.total_bb,
            bb_per_100=s.bb_per_100,
            ci95=s.ci95,
            samples=s.samples,
        )
    if total != 0:
        raise AccountingError(f"report totals do not sum to zero: {total} chips")
    return stats


# --- the match ----------------------------------------------------------------------


def play_match(
    lineup: Sequence[Union[Entrant, str]],
    n_deals: int,
    seed: int,
    *,
    table_size: Optional[int] = None,
    stacks: Union[float, Sequence[float], StackConfig] = 100.0,
    n_histories: int = 20,
    n_tables: int = 256,
    record_hands: bool = False,
) -> MatchResult:
    """Play a duplicate match (see the module docstring).

    `lineup`: one entrant per position (a plain string `s` is
    `Entrant(name=s, bot=s)`); `table_size`, if given, must equal its length.
    `stacks`: a number (every seat, BB), a sequence (per seat) or a
    `StackConfig` (drawn per deal). The same inputs, including `n_tables`,
    give exactly the same match."""
    entrants = _entrants(lineup)
    n = len(entrants)
    if not 2 <= n <= 10:
        raise ValueError(f"a lineup has 2..10 entrants, got {n}")
    if table_size is not None and table_size != n:
        raise ValueError(f"table_size {table_size} != lineup length {n}")
    if isinstance(n_deals, bool) or not isinstance(n_deals, (int, np.integer)) or n_deals < 1:
        raise ValueError(f"n_deals must be a positive int, got {n_deals!r}")
    if isinstance(n_tables, bool) or not isinstance(n_tables, int) or n_tables < 1:
        raise ValueError(f"n_tables must be a positive int, got {n_tables!r}")
    n_deals = int(n_deals)

    ss = np.random.SeedSequence(seed)
    deal_ss, bot_ss, hist_ss = ss.spawn(3)
    specs = _deals(n, n_deals, stacks, deal_ss)
    controllers, pos_keys = _controllers(entrants, bot_ss)
    names = tuple(e.name for e in entrants)

    total_jobs = n_deals * n
    n_hist = min(max(int(n_histories), 0), total_jobs)
    sampled: set[int] = set()
    if n_hist:
        hist_rng = np.random.Generator(np.random.PCG64(hist_ss))
        picks = hist_rng.choice(total_jobs, size=n_hist, replace=False)
        sampled = {int(j) for j in picks}
    histories: dict[int, str] = {}
    records: dict[int, HandRecord] = {}

    needs_obs = any(getattr(c, "wants_obs", True) for c in controllers.values())
    batch = TableBatch(
        min(n_tables, total_jobs), chips_per_bb=CHIPS_PER_BB, observations=needs_obs
    )
    position_chips = np.zeros((n, n_deals), dtype=np.int64)
    jobs_of: dict[int, int] = {}  # hand id -> job index (d * n + r)

    def seat_keys(r: int) -> tuple[int, ...]:
        return tuple(pos_keys[(s - r) % n] for s in range(n))

    def account(fh: FinishedHand) -> None:
        job = jobs_of.pop(fh.hand_id)
        d, r = divmod(job, n)
        chips = [to_chips(x, CHIPS_PER_BB) for x in fh.net_bb.tolist()]
        if len(chips) != n or sum(chips) != 0:
            raise AccountingError(
                f"deal {d}, rotation {r}: nets do not sum to zero in chips: "
                f"{chips} (BB {fh.net_bb.tolist()})"
            )
        for s in range(n):
            position_chips[(s - r) % n, d] += chips[s]
        if job in sampled or record_hands:
            hole = batch.finished_hole_cards(fh.table)
            seat_names = tuple(names[(s - r) % n] for s in range(n))
            if job in sampled:
                histories[job] = format_hand(d, r, fh.spec, seat_names, hole, fh.result)
            if record_hands:
                records[job] = HandRecord(
                    deal=d,
                    rotation=r,
                    spec=fh.spec,
                    names=seat_names,
                    positions=tuple((s - r) % n for s in range(n)),
                    chips=tuple(chips),
                    hole_cards=tuple(str(hole[s]) for s in range(n)),
                    board=str(fh.result.board),
                )

    t0 = time.perf_counter()
    next_job = 0
    done = 0
    while done < total_jobs:
        for t in batch.idle_tables():
            if next_job >= total_jobs:
                break
            d, r = divmod(next_job, n)
            hand_id = batch.start_hand(t, specs[d], seat_keys(r))
            jobs_of[hand_id] = next_job
            next_job += 1
        finished = batch.step(controllers)
        if not finished and not batch.any_in_hand() and next_job >= total_jobs:
            raise RuntimeError("match stalled: no hand running and jobs unfinished")
        for fh in finished:  # before any of these tables is restarted
            account(fh)
            done += 1
    wall = time.perf_counter() - t0

    lineup_names = names
    stats = _name_stats(lineup_names, position_chips)
    position_chips.setflags(write=False)
    return MatchResult(
        table_size=n,
        n_deals=n_deals,
        seed=int(seed),
        lineup=lineup_names,
        stats=stats,
        position_chips=position_chips,
        histories=tuple(histories[j] for j in sorted(histories)),
        hands=tuple(records[j] for j in sorted(records)) if record_hands else None,
        wall_time=wall,
        hands_per_second=total_jobs / wall if wall > 0 else float("inf"),
    )


def combine(*results: MatchResult) -> MatchResult:
    """Concatenate the deals of matches with the same lineup names and table
    size (different seeds or more deals); stats are recomputed from the
    concatenated chips per deal, histories (and hand records) concatenated."""
    if not results:
        raise ValueError("need at least one match result")
    first = results[0]
    for r in results[1:]:
        if r.lineup != first.lineup or r.table_size != first.table_size:
            raise ValueError(
                f"cannot combine {r.lineup} ({r.table_size}p) with "
                f"{first.lineup} ({first.table_size}p)"
            )
    chips = np.concatenate([r.position_chips for r in results], axis=1).astype(np.int64)
    stats = _name_stats(first.lineup, chips)
    chips.setflags(write=False)
    n_deals = sum(r.n_deals for r in results)
    wall = sum(r.wall_time for r in results)
    hands: Optional[tuple[HandRecord, ...]] = None
    if all(r.hands is not None for r in results):
        hands = tuple(h for r in results for h in r.hands)  # type: ignore[union-attr]
    return MatchResult(
        table_size=first.table_size,
        n_deals=n_deals,
        seed=first.seed,
        lineup=first.lineup,
        stats=stats,
        position_chips=chips,
        histories=tuple(h for r in results for h in r.histories),
        hands=hands,
        wall_time=wall,
        hands_per_second=n_deals * first.table_size / wall if wall > 0 else float("inf"),
    )
