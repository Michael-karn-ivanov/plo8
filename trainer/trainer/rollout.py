"""The rollout loop: the learner against arena's bots in one rl_env `TableBatch`.

`RolloutWorker(seed=...)` owns a TableBatch, a Recorder that records only the
learner (`LEARNER_KEY`), a `HandSampler`, a lineup generator, one arena bot per
opponent name and the learner `NeuralPolicy`. `collect(weights, n_decisions)`
loads the weights (validated: a mismatch raises before any hand), deals hands
on idle tables while fewer than `n_decisions` learner decisions exist (ready or
pending), then stops dealing and plays every running hand to its end, and
returns `(TrainingRecords, RolloutStats)`. Records only ever come from complete
hands; nothing is carried from one collection to the next except the
generators, so a sequence of collections is reproducible from the seed.

The worker only chooses hand specs, lineups and controller keys and calls
`TableBatch.start_hand` / `step`; batching per controller, mask checks,
recording and reward attachment are rl_env's.
"""

from __future__ import annotations

import math
import numbers
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

from arena import BOT_NAMES, make_bot
from rl_env import FinishedHand, HandSampler, HandSpec, Recorder, TableBatch, TrainingRecords

from .lineup import DEFAULT_OPPONENTS, LEARNER_KEY, MAX_LEARNER_SEATS, draw_lineup
from .nets import DEFAULT_HIDDEN, DEFAULT_LAYERS, PolicyNet
from .policy import NeuralPolicy

__all__ = ["RolloutWorker", "RolloutStats", "ResultStats"]

CHIPS_PER_BB = 100  # TableBatch's default
CHIPS_PER_BIG_BET = 2 * CHIPS_PER_BB


def _int_arg(name: str, value: Any, low: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an int, got {value!r}")
    value = int(value)
    if value < low:
        raise ValueError(f"{name} must be >= {low}, got {value}")
    return value


def _seed_int(ss: np.random.SeedSequence, dtype: Any = np.uint64) -> int:
    return int(ss.generate_state(1, dtype)[0])


# --- statistics --------------------------------------------------------------------


@dataclass(frozen=True)
class ResultStats:
    """The learner's results over one group of hands."""

    hands: int  # hands in the group
    learner_seats: int  # learner seats over those hands
    decisions: int  # learner decisions in those hands
    total_chips: int  # learner's net over all its seats, chips (100 per BB)
    total_bb: float  # big bets: total_chips / 200
    bb_per_100: float  # 100 * total_bb / hands (nan when hands == 0)


@dataclass(frozen=True)
class RolloutStats:
    """Logging figures of one collection (not an evaluation)."""

    overall: ResultStats
    by_size: dict  # int -> ResultStats, every size of `sizes`, ascending
    by_opponent: dict  # str -> ResultStats, every opponent, in `opponents` order
    all_decisions: int  # every seat's decisions, bots included
    wall_time: float  # seconds, time.perf_counter
    decisions_per_second: float  # learner decisions / wall_time
    all_decisions_per_second: float

    @property
    def hands(self) -> int:
        return self.overall.hands

    @property
    def decisions(self) -> int:
        return self.overall.decisions

    @property
    def total_bb(self) -> float:
        return self.overall.total_bb

    @property
    def bb_per_100(self) -> float:
        return self.overall.bb_per_100


class _Group:
    __slots__ = ("hands", "seats", "decisions", "chips")

    def __init__(self) -> None:
        self.hands = 0
        self.seats = 0
        self.decisions = 0
        self.chips = 0

    def add(self, seats: int, decisions: int, chips: int) -> None:
        self.hands += 1
        self.seats += seats
        self.decisions += decisions
        self.chips += chips

    def result(self) -> ResultStats:
        total_bb = self.chips / CHIPS_PER_BIG_BET
        rate = 100.0 * total_bb / self.hands if self.hands else math.nan
        return ResultStats(
            hands=self.hands,
            learner_seats=self.seats,
            decisions=self.decisions,
            total_chips=self.chips,
            total_bb=total_bb,
            bb_per_100=rate,
        )


class _StatsAccumulator:
    """Per hand of one collection: table size, learner seats, bot names present
    (`start`), the learner's chips (`add`); decisions from the records (`finish`)."""

    def __init__(self, opponents: Sequence[str], sizes: Iterable[int]) -> None:
        self._opponents = tuple(opponents)
        self._sizes = tuple(sorted(set(int(s) for s in sizes)))
        self._started: dict[int, tuple[int, tuple[int, ...], tuple[str, ...]]] = {}
        self._finished: dict[int, int] = {}  # hand_id -> learner chips

    def start(self, hand_id: int, spec: HandSpec, keys: Sequence[Any]) -> None:
        seats = tuple(s for s, k in enumerate(keys) if k == LEARNER_KEY)
        present = set(k for k in keys if k != LEARNER_KEY)
        bots = tuple(name for name in self._opponents if name in present)
        self._started[int(hand_id)] = (int(spec.n_seats), seats, bots)

    def add(self, fh: FinishedHand) -> None:
        hand_id = int(fh.hand_id)
        _, seats, _ = self._started[hand_id]
        # Integer chips, as arena does: sums are exact and order-free.
        chips = sum(int(round(float(fh.net_bb[s]) * CHIPS_PER_BB)) for s in seats)
        self._finished[hand_id] = chips

    def finish(self, records: TrainingRecords, *, wall: float, all_decisions: int) -> RolloutStats:
        per_hand: dict[int, int] = {}
        if len(records):
            ids, counts = np.unique(records.hand_id, return_counts=True)
            per_hand = {int(h): int(c) for h, c in zip(ids, counts)}
        overall = _Group()
        by_size = {s: _Group() for s in self._sizes}
        by_opp = {name: _Group() for name in self._opponents}
        for hand_id in sorted(self._finished):
            n_seats, seats, bots = self._started[hand_id]
            chips = self._finished[hand_id]
            dec = per_hand.get(hand_id, 0)
            overall.add(len(seats), dec, chips)
            by_size[n_seats].add(len(seats), dec, chips)
            for name in bots:  # once per distinct bot name
                by_opp[name].add(len(seats), dec, chips)
        n_dec = overall.decisions
        return RolloutStats(
            overall=overall.result(),
            by_size={s: g.result() for s, g in by_size.items()},
            by_opponent={name: g.result() for name, g in by_opp.items()},
            all_decisions=int(all_decisions),
            wall_time=float(wall),
            decisions_per_second=n_dec / wall if wall > 0 else math.nan,
            all_decisions_per_second=all_decisions / wall if wall > 0 else math.nan,
        )


# --- the worker --------------------------------------------------------------------


class RolloutWorker:
    """Plays the learner against arena's bots in one TableBatch and returns the
    learner's training records (see the module docstring). Single-threaded, in
    the caller's process; later wrapped as a Ray actor."""

    def __init__(
        self,
        *,
        seed: int,
        n_tables: int = 256,
        opponents: Sequence[str] = DEFAULT_OPPONENTS,
        sizes: Iterable[int] = range(2, 11),
        max_learner_seats: int = MAX_LEARNER_SEATS,
        hidden: int = DEFAULT_HIDDEN,
        layers: int = DEFAULT_LAYERS,
        num_threads: Optional[int] = 1,
    ) -> None:
        seed = _int_arg("seed", seed, 0)
        n_tables = _int_arg("n_tables", n_tables, 1)
        self.max_learner_seats = _int_arg("max_learner_seats", max_learner_seats, 1)
        if isinstance(opponents, str):
            raise TypeError("opponents must be a sequence of bot names, not a string")
        opponents = tuple(opponents)
        if not opponents:
            raise ValueError("opponents must not be empty")
        for name in opponents:
            if not isinstance(name, str):
                raise TypeError(f"opponent names must be strings, got {name!r}")
            if name == LEARNER_KEY:
                raise ValueError(f"{LEARNER_KEY!r} is the learner's key, not an opponent")
            if name not in BOT_NAMES:
                raise ValueError(f"unknown bot {name!r}; known bots: {', '.join(BOT_NAMES)}")
        if len(set(opponents)) != len(opponents):
            raise ValueError(f"duplicate opponent names in {opponents}")
        self._opponents = opponents
        sizes = tuple(sizes)
        if not sizes:
            raise ValueError("sizes must not be empty")
        for s in sizes:
            if isinstance(s, (bool, np.bool_)) or not isinstance(s, numbers.Integral):
                raise TypeError(f"table sizes must be ints, got {s!r}")

        hand_ss, lineup_ss, policy_ss, bot_ss = np.random.SeedSequence(seed).spawn(4)
        self._sampler = HandSampler(_seed_int(hand_ss), sizes)  # checks 2..10
        self._lineup_rng = np.random.Generator(np.random.PCG64(_seed_int(lineup_ss)))
        bot_seeds = bot_ss.spawn(len(opponents))
        self._bots = {
            name: make_bot(name, seed=_seed_int(bot_seeds[i])) for i, name in enumerate(opponents)
        }
        # Initial weights are irrelevant (every collection loads weights first);
        # building them must not move torch's global RNG.
        with torch.random.fork_rng(devices=[]):
            net = PolicyNet(hidden=hidden, layers=layers)
        self._policy = NeuralPolicy(net, seed=_seed_int(policy_ss, np.uint32), num_threads=num_threads)
        self._controllers: dict = {LEARNER_KEY: self._policy, **self._bots}
        self._recorder = Recorder(keys=[LEARNER_KEY])  # bots are not recorded
        self._batch = TableBatch(n_tables, recorder=self._recorder)

    # --- read-only views ---------------------------------------------------------

    @property
    def policy(self) -> NeuralPolicy:
        return self._policy

    @property
    def batch(self) -> TableBatch:
        return self._batch

    @property
    def recorder(self) -> Recorder:
        return self._recorder

    @property
    def opponents(self) -> tuple[str, ...]:
        return self._opponents

    @property
    def sizes(self) -> tuple[int, ...]:
        return self._sampler.sizes

    # --- collection --------------------------------------------------------------

    def collect(
        self,
        weights: Mapping[str, Any],
        n_decisions: int,
        *,
        on_finished: Optional[Callable[[FinishedHand], None]] = None,
    ) -> tuple[TrainingRecords, RolloutStats]:
        """At least `n_decisions` learner records from finished hands, played with
        exactly `weights`, and the collection's statistics. `on_finished` sees
        every finished hand before its table is restarted."""
        raise NotImplementedError
