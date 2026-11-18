"""Step-mode self-play benchmark workload (standard library only).

`run_benchmark(players, ...)` builds `tables` environments of `players` seats
(players without agents), each with its own `Deck(random.Random(seed + t))`,
and plays random legal actions in step mode (one `random.Random(seed)` chooses
them), round-robin over the tables as a batched caller would. Between hands
every player below the starting stack is topped up with `rebuy()`, so the table
size stays constant and nobody sits out. The first `warmup` hands are not timed.

`views=True` also requests the acting player's view at every decision;
`callback=True` plays the same workload in callback mode with a random agent.

The hands and decisions are deterministic for given arguments; only the
timings vary. `format_report(results)` formats one line per result. Used by
`buck2 run //engine:bench` and by other projects (e.g. `//rl_env`) that report
the engine-only number next to their own.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from entities import Deck

from .actions import Action
from .environment import MAX_PLAYERS, MIN_PLAYERS, Environment
from .player import Player
from .step import Decision

__all__ = ["BenchResult", "run_benchmark", "format_report", "WARMUP_HANDS"]

WARMUP_HANDS = 1000


@dataclass(frozen=True)
class BenchResult:
    """One benchmark run at one table size (timings cover the timed hands only)."""

    players: int
    hands: int
    decisions: int
    seconds: float
    hands_per_s: float
    decisions_per_s: float
    showdown_share: float
    tables: int
    mode: str  # "step", "step+views" or "callback"


def _tables(
    n_tables: int, n_players: int, stack: float, seed: int, agent: Optional[Callable[[], Any]] = None
) -> list[Environment]:
    envs = []
    for t in range(n_tables):
        players = [
            Player(f"p{i}", stack, agent() if agent is not None else None) for i in range(n_players)
        ]
        envs.append(Environment(players, deck=Deck(random.Random(seed + t))))
    return envs


def _top_up(env: Environment, stack_chips: int) -> None:
    cpb = env.chips_per_bb
    for p in env.players:
        if p.chips < stack_chips:
            p.rebuy((stack_chips - p.chips) / cpb)


def _run_step(
    n_tables: int, n_players: int, hands: int, stack: float, seed: int, views: bool, warmup: int
) -> tuple[int, int, int, float]:
    """Play `warmup + hands` step-mode hands; time the last `hands`.
    Returns (hands, decisions, showdowns, seconds)."""
    rng = random.Random(seed)
    choice = rng.choice
    envs = _tables(n_tables, n_players, stack, seed)
    stack_chips = envs[0].dealer.to_chips(stack)
    pending: list[Optional[Decision]] = [env.start_hand() for env in envs]
    counts = {"hands": 0, "decisions": 0, "showdowns": 0}
    done = 0

    def play(until: int, counting: bool) -> None:
        nonlocal done
        while done < until:
            for k, env in enumerate(envs):
                decision = pending[k]
                if decision is None:
                    done += 1
                    if counting:
                        counts["hands"] += 1
                        counts["showdowns"] += env.last_result.showdown  # type: ignore[union-attr]
                    _top_up(env, stack_chips)
                    pending[k] = env.start_hand()
                    if done >= until:
                        return
                else:
                    if views:
                        env.view()
                    pending[k] = env.step(choice(decision.legal.actions))
                    if counting:
                        counts["decisions"] += 1

    play(warmup, counting=False)
    t0 = time.perf_counter()
    play(warmup + hands, counting=True)
    seconds = time.perf_counter() - t0
    return counts["hands"], counts["decisions"], counts["showdowns"], seconds


class _RandomAgent:
    """Callback-mode agent: random legal action, tops up its stack after each hand."""

    def __init__(self, rng: random.Random, stack: float, counter: list[int]) -> None:
        self.rng = rng
        self.stack = stack
        self.counter = counter

    def on_turn(self, player: Player, view: Any, legal: Any) -> None:
        self.counter[0] += 1
        player.act(Action.of(self.rng.choice(legal.actions)))

    def on_hand_end(self, player: Player, result: Any) -> None:
        dealer = player._dealer
        assert dealer is not None
        missing = dealer.to_chips(self.stack) - player.chips
        if missing > 0:
            player.rebuy(dealer.to_bb(missing))


def _run_callback(
    n_tables: int, n_players: int, hands: int, stack: float, seed: int, warmup: int
) -> tuple[int, int, int, float]:
    rng = random.Random(seed)
    counter = [0]
    envs = _tables(n_tables, n_players, stack, seed, agent=lambda: _RandomAgent(rng, stack, counter))
    counts = {"hands": 0, "showdowns": 0}
    done = 0

    def play(until: int, counting: bool) -> None:
        nonlocal done
        while done < until:
            for env in envs:
                result = env.start_round()
                done += 1
                if counting:
                    counts["hands"] += 1
                    counts["showdowns"] += result.showdown
                if done >= until:
                    return

    play(warmup, counting=False)
    counter[0] = 0
    t0 = time.perf_counter()
    play(warmup + hands, counting=True)
    seconds = time.perf_counter() - t0
    return counts["hands"], counter[0], counts["showdowns"], seconds


def run_benchmark(
    players: int,
    *,
    tables: int = 512,
    hands: int = 20_000,
    stack: float = 100.0,
    seed: int = 0,
    views: bool = False,
    callback: bool = False,
    warmup: int = WARMUP_HANDS,
) -> BenchResult:
    """One benchmark run at one table size (see the module docstring)."""
    if isinstance(players, bool) or not isinstance(players, int) or not (
        MIN_PLAYERS <= players <= MAX_PLAYERS
    ):
        raise ValueError(f"players must be {MIN_PLAYERS}-{MAX_PLAYERS}, got {players!r}")
    if tables < 1 or hands < 1 or warmup < 0:
        raise ValueError("need tables >= 1, hands >= 1 and warmup >= 0")
    if not stack > 0:
        raise ValueError(f"stack must be positive, got {stack!r}")
    if callback:
        mode = "callback"
        n_hands, decisions, showdowns, seconds = _run_callback(
            tables, players, hands, stack, seed, warmup
        )
    else:
        mode = "step+views" if views else "step"
        n_hands, decisions, showdowns, seconds = _run_step(
            tables, players, hands, stack, seed, views, warmup
        )
    secs = seconds if seconds > 0 else 1e-9
    return BenchResult(
        players=players,
        hands=n_hands,
        decisions=decisions,
        seconds=seconds,
        hands_per_s=n_hands / secs,
        decisions_per_s=decisions / secs,
        showdown_share=showdowns / n_hands if n_hands else 0.0,
        tables=tables,
        mode=mode,
    )


def format_line(result: BenchResult) -> str:
    """One report line for `result`."""
    return (
        f"{result.mode:10s} {result.players:2d} players: {result.hands:7d} hands, "
        f"{result.decisions:8d} decisions, {result.seconds:7.2f} s, "
        f"{result.hands_per_s:9.0f} hands/s, {result.decisions_per_s:9.0f} decisions/s, "
        f"showdown {result.showdown_share:.1%}"
    )


def format_report(results: Sequence[BenchResult]) -> str:
    """The report table: one line per result (table size / mode)."""
    return "\n".join(format_line(r) for r in results)
