"""Step-mode self-play benchmark for the engine (standard library only).

    buck2 run //engine:bench -- [--tables N] [--players 2,6,9] [--hands H]
                                [--stack BB] [--seed S] [--views] [--callback]
                                [--profile]

For every table size it builds N environments of players without agents, each
with its own `Deck(random.Random(seed + t))`, and plays random legal actions in
step mode (one `random.Random(seed)` chooses them), round-robin over the tables
as a batched caller would. Between hands every player below the starting stack
is topped up with `rebuy()`, so the table size stays constant and nobody sits
out. The first 1,000 hands are a warm-up and are not timed.

`--views` also requests the acting player's view at every decision;
`--callback` plays the same workload in callback mode with a random agent, for
comparison; `--profile` wraps the timed part in cProfile.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import random
import sys
import time
from typing import Any, Callable, Optional

from engine import Action, Deck, Environment, Player

WARMUP_HANDS = 1000


class _Stats:
    def __init__(self) -> None:
        self.hands = 0
        self.decisions = 0
        self.showdowns = 0
        self.seconds = 0.0


def _top_up(env: Environment, stack_chips: int) -> None:
    cpb = env.chips_per_bb
    for p in env.players:
        if p.chips < stack_chips:
            p.rebuy((stack_chips - p.chips) / cpb)


def _tables(n_tables: int, n_players: int, stack: float, seed: int, agent: Callable[[], Any] = None):
    envs = []
    for t in range(n_tables):
        players = [
            Player(f"p{i}", stack, agent() if agent is not None else None) for i in range(n_players)
        ]
        envs.append(Environment(players, deck=Deck(random.Random(seed + t))))
    return envs


def _timed(fn: Callable[[], None], profile: bool) -> None:
    if not profile:
        fn()
        return
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        fn()
    finally:
        profiler.disable()
    for key in ("cumulative", "tottime"):
        out = io.StringIO()
        pstats.Stats(profiler, stream=out).sort_stats(key).print_stats(25)
        print(f"--- profile, sorted by {key} ---")
        print(out.getvalue())


def run_step(
    n_tables: int,
    n_players: int,
    hands: int,
    stack: float,
    seed: int,
    views: bool = False,
    profile: bool = False,
    warmup: int = WARMUP_HANDS,
) -> _Stats:
    """Play `warmup + hands` step-mode hands; time the last `hands`."""
    rng = random.Random(seed)
    choice = rng.choice
    envs = _tables(n_tables, n_players, stack, seed)
    stack_chips = envs[0].dealer.to_chips(stack)
    stats = _Stats()
    pending: list[Optional[Any]] = [env.start_hand() for env in envs]
    total = warmup + hands
    done = 0
    t0: list[float] = []

    def play(until: int, counting: bool) -> None:
        nonlocal done
        while done < until:
            for k, env in enumerate(envs):
                decision = pending[k]
                if decision is None:
                    done += 1
                    if counting:
                        stats.hands += 1
                        stats.showdowns += env.last_result.showdown  # type: ignore[union-attr]
                    _top_up(env, stack_chips)
                    pending[k] = env.start_hand()
                    if done >= until:
                        return
                else:
                    if views:
                        env.view()
                    pending[k] = env.step(choice(decision.legal.actions))
                    if counting:
                        stats.decisions += 1

    play(warmup, counting=False)

    def timed_part() -> None:
        t0.append(time.perf_counter())
        play(total, counting=True)
        t0.append(time.perf_counter())

    _timed(timed_part, profile)
    stats.seconds = t0[1] - t0[0]
    return stats


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
        if player.stack < self.stack:
            player.rebuy(self.stack - player.stack)


def run_callback(
    n_tables: int,
    n_players: int,
    hands: int,
    stack: float,
    seed: int,
    profile: bool = False,
    warmup: int = WARMUP_HANDS,
) -> _Stats:
    rng = random.Random(seed)
    counter = [0]
    envs = _tables(n_tables, n_players, stack, seed, agent=lambda: _RandomAgent(rng, stack, counter))
    stats = _Stats()
    done = 0
    t0: list[float] = []

    def play(until: int, counting: bool) -> None:
        nonlocal done
        while done < until:
            for env in envs:
                result = env.start_round()
                done += 1
                if counting:
                    stats.hands += 1
                    stats.showdowns += result.showdown
                if done >= until:
                    return

    play(warmup, counting=False)

    def timed_part() -> None:
        counter[0] = 0
        t0.append(time.perf_counter())
        play(warmup + hands, counting=True)
        t0.append(time.perf_counter())

    _timed(timed_part, profile)
    stats.decisions = counter[0]
    stats.seconds = t0[1] - t0[0]
    return stats


def _report(mode: str, n_players: int, stats: _Stats) -> None:
    secs = stats.seconds or 1e-9
    share = stats.showdowns / stats.hands if stats.hands else 0.0
    print(
        f"{mode:8s} {n_players:2d} players: {stats.hands:7d} hands, {stats.decisions:8d} decisions, "
        f"{secs:7.2f} s, {stats.hands / secs:9.0f} hands/s, "
        f"{stats.decisions / secs:9.0f} decisions/s, showdown {share:.1%}"
    )
    sys.stdout.flush()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Engine step-mode self-play benchmark.")
    parser.add_argument("--tables", type=int, default=512, help="tables played side by side")
    parser.add_argument(
        "--players", default="2,6,9", help="comma-separated table sizes (one run per size)"
    )
    parser.add_argument("--hands", type=int, default=20_000, help="timed hands per run")
    parser.add_argument("--stack", type=float, default=100.0, help="starting stack in BB")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--views", action="store_true", help="request a view at every decision")
    parser.add_argument(
        "--callback", action="store_true", help="also run the workload in callback mode"
    )
    parser.add_argument("--profile", action="store_true", help="profile the timed part")
    args = parser.parse_args(argv)

    sizes = [int(s) for s in args.players.split(",") if s.strip()]
    if args.tables < 1 or args.hands < 1 or any(not 2 <= n <= 10 for n in sizes):
        parser.error("need --tables >= 1, --hands >= 1 and table sizes in 2..10")
    print(
        f"tables={args.tables} hands={args.hands} stack={args.stack:g} seed={args.seed} "
        f"views={args.views} (warm-up {WARMUP_HANDS} hands not timed)"
    )
    for n in sizes:
        stats = run_step(
            args.tables, n, args.hands, args.stack, args.seed, views=args.views, profile=args.profile
        )
        _report("step", n, stats)
        if args.callback:
            stats = run_callback(
                args.tables, n, args.hands, args.stack, args.seed, profile=args.profile
            )
            _report("callback", n, stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
