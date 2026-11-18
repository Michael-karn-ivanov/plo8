"""Rollout benchmark for rl_env.

    buck2 run //rl_env:bench -- [--tables 512] [--players 2,6,9] [--hands 20000]
                                [--seed 0] [--stacks deep|mixed] [--no-obs]
                                [--record] [--profile]

The workload of the spec's speed target: `--tables` tables in one `TableBatch`,
a `RandomController` at every seat, the observation and the mask built at every
decision (`--no-obs`: mask only, to isolate the observation cost). For each table
size it reports hands, decisions, wall time, decisions/s and hands/s, next to the
engine-only number for the same size from the engine's own workload
(`engine.benchmark.run_benchmark`), and the share of time spent in rl_env,
`1 - engine_time / total_time` with the engine time taken per decision
(`= 1 - rl_env decisions/s / engine decisions/s`). Timing with
`time.perf_counter()`; the first 1,000 hands are not timed.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from typing import Callable, Optional, TypeVar

from engine.benchmark import WARMUP_HANDS, format_line, run_benchmark

from rl_env import HandSampler, RandomController, Recorder, StackConfig, TableBatch

T = TypeVar("T")

DRAIN_EVERY = 100_000  # drained records, so --record does not grow without bound


def _profiled(fn: Callable[[], T], profile: bool) -> T:
    if not profile:
        return fn()
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        return fn()
    finally:
        profiler.disable()
        for key in ("cumulative", "tottime"):
            out = io.StringIO()
            pstats.Stats(profiler, stream=out).sort_stats(key).print_stats(30)
            print(f"--- profile, sorted by {key} ---")
            print(out.getvalue())


def run_rl_env(
    players: int,
    *,
    tables: int,
    hands: int,
    seed: int,
    cfg: StackConfig,
    observations: bool = True,
    record: bool = False,
    warmup: int = WARMUP_HANDS,
) -> dict:
    """One run at one table size; returns hands, decisions and seconds of the
    timed part."""
    recorder = Recorder() if record else None
    batch = TableBatch(tables, recorder=recorder, observations=observations)
    sampler = HandSampler(seed, sizes=(players,), cfg=cfg)
    controllers = {"random": RandomController(seed)}
    keys = ("random",) * players

    def seat_controllers(spec):
        return keys

    def drain(_fh) -> None:
        if recorder is not None and recorder.n_ready >= DRAIN_EVERY:
            recorder.drain()

    if warmup:
        batch.run(controllers, sampler, warmup, seat_controllers, drain)
    decisions0 = batch.n_decisions
    t0 = time.perf_counter()
    done = batch.run(controllers, sampler, hands, seat_controllers, drain)
    seconds = time.perf_counter() - t0
    return {"hands": done, "decisions": batch.n_decisions - decisions0, "seconds": seconds}


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="rl_env rollout benchmark.")
    parser.add_argument("--tables", type=int, default=512, help="tables in one batch")
    parser.add_argument("--players", default="2,6,9", help="comma-separated table sizes")
    parser.add_argument("--hands", type=int, default=20_000, help="timed hands per size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--stacks", choices=("deep", "mixed"), default="mixed", help="deep: 100 BB everywhere"
    )
    parser.add_argument("--no-obs", action="store_true", help="mask only, no observation")
    parser.add_argument("--record", action="store_true", help="record with a Recorder")
    parser.add_argument("--profile", action="store_true", help="profile the rl_env runs")
    args = parser.parse_args(argv)

    try:
        sizes = [int(s) for s in args.players.split(",") if s.strip()]
    except ValueError:
        parser.error("--players must be comma-separated integers")
    if args.tables < 1 or args.hands < 1 or not sizes or any(not 2 <= n <= 10 for n in sizes):
        parser.error("need --tables >= 1, --hands >= 1 and table sizes in 2..10")
    cfg = StackConfig(deep_share=1.0) if args.stacks == "deep" else StackConfig()
    print(
        f"tables={args.tables} hands={args.hands} seed={args.seed} stacks={args.stacks} "
        f"obs={not args.no_obs} record={args.record} (warm-up {WARMUP_HANDS} hands not timed)"
    )
    sys.stdout.flush()

    for n in sizes:
        r = _profiled(
            lambda: run_rl_env(
                n,
                tables=args.tables,
                hands=args.hands,
                seed=args.seed,
                cfg=cfg,
                observations=not args.no_obs,
                record=args.record,
            ),
            args.profile,
        )
        secs = r["seconds"] if r["seconds"] > 0 else 1e-9
        dps = r["decisions"] / secs
        hps = r["hands"] / secs
        engine = run_benchmark(n, tables=args.tables, hands=args.hands, seed=args.seed)
        share = 1.0 - dps / engine.decisions_per_s if engine.decisions_per_s > 0 else float("nan")
        print(
            f"rl_env     {n:2d} players: {r['hands']:7d} hands, {r['decisions']:8d} decisions, "
            f"{r['seconds']:7.2f} s, {hps:9.0f} hands/s, {dps:9.0f} decisions/s | "
            f"engine-only {engine.decisions_per_s:9.0f} decisions/s | rl_env share {share:.1%}"
        )
        print(format_line(engine))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
