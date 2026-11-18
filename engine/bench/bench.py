"""Step-mode self-play benchmark for the engine (standard library only).

    buck2 run //engine:bench -- [--tables N] [--players 2,6,9] [--hands H]
                                [--stack BB] [--seed S] [--views] [--callback]
                                [--profile]

A thin command-line wrapper around `engine.benchmark`: one `run_benchmark` per
table size (and, with `--callback`, one more in callback mode), then the report
from `format_report`. `--profile` wraps the runs in cProfile and prints the top
functions by cumulative and total time.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
from typing import Callable, Optional, TypeVar

from engine.benchmark import WARMUP_HANDS, BenchResult, format_line, run_benchmark

T = TypeVar("T")


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
            pstats.Stats(profiler, stream=out).sort_stats(key).print_stats(25)
            print(f"--- profile, sorted by {key} ---")
            print(out.getvalue())


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
    parser.add_argument("--profile", action="store_true", help="profile the runs")
    args = parser.parse_args(argv)

    try:
        sizes = [int(s) for s in args.players.split(",") if s.strip()]
    except ValueError:
        parser.error("--players must be comma-separated integers")
    if args.tables < 1 or args.hands < 1 or not sizes or any(not 2 <= n <= 10 for n in sizes):
        parser.error("need --tables >= 1, --hands >= 1 and table sizes in 2..10")
    print(
        f"tables={args.tables} hands={args.hands} stack={args.stack:g} seed={args.seed} "
        f"views={args.views} (warm-up {WARMUP_HANDS} hands not timed)"
    )
    sys.stdout.flush()

    results: list[BenchResult] = []
    common = dict(tables=args.tables, hands=args.hands, stack=args.stack, seed=args.seed)

    def runs() -> None:
        for n in sizes:
            modes = [False, True] if args.callback else [False]
            for callback in modes:
                result = run_benchmark(n, views=args.views, callback=callback, **common)
                results.append(result)
                print(format_line(result))  # progress: one line as soon as it is done
                sys.stdout.flush()

    _profiled(runs, args.profile)
    return 0


if __name__ == "__main__":
    sys.exit(main())
