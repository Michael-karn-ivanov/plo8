"""Command-line evaluator (the code of `//arena:match`, `bin/match.py`).

    buck2 run //arena:match -- --agent rule --opponents random --players 2,6,9 \
                               --deals 2000 --seed 0 [--histories N] \
                               [--stacks fixed|training] [--tables 256]

Lineup per table size `n`: position 0 = the agent, positions 1..n-1 = the
opponents in order, repeated. Report names are the bot names; if the agent's
name also appears among the opponents it is reported as `NAME*`, so the
measured bot is not pooled with its copies.
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Optional, TextIO

from rl_env import StackConfig

from .bots import BOT_NAMES
from .match import Entrant, MatchResult, play_match

__all__ = ["main", "lineup_for", "format_results"]

HEADER = (
    "results in big bets (2 BB) per 100 hands; 95% CI = 1.96*sd/sqrt(deals), "
    "one sample per rotation group"
)
COLUMNS = ("agent", "opponents", "players", "hands", "bb/100", "95% CI")


def lineup_for(agent: str, opponents: list[str], n: int) -> list[Entrant]:
    """Position 0 = agent, then the opponents in order, repeated."""
    agent_name = f"{agent}*" if agent in opponents else agent
    lineup = [Entrant(agent_name, agent)]
    for i in range(n - 1):
        b = opponents[i % len(opponents)]
        lineup.append(Entrant(b, b))
    return lineup


def _fmt_ci(ci: float) -> str:
    return "±nan" if math.isnan(ci) else f"±{ci:.2f}"


def format_results(results: list[MatchResult]) -> list[str]:
    """The results table: one row per report name per table size."""
    rows = [COLUMNS]
    for res in results:
        for name, st in res.stats.items():
            others = ",".join(res.lineup[p] for p in range(res.table_size) if p not in st.positions)
            rows.append(
                (
                    name,
                    others,
                    str(res.table_size),
                    str(st.hands),
                    f"{st.bb_per_100:+.2f}",
                    _fmt_ci(st.ci95),
                )
            )
    widths = [max(len(r[i]) for r in rows) for i in range(len(COLUMNS))]
    right = {2, 3, 4, 5}
    out = []
    for r in rows:
        cells = [
            r[i].rjust(widths[i]) if i in right else r[i].ljust(widths[i])
            for i in range(len(COLUMNS))
        ]
        out.append("  ".join(cells).rstrip())
    return out


def main(argv: Optional[list[str]] = None, out: Optional[TextIO] = None) -> int:
    out = out if out is not None else sys.stdout
    parser = argparse.ArgumentParser(description="arena duplicate-match evaluator.")
    parser.add_argument("--agent", required=True, choices=BOT_NAMES, help="the measured bot")
    parser.add_argument(
        "--opponents", required=True, help="comma-separated bot names filling the other seats"
    )
    parser.add_argument("--players", default="2,6,9", help="comma-separated table sizes")
    parser.add_argument("--deals", type=int, default=2000, help="deals per table size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--histories", type=int, default=0, help="sample histories per match")
    parser.add_argument(
        "--stacks",
        choices=("fixed", "training"),
        default="fixed",
        help="fixed: 100 BB every seat; training: rl_env's StackConfig()",
    )
    parser.add_argument("--tables", type=int, default=256, help="tables in the batch")
    args = parser.parse_args(argv)

    opponents = [s.strip() for s in args.opponents.split(",") if s.strip()]
    unknown = [o for o in opponents if o not in BOT_NAMES]
    if not opponents or unknown:
        parser.error(f"--opponents must be names from {', '.join(BOT_NAMES)}")
    try:
        sizes = [int(s) for s in args.players.split(",") if s.strip()]
    except ValueError:
        parser.error("--players must be comma-separated integers")
    if not sizes or any(not 2 <= n <= 10 for n in sizes):
        parser.error("table sizes must be in 2..10")
    if args.deals < 1 or args.tables < 1:
        parser.error("need --deals >= 1 and --tables >= 1")
    stacks = StackConfig() if args.stacks == "training" else 100.0
    n_hist = max(args.histories, 0)

    print(HEADER, file=out)
    results = []
    for n in sizes:
        res = play_match(
            lineup_for(args.agent, opponents, n),
            args.deals,
            args.seed,
            stacks=stacks,
            n_histories=n_hist,
            n_tables=args.tables,
        )
        results.append(res)
    for line in format_results(results):
        print(line, file=out)
    for res in results:
        print(
            f"{res.table_size} players: wall {res.wall_time:.1f} s, "
            f"{res.hands_per_second:,.0f} hands/s",
            file=out,
        )
    for res in results:
        for h in res.histories[:n_hist]:
            print("", file=out)
            print(h, file=out)
    out.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
