"""cli_server entry point: command line, logging, signal handling.

    cli_server [--port PORT] [--grace-period SECONDS] [--action-timeout SECONDS]
               [--hand-pause SECONDS] [--max-seats N] [--max-buy-in BB] [--seed N]
               [--log-level LEVEL] [--e2e_test [--case NAME]]

With `--e2e_test` the binary does not serve a game: it replays the recorded
hands of `e2e_test_sets` through a real server and scripted client processes
(see `cli_server.e2e`) and exits 0 if all of them passed.
"""

from __future__ import annotations

import argparse
import logging
import math
import signal
import sys
from typing import Any, Optional, Sequence

from engine import MAX_PLAYERS, MIN_PLAYERS

from cli_server.e2e.cases import UnknownCaseError, case_name, case_path
from cli_server.e2e.driver import run_cases
from cli_server.server import DEFAULT_GRACE_PERIOD, DEFAULT_HAND_PAUSE, HOST, GameServer

__all__ = [
    "DEFAULT_PORT",
    "DEFAULT_GRACE_PERIOD",
    "DEFAULT_ACTION_TIMEOUT",
    "DEFAULT_HAND_PAUSE",
    "E2E_ACTION_TIMEOUT",
    "E2E_SEED",
    "build_parser",
    "parse_args",
    "env_kwargs",
    "main",
]

_log = logging.getLogger("cli_server")

DEFAULT_PORT = 8008
DEFAULT_ACTION_TIMEOUT = 180.0  # spec: time to act for a client (overrides the engine's 600 s)
E2E_ACTION_TIMEOUT = 10.0  # e2e mode: a stuck replay fails quickly
E2E_SEED = 0  # e2e mode: the cards a case does not fix are reproducible
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def _port(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a port number: {text!r}") from None
    if not 0 <= value <= 65535:
        raise argparse.ArgumentTypeError(f"port must be 0-65535, got {value}")
    return value


def _number(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"not a finite number: {text!r}")
    return value


def _non_negative(text: str) -> float:
    value = _number(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {text}")
    return value


def _positive(text: str) -> float:
    value = _number(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {text}")
    return value


def _seats(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None
    if not MIN_PLAYERS <= value <= MAX_PLAYERS:
        raise argparse.ArgumentTypeError(f"must be {MIN_PLAYERS}-{MAX_PLAYERS}, got {value}")
    return value


def _seed(text: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli_server",
        description="Host one fixed-limit Omaha Hi-Lo 8-or-better table on localhost.",
    )
    parser.add_argument(
        "--port",
        type=_port,
        default=DEFAULT_PORT,
        help=f"TCP port for client requests (default {DEFAULT_PORT}; 0 picks a free port)",
    )
    parser.add_argument(
        "--grace-period",
        type=_non_negative,
        default=DEFAULT_GRACE_PERIOD,
        metavar="SECONDS",
        help="wait after the last join before the first hand (default %(default)s)",
    )
    parser.add_argument(
        "--action-timeout",
        type=_positive,
        default=None,  # resolved in parse_args: depends on --e2e_test
        metavar="SECONDS",
        help=f"time a client has to answer a turn or hand_end (default {DEFAULT_ACTION_TIMEOUT:g}; "
        f"{E2E_ACTION_TIMEOUT:g} with --e2e_test)",
    )
    parser.add_argument(
        "--hand-pause",
        type=_non_negative,
        default=DEFAULT_HAND_PAUSE,
        metavar="SECONDS",
        help="pause between hands so players can see the showdown; 0 disables it "
        "(default %(default)s)",
    )
    parser.add_argument(
        "--max-seats",
        type=_seats,
        default=MAX_PLAYERS,
        metavar="N",
        help=f"seated + waiting players ({MIN_PLAYERS}-{MAX_PLAYERS}, default %(default)s)",
    )
    parser.add_argument(
        "--max-buy-in",
        type=_positive,
        default=None,
        metavar="BB",
        help="largest stack in big blinds (default: no limit)",
    )
    parser.add_argument(
        "--seed",
        type=_seed,
        default=None,
        help=f"seed for the deck shuffles (default: random; {E2E_SEED} with --e2e_test)",
    )
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=LOG_LEVELS,
        default="INFO",
        help="logging level (default %(default)s)",
    )
    parser.add_argument(
        "--e2e_test",
        "--e2e-test",
        dest="e2e_test",
        action="store_true",
        help="run the end-to-end tests (recorded hands in e2e_test_sets) instead of serving",
    )
    parser.add_argument(
        "--case",
        default=None,
        metavar="NAME",
        help="with --e2e_test: run only this case (a file in e2e_test_sets, with or "
        "without .json); default: all cases",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.case is not None:
        if not args.e2e_test:
            parser.error("--case needs --e2e_test")
        try:
            case_path(args.case)
        except UnknownCaseError as e:
            parser.error(str(e))
        args.case = case_name(args.case)
    if args.action_timeout is None:
        args.action_timeout = E2E_ACTION_TIMEOUT if args.e2e_test else DEFAULT_ACTION_TIMEOUT
    if args.seed is None and args.e2e_test:
        args.seed = E2E_SEED
    return args


def env_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """`Environment` keyword arguments from the parsed command line."""
    return {
        "action_timeout": args.action_timeout,
        "max_seats": args.max_seats,
        "max_buy_in": args.max_buy_in,
        "seed": args.seed,
    }


def _interrupt(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt(f"signal {signum}")


def _install_signal_handlers() -> dict:
    """SIGINT / SIGTERM raise KeyboardInterrupt in the main thread."""
    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[sig] = signal.signal(sig, _interrupt)
        except ValueError:  # not the main thread (e.g. embedded use)
            pass
    return previous


def _restore_signal_handlers(previous: dict) -> None:
    for sig, handler in previous.items():
        signal.signal(sig, handler)


def run_e2e(args: argparse.Namespace) -> int:
    """`--e2e_test`: every case (or `--case`) on its own server; 0 if all passed.
    An interrupt stops the current case, prints the report so far and exits 1."""
    previous = _install_signal_handlers()
    try:
        return run_cases(
            None if args.case is None else [args.case],
            seed=args.seed,
            action_timeout=args.action_timeout,
        )
    except KeyboardInterrupt:
        return 1
    finally:
        _restore_signal_handlers(previous)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        stream=sys.stderr,
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.e2e_test:
        return run_e2e(args)
    try:
        server = GameServer(
            port=args.port,
            grace_period=args.grace_period,
            hand_pause=args.hand_pause,
            env_kwargs=env_kwargs(args),
        )
    except OSError as e:
        _log.error("cannot listen on %s:%d: %s", HOST, args.port, e)
        return 1
    s = server.settings
    _log.info(
        "listening on %s:%d; blinds %g/%g BB, bets %g/%g BB, %d seats, max buy-in %s, "
        "action timeout %gs, grace period %gs, pause between hands %gs",
        HOST,
        server.port,
        s.small_blind_bb,
        s.big_blind_bb,
        s.small_bet_bb,
        s.big_bet_bb,
        s.max_seats,
        "none" if s.max_buy_in_bb is None else f"{s.max_buy_in_bb:g} BB",
        args.action_timeout,
        args.grace_period,
        args.hand_pause,
    )

    # SIGINT / SIGTERM raise KeyboardInterrupt in the main thread, which runs the
    # game loop: a running hand is aborted there (the engine refunds the bets),
    # and a lobby wait or a pause between hands ends at once.
    previous = _install_signal_handlers()
    code = 0
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log.info("interrupted, shutting down")
    except Exception:  # noqa: BLE001
        _log.exception("the game loop failed")
        code = 1
    finally:
        server.shutdown()
        _restore_signal_handlers(previous)
    return code


if __name__ == "__main__":
    sys.exit(main())
