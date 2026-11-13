"""cli_server entry point: command line, logging, signal handling.

    cli_server [--port PORT] [--grace-period SECONDS] [--action-timeout SECONDS]
               [--max-seats N] [--max-buy-in BB] [--seed N] [--log-level LEVEL]
"""

from __future__ import annotations

import argparse
import logging
import math
import signal
import sys
from typing import Any, Optional, Sequence

from engine import DEFAULT_ACTION_TIMEOUT, MAX_PLAYERS, MIN_PLAYERS

from cli_server.server import HOST, GameServer

__all__ = ["DEFAULT_PORT", "DEFAULT_GRACE_PERIOD", "build_parser", "parse_args", "env_kwargs", "main"]

_log = logging.getLogger("cli_server")

DEFAULT_PORT = 8008
DEFAULT_GRACE_PERIOD = 15.0
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
        default=DEFAULT_ACTION_TIMEOUT,
        metavar="SECONDS",
        help="time a client has to answer a turn or hand_end (default %(default)s)",
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
    parser.add_argument("--seed", type=_seed, default=None, help="seed for the deck shuffles")
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=LOG_LEVELS,
        default="INFO",
        help="logging level (default %(default)s)",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


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


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        stream=sys.stderr,
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        server = GameServer(
            port=args.port, grace_period=args.grace_period, env_kwargs=env_kwargs(args)
        )
    except OSError as e:
        _log.error("cannot listen on %s:%d: %s", HOST, args.port, e)
        return 1
    _log.info(
        "listening on %s:%d (grace period %gs, action timeout %gs)",
        HOST,
        server.port,
        args.grace_period,
        args.action_timeout,
    )

    # SIGINT / SIGTERM raise KeyboardInterrupt in the main thread, which runs the
    # game loop: a running hand is aborted there (the engine refunds the bets).
    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[sig] = signal.signal(sig, _interrupt)
        except ValueError:  # not the main thread (e.g. embedded use)
            pass
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
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return code


if __name__ == "__main__":
    sys.exit(main())
