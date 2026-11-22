"""bot_client entry point: command line, logging, startup / join, signals.

    bot_client [--server-port PORT] [--port PORT] [--name NAME] [--stack BB]
               [--bot NAME] [--seed N] [--rebuy topup|bust|never] [--hands N]
               [--log-level LEVEL]

Exit codes (as cli_client): 0 for a normal end (hands reached, busted under
`--rebuy never`, left, the server closed the table, a confirmed user stop), 1
for a lost connection, 2 for invalid arguments or a failed startup / join.
"""

from __future__ import annotations

import argparse
import logging
import math
import queue
import signal
import sys
import threading
from typing import Any, Callable, Optional, Sequence

from arena import BOT_NAMES
from cli_client.net import CallbackListener, ConnectionLost, ServerConnection
from cli_server.protocol import ProtocolError, TableSettings, decode_join_settings, join_request

from .client import BotClient
from .players import make_player
from .policy import POLICIES

__all__ = [
    "DEFAULT_SERVER_PORT",
    "DEFAULT_STACK",
    "LOG_LEVELS",
    "StartupError",
    "build_parser",
    "parse_args",
    "setup_logging",
    "bot_logger",
    "start_bot",
    "main",
]

DEFAULT_SERVER_PORT = 8008
DEFAULT_STACK = 100.0
LOG_LEVELS = {"warning": logging.WARNING, "info": logging.INFO, "debug": logging.DEBUG}
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


class StartupError(Exception):
    """Startup failed (port, server, join): exit status 2."""


# --- command line -------------------------------------------------------------------


def _int(text: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None


def _port(text: str) -> int:
    value = _int(text)
    if not 0 <= value <= 65535:
        raise argparse.ArgumentTypeError(f"port must be 0-65535, got {value}")
    return value


def _server_port(text: str) -> int:
    value = _int(text)
    if not 1 <= value <= 65535:
        raise argparse.ArgumentTypeError(f"server port must be 1-65535, got {value}")
    return value


def _stack(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"must be a finite number > 0, got {text}")
    return value


def _hands(text: str) -> int:
    value = _int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def _name(text: str) -> str:
    name = text.strip()
    if not name:
        raise argparse.ArgumentTypeError("the name cannot be empty")
    return name


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bot_client",
        description="Seat an arena bot at a running cli_server table and play it there.",
    )
    p.add_argument(
        "--server-port",
        type=_server_port,
        default=DEFAULT_SERVER_PORT,
        metavar="PORT",
        help=f"port of the running server (default {DEFAULT_SERVER_PORT})",
    )
    p.add_argument(
        "--port",
        type=_port,
        default=0,
        metavar="PORT",
        help="own callback port for server messages (default 0: a free port, logged)",
    )
    p.add_argument(
        "--name", type=_name, default=None, help="player name (default: <bot>-bot, e.g. rule-bot)"
    )
    p.add_argument(
        "--stack",
        type=_stack,
        default=DEFAULT_STACK,
        metavar="BB",
        help="buy-in in big blinds, also the rebuy target (default %(default)g)",
    )
    # Ways of choosing the player; `--model PATH` joins this group later.
    who = p.add_mutually_exclusive_group()
    who.add_argument(
        "--bot",
        choices=BOT_NAMES,
        default="rule",
        help="the arena bot to play (default %(default)s)",
    )
    p.add_argument(
        "--seed", type=_int, default=0, help="seed for bots that use randomness (default 0)"
    )
    p.add_argument(
        "--rebuy",
        choices=POLICIES,
        default="topup",
        help="rebuy policy: topup to --stack, rebuy only when busted, or never (default topup)",
    )
    p.add_argument(
        "--hands",
        type=_hands,
        default=None,
        metavar="N",
        help="leave after N hands dealt in (default: play until stopped)",
    )
    p.add_argument(
        "--log-level",
        type=str.lower,
        choices=tuple(LOG_LEVELS),
        default="info",
        help="warning, info (one line per hand; default) or debug (every message)",
    )
    return p


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    if args.name is None:
        args.name = f"{args.bot}-bot"
    return args


# --- logging ------------------------------------------------------------------------


def setup_logging(level: str) -> None:
    """stderr handler on the root logger (once per process); `level` for
    bot_client's loggers and cli_client's networking (debug: every message)."""
    root = logging.getLogger()
    if not any(getattr(h, "_bot_client", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        handler._bot_client = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    value = LOG_LEVELS[level]
    logging.getLogger("bot_client").setLevel(value)
    logging.getLogger("cli_client.net").setLevel(value)


def bot_logger(name: str) -> logging.Logger:
    """The logger of one bot (a child of `bot_client`)."""
    return logging.getLogger("bot_client").getChild(name.replace(".", "_"))


def _describe_settings(s: TableSettings) -> str:
    max_buy_in = "none" if s.max_buy_in_bb is None else f"{s.max_buy_in_bb:g} BB"
    timeout = "none" if s.action_timeout is None else f"{s.action_timeout:g}s"
    return (
        f"table settings: blinds {s.small_blind_bb:g}/{s.big_blind_bb:g} BB, "
        f"bets {s.small_bet_bb:g}/{s.big_bet_bb:g} BB, {s.max_seats} seats, "
        f"max buy-in {max_buy_in}, time to act {timeout}"
    )


# --- startup ------------------------------------------------------------------------


def start_bot(
    args: argparse.Namespace,
    inbox: Optional["queue.Queue[Any]"] = None,
    *,
    leave_timeout: float = 5.0,
) -> BotClient:
    """Create the player, open the callback listener, join. Returns the client
    (not yet running; it owns the listener and the connection). Raises
    StartupError (exit status 2)."""
    log = bot_logger(args.name)
    player = make_player(args)  # before listening: a failing bot fails before joining
    inbox = inbox if inbox is not None else queue.Queue()
    try:
        listener = CallbackListener(args.name, inbox, port=args.port).start()
    except OSError as e:
        raise StartupError(f"cannot listen on port {args.port}: {e}") from None
    log.info("listening on port %d", listener.port)
    conn = ServerConnection(args.server_port)
    try:
        try:
            response = conn.request(join_request(args.name, args.stack, listener.port, None))
        except ConnectionLost as e:
            raise StartupError(f"cannot reach the server on port {args.server_port}: {e}") from None
        if not response.get("ok"):
            raise StartupError(
                f"join of {args.name} rejected: {response.get('error') or 'join rejected'}"
            )
    except BaseException:
        listener.close()
        conn.close()
        raise
    queued = bool(response.get("queued"))
    if queued:
        log.info("joined as %s: queued until the current hand ends", args.name)
    else:
        log.info("joined as %s: seat %s", args.name, response.get("seat"))
    try:
        settings = decode_join_settings(response)
    except ProtocolError as e:
        log.warning("table settings not readable: %s", e)
        settings = None
    if settings is None:
        log.info("table settings: not sent")
    else:
        log.info("%s", _describe_settings(settings))
    return BotClient(
        args.name,
        player,
        conn,
        inbox,
        stack_bb=args.stack,
        rebuy=args.rebuy,
        max_hands=args.hands,
        settings=settings,
        queued=queued,
        leave_timeout=leave_timeout,
        listener=listener,
        logger=log,
    )


def _install_signal_handlers(client: BotClient) -> dict:
    """SIGINT / SIGTERM ask the client to stop (only on the main thread)."""
    if threading.current_thread() is not threading.main_thread():
        return {}

    def handler(signum: int, frame: Any) -> None:
        client.request_stop()

    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous[sig] = signal.signal(sig, handler)
        except (ValueError, OSError):
            pass
    return previous


def _restore_signal_handlers(previous: dict) -> None:
    for sig, handler in previous.items():
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass


def main(
    argv: Optional[Sequence[str]] = None,
    *,
    on_start: Optional[Callable[[BotClient], None]] = None,
    leave_timeout: float = 5.0,
) -> int:
    """Run one bot until an end condition; returns the exit status. Invalid
    arguments exit through argparse (SystemExit 2). `on_start` (tests) gets the
    client once it has joined, before it runs."""
    args = parse_args(argv)
    setup_logging(args.log_level)
    log = bot_logger(args.name)
    try:
        client = start_bot(args, leave_timeout=leave_timeout)
    except StartupError as e:
        log.error("%s", e)
        return 2
    previous = _install_signal_handlers(client)
    code = 1
    try:
        if on_start is not None:
            on_start(client)
        code = client.run()
    except KeyboardInterrupt:
        code = 1
    finally:
        _restore_signal_handlers(previous)
        client.close()
        # The summary is shown at every level, also at `warning`.
        log.log(max(logging.INFO, log.getEffectiveLevel()), "%s", client.summary())
    return code


if __name__ == "__main__":
    sys.exit(main())
