"""cli_client entry point: command line, joining the table, running the UI.

    cli_client --server-port PORT --port PORT --name NICK [--stack BB] [--seat N]
               [--ascii] [--hand-end-timeout SECONDS] [--log-file PATH]

Exit codes: 0 after a normal leave, 1 if the server connection is lost or the
server closes the table, 2 if startup or the join fails.
"""

from __future__ import annotations

import argparse
import logging
import math
import queue
import sys
import time
from typing import Any, Callable, Optional, Sequence

from cli_server.protocol import ProtocolError, decode_join_settings, join_request

from .controller import Client, ClientSettings
from .net import HOST, CallbackListener, ConnectionLost, ServerConnection

__all__ = ["DEFAULT_SERVER_PORT", "JoinError", "build_parser", "parse_args", "join_table", "main"]

DEFAULT_SERVER_PORT = 8008
DEFAULT_STACK = 100.0
DEFAULT_HAND_END_TIMEOUT = 20.0

_log = logging.getLogger("cli_client")


class JoinError(Exception):
    """The server cannot be reached or rejected the join."""


def _port(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a port number: {text!r}") from None
    if not 0 <= value <= 65535:
        raise argparse.ArgumentTypeError(f"port must be 0-65535, got {value}")
    return value


def _server_port(text: str) -> int:
    value = _port(text)
    if value == 0:
        raise argparse.ArgumentTypeError("the server port cannot be 0")
    return value


def _positive(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {text}")
    return value


def _name(text: str) -> str:
    name = text.strip()
    if not name:
        raise argparse.ArgumentTypeError("the name cannot be empty")
    return name


def _seat(text: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {text!r}") from None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cli_client",
        description="Play at a cli_server Omaha Hi-Lo 8-or-better table from the terminal.",
    )
    p.add_argument(
        "--server-port",
        type=_server_port,
        default=DEFAULT_SERVER_PORT,
        help=f"port of the running server (default {DEFAULT_SERVER_PORT})",
    )
    p.add_argument(
        "--port",
        type=_port,
        required=True,
        help="own port to listen for server events (0 picks a free port)",
    )
    p.add_argument("--name", type=_name, required=True, help="player name (nick)")
    p.add_argument(
        "--stack",
        type=_positive,
        default=DEFAULT_STACK,
        metavar="BB",
        help="buy-in in big blinds, also the default rebuy (default %(default)g)",
    )
    p.add_argument(
        "--seat",
        type=_seat,
        default=None,
        metavar="N",
        help="join offset from the button (default: join as the big blind)",
    )
    p.add_argument("--ascii", action="store_true", help="ASCII suits and boxes")
    p.add_argument(
        "--hand-end-timeout",
        type=_positive,
        default=DEFAULT_HAND_END_TIMEOUT,
        metavar="SECONDS",
        help="time for the rebuy / sit-in menu before the default answer (default %(default)g)",
    )
    p.add_argument("--log-file", default=None, metavar="PATH", help="debug log of all messages")
    return p


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def join_table(
    name: str,
    server_port: int,
    port: int = 0,
    stack_bb: float = DEFAULT_STACK,
    seat: Optional[int] = None,
    settings: Optional[ClientSettings] = None,
    host: str = HOST,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[Client, CallbackListener]:
    """Start the callback listener, join the table and return a headless
    `Client` with its listener. Raises JoinError (or OSError if `port` cannot
    be bound)."""
    inbox: "queue.Queue[Any]" = queue.Queue()
    # Listening before `join`: the server connects to it while handling the join.
    listener = CallbackListener(name, inbox, host=host, port=port).start()
    conn = ServerConnection(server_port, host=host)
    try:
        response = conn.request(join_request(name, stack_bb, listener.port, seat))
    except ConnectionLost as e:
        listener.close()
        conn.close()
        raise JoinError(f"cannot reach the server on port {server_port}: {e}") from None
    if not response.get("ok"):
        listener.close()
        conn.close()
        raise JoinError(str(response.get("error") or "join rejected"))
    try:
        table = decode_join_settings(response)
    except ProtocolError:
        table = None
    if settings is None:
        settings = ClientSettings(stack_bb=stack_bb)
    client = Client(
        name,
        conn,
        inbox,
        clock=clock,
        settings=settings,
        table=table,
        queued=bool(response.get("queued")),
    )
    return client, listener


def _setup_logging(path: Optional[str]) -> None:
    # curses owns the terminal: no logging to stderr.
    logger = logging.getLogger("cli_client")
    logger.propagate = False
    if path:
        handler: logging.Handler = logging.FileHandler(path)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.setLevel(logging.DEBUG)
    else:
        handler = logging.NullHandler()
    logger.addHandler(handler)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print("cli_client: stdin and stdout must be a terminal", file=sys.stderr)
        return 2
    _setup_logging(args.log_file)
    settings = ClientSettings(
        stack_bb=args.stack, hand_end_timeout=args.hand_end_timeout, ascii=args.ascii
    )
    try:
        client, listener = join_table(
            args.name, args.server_port, args.port, args.stack, args.seat, settings
        )
    except JoinError as e:
        print(f"cli_client: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"cli_client: cannot listen on port {args.port}: {e}", file=sys.stderr)
        return 2

    from . import ui  # curses only once we know we have a terminal

    try:
        code = ui.run(client)
    finally:
        listener.close()
        client.conn.close()
    print(client.summary())
    return code


if __name__ == "__main__":
    sys.exit(main())
