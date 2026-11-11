"""Adapter for agents running in separate processes.

The host process owns the Environment and Dealer. Each remote seat gets a
`RemoteAgent` proxy that talks to `agent_loop` running in the agent's process
over a `multiprocessing` connection.

Protocol (all messages are picklable tuples):
  host -> agent: ("turn", seq, TableView, LegalActions)
                 ("hand_end", seq, seat, HandResult)
                 ("event", Event)
                 ("close",)
  agent -> host: ("action", seq, Action | None)
                 ("rebuy", seq, amount_bb | None)

Replies carry the request's sequence number, so a late reply to a request that
already timed out is discarded instead of being taken as the next answer.
"""

from __future__ import annotations

import multiprocessing
import time
from typing import Any, Optional

from entities import Hand

from .actions import Action, ActionType, IllegalActionError, LegalActions
from .player import Player, fallback_action
from .views import Event, HandResult, TableView

__all__ = ["RemoteAgent", "RemotePlayer", "agent_loop", "start_agent_process"]


class RemoteAgent:
    """Host-side `Agent` that forwards callbacks to an agent in another process."""

    def __init__(self, conn: Any, timeout: Optional[float] = None) -> None:
        self.conn = conn
        self.timeout = timeout
        self._seq = 0
        self.closed = False

    def _request(self, kind: str, *payload: Any) -> Optional[tuple]:
        if self.closed:
            return None
        self._seq += 1
        seq = self._seq
        try:
            self.conn.send((kind, seq, *payload))
        except (OSError, EOFError, ValueError):
            self.closed = True
            return None
        deadline = None if self.timeout is None else time.monotonic() + self.timeout
        while True:
            if deadline is not None:
                remaining = max(0.0, deadline - time.monotonic())
                if not self.conn.poll(remaining):
                    return None  # timed out
            try:
                reply = self.conn.recv()
            except (OSError, EOFError):
                self.closed = True
                return None
            if isinstance(reply, tuple) and len(reply) == 3 and reply[1] == seq:
                return reply
            # Stale or malformed reply: discard and keep waiting.

    def on_turn(self, player: Player, view: TableView, legal: LegalActions) -> None:
        reply = self._request("turn", view, legal)
        action = reply[2] if reply is not None and reply[0] == "action" else None
        if not isinstance(action, Action) or not legal.allows(action.type):
            action = fallback_action(legal)  # no reply in time: check if legal, else fold
        player.act(action)

    def on_hand_end(self, player: Player, result: HandResult) -> None:
        reply = self._request("hand_end", player.seat, result)
        amount = reply[2] if reply is not None and reply[0] == "rebuy" else None
        if amount:
            try:
                player.rebuy(amount)
            except (ValueError, TypeError, IllegalActionError):
                pass  # invalid rebuy requests are ignored

    def on_event(self, event: Event) -> None:
        if self.closed:
            return
        try:
            self.conn.send(("event", event))
        except (OSError, EOFError, ValueError):
            self.closed = True

    def close(self) -> None:
        if not self.closed:
            try:
                self.conn.send(("close",))
            except (OSError, EOFError, ValueError):
                pass
            self.closed = True


class RemotePlayer:
    """Agent-side stand-in for `Player`, with the same action/rebuy API.

    It records the agent's decision, which `agent_loop` sends back to the host;
    the host re-validates it against the real rules.
    """

    def __init__(self) -> None:
        self.seat: Optional[int] = None
        self.name: Optional[str] = None
        self.stack = 0.0
        self.hole_cards: Optional[Hand] = None
        self._legal: Optional[LegalActions] = None
        self._pending: Optional[Action] = None
        self._rebuy_open = False
        self._rebuy = 0.0

    def _update_from_view(self, view: TableView) -> None:
        self.seat = view.seat
        me = view.seats[view.seat]
        self.name = me.name
        self.stack = me.stack
        self.hole_cards = view.hole_cards

    def fold(self) -> None:
        self._record(ActionType.FOLD)

    def check(self) -> None:
        self._record(ActionType.CHECK)

    def call(self) -> None:
        self._record(ActionType.CALL)

    def raise_(self) -> None:
        self._record(ActionType.RAISE)

    def act(self, action: Action) -> None:
        if not isinstance(action, Action):
            raise IllegalActionError(f"not an Action: {action!r}")
        self._record(action.type)

    def _record(self, action_type: ActionType) -> None:
        if self._legal is None:
            raise IllegalActionError("it is not this player's turn")
        if self._pending is not None:
            raise IllegalActionError("already acted this turn")
        if not self._legal.allows(action_type):
            raise IllegalActionError(f"{action_type.value} is not legal now")
        self._pending = Action(action_type)

    def rebuy(self, amount_bb: float) -> None:
        if not self._rebuy_open:
            raise IllegalActionError("cannot rebuy during a hand")
        if isinstance(amount_bb, bool) or not isinstance(amount_bb, (int, float)) or amount_bb <= 0:
            raise ValueError(f"rebuy amount must be positive, got {amount_bb!r}")
        self._rebuy += amount_bb
        self.stack += amount_bb

    def show_cards(self) -> Optional[Hand]:
        return self.hole_cards


def agent_loop(conn: Any, agent: Any) -> None:
    """Run in the agent process: dispatch host messages to a local agent.

    `agent` has the same callbacks as an in-process agent (`on_turn`,
    `on_hand_end`, optional `on_event`) and receives a `RemotePlayer`.
    Returns when the host sends "close" or the connection is closed.
    """
    player = RemotePlayer()
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        kind = msg[0]
        if kind == "close":
            return
        if kind == "event":
            callback = getattr(agent, "on_event", None)
            if callable(callback):
                callback(msg[1])
        elif kind == "turn":
            _, seq, view, legal = msg
            player._update_from_view(view)
            player._legal = legal
            player._pending = None
            try:
                agent.on_turn(player, view, legal)
            except IllegalActionError:
                pass
            action = player._pending
            player._legal = None
            player._pending = None
            conn.send(("action", seq, action))
        elif kind == "hand_end":
            _, seq, seat, result = msg
            player.seat = seat
            player.stack = result.stacks.get(seat, player.stack)
            player._rebuy = 0.0
            player._rebuy_open = True
            try:
                agent.on_hand_end(player, result)
            finally:
                player._rebuy_open = False
                player.hole_cards = None
            conn.send(("rebuy", seq, player._rebuy or None))


def start_agent_process(
    agent: Any, timeout: Optional[float] = None, context: Any = None
) -> tuple[RemoteAgent, Any]:
    """Start `agent` in a child process; return (RemoteAgent for the host, Process).

    With the "spawn" start method (macOS/Windows default) the agent must be
    picklable and importable in the child.
    """
    ctx = context if context is not None else multiprocessing.get_context()
    host_conn, agent_conn = ctx.Pipe()
    proc = ctx.Process(target=agent_loop, args=(agent_conn, agent), daemon=True)
    proc.start()
    agent_conn.close()  # the child owns its end now
    return RemoteAgent(host_conn, timeout=timeout), proc
