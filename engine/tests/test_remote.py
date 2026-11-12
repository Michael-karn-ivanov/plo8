import multiprocessing
import pickle
import random
import threading
import time
import unittest

from entities import Board, Deck, Hand
from engine import (
    Action,
    ActionType,
    Environment,
    EventType,
    HandEndReply,
    Join,
    JoinListener,
    Player,
    PlayerStatus,
    RemoteAgent,
    agent_loop,
    connect_and_join,
    handle_join,
    join_table,
    start_agent_process,
)

PLAYER_ACTIONS = (EventType.FOLD, EventType.CHECK, EventType.CALL, EventType.RAISE)


class PassiveLocal:
    def __init__(self):
        self.hole_cards = {}
        self.events = 0
        self.results = 0
        self.seated = []
        self.replies = []

    def on_turn(self, player, view, legal):
        self.hole_cards[view.hand_number] = view.hole_cards
        if legal.can_check:
            player.check()
        else:
            player.call()

    def on_hand_end(self, player, result):
        self.results += 1

    def on_event(self, event):
        self.events += 1

    def on_seated(self, player, view):
        self.seated.append((player.seat, player.name, view.seat))

    def on_join_reply(self, reply):
        self.replies.append(reply)


class RaiseOnceLocal(PassiveLocal):
    """Raises the first time it can in every hand."""

    def __init__(self):
        super().__init__()
        self.raised_in = set()

    def on_turn(self, player, view, legal):
        if legal.can_raise and view.hand_number not in self.raised_in:
            self.raised_in.add(view.hand_number)
            self.hole_cards[view.hand_number] = view.hole_cards
            player.raise_()
        else:
            super().on_turn(player, view, legal)


class RebuyLocal(PassiveLocal):
    def on_hand_end(self, player, result):
        super().on_hand_end(player, result)
        player.rebuy(10)


class BustedRebuyLocal(PassiveLocal):
    """Rebuys after hand 2 if busted; records status notifications."""

    def __init__(self):
        super().__init__()
        self.dealt = []
        self.statuses = []

    def on_hand_end(self, player, result):
        super().on_hand_end(player, result)
        self.dealt.append(result.dealt_in[player.seat])
        if result.hand_number == 2 and player.stack == 0:
            player.rebuy(20)

    def on_sit_out(self, player):
        self.statuses.append(("out", player.sitting_out))

    def on_return(self, player):
        self.statuses.append(("back", player.sitting_out))


class LeaveOnTurnLocal(PassiveLocal):
    def __init__(self):
        super().__init__()
        self.left = []

    def on_turn(self, player, view, legal):
        player.leave()

    def on_left(self, player, final_stack_bb):
        self.left.append((final_stack_bb, player.status))


class LeaveOnJoinEventLocal(PassiveLocal):
    """Leaves (unsolicited, outside any turn) when someone joins the table."""

    def __init__(self):
        super().__init__()
        self.player = None
        self.left = []

    def on_hand_end(self, player, result):
        super().on_hand_end(player, result)
        self.player = player

    def on_event(self, event):
        super().on_event(event)
        if event.type is EventType.PLAYER_JOINED and self.player is not None:
            self.player.leave()

    def on_left(self, player, final_stack_bb):
        self.left.append(final_stack_bb)


class StackedDeck(Deck):
    """A deck whose top cards are fixed after every shuffle."""

    def __init__(self, top):
        super().__init__(random.Random(0))
        self.top = list(top)
        self.queue = []

    def shuffle(self):
        super().shuffle()
        self.remove(self.top)
        self.queue = list(self.top)

    def deal(self, n=1):
        out = []
        while self.queue and len(out) < n:
            out.append(self.queue.pop(0))
        if len(out) < n:
            out += super().deal(n - len(out))
        return out


def stacked(hands_in_deal_order, board):
    hands = [Hand.parse(h) for h in hands_in_deal_order]
    top = [h.cards[r] for r in range(4) for h in hands]
    return StackedDeck(top + list(Board.parse(board)))


def _local_agents():
    return [PassiveLocal(), RaiseOnceLocal(), RebuyLocal()]


def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


def _threaded_remote(agent, timeout=10):
    host, peer = multiprocessing.Pipe()
    t = threading.Thread(target=agent_loop, args=(peer, agent), daemon=True)
    t.start()
    return RemoteAgent(host, timeout=timeout), t, peer


class ThreadedRemoteTest(unittest.TestCase):
    """The full protocol over multiprocessing pipes, with agents on threads."""

    def test_hands_with_remote_agents(self):
        local = _local_agents()
        remotes, threads = [], []
        for agent in local:
            remote, t, _ = _threaded_remote(agent)
            remotes.append(remote)
            threads.append(t)
        players = [Player(f"p{i}", 100, r) for i, r in enumerate(remotes)]
        env = Environment(players, seed=3)
        try:
            results = env.run(3)
        finally:
            for r in remotes:
                r.close()
            for t in threads:
                t.join(5)
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertTrue(
                any(e.type is EventType.RAISE and e.seat == 1 for e in result.history)
            )
        self.assertEqual(players[2].rebuy_chips, 3 * 1000)
        self.assertEqual(
            sum(p.chips for p in players), 30000 + sum(p.rebuy_chips for p in players)
        )
        # Each agent got its own hole cards, the end-of-hand results and the events.
        for seat, agent in enumerate(local):
            self.assertEqual(agent.hole_cards[3], players[seat].hole_cards)
            self.assertEqual(agent.results, 3)
            self.assertGreater(agent.events, 0)
        self.assertTrue(all(p.status is PlayerStatus.ACTIVE for p in players))

    def test_timeout_folds_sits_out_and_late_reply_is_discarded(self):
        host, peer = multiprocessing.Pipe()

        def late_responder():
            # Answers turns too late (with the right sequence number), hand ends at once.
            while True:
                try:
                    msg = peer.recv()
                except (EOFError, OSError):
                    return
                if msg[0] == "close":
                    return
                if msg[0] == "turn":
                    time.sleep(0.3)
                    peer.send(("action", msg[1], Action(ActionType.CALL)))
                elif msg[0] == "hand_end":
                    peer.send(("hand_end", msg[1], HandEndReply()))

        t = threading.Thread(target=late_responder, daemon=True)
        t.start()
        remote = RemoteAgent(host)  # timeout taken from the environment
        players = [Player("remote", 100, remote), Player("b", 100), Player("c", 100)]
        env = Environment(players, seed=1, action_timeout=0.05)
        try:
            result = env.start_round()
            status_after_first = players[0].status
            time.sleep(0.5)  # let the late replies arrive
            second = env.start_round()
        finally:
            remote.close()
            t.join(5)
        seat0 = [e.type for e in result.history if e.seat == 0 and e.type in PLAYER_ACTIONS]
        self.assertEqual(seat0, [EventType.FOLD])  # no reply in time -> folded
        self.assertEqual(result.net[0], 0.0)
        self.assertIs(status_after_first, PlayerStatus.SITTING_OUT)
        self.assertGreater(players[0].chips, 0)
        # The late CALL was discarded, not applied to a later request.
        self.assertFalse(second.dealt_in[0])
        self.assertGreaterEqual(remote.discarded, 1)
        self.assertFalse(
            any(e.seat == 0 and e.type in PLAYER_ACTIONS for e in second.history)
        )

    def test_sitting_out_remote_agent_rebuys_from_hand_end(self):
        # Hand 1: the remote agent (seat 2) is all-in on the BB and loses it all.
        deck = stacked(["9c9s5c5s", "KhKdQcQh", "AcAs3d4d"], "2s7d8h Tc Ts")
        local = BustedRebuyLocal()
        remote, t, _ = _threaded_remote(local)
        players = [
            Player("p0", 100, PassiveLocal()),
            Player("p1", 100, PassiveLocal()),
            Player("remote", 1, remote),
        ]
        env = Environment(players, deck=deck, seed=1)
        try:
            results = env.run(3)
        finally:
            remote.close()
            t.join(5)
        self.assertEqual(len(results), 3)
        self.assertEqual([r.dealt_in[2] for r in results], [True, False, True])
        self.assertEqual(local.dealt, [True, False, True])
        self.assertEqual(local.statuses, [("out", True), ("back", False)])
        self.assertEqual(players[2].rebuy_chips, 2000)
        self.assertIs(players[2].status, PlayerStatus.ACTIVE)

    def test_remote_agent_leaves_during_turn(self):
        local = LeaveOnTurnLocal()
        remote, t, _ = _threaded_remote(local)
        players = [Player("remote", 100, remote), Player("b", 100), Player("c", 100)]
        env = Environment(players, seed=1)
        result = env.start_round()
        t.join(5)
        self.assertFalse(t.is_alive())  # agent_loop returned after "left"
        seat0 = [e.type for e in result.history if e.seat == 0 and e.type in PLAYER_ACTIONS]
        self.assertEqual(seat0, [EventType.FOLD])
        self.assertIs(players[0].status, PlayerStatus.LEFT)
        self.assertEqual(local.left, [(100.0, PlayerStatus.LEFT)])
        self.assertEqual([p.name for p in env.players], ["b", "c"])
        self.assertTrue(remote.closed)

    def test_remote_agent_leaves_outside_its_turn(self):
        local = LeaveOnJoinEventLocal()
        remote, t, _ = _threaded_remote(local)
        players = [Player("remote", 100, remote), Player("b", 100), Player("c", 100)]
        env = Environment(players, seed=1)
        env.start_round()
        env.join(10, name="x")  # the remote agent reacts by sending Leave
        self.assertTrue(_wait_for(lambda: players[0].status is PlayerStatus.LEFT))
        t.join(5)
        self.assertEqual(sorted(p.name for p in env.players), ["b", "c", "x"])
        self.assertEqual(len(local.left), 1)

    def test_closed_connection_leaves(self):
        host, peer = multiprocessing.Pipe()
        players = [Player("remote", 100, RemoteAgent(host)), Player("b", 100), Player("c", 100)]
        env = Environment(players, seed=1)
        peer.close()  # the agent process went away
        self.assertTrue(_wait_for(lambda: players[0].status is PlayerStatus.LEFT))
        self.assertEqual([p.name for p in env.players], ["b", "c"])

    def test_views_and_results_pickle(self):
        captured = {}

        class Capture(PassiveLocal):
            def on_turn(self, player, view, legal):
                captured.setdefault("view", view)
                captured.setdefault("legal", legal)
                super().on_turn(player, view, legal)

        env = Environment([Player("a", 100, Capture()), Player("b", 100, Capture())], seed=2)
        captured["result"] = env.start_round()
        captured["reply"] = HandEndReply(5.0, True, False)
        for value in captured.values():
            self.assertEqual(pickle.loads(pickle.dumps(value)), value)


class ThreadedJoinTest(unittest.TestCase):
    """Join messages over pipes / a listener, with the joining agent on a thread."""

    def test_agent_joins_mid_hand_via_join_message(self):
        host, peer = multiprocessing.Pipe()
        joiner = PassiveLocal()
        out = {}
        t = threading.Thread(
            target=lambda: out.update(reply=join_table(peer, Join("remote", 50), joiner)),
            daemon=True,
        )
        t.start()

        env_ref, holder = [], {}

        class Host(PassiveLocal):
            def on_turn(self, player, view, legal):
                if "player" not in holder:
                    # What the host's listener does when a Join arrives mid-hand.
                    p = handle_join(env_ref[0], host, timeout=10)
                    holder["player"] = p
                    holder["status"] = p.status
                super().on_turn(player, view, legal)

        players = [Player("a", 100, Host()), Player("b", 100, PassiveLocal())]
        env = Environment(players, seed=7)
        env_ref.append(env)
        try:
            results = env.run(2)
        finally:
            if holder.get("player") is not None:
                holder["player"].agent.close()
            t.join(5)

        p = holder["player"]
        self.assertIs(holder["status"], PlayerStatus.WAITING)
        reply = out["reply"]
        self.assertTrue(reply.accepted)
        self.assertTrue(reply.queued)
        self.assertEqual((reply.name, reply.seat), ("remote", None))
        self.assertEqual(joiner.replies, [reply])
        # Not in hand 1; seated after it as big blind, and played hand 2.
        self.assertNotIn("remote", results[0].names.values())
        self.assertIs(p.status, PlayerStatus.ACTIVE)
        self.assertEqual(joiner.seated, [(p.seat, "remote", p.seat)])
        self.assertEqual(results[1].names[p.seat], "remote")
        self.assertEqual(results[1].bb_seat, p.seat)
        self.assertEqual(joiner.results, 1)
        self.assertEqual(set(joiner.hole_cards), {2})
        self.assertEqual(joiner.hole_cards[2], p.hole_cards)

    def test_join_between_hands_is_seated_at_once(self):
        host, peer = multiprocessing.Pipe()
        joiner = PassiveLocal()
        out = {}
        t = threading.Thread(
            target=lambda: out.update(reply=join_table(peer, Join(None, 30, seat=1), joiner)),
            daemon=True,
        )
        t.start()
        env = Environment([Player("a", 100), Player("b", 100), Player("c", 100)])
        p = handle_join(env, host, timeout=10)
        try:
            self.assertIs(p.status, PlayerStatus.ACTIVE)
            self.assertEqual(p.name, "Player1")
            self.assertEqual(p.seat, 1)
            result = env.start_round()
            self.assertEqual(result.sb_seat, p.seat)
        finally:
            p.agent.close()
            t.join(5)
        reply = out["reply"]
        self.assertTrue(reply.accepted)
        self.assertFalse(reply.queued)
        self.assertEqual(reply.seat, 1)
        # The "seated" notification may arrive before the reply; it is still delivered.
        self.assertEqual(joiner.seated, [(1, "Player1", 1)])
        self.assertEqual(joiner.results, 1)

    def test_rejected_join(self):
        env = Environment([Player("a", 100), Player("b", 100)], max_seats=3)
        cases = [Join("a", 10), Join("x", 0), Join("y", 10, seat=7)]
        for join in cases:
            host, peer = multiprocessing.Pipe()
            out = {}
            t = threading.Thread(
                target=lambda: out.update(reply=join_table(peer, join, PassiveLocal())),
                daemon=True,
            )
            t.start()
            self.assertIsNone(handle_join(env, host))
            t.join(5)
            self.assertFalse(out["reply"].accepted, join)
            self.assertTrue(out["reply"].error)
        self.assertEqual(len(env.players), 2)

        # Not a Join message.
        host, peer = multiprocessing.Pipe()
        peer.send(("hello",))
        self.assertIsNone(handle_join(env, host))
        kind, reply = peer.recv()
        self.assertEqual(kind, "joined")
        self.assertFalse(reply.accepted)

    def test_listener_accepts_and_rejects(self):
        env = Environment([Player("a", 100), Player("b", 100)], max_seats=3)
        listener = JoinListener(env, address=("127.0.0.1", 0), family="AF_INET", timeout=10)
        listener.start()
        try:
            dup = connect_and_join(listener.address, Join("a", 10), PassiveLocal())
            self.assertFalse(dup.accepted)
            self.assertIn("taken", dup.error)

            joiner = PassiveLocal()
            out = {}
            t = threading.Thread(
                target=lambda: out.update(
                    reply=connect_and_join(listener.address, Join("c", 20), joiner)
                ),
                daemon=True,
            )
            t.start()
            self.assertTrue(_wait_for(lambda: len(listener.players) == 1))
            c = listener.players[0]
            self.assertIs(c.status, PlayerStatus.ACTIVE)
            self.assertEqual(len(env.players), 3)

            full = connect_and_join(listener.address, Join("d", 20), PassiveLocal())
            self.assertFalse(full.accepted)
            self.assertTrue(_wait_for(lambda: listener.rejected == 2))

            env.start_round()
            c.agent.close()
            t.join(5)
        finally:
            listener.close()
        self.assertTrue(out["reply"].accepted)
        self.assertEqual(joiner.results, 1)


@unittest.skipUnless(
    "fork" in multiprocessing.get_all_start_methods(), "needs the fork start method"
)
class ProcessRemoteTest(unittest.TestCase):
    """Agents in child processes."""

    def test_hands_with_agents_in_child_processes(self):
        ctx = multiprocessing.get_context("fork")
        remotes, procs = [], []
        for agent in _local_agents():
            remote, proc = start_agent_process(agent, timeout=10, context=ctx)
            remotes.append(remote)
            procs.append(proc)
        players = [Player(f"p{i}", 100, r) for i, r in enumerate(remotes)]
        env = Environment(players, seed=5)
        try:
            results = env.run(2)
        finally:
            for r in remotes:
                r.close()
            for proc in procs:
                proc.join(5)
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertTrue(
                any(e.type is EventType.RAISE and e.seat == 1 for e in result.history)
            )
        self.assertEqual(players[2].rebuy_chips, 2 * 1000)
        self.assertEqual(
            sum(p.chips for p in players), 30000 + sum(p.rebuy_chips for p in players)
        )
        for proc in procs:
            self.assertEqual(proc.exitcode, 0)
        # Closing on purpose is not a leave.
        self.assertEqual(len(env.players), 3)

    def test_agent_process_joins_mid_hand(self):
        ctx = multiprocessing.get_context("fork")
        state = {}

        class Host(PassiveLocal):
            def on_turn(self, player, view, legal):
                if "proc" not in state:
                    listener = state["listener"]
                    proc = ctx.Process(
                        target=connect_and_join,
                        args=(listener.address, Join("child", 40), PassiveLocal()),
                        daemon=True,
                    )
                    proc.start()
                    state["proc"] = proc
                    state["arrived"] = _wait_for(lambda: len(listener.players) == 1)
                    if state["arrived"]:
                        state["status"] = listener.players[0].status
                super().on_turn(player, view, legal)

        env = Environment([Player("a", 100, Host()), Player("b", 100, PassiveLocal())], seed=4)
        listener = JoinListener(env, address=("127.0.0.1", 0), family="AF_INET", timeout=10)
        state["listener"] = listener.start()
        try:
            results = env.run(2)
        finally:
            for p in listener.players:
                p.agent.close()
            listener.close()
            if "proc" in state:
                state["proc"].join(5)

        self.assertTrue(state["arrived"])
        self.assertIs(state["status"], PlayerStatus.WAITING)
        child = listener.players[0]
        self.assertNotIn("child", results[0].names.values())
        self.assertEqual(results[1].names[child.seat], "child")
        self.assertEqual(results[1].bb_seat, child.seat)
        self.assertIsNotNone(child.hole_cards)
        self.assertEqual(sum(p.chips for p in env.players), 20000 + 4000)
        self.assertEqual(state["proc"].exitcode, 0)


if __name__ == "__main__":
    unittest.main()
