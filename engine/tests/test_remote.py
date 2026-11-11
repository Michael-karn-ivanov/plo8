import multiprocessing
import pickle
import threading
import unittest

from engine import (
    Action,
    ActionType,
    Environment,
    EventType,
    Player,
    RemoteAgent,
    agent_loop,
    start_agent_process,
)


class PassiveLocal:
    def __init__(self):
        self.hole_cards = {}
        self.events = 0
        self.results = 0

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


def _local_agents():
    return [PassiveLocal(), RaiseOnceLocal(), RebuyLocal()]


class ThreadedRemoteTest(unittest.TestCase):
    """The full protocol over multiprocessing pipes, with agents on threads."""

    def test_hands_with_remote_agents(self):
        local = _local_agents()
        remotes, threads = [], []
        for agent in local:
            host, peer = multiprocessing.Pipe()
            t = threading.Thread(target=agent_loop, args=(peer, agent), daemon=True)
            t.start()
            remotes.append(RemoteAgent(host, timeout=10))
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

    def test_timeout_falls_back_and_stale_replies_are_ignored(self):
        host, peer = multiprocessing.Pipe()

        def stale_responder():
            # Answers every turn with a raise, but with the wrong sequence number.
            while True:
                try:
                    msg = peer.recv()
                except (EOFError, OSError):
                    return
                if msg[0] == "close":
                    return
                if msg[0] == "turn":
                    peer.send(("action", msg[1] - 1000, Action(ActionType.RAISE)))

        t = threading.Thread(target=stale_responder, daemon=True)
        t.start()
        remote = RemoteAgent(host, timeout=0.05)
        players = [Player("remote", 100, remote), Player("b", 100), Player("c", 100)]
        env = Environment(players, seed=1)
        try:
            result = env.start_round()
        finally:
            remote.close()
            t.join(5)
        seat0 = [e for e in result.history if e.seat == 0 and e.type is not EventType.HAND_START]
        self.assertEqual(seat0[0].type, EventType.FOLD)  # facing the BB: no reply -> fold
        self.assertEqual(result.net[0], 0.0)

    def test_views_and_results_pickle(self):
        captured = {}

        class Capture(PassiveLocal):
            def on_turn(self, player, view, legal):
                captured.setdefault("view", view)
                captured.setdefault("legal", legal)
                super().on_turn(player, view, legal)

        env = Environment([Player("a", 100, Capture()), Player("b", 100, Capture())], seed=2)
        captured["result"] = env.start_round()
        for value in captured.values():
            self.assertEqual(pickle.loads(pickle.dumps(value)), value)


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


if __name__ == "__main__":
    unittest.main()
