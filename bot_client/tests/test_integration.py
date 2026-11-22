"""Bot clients at a real in-process GameServer (short grace period, no pause
between hands), run in-process on threads."""

import unittest

from cli_client.net import ServerConnection
from cli_server.protocol import action_request
from engine import PlayerActed

from bot_client.client import is_stale_turn_error
from bot_client.main import main

try:
    from . import helpers
except ImportError:  # run as a top-level module
    import helpers  # type: ignore[no-redef]

ScriptedClient = helpers.ScriptedClient
bot_argv = helpers.bot_argv
participants = helpers.participants
run_bot = helpers.run_bot
seat_of = helpers.seat_of
server = helpers.server
wait_for = helpers.wait_for

BUY_IN = 100.0


def chips(bb):
    return int(round(bb * 100))


class IntegrationBase(unittest.TestCase):
    def start_server(self, **env):
        srv = server(**env)
        self.addCleanup(srv.shutdown)
        return srv

    def scripted(self, srv, name, stack=BUY_IN, auto=True):
        c = ScriptedClient(srv.port, name, stack, auto=auto)
        self.addCleanup(c.close)
        self.assertTrue(c.ok, c.response)
        return c

    def bot(self, srv, name, *extra):
        thread, box, client = run_bot(bot_argv(srv.port, name, *extra))
        self.assertIsNotNone(client, f"{name} did not start: {box}")

        def stop():
            if thread.is_alive():
                client.request_stop()
                thread.join(10)

        self.addCleanup(stop)
        return thread, box, client

    def finish(self, thread, box, timeout=240.0):
        thread.join(timeout)
        self.assertFalse(thread.is_alive(), "the bot client did not end")
        return box.get("code")

    def results(self, srv):
        return sorted(srv.results, key=lambda r: r.hand_number)

    def dealt_in_results(self, srv, name):
        out = []
        for r in self.results(srv):
            s = seat_of(r, name)
            if s is not None and r.dealt_in.get(s):
                out.append(r)
        return out

    def assert_clean(self, client):
        self.assertEqual(client.rejected_actions, 0)
        self.assertEqual(client.rejected_hand_ends, 0)
        self.assertEqual(client.fallbacks, 0)
        self.assertEqual(client.sat_out_count, 0)
        self.assertEqual(client.problems, [])

    def assert_no_forced_folds(self, srv, name):
        for r in self.results(srv):
            for e in r.history:
                if isinstance(e, PlayerActed) and e.name == name:
                    self.assertIsNone(e.reason, f"forced fold of {name} in hand {r.hand_number}")


class RuleBotAgainstScriptedClientsTest(IntegrationBase):
    def test_rule_bot_plays_200_hands_without_errors(self):
        srv = self.start_server()
        self.scripted(srv, "Alice")
        self.scripted(srv, "Bob")
        thread, box, client = self.bot(srv, "RuleBot", "--bot", "rule", "--hands", "200")
        self.assertEqual(self.finish(thread, box), 0)
        self.assertEqual(client.hands_played, 200)
        self.assert_clean(client)
        self.assertEqual(client.status, "left")
        # Every hand the bot was dealt into on the server is one it counted.
        wait_for(lambda: len(self.dealt_in_results(srv, "RuleBot")) == 200, 10, "the last result")
        self.assertEqual(
            [r.hand_number for r in self.dealt_in_results(srv, "RuleBot")],
            [h for h, _ in client.stats.nets],
        )
        # Every hand_end the server sent (one per result the bot was seated in)
        # was answered, and every turn too.
        seated = [r for r in self.results(srv) if seat_of(r, "RuleBot") is not None]
        self.assertEqual(client.hand_ends_answered, len(seated))
        acted = sum(
            1
            for r in seated
            for e in r.history
            if isinstance(e, PlayerActed) and e.name == "RuleBot"
        )
        self.assertEqual(client.turns_answered, acted)
        self.assert_no_forced_folds(srv, "RuleBot")
        # Nets agree with the server, hand by hand.
        for r in self.dealt_in_results(srv, "RuleBot"):
            self.assertIn((r.hand_number, chips(r.net[seat_of(r, "RuleBot")])), client.stats.nets)
        wait_for(lambda: "RuleBot" not in srv.session_names(), 10, "the session to end")


class BotAgainstBotTest(IntegrationBase):
    BOTS = {
        "Rule": ("rule", "0"),
        "Calling": ("calling", "0"),
        "Random": ("random", "3"),
        "Maniac": ("maniac", "0"),
    }

    def test_bots_play_and_chips_are_conserved(self):
        srv = self.start_server()
        bots = {}
        for name, (bot, seed) in self.BOTS.items():
            bots[name] = self.bot(srv, name, "--bot", bot, "--seed", seed)
        clients = {name: b[2] for name, b in bots.items()}
        wait_for(
            lambda: min(c.hands_played for c in clients.values()) >= 200, 240, "200 hands each"
        )
        for _thread, _box, client in bots.values():
            client.request_stop()
        for name, (thread, box, client) in bots.items():
            self.assertEqual(self.finish(thread, box, 30), 0, name)
            self.assertEqual(client.status, "left", name)
            self.assertIsNotNone(client.final_stack, name)
            self.assertEqual(client.rejected_actions, 0, name)
            self.assertEqual(client.fallbacks, 0, name)
            self.assertEqual(client.sat_out_count, 0, name)
            self.assertEqual(client.problems, [], name)
        last_hand = max(c.stats.nets[-1][0] for c in clients.values())
        wait_for(
            lambda: self.results(srv) and self.results(srv)[-1].hand_number >= last_hand,
            10,
            "the results",
        )
        results = self.results(srv)
        self.assertGreaterEqual(len(results), 200)
        rebuys = {
            name: [(-1 if h is None else h, c) for h, c in cl.stats.rebuys]
            for name, cl in clients.items()
        }

        # Hand by hand: each participant starts with its stack after the last
        # hand it was seated in (or its buy-in) plus the rebuys made since.
        last = {}
        for r in results:
            n = r.hand_number
            for name, start in participants(r).items():
                p, base = last.get(name, (-1, chips(BUY_IN)))
                added = sum(c for h, c in rebuys[name] if p <= h < n)
                self.assertEqual(chips(start), base + added, f"{name} at the start of hand {n}")
            for s, name in r.names.items():
                last[name] = (n, chips(r.stacks[s]))

        # Over the run: what the leavers took away = buy-ins + rebuys.
        total_rebuys = sum(cl.stats.rebought_chips for cl in clients.values())
        taken = sum(chips(cl.final_stack) for cl in clients.values())
        self.assertEqual(taken, len(clients) * chips(BUY_IN) + total_rebuys)

        for name, cl in clients.items():
            server_nets = []
            for r in results:
                s = seat_of(r, name)
                if s is not None and r.dealt_in.get(s):
                    server_nets.append((r.hand_number, chips(r.net[s])))
            # The bot counts every dealt-in hand it got a hand_end for: all of
            # them except a last one it was folded out of by its leave.
            self.assertEqual(cl.stats.nets, server_nets[: len(cl.stats.nets)], name)
            self.assertLessEqual(len(server_nets) - len(cl.stats.nets), 1, name)
            self.assertEqual(
                chips(cl.final_stack),
                chips(BUY_IN) + sum(c for _, c in server_nets) + cl.stats.rebought_chips,
                name,
            )


class RebuyPolicyTest(IntegrationBase):
    def opponents(self, srv, stack="3"):
        for name in ("ManiacA", "ManiacB"):
            self.bot(srv, name, "--bot", "maniac", "--stack", stack)

    def history_of(self, srv, name):
        """[(hand number, start stack, end stack)] of the hands `name` was dealt into."""
        return [
            (r.hand_number, participants(r)[name], r.stacks[seat_of(r, name)])
            for r in self.dealt_in_results(srv, name)
        ]

    def test_topup_starts_every_hand_at_the_buy_in_never_above_the_maximum(self):
        srv = self.start_server(max_buy_in=3.0)
        self.opponents(srv)
        thread, box, client = self.bot(
            srv, "TopUp", "--bot", "calling", "--stack", "3", "--hands", "30"
        )
        self.assertEqual(self.finish(thread, box, 120), 0)
        self.assert_clean(client)
        wait_for(lambda: len(self.dealt_in_results(srv, "TopUp")) == 30, 10, "the results")
        prev_end = 3.0
        expected_rebuys = 0
        for n, start, end in self.history_of(srv, "TopUp"):
            # Topped up to 3 BB when below it, never above the 3 BB maximum.
            self.assertEqual(chips(start), max(300, chips(prev_end)), f"start of hand {n}")
            expected_rebuys += max(0, 300 - chips(prev_end))
            prev_end = end
        self.assertGreater(expected_rebuys, 0, "the test bot never lost chips")
        self.assertEqual(client.stats.rebought_chips, expected_rebuys)

    def test_bust_rebuys_only_at_zero(self):
        srv = self.start_server()
        self.opponents(srv)
        thread, box, client = self.bot(
            srv, "Buster", "--bot", "calling", "--stack", "3", "--rebuy", "bust", "--hands", "30"
        )
        self.assertEqual(self.finish(thread, box, 120), 0)
        self.assert_clean(client)
        wait_for(lambda: len(self.dealt_in_results(srv, "Buster")) == 30, 10, "the results")
        ends = {n: end for n, _start, end in self.history_of(srv, "Buster")}
        rebuy_hands = [h for h, _ in client.stats.rebuys]
        self.assertGreater(len(rebuy_hands), 0, "the test bot never busted")
        for h, c in client.stats.rebuys:
            self.assertEqual(ends.get(h), 0.0, f"rebuy after hand {h}")
            self.assertEqual(c, 300)
        last = max(ends)
        busts = [n for n, end in ends.items() if end == 0.0 and n != last]
        self.assertEqual(sorted(rebuy_hands), sorted(busts))

    def test_never_leaves_once_busted(self):
        srv = self.start_server()
        self.opponents(srv)
        thread, box, client = self.bot(
            srv, "Never", "--bot", "calling", "--stack", "3", "--rebuy", "never"
        )
        self.assertEqual(self.finish(thread, box, 120), 0)
        self.assert_clean(client)
        self.assertEqual(client.stats.rebought_chips, 0)
        self.assertEqual(client.status, "left")
        self.assertEqual(client.final_stack, 0.0)
        wait_for(lambda: "Never" not in srv.session_names(), 10, "the session to end")
        wait_for(
            lambda: len(self.dealt_in_results(srv, "Never")) == client.hands_played, 10, "results"
        )
        hist = self.history_of(srv, "Never")
        self.assertEqual(hist[-1][2], 0.0)
        self.assertTrue(all(end > 0 for _n, _s, end in hist[:-1]))


class LifecycleTest(IntegrationBase):
    def test_hands_limit_leaves_after_exactly_n_hands(self):
        srv = self.start_server()
        self.scripted(srv, "Alice")
        self.scripted(srv, "Bob")
        thread, box, client = self.bot(srv, "Five", "--hands", "5")
        self.assertEqual(self.finish(thread, box, 60), 0)
        self.assertEqual(client.hands_played, 5)
        self.assert_clean(client)
        last = client.stats.nets[-1][0]
        later = wait_for(
            lambda: [r for r in self.results(srv) if r.hand_number > last], 20, "a later hand"
        )
        self.assertIsNone(seat_of(later[0], "Five"))
        wait_for(lambda: "Five" not in srv.session_names(), 10, "the session to end")
        self.assertEqual(len(self.dealt_in_results(srv, "Five")), 5)

    def test_duplicate_name_is_rejected(self):
        srv = self.start_server()
        self.scripted(srv, "Dup")
        with self.assertLogs("bot_client", level="ERROR") as logs:
            code = main(bot_argv(srv.port, "Dup"))
        self.assertEqual(code, 2)
        self.assertIn("already taken", "\n".join(logs.output))

    def test_stack_above_the_maximum_is_rejected(self):
        srv = self.start_server(max_buy_in=50.0)
        with self.assertLogs("bot_client", level="ERROR") as logs:
            code = main(bot_argv(srv.port, "Rich", "--stack", "100"))
        self.assertEqual(code, 2)
        self.assertIn("exceeds the max buy-in", "\n".join(logs.output))
        self.assertEqual(srv.session_names(), [])

    def test_unreachable_server(self):
        srv = self.start_server()
        port = srv.port
        srv.shutdown()
        with self.assertLogs("bot_client", level="ERROR") as logs:
            code = main(bot_argv(port, "Lonely"))
        self.assertEqual(code, 2)
        self.assertTrue(
            any("cannot reach the server" in line or "rejected" in line for line in logs.output),
            logs.output,
        )

    def test_queued_join_waits_then_plays(self):
        srv = self.start_server()
        a = self.scripted(srv, "Alice", auto=False)
        b = self.scripted(srv, "Bob", auto=False)
        wait_for(lambda: a.pending_turn or b.pending_turn, 10, "a turn")
        thread, box, client = self.bot(srv, "Late", "--hands", "2")
        self.assertEqual(client.status, "waiting")
        a.release()
        b.release()
        self.assertEqual(self.finish(thread, box, 60), 0)
        self.assertEqual(client.hands_played, 2)
        self.assert_clean(client)

    def test_stale_turn_wording_of_the_real_server(self):
        srv = self.start_server()
        a = self.scripted(srv, "Alice", auto=False)
        b = self.scripted(srv, "Bob", auto=False)
        actor = wait_for(
            lambda: a if a.pending_turn else b if b.pending_turn else None, 10, "a turn"
        )
        other = b if actor is a else a
        seq, legal = actor.pending_turn
        conn = ServerConnection(srv.port)
        self.addCleanup(conn.close)

        wrong_seq = conn.request(action_request(actor.name, seq + 100, "call"))
        self.assertFalse(wrong_seq["ok"])
        self.assertTrue(is_stale_turn_error(wrong_seq["error"]), wrong_seq)

        no_turn = conn.request(action_request(other.name, seq, "call"))
        self.assertFalse(no_turn["ok"])
        self.assertTrue(is_stale_turn_error(no_turn["error"]), no_turn)

        if not legal.can_check:
            illegal_action = "check"
        elif not legal.can_raise:
            illegal_action = "raise"
        else:
            self.fail(f"no illegal action to try for {legal}")
        illegal = conn.request(action_request(actor.name, seq, illegal_action))
        self.assertFalse(illegal["ok"])
        self.assertIn("is not legal now", illegal["error"])
        self.assertFalse(is_stale_turn_error(illegal["error"]), illegal)
        a.release()
        b.release()


if __name__ == "__main__":
    unittest.main()
