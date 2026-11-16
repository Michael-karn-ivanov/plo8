"""Headless clients against an in-process GameServer, driven by scripted keys."""

import time
import unittest

from cli_server.server import GameServer

from cli_client.controller import ClientSettings
from cli_client.main import JoinError, join_table

GRACE = 0.2
SEED = 7


def passive(client):
    """Check / call; take the first hand-end option (rebuy / sit in)."""
    if client.menu_kind == "action":
        return "c"
    if client.menu_kind == "hand_end":
        return "1"
    return None


def aggressive(client):
    if client.menu_kind == "action":
        return "r" if any(i.kind == "raise" for i in client.menu) else "c"
    if client.menu_kind == "hand_end":
        return "1"
    return None


def idle(client):
    return None


class IntegrationTest(unittest.TestCase):
    def start_server(self, **env):
        kwargs = {"seed": SEED, "action_timeout": 5.0}
        kwargs.update(env)
        server = GameServer(port=0, grace_period=GRACE, hand_pause=0.0, env_kwargs=kwargs).start()
        self.addCleanup(server.shutdown)
        return server

    def join(self, server, name, stack=40.0):
        client, listener = join_table(
            name,
            server.port,
            0,
            stack,
            settings=ClientSettings(stack_bb=stack, hand_end_timeout=5.0, exit_delay=0.0),
        )
        self.addCleanup(client.conn.close)
        self.addCleanup(listener.close)
        return client

    def pump(self, players, until, timeout=20.0, what="condition"):
        """Run the clients' UI loops (drain, scripted key, tick) until `until()`."""
        deadline = time.monotonic() + timeout
        while True:
            for client, policy in players:
                if client.done:
                    continue
                client.drain()
                key = policy(client)
                if key is not None:
                    client.handle_key(key)
                client.tick()
            if until():
                return
            if time.monotonic() > deadline:
                raise AssertionError(f"timed out waiting for {what}")
            time.sleep(0.005)

    def server_result(self, server, hand_number, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for r in list(server.results):
                if r.hand_number == hand_number:
                    return r
            time.sleep(0.01)
        raise AssertionError(f"no server result for hand {hand_number}")

    def test_play_hands_with_consistent_models(self):
        server = self.start_server()
        a = self.join(server, "Alice")
        b = self.join(server, "Bob")
        self.assertTrue(a.state.has_settings)
        self.assertEqual(a.state.settings.big_blind_bb, 1.0)
        self.pump(
            [(a, passive), (b, passive)],
            lambda: all(c.last_result is not None and c.last_result.hand_number >= 2 for c in (a, b)),
            what="two hands",
        )
        for client in (a, b):
            result = client.last_result
            expected = self.server_result(server, result.hand_number)
            by_name = {expected.names[s]: st for s, st in expected.stacks.items()}
            # The model built from events agrees with the server before the resync.
            self.assertEqual(client.model_stacks_at_hand_end, by_name)
            self.assertEqual({result.names[s]: st for s, st in result.stacks.items()}, by_name)
            self.assertGreaterEqual(client.hands_played, 2)
            self.assertFalse(client.server_closed)

    def test_mid_hand_join_is_queued_then_seated(self):
        server = self.start_server()
        a = self.join(server, "Alice")
        b = self.join(server, "Bob")
        # Nobody acts: the first hand waits on a turn.
        self.pump(
            [(a, idle), (b, idle)],
            lambda: a.menu_kind == "action" or b.menu_kind == "action",
            what="a turn",
        )
        c = self.join(server, "Carol")
        self.assertTrue(c.state.queued)
        self.assertEqual(c.lobby_text(), "Waiting for the current hand to finish")
        players = [(a, passive), (b, passive), (c, passive)]
        self.pump(players, lambda: not c.state.queued and c.hands_played >= 1, what="Carol dealt in")
        self.assertIn("Carol", [s.name for s in a.state.seats])
        self.assertTrue(any("You sit down" in line.text for line in c.state.log))

    def test_duplicate_name_is_rejected(self):
        server = self.start_server()
        self.join(server, "Alice")
        with self.assertRaises(JoinError):
            self.join(server, "Alice")

    def test_busted_client_rebuys_through_the_menu(self):
        server = self.start_server()
        a = self.join(server, "Alice", stack=2.0)
        b = self.join(server, "Bob", stack=2.0)
        busted = []

        def policy(client):
            if client.menu_kind == "hand_end" and client.menu and client.menu[0].kind == "rebuy":
                if client.name not in busted:
                    busted.append(client.name)
            return aggressive(client)

        def rebought():
            for c in (a, b):
                if c.name in busted and any(line.text == f"{c.name} rebuys 2" for line in c.state.log):
                    me = c.state.me
                    if me is not None and me.stack > 0:
                        return True
            return False

        self.pump([(a, policy), (b, policy)], rebought, timeout=30.0, what="a rebuy")
        self.assertTrue(busted)
        self.assertFalse(a.server_closed or b.server_closed)

    def test_leave_receives_left(self):
        server = self.start_server()
        a = self.join(server, "Alice")
        b = self.join(server, "Bob")
        self.pump(
            [(a, passive), (b, passive)],
            lambda: a.hands_played >= 1 and b.hands_played >= 1,
            what="a hand",
        )

        def leaver(client):
            if client.confirm is not None:
                return "y"
            if not client.leaving and not client.state.left:
                return "l"
            return passive(client)

        self.pump([(a, leaver), (b, passive)], lambda: a.done, what="Alice to leave")
        self.assertTrue(a.state.left)
        self.assertEqual(a.exit_code, 0)
        self.assertIsNotNone(a.final_stack)
        self.pump([(b, passive)], lambda: b.state.find("Alice") is None, what="Alice gone for Bob")
        self.assertTrue(a.summary().startswith("Alice: "))


if __name__ == "__main__":
    unittest.main()
