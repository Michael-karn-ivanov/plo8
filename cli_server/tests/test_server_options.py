"""GameServer options used by the e2e driver: required_players, max_hands, and
env_kwargs carrying chips_per_bb / deck."""

import random
import threading
import time
import unittest

from entities import Card

from cli_server.e2e.deck import StackedDeck
from cli_server.server import STOPPED, GameServer

try:
    from cli_server_tests.tests.fake_client import FakeClient, is_event, is_type
except ImportError:  # run directly from the tests directory
    from fake_client import FakeClient, is_event, is_type


def is_lobby(state, **fields):
    def pred(m):
        if m.get("type") != "lobby" or m["state"] != state:
            return False
        return all(m.get(k) == v for k, v in fields.items())

    return pred


class ServerOptionsTest(unittest.TestCase):
    def server(self, **kwargs):
        env = {"seed": 3, "action_timeout": 5.0}
        env.update(kwargs.pop("env", {}))
        server = GameServer(port=0, hand_pause=0.0, env_kwargs=env, **kwargs)
        self.addCleanup(server.shutdown)
        return server

    def client(self, server, name, **kwargs):
        c = FakeClient(server.port, name, **kwargs)
        self.addCleanup(c.close)
        return c

    def test_invalid_options(self):
        for kwargs in ({"required_players": 1}, {"required_players": 11}, {"max_hands": 0}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    GameServer(port=0, **kwargs)

    def test_required_players(self):
        server = self.server(grace_period=0.1, required_players=3).start()
        a = self.client(server, "A", auto=True)
        b = self.client(server, "B", auto=True)
        self.assertTrue(a.join(40)["ok"])
        self.assertTrue(b.join(40)["ok"])
        m = b.wait_for(is_lobby("waiting", active=2))
        self.assertEqual((m["needed"], m["starts_in"]), (3, None))
        time.sleep(0.5)  # two players are not enough
        self.assertEqual(list(server.hand_start_times), [])
        self.assertEqual(a.of_type("hole_cards"), [])

        c = self.client(server, "C", auto=True)
        self.assertTrue(c.join(40)["ok"])
        a.wait_for(is_lobby("starting", active=3, needed=3))
        started = c.wait_for(is_event("hand_started", hand_no=1))
        self.assertEqual(len(started["event"]["participants"]), 3)

    def test_max_hands_returns_after_one_hand(self):
        server = self.server(grace_period=0.0, required_players=2, max_hands=1)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        a = self.client(server, "A", auto=True)
        b = self.client(server, "B", auto=True)
        self.assertTrue(a.join(40)["ok"])
        self.assertTrue(b.join(40)["ok"])
        thread.join(15)
        self.assertFalse(thread.is_alive())  # serve_forever returned
        self.assertEqual(server.hands_played, 1)
        self.assertEqual(server.state, STOPPED)
        self.assertEqual(len(server.results), 1)
        # The sessions stay open until shutdown(), which sends `close`.
        time.sleep(0.2)
        self.assertEqual(a.of_type("close"), [])
        self.assertEqual(sorted(server.session_names()), ["A", "B"])
        server.shutdown()
        for c in (a, b):
            c.wait_type("close")
            c.wait_eof()
            self.assertEqual(len(c.of_type("hand_end")), 1)

    def test_chips_per_bb_and_deck_reach_the_environment(self):
        # Two players: A (button / SB, engine seat 0) and B (seat 1). B is dealt first.
        a_hole = [Card.parse(c) for c in ("Ah", "Ad", "Kc", "Qc")]
        b_hole = [Card.parse(c) for c in ("7c", "6h", "3s", "4d")]
        script = [x for pair in zip(b_hole, a_hole) for x in pair] + [None] * 5
        deck = StackedDeck(script, random.Random(0))
        server = self.server(
            grace_period=0.0, max_hands=1, env={"chips_per_bb": 2, "deck": deck}
        ).start()
        self.assertIs(server.env.deck, deck)
        self.assertEqual(server.env.chips_per_bb, 2)
        self.assertEqual((server.settings.small_blind_bb, server.settings.big_blind_bb), (0.5, 1.0))
        a = self.client(server, "A", auto=True)
        b = self.client(server, "B", auto=True)
        self.assertTrue(a.join(50)["ok"])
        self.assertTrue(b.join(50)["ok"])
        self.assertEqual(a.wait_type("hole_cards")["cards"], ["Ah", "Ad", "Kc", "Qc"])
        self.assertEqual(b.wait_type("hole_cards")["cards"], ["7c", "6h", "3s", "4d"])
        sb = a.wait_for(is_event("blind_posted", blind="SB"))["event"]
        bb = a.wait_for(is_event("blind_posted", blind="BB"))["event"]
        self.assertEqual((sb["name"], sb["amount_bb"], sb["stack_bb"]), ("A", 0.5, 49.5))
        self.assertEqual((bb["name"], bb["amount_bb"], bb["stack_bb"]), ("B", 1.0, 49.0))
        a.wait_for(is_type("hand_end"))
        self.assertEqual(deck.shuffles, 1)
        self.assertEqual(server.env.dealer.big_blind, 2)  # chips


if __name__ == "__main__":
    unittest.main()
