import json
import socket
import time
import unittest

from engine import PlayerStatus

from cli_server.protocol import action_request, decode_hand_result, decode_join_settings, rebuy_request
from cli_server.server import LOBBY, STOPPED, GameServer

try:
    from cli_server_tests.tests.fake_client import FakeClient, is_event, is_type, unused_port, wait_until
except ImportError:  # run directly from the tests directory
    from fake_client import FakeClient, is_event, is_type, unused_port, wait_until

GRACE = 0.2
SEED = 7


def hole_cards_of(hand_number):
    return lambda m: m["type"] == "hole_cards" and m["hand_number"] == hand_number


def hand_end_of(hand_number):
    return lambda m: m["type"] == "hand_end" and m["result"]["hand_number"] == hand_number


def is_lobby(state=None, **fields):
    def pred(m):
        if m.get("type") != "lobby" or (state is not None and m["state"] != state):
            return False
        return all(m.get(k) == v for k, v in fields.items())

    return pred


def index_of(msgs, pred, start=0):
    return next(i for i in range(start, len(msgs)) if pred(msgs[i]))


def any_turn(clients, timeout=10.0):
    """(client, turn message) for the first client that got a turn."""

    def find():
        for c in clients:
            for m in c.snapshot():
                if m["type"] == "turn":
                    return c, m
        return None

    return wait_until(find, timeout, message="a turn")


def retry_ok(fn, timeout=5.0):
    """Repeat a request until it succeeds (e.g. a rebuy that raced the end of a hand)."""
    last = {}

    def attempt():
        nonlocal last
        last = fn()
        return last.get("ok")

    try:
        wait_until(attempt, timeout, interval=0.02, message="a successful request")
    except AssertionError:
        raise AssertionError(f"request kept failing: {last}") from None
    return last


class TimedServer(GameServer):
    """Also records when each hand is over (all hand_end replies in)."""

    def __init__(self, *args, **kwargs):
        self.hand_end_times = []
        super().__init__(*args, **kwargs)

    def _play_hand(self):
        played = super()._play_hand()
        if played:
            self.hand_end_times.append(time.monotonic())
        return played


class ServerTestCase(unittest.TestCase):
    def start_server(self, grace=GRACE, action_timeout=5.0, hand_pause=0.0, **env):
        kwargs = {"seed": SEED, "action_timeout": action_timeout}
        kwargs.update(env)
        server = TimedServer(
            port=0, grace_period=grace, hand_pause=hand_pause, env_kwargs=kwargs
        ).start()
        self.addCleanup(server.shutdown)
        return server

    def client(self, server, name, **kwargs):
        c = FakeClient(server.port, name, **kwargs)
        self.addCleanup(c.close)
        return c

    def seated_names(self, server):
        return [p.name for p in server.env.players]

    def engine_player(self, server, name):
        return next(p for p in server.env.players if p.name == name)


class LobbyTest(ServerTestCase):
    def test_no_hand_with_one_client_then_grace_period(self):
        server = self.start_server(grace=0.3)
        a = self.client(server, "A", auto=True)
        r = a.join(40)
        self.assertTrue(r["ok"], r)
        self.assertEqual((r["queued"], r["name"], r["seat"]), (False, "A", 0))
        a.wait_type("seated")
        time.sleep(0.6)
        self.assertEqual(list(server.hand_start_times), [])
        self.assertEqual(a.of_type("hole_cards"), [])
        self.assertEqual(server.state, LOBBY)

        b = self.client(server, "B", auto=True)
        t0 = time.monotonic()
        self.assertTrue(b.join(40)["ok"])
        a.wait_for(hole_cards_of(1))
        start = server.hand_start_times[0]
        self.assertGreaterEqual(start, t0 + 0.3)
        self.assertLess(start, t0 + 3.0)

    def test_each_join_restarts_the_grace_period(self):
        grace = 0.8
        server = self.start_server(grace=grace)
        a = self.client(server, "A", auto=True)
        b = self.client(server, "B", auto=True)
        c = self.client(server, "C", auto=True)
        self.assertTrue(a.join(40)["ok"])
        self.assertTrue(b.join(40)["ok"])
        time.sleep(0.3)
        self.assertEqual(list(server.hand_start_times), [])
        t1 = time.monotonic()
        r = c.join(40)
        self.assertTrue(r["ok"])
        self.assertFalse(r["queued"])  # still in the lobby: seated at once
        hole = c.wait_for(hole_cards_of(1))  # dealt into the first hand
        self.assertEqual(len(hole["cards"]), 4)
        self.assertGreaterEqual(server.hand_start_times[0], t1 + grace)
        started = c.wait_for(is_event("hand_started", hand_no=1))
        self.assertEqual(sorted(p[1] for p in started["event"]["participants"]), ["A", "B", "C"])

    def test_join_mid_hand_is_queued(self):
        server = self.start_server(action_timeout=10.0)
        a = self.client(server, "A")
        b = self.client(server, "B")
        self.assertTrue(a.join(40)["ok"])
        self.assertTrue(b.join(40)["ok"])
        any_turn([a, b])  # hand 1 is waiting for an action

        d = self.client(server, "D", auto=True)
        r = d.join(40)
        self.assertTrue(r["ok"], r)
        self.assertTrue(r["queued"])
        self.assertIsNone(r["seat"])
        self.assertIn("table", r)
        time.sleep(0.3)
        self.assertEqual(d.snapshot(), [])  # nothing about the hand in progress
        self.assertEqual(server.env.dealer.hand_number, 1)
        self.assertNotIn("D", self.seated_names(server))

        a.set_auto(True)
        b.set_auto(True)
        d.wait_for(hole_cards_of(2))
        msgs = d.snapshot()
        self.assertEqual(msgs[0]["type"], "seated")
        self.assertFalse(any(m["type"] == "event" and m["event"]["hand_no"] == 1 for m in msgs))
        self.assertFalse(any(hand_end_of(1)(m) for m in msgs))
        # no turn announcements while waiting for a seat
        self.assertFalse(any(m["type"] == "acting" and m["hand_number"] == 1 for m in msgs))
        started = d.wait_for(is_event("hand_started", hand_no=2))
        self.assertIn("D", [p[1] for p in started["event"]["participants"]])
        a.wait_for(is_event("player_joined", name="D"))


class TableSettingsTest(ServerTestCase):
    def test_join_response_carries_table_settings(self):
        server = self.start_server(action_timeout=7.0, hand_pause=0.25, max_seats=6, max_buy_in=50.0)
        a = self.client(server, "A")
        r = a.join(40)
        self.assertTrue(r["ok"], r)
        self.assertEqual(
            r["table"],
            {
                "small_blind_bb": 0.5,
                "big_blind_bb": 1.0,
                "small_bet_bb": 1.0,
                "big_bet_bb": 2.0,
                "max_seats": 6,
                "max_buy_in_bb": 50.0,
                "action_timeout": 7.0,
                "grace_period": GRACE,
                "hand_pause": 0.25,
            },
        )
        self.assertEqual(decode_join_settings(r), server.settings)

        x = self.client(server, "X")
        r = x.join(80)  # above the max buy-in: rejected by the engine
        self.assertFalse(r["ok"])
        self.assertNotIn("table", r)
        r = self.client(server, "A").join(40)  # name taken
        self.assertFalse(r["ok"])
        self.assertNotIn("table", r)

    def test_no_max_buy_in(self):
        server = self.start_server()
        r = self.client(server, "A").join(40)
        self.assertIsNone(r["table"]["max_buy_in_bb"])
        self.assertEqual(r["table"]["max_seats"], 10)


class LobbyStatusTest(ServerTestCase):
    def test_waiting_starting_and_no_lobby_during_a_hand(self):
        grace = 1.0
        server = self.start_server(grace=grace, action_timeout=10.0)
        a = self.client(server, "A")
        self.assertTrue(a.join(40)["ok"])
        m = a.wait_for(is_lobby("waiting"))
        self.assertEqual((m["active"], m["needed"], m["starts_in"]), (1, 2, None))
        self.assertEqual(m["to"], "A")

        # A wake-up that changes nothing (a rejected join) does not re-send it.
        x = self.client(server, "X")
        self.assertFalse(x.join(-5)["ok"])
        time.sleep(0.3)
        self.assertEqual(len(a.of_type("lobby")), 1)
        self.assertEqual(x.of_type("lobby"), [])

        b = self.client(server, "B")
        self.assertTrue(b.join(40)["ok"])
        for c in (a, b):
            m = c.wait_for(is_lobby("starting"))
            self.assertEqual((m["active"], m["needed"]), (2, 2))
            self.assertGreater(m["starts_in"], grace - 0.5)
            self.assertLessEqual(m["starts_in"], grace)
        self.assertEqual(b.of_type("lobby")[0]["state"], "starting")  # its first status

        time.sleep(0.3)
        c = self.client(server, "C")
        self.assertTrue(c.join(40)["ok"])
        a.wait_for(is_lobby("starting", active=3))
        c.wait_for(is_lobby("starting", active=3))
        time.sleep(0.05)
        last = [m for m in a.of_type("lobby") if m["active"] == 3][-1]
        self.assertGreater(last["starts_in"], grace - 0.25)  # the grace period restarted

        # The first hand starts; nobody answers the first turn, so it keeps running.
        any_turn([a, b, c])
        time.sleep(0.3)
        for client in (a, b, c):
            msgs = client.snapshot()
            started = index_of(msgs, is_event("hand_started", hand_no=1))
            self.assertFalse([m for m in msgs[started:] if m["type"] == "lobby"])

    def test_leave_to_one_player_sends_waiting(self):
        server = self.start_server(grace=5.0)
        a = self.client(server, "A")
        b = self.client(server, "B")
        self.assertTrue(a.join(40)["ok"])
        self.assertTrue(b.join(40)["ok"])
        a.wait_for(is_lobby("starting"))
        start = len(a.snapshot())
        self.assertTrue(b.leave()["ok"])
        m = a.wait_for(is_lobby("waiting"), start=start)
        self.assertEqual((m["active"], m["starts_in"]), (1, None))
        self.assertEqual(server.env.dealer.hand_number, 0)


class ActingTest(ServerTestCase):
    def test_every_turn_is_announced_to_the_others(self):
        server = self.start_server()
        clients = [self.client(server, n, auto=True) for n in "ABC"]
        for c in clients:
            self.assertTrue(c.join(40)["ok"])
        for c in clients:
            c.wait_for(hand_end_of(1))

        for c in clients:
            msgs = c.snapshot()
            end = index_of(msgs, hand_end_of(1))
            bb = index_of(msgs, is_event("blind_posted", hand_no=1, blind="BB"))
            sequence = []  # ("ask" | "acted", name, seat, index)
            for i, m in enumerate(msgs[: end + 1]):
                if m["type"] == "acting":
                    self.assertNotEqual(m["name"], c.name)  # the actor gets `turn` instead
                    self.assertEqual((m["hand_number"], m["timeout"]), (1, 5.0))
                    sequence.append(("ask", m["name"], m["seat"], i))
                elif m["type"] == "turn":
                    self.assertEqual(m["timeout"], 5.0)
                    sequence.append(("ask", c.name, m["view"]["seat"], i))
                elif is_event("player_acted", hand_no=1)(m):
                    e = m["event"]
                    sequence.append(("acted", e["name"], e["seat"], i))
            self.assertTrue(sequence)
            self.assertGreater(sequence[0][3], bb)  # after the events that preceded it
            # Each turn is announced once, after the previous action and before
            # the actor's own action.
            self.assertEqual(len(sequence) % 2, 0, sequence)
            asks, acts = sequence[0::2], sequence[1::2]
            self.assertTrue(all(s[0] == "ask" for s in asks), sequence)
            self.assertTrue(all(s[0] == "acted" for s in acts), sequence)
            self.assertEqual([s[1:3] for s in asks], [s[1:3] for s in acts])

    def test_sitting_out_players_are_told_too(self):
        server = self.start_server(action_timeout=0.5)
        a = self.client(server, "A")  # never answers: times out and sits out
        b = self.client(server, "B", auto=True)
        c = self.client(server, "C", auto=True)
        for client in (a, b, c):
            self.assertTrue(client.join(40)["ok"])
        a.wait_type("sat_out")
        m = a.wait_for(lambda m: m["type"] == "acting" and m["hand_number"] == 2, timeout=15)
        self.assertIn(m["name"], ("B", "C"))
        self.assertIs(self.engine_player(server, "A").status, PlayerStatus.SITTING_OUT)
        self.assertFalse([m for m in a.of_type("turn") if m["view"]["hand_number"] == 2])


class PauseTest(ServerTestCase):
    def test_pause_between_hands(self):
        pause = 0.3
        server = self.start_server(hand_pause=pause)
        a = self.client(server, "A", auto=True)
        b = self.client(server, "B", auto=True)
        self.assertTrue(a.join(40)["ok"])
        self.assertTrue(b.join(40)["ok"])
        for c in (a, b):
            c.wait_for(hole_cards_of(2))
        self.assertGreaterEqual(server.hand_start_times[1] - server.hand_end_times[0], pause)
        for c in (a, b):
            msgs = c.snapshot()
            end = index_of(msgs, hand_end_of(1))
            started = index_of(msgs, is_event("hand_started", hand_no=2))
            lobby = [m for m in msgs[end:started] if m["type"] == "lobby"]
            self.assertEqual(len(lobby), 1, lobby)  # announced once
            self.assertEqual((lobby[0]["state"], lobby[0]["active"]), ("next_hand", 2))
            self.assertGreater(lobby[0]["starts_in"], 0)
            self.assertLessEqual(lobby[0]["starts_in"], pause)

    def test_join_during_the_pause_is_seated_at_once(self):
        server = self.start_server(hand_pause=1.5)
        a = self.client(server, "A", auto=True)
        b = self.client(server, "B", auto=True)
        self.assertTrue(a.join(40)["ok"])
        self.assertTrue(b.join(40)["ok"])
        a.wait_for(is_lobby("next_hand"))
        self.assertEqual(server.env.dealer.hand_number, 1)

        c = self.client(server, "C", auto=True)
        r = c.join(40)
        self.assertTrue(r["ok"], r)
        self.assertFalse(r["queued"])
        self.assertIsNotNone(r["seat"])
        c.wait_for(is_lobby("next_hand"))  # gets the status at once
        a.wait_for(is_lobby("next_hand", active=3))  # re-sent: one more player
        started = c.wait_for(is_event("hand_started", hand_no=2))
        self.assertIn("C", [p[1] for p in started["event"]["participants"]])
        c.wait_for(hole_cards_of(2))

    def test_leaving_during_the_pause_goes_back_to_the_lobby(self):
        server = self.start_server(hand_pause=1.0)
        a = self.client(server, "A", auto=True)
        b = self.client(server, "B", auto=True)
        self.assertTrue(a.join(40)["ok"])
        self.assertTrue(b.join(40)["ok"])
        a.wait_for(is_lobby("next_hand"))
        start = len(a.snapshot())
        self.assertTrue(b.leave()["ok"])
        m = a.wait_for(is_lobby("waiting"), start=start)
        self.assertEqual(m["active"], 1)
        wait_until(lambda: server.state == LOBBY, message="back in the lobby")
        time.sleep(1.2)
        self.assertEqual(server.env.dealer.hand_number, 1)


class StopTest(ServerTestCase):
    def stop_and_time(self, server):
        t0 = time.monotonic()
        server.stop()
        server._game_thread.join(5)
        self.assertFalse(server._game_thread.is_alive())
        self.assertEqual(server.state, STOPPED)
        return time.monotonic() - t0

    def test_stop_ends_a_lobby_wait_at_once(self):
        server = self.start_server()
        a = self.client(server, "A")
        self.assertTrue(a.join(40)["ok"])
        a.wait_for(is_lobby("waiting"))  # waits with no timeout
        self.assertLess(self.stop_and_time(server), 1.0)

    def test_stop_ends_a_pause_at_once(self):
        server = self.start_server(hand_pause=60)
        a = self.client(server, "A", auto=True)
        b = self.client(server, "B", auto=True)
        self.assertTrue(a.join(40)["ok"])
        self.assertTrue(b.join(40)["ok"])
        a.wait_for(is_lobby("next_hand"))
        self.assertLess(self.stop_and_time(server), 1.0)
        self.assertEqual(server.env.dealer.hand_number, 1)


class HandMessagesTest(ServerTestCase):
    def test_every_client_sees_the_hand(self):
        server = self.start_server()
        clients = [self.client(server, n, auto=True) for n in "ABC"]
        for c in clients:
            self.assertTrue(c.join(40)["ok"])
        for c in clients:
            c.wait_for(hand_end_of(1))
        wait_until(lambda: len(server.results) >= 1)
        engine_result = server.results[0]

        holes = {}
        seqs = []
        shown = {}
        for c in clients:
            msgs = c.snapshot()
            end = next(i for i, m in enumerate(msgs) if hand_end_of(1)(m))
            hand = msgs[: end + 1]
            self.assertTrue(all(m["to"] == c.name for m in hand))

            def where(pred):
                return [i for i, m in enumerate(hand) if pred(m)]

            blinds = where(is_event("blind_posted", hand_no=1))
            self.assertEqual(len(blinds), 2)
            sb, bb = (hand[i]["event"] for i in blinds)
            self.assertEqual((sb["blind"], sb["amount_bb"], sb["stack_bb"]), ("SB", 0.5, 39.5))
            self.assertEqual((bb["blind"], bb["amount_bb"], bb["stack_bb"]), ("BB", 1.0, 39.0))

            hole = where(is_type("hole_cards"))
            self.assertEqual(len(hole), 1)  # its own cards only
            hole_msg = hand[hole[0]]
            self.assertEqual(hole_msg["hand_number"], 1)
            self.assertEqual(len(hole_msg["cards"]), 4)
            self.assertEqual(hole_msg["view"]["hole_cards"], hole_msg["cards"])
            holes[c.name] = hole_msg["cards"]

            turns = where(is_type("turn"))
            self.assertTrue(turns)
            self.assertLess(max(blinds), hole[0])  # blinds, then cards, then action
            self.assertLess(hole[0], min(turns))
            for i in turns:
                self.assertEqual(hand[i]["view"]["hole_cards"], holes[c.name])

            hand_events = [
                m["event"] for m in hand if m["type"] == "event" and m["event"]["hand_no"] == 1
            ]
            seqs.append([e["seq"] for e in hand_events])
            last_stack = {}
            for e in hand_events:
                if "stack_bb" in e:
                    last_stack[e["name"]] = e["stack_bb"]
                if e["type"] == "hand_shown":
                    shown[e["name"]] = e["hand"]

            result = decode_hand_result(hand[end]["result"])
            self.assertEqual(result, engine_result)
            my_seat = [s for s, n in result.names.items() if n == c.name][0]
            self.assertEqual(hand[end]["seat"], my_seat)
            self.assertEqual(last_stack, {result.names[s]: st for s, st in result.stacks.items()})

        self.assertEqual(seqs[0], seqs[1])  # everyone gets the same event stream
        self.assertEqual(seqs[0], seqs[2])
        self.assertEqual(len({card for h in holes.values() for card in h}), 12)
        for name, cards in shown.items():
            self.assertEqual(cards, holes[name])

    def test_chips_are_conserved(self):
        server = self.start_server()
        clients = [self.client(server, n, auto=True) for n in "ABC"]
        for c in clients:
            self.assertTrue(c.join(40)["ok"])
        wait_until(lambda: len(server.results) >= 3, timeout=20)
        results = list(server.results)[:3]
        for r in results:
            self.assertAlmostEqual(sum(r.stacks.values()), 120.0)
        for c in clients:
            for i, r in enumerate(results, start=1):
                msg = c.wait_for(hand_end_of(i))
                self.assertEqual(decode_hand_result(msg["result"]).stacks, r.stacks)


class RequestsTest(ServerTestCase):
    def test_actions_are_routed_by_name(self):
        server = self.start_server(action_timeout=10.0)
        a = self.client(server, "A")
        b = self.client(server, "B")
        self.assertTrue(a.join(40)["ok"])
        self.assertTrue(b.join(40)["ok"])
        actor, turn = any_turn([a, b])
        other = b if actor is a else a
        seq, legal = turn["seq"], turn["legal"]
        self.assertEqual(turn["timeout"], 10.0)
        acting = other.wait_for(is_type("acting"))
        self.assertEqual((acting["name"], acting["seat"]), (actor.name, turn["view"]["seat"]))

        r = actor.act(seq + 100, "fold")
        self.assertFalse(r["ok"])
        self.assertIn("pending turn", r["error"])
        illegal = "call" if legal["can_check"] else "check"
        r = actor.act(seq, illegal)
        self.assertFalse(r["ok"])
        self.assertIn("not legal", r["error"])
        r = other.act(seq, "fold")  # not other's turn
        self.assertFalse(r["ok"])
        r = actor.request(action_request("Nobody", seq, "fold"))
        self.assertFalse(r["ok"])
        self.assertIn("unknown player", r["error"])
        r = actor.rebuy(5)  # not during betting
        self.assertFalse(r["ok"])
        self.assertIn("during a hand", r["error"])
        r = actor.request(rebuy_request("Nobody", 5))
        self.assertFalse(r["ok"])

        good = "check" if legal["can_check"] else "call"
        r = actor.act(seq, good)  # corrected action, still within the timeout
        self.assertTrue(r["ok"], r)
        self.assertFalse(actor.act(seq, good)["ok"])  # duplicate
        ev = other.wait_for(is_event("player_acted", name=actor.name))["event"]
        self.assertEqual(ev["action"], good)
        self.assertIsNone(ev["reason"])

    def test_join_rejections(self):
        server = self.start_server()
        a = self.client(server, "A")
        self.assertTrue(a.join(40)["ok"])

        dup = self.client(server, "A")
        r = dup.join(40)
        self.assertFalse(r["ok"])
        self.assertIn("taken", r["error"])

        x = self.client(server, "X")
        r = x.join(40, callback_port=unused_port())
        self.assertFalse(r["ok"])
        self.assertIn("callback port", r["error"])

        for bad in (
            {"type": "join"},
            {"type": "join", "name": "", "stack_bb": 40, "callback_port": x.callback_port},
            {"type": "join", "name": "X", "stack_bb": "abc", "callback_port": x.callback_port},
            {"type": "join", "name": "X", "stack_bb": 40, "callback_port": "x"},
            {"type": "bogus", "name": "X"},
            {"name": "X"},
        ):
            with self.subTest(request=bad):
                r = x.request(bad)
                self.assertFalse(r["ok"])
                self.assertEqual(r["type"], "response")
                self.assertNotIn("table", r)

        r = x.join(-5)  # rejected by the engine
        self.assertFalse(r["ok"])
        self.assertIn("buy-in", r["error"])

        r = x.raw(b"this is not json\n")  # the connection stays usable
        self.assertFalse(r["ok"])
        r = x.join(40)
        self.assertTrue(r["ok"], r)  # "X" was not left reserved by the failures
        self.assertEqual(sorted(server.session_names()), ["A", "X"])

    def test_unknown_names(self):
        server = self.start_server()
        a = self.client(server, "A")
        for msg in (
            {"type": "leave", "name": "Ghost"},
            {"type": "sit_in", "name": "Ghost"},
            {"type": "hand_end", "name": "Ghost", "seq": 1},
        ):
            r = a.request(msg)
            self.assertFalse(r["ok"])
            self.assertIn("unknown player", r["error"])


class LeavingTest(ServerTestCase):
    def test_unresponsive_client_is_folded_and_sits_out(self):
        server = self.start_server(action_timeout=0.5)
        a = self.client(server, "A")  # never answers
        b = self.client(server, "B", auto=True)
        self.assertTrue(a.join(40)["ok"])
        self.assertTrue(b.join(40)["ok"])
        b.wait_for(is_event("player_acted", name="A", action="fold", reason="timeout"))
        a.wait_type("sat_out")
        wait_until(lambda: server.state == LOBBY and not server.env.dealer.in_hand)
        self.assertIs(self.engine_player(server, "A").status, PlayerStatus.SITTING_OUT)
        time.sleep(3 * GRACE)
        self.assertEqual(server.env.dealer.hand_number, 1)  # nobody to play with

        a.set_auto(True, answer_pending=False)
        r = retry_ok(a.sit_in)
        self.assertTrue(r["ok"])
        a.wait_type("returned")
        a.wait_for(hole_cards_of(2))

    def test_closed_callback_between_hands_removes_the_player(self):
        server = self.start_server()
        a = self.client(server, "A")
        self.assertTrue(a.join(40)["ok"])
        a.wait_type("seated")
        a.close_callback()
        wait_until(
            lambda: "A" not in self.seated_names(server) and "A" not in server.session_names(),
            message="A removed",
        )
        again = self.client(server, "A")
        self.assertTrue(again.join(40)["ok"])

    def test_closed_callback_mid_hand_removes_the_player(self):
        server = self.start_server(action_timeout=10.0)
        a = self.client(server, "A")
        b = self.client(server, "B", auto=True)
        self.assertTrue(a.join(40)["ok"])
        self.assertTrue(b.join(40)["ok"])
        a.wait_type("turn")  # first joiner: button / small blind, acts first heads-up
        a.close_callback()
        b.wait_for(is_event("player_left", name="A"))
        wait_until(lambda: "A" not in self.seated_names(server), message="A removed")

    def test_leave_between_hands_frees_the_name(self):
        server = self.start_server()
        a = self.client(server, "A")
        self.assertTrue(a.join(40)["ok"])
        a.wait_type("seated")
        self.assertTrue(a.leave()["ok"])
        left = a.wait_type("left")
        self.assertEqual(left["final_stack_bb"], 40.0)
        a.wait_type("close")
        a.wait_eof()
        self.assertEqual(self.seated_names(server), [])
        again = self.client(server, "A")
        self.assertTrue(again.join(40)["ok"])

    def test_leave_mid_hand(self):
        server = self.start_server(action_timeout=10.0)
        a = self.client(server, "A")
        b = self.client(server, "B")
        self.assertTrue(a.join(40)["ok"])
        self.assertTrue(b.join(40)["ok"])
        actor, _ = any_turn([a, b])
        other = b if actor is a else a
        self.assertTrue(other.leave()["ok"])
        # The engine applies the leave on the RemoteAgent's reader thread.
        leaver = self.engine_player(server, other.name)
        wait_until(lambda: leaver.leave_requested, message="leave requested")
        actor.set_auto(True)  # answers the pending turn
        actor.wait_for(is_event("player_acted", name=other.name, action="fold", reason="leave"))
        left = other.wait_type("left")
        self.assertLess(left["final_stack_bb"], 40.0)  # the posted blind stays in the pot
        other.wait_type("close")
        self.assertFalse(other.of_type("hand_end"))
        actor.wait_for(is_event("player_left", name=other.name))
        # The leaver was folded without being asked: no turn announcement for it.
        self.assertFalse(
            [m for m in actor.of_type("acting") if m["name"] == other.name and m["hand_number"] == 1]
        )

        back = self.client(server, other.name, auto=True)
        self.assertTrue(back.join(40)["ok"])  # the name can be used again
        back.wait_for(hole_cards_of(2))

    def test_busted_player_rebuys_and_play_resumes_after_grace(self):
        server = self.start_server()
        a = self.client(server, "A", auto=True)
        b = self.client(server, "B", auto=True)
        self.assertTrue(a.join(1.0)["ok"])  # all-in every hand until one busts
        self.assertTrue(b.join(1.0)["ok"])
        busted = wait_until(
            lambda: next((c for c in (a, b) if c.of_type("sat_out")), None),
            timeout=30,
            message="a busted player",
        )
        other = b if busted is a else a
        wait_until(lambda: server.state == LOBBY and not server.env.dealer.in_hand)
        hands = server.env.dealer.hand_number
        time.sleep(2 * GRACE)
        self.assertEqual(server.env.dealer.hand_number, hands)  # one active player left
        for c in (a, b):  # back in the lobby: waiting for a second active player
            last_end = index_of(c.snapshot(), hand_end_of(hands))
            m = c.wait_for(is_lobby("waiting"), start=last_end)
            self.assertEqual(m["active"], 1)

        starts = {c.name: len(c.snapshot()) for c in (a, b)}
        t0 = time.monotonic()
        r = retry_ok(lambda: busted.rebuy(1.0))
        self.assertTrue(r["ok"])
        busted.wait_type("returned")
        for c in (busted, other):
            m = c.wait_for(is_lobby("starting", active=2), start=starts[c.name])
            self.assertLessEqual(m["starts_in"], GRACE)
        busted.wait_for(hole_cards_of(hands + 1))
        self.assertGreaterEqual(server.hand_start_times[-1], t0 + GRACE)


class ProtocolOverTcpTest(ServerTestCase):
    def test_one_connection_per_request(self):
        """Clients may also open a new connection for every call."""
        server = self.start_server()
        a = self.client(server, "A")
        join = b'{"type": "join", "name": "A", "stack_bb": 40, "callback_port": %d}\n' % a.callback_port
        with socket.create_connection(("127.0.0.1", server.port), timeout=5) as s:
            s.sendall(join)
            r = json.loads(s.makefile("rb").readline())
        self.assertTrue(r["ok"], r)
        a.wait_type("seated")
        with socket.create_connection(("127.0.0.1", server.port), timeout=5) as s:
            s.sendall(b'{"type": "leave", "name": "A"}\n')
            r = json.loads(s.makefile("rb").readline())
        self.assertTrue(r["ok"], r)
        a.wait_type("left")


if __name__ == "__main__":
    unittest.main()
