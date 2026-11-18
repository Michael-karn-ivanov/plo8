import dataclasses
import io
import json
import random
import unittest

from entities import Board, Card, Hand, Street
from engine import (
    Action,
    ActionType,
    BetReturned,
    BlindPosted,
    BoardDealt,
    Environment,
    HandEndReply,
    HandResult,
    HandShown,
    HandStarted,
    JoinReply,
    LegalActions,
    Leave,
    Player,
    PlayerActed,
    PlayerJoined,
    PlayerLeft,
    PlayerRebought,
    PlayerReturned,
    PlayerSatOut,
    PlayerStatus,
    PotAwarded,
    PotResult,
    PotView,
    SeatView,
    TableView,
)

from cli_server import protocol as p
from cli_server.protocol import LineTooLongError, ProtocolError


def through_json(obj):
    """What the other side gets: the object serialised and parsed again."""
    return p.decode_line(p.encode_line(obj)) if isinstance(obj, dict) and "type" in obj else json.loads(
        json.dumps(obj)
    )


class Recorder:
    """In-process agent recording everything the engine hands out. Raises once per
    street when it can, otherwise checks / calls, so hands reach showdown."""

    def __init__(self):
        self.events = []
        self.views = []
        self.legals = []
        self.holes = []
        self.results = []

    def on_turn(self, player, view, legal):
        self.views.append(view)
        self.legals.append(legal)
        if legal.can_raise and view.raises < 2:
            player.raise_()
        elif legal.can_check:
            player.check()
        else:
            player.call()

    def on_hand_end(self, player, result):
        self.results.append(result)

    def on_hole_cards(self, player, hand, view):
        self.holes.append((hand, view))

    def on_event(self, player, event):
        self.events.append(event)


def play_hands(n=4, seed=3):
    agents = [Recorder() for _ in range(3)]
    players = [Player(name, 40, agent) for name, agent in zip("ABC", agents)]
    env = Environment(players, seed=seed)
    env.run(n)
    return agents


def step_decisions(chips_per_bb, stacks_chips, seed, max_hands=20):
    """Play hands in step mode at `chips_per_bb` with players joining with the
    given stacks (in chips); returns (TableView, LegalActions) of every decision.
    Aggressive play with short stacks, so all-ins and side pots happen."""
    env = Environment(chips_per_bb=chips_per_bb, seed=seed)
    for k, chips in enumerate(stacks_chips):
        env.join(chips / chips_per_bb, name=f"P{k}")
    rng = random.Random(seed)
    out = []
    for _ in range(max_hands):
        if not env.can_start():
            break
        decision = env.start_hand()
        while decision is not None:
            legal = decision.legal
            out.append((env.view(), legal))
            r = rng.random()
            if legal.can_raise and r < 0.6:
                action = ActionType.RAISE
            elif legal.can_check:
                action = ActionType.CHECK
            elif r < 0.9:
                action = ActionType.CALL
            else:
                action = ActionType.FOLD
            decision = env.step(action)
    return out


BOARD = Board.parse("Ah7d2c 9s Td")

SAMPLE_EVENTS = [
    HandStarted(0, 1, 2, ((0, "A", 40.0), (1, "B", 40.0), (2, "C", 40.0)), hand_no=1, seq=1),
    BlindPosted(1, "B", "SB", 0.5, False, 39.5, hand_no=1, seq=2),
    BlindPosted(2, "C", "BB", 1.0, True, 0.0, hand_no=1, seq=3),
    PlayerActed(1, "B", Street.FLOP, ActionType.RAISE, 1.0, 1.0, False, 37.0, hand_no=3, seq=41),
    PlayerActed(2, "C", Street.PREFLOP, ActionType.FOLD, 0.0, 0.0, False, 39.0, "timeout", hand_no=1, seq=5),
    BoardDealt(Street.FLOP, BOARD.flop, Board(flop=BOARD.flop), hand_no=1, seq=6),
    BoardDealt(Street.RIVER, (BOARD.river,), BOARD, hand_no=1, seq=8),
    HandShown(0, "A", Hand.parse("AsKd7h2c"), 38.0, hand_no=1, seq=9),
    BetReturned(1, "B", 2.0, 30.0, hand_no=1, seq=10),
    PotAwarded(0, "A", 0, "high", 5.0, 43.0, hand_no=1, seq=11),
    PotAwarded(0, "A", 1, "whole", 2.5, 45.5, hand_no=1, seq=12),
    PlayerJoined(3, "D", 40.0, seq=13),
    PlayerSatOut(1, "B", 0.0, seq=14),
    PlayerReturned(1, "B", 10.0, seq=15),
    PlayerRebought(1, "B", 10.0, 10.0, seq=16),
    PlayerLeft(2, "C", 39.0, seq=17),
]

LEGAL = LegalActions(
    can_check=False, call_amount=0.5, can_raise=True, raise_to=2.0, bet_unit=1.0, raises_left=3
)

SETTINGS = p.TableSettings(
    small_blind_bb=0.5,
    big_blind_bb=1.0,
    small_bet_bb=1.0,
    big_bet_bb=2.0,
    max_seats=10,
    max_buy_in_bb=100.0,
    action_timeout=180.0,
    grace_period=15.0,
    hand_pause=3.0,
)


def sample_view(hole="AsKd7h2c"):
    return TableView(
        hand_number=2,
        seat=0,
        hole_cards=None if hole is None else Hand.parse(hole),
        street=Street.TURN,
        board=Board.parse("Ah7d2c 9s"),
        pot=6.5,
        side_pots=(PotView(4.0, (0, 1, 2)), PotView(2.5, (0, 1))),
        current_bet=2.0,
        bet_unit=2.0,
        raises=1,
        button=2,
        sb_seat=0,
        bb_seat=1,
        seats=(
            SeatView(0, "A", 30.0, 2.0, 5.0, False, False, True),
            SeatView(1, "B", 0.0, 0.0, 3.0, False, True, True),
            SeatView(2, "C", 12.0, 0.0, 0.0, True, False, True, PlayerStatus.ACTIVE, True),
            SeatView(3, "D", 0.0, 0.0, 0.0, False, False, False, PlayerStatus.SITTING_OUT),
        ),
        history=tuple(SAMPLE_EVENTS[:4]),
        waiting=("E",),
    )


def sample_result():
    return HandResult(
        hand_number=4,
        button=0,
        sb_seat=1,
        bb_seat=2,
        board=BOARD,
        showdown=True,
        shown={0: Hand.parse("AsKd7h2c"), 1: Hand.parse("QsQd3h4c")},
        pots=(
            PotResult(6.0, (0, 1), ((0, 3.0),), ((1, 3.0),), False),
            PotResult(1.0, (0,), ((0, 1.0),), (), True),
        ),
        net={0: 1.0, 1: 0.0, 2: -1.0, 3: 0.0},
        stacks={0: 41.0, 1: 40.0, 2: 39.0, 3: 0.0},
        history=tuple(SAMPLE_EVENTS),
        names={0: "A", 1: "B", 2: "C", 3: "D"},
        dealt_in={0: True, 1: True, 2: True, 3: False},
        returned={1: 2.0},
    )


class FramingTest(unittest.TestCase):
    def test_line_round_trip(self):
        msg = {"type": "x", "name": "Zoë", "n": 1.5, "nested": [None, True]}
        line = p.encode_line(msg)
        self.assertTrue(line.endswith(b"\n"))
        self.assertEqual(line.count(b"\n"), 1)
        self.assertEqual(p.decode_line(line), msg)
        self.assertEqual(p.decode_line(line.decode("utf-8")), msg)

    def test_newline_in_string_stays_on_one_line(self):
        line = p.encode_line({"type": "x", "text": "a\nb"})
        self.assertEqual(line.count(b"\n"), 1)

    def test_invalid_lines(self):
        for bad in (b"not json\n", b"[1, 2]\n", b'{"name": "x"}\n', b'{"type": ""}\n', b'{"type": 3}\n', b"\xff\n"):
            with self.subTest(bad=bad):
                with self.assertRaises(ProtocolError):
                    p.decode_line(bad)

    def test_encode_needs_type(self):
        with self.assertRaises(ProtocolError):
            p.encode_line({"name": "x"})
        with self.assertRaises(ProtocolError):
            p.encode_line({"type": "x", "bad": object()})

    def test_read_and_write_messages(self):
        buf = io.BytesIO()
        p.write_message(buf, {"type": "a"})
        buf.write(b"\n  \n")  # blank lines are skipped
        p.write_message(buf, {"type": "b", "k": 1})
        buf.seek(0)
        self.assertEqual(p.read_message(buf), {"type": "a"})
        self.assertEqual(p.read_message(buf), {"type": "b", "k": 1})
        self.assertIsNone(p.read_message(buf))

    def test_line_too_long(self):
        buf = io.BytesIO(b'{"type": "x", "pad": "' + b"a" * (p.MAX_LINE_BYTES + 10) + b'"}\n')
        with self.assertRaises(LineTooLongError):
            p.read_message(buf)


class EngineTypesTest(unittest.TestCase):
    def test_cards(self):
        for i in range(52):
            card = Card.from_index(i)
            self.assertEqual(p.decode_card(through_json(p.encode_card(card))), card)
        self.assertEqual(p.encode_card(Card.parse("As")), "As")
        for bad in ("Zz", "", 5, None):
            with self.assertRaises(ProtocolError):
                p.decode_card(bad)

    def test_hand(self):
        hand = Hand.parse("AsKd7h2c")
        self.assertEqual(p.encode_hand(hand), ["As", "Kd", "7h", "2c"])
        self.assertEqual(p.decode_hand(through_json(p.encode_hand(hand))), hand)
        with self.assertRaises(ProtocolError):
            p.decode_hand(["As", "Kd", "7h"])
        with self.assertRaises(ProtocolError):
            p.decode_hand(["As", "As", "7h", "2c"])

    def test_board_at_every_street(self):
        for text in ("", "Ah7d2c", "Ah7d2c 9s", "Ah7d2c 9s Td"):
            board = Board.parse(text)
            with self.subTest(board=text):
                self.assertEqual(p.decode_board(through_json(p.encode_board(board))), board)
        self.assertEqual(
            p.encode_board(Board.parse("Ah7d2c 9s")),
            {"flop": ["Ah", "7d", "2c"], "turn": "9s", "river": None},
        )
        with self.assertRaises(ProtocolError):
            p.decode_board({"flop": None, "turn": "9s", "river": None})

    def test_enums(self):
        for street in Street:
            self.assertEqual(p.decode_street(p.encode_street(street)), street)
        self.assertEqual(p.encode_street(Street.FLOP), "flop")
        for action in ActionType:
            self.assertEqual(p.decode_action_type(p.encode_action_type(action)), action)
        for status in PlayerStatus:
            self.assertEqual(p.decode_status(p.encode_status(status)), status)
        with self.assertRaises(ProtocolError):
            p.decode_street("FLOP")
        with self.assertRaises(ProtocolError):
            p.decode_action_type("bet")

    def test_actions(self):
        for name in ("fold", "check", "call", "raise"):
            self.assertEqual(p.decode_action(name), Action(ActionType(name)))
            self.assertEqual(p.encode_action(p.decode_action(name)), name)
        self.assertEqual(p.decode_action("leave"), Leave())
        self.assertEqual(p.encode_action(Leave()), "leave")
        for bad in ("bet", "", None, 1):
            with self.assertRaises(ProtocolError):
                p.decode_action(bad)

    def test_every_event_type(self):
        covered = {p.encode_event(e)["type"] for e in SAMPLE_EVENTS}
        self.assertEqual(covered, set(p.EVENT_TYPES))
        for event in SAMPLE_EVENTS:
            with self.subTest(event=event):
                self.assertEqual(p.decode_event(through_json(p.encode_event(event))), event)

    def test_event_wire_format(self):
        event = PlayerActed(1, "B", Street.FLOP, ActionType.RAISE, 1.0, 1.0, False, 37.0, hand_no=3, seq=41)
        self.assertEqual(
            p.encode_event(event),
            {
                "type": "player_acted",
                "hand_no": 3,
                "seq": 41,
                "seat": 1,
                "name": "B",
                "street": "flop",
                "action": "raise",
                "amount_bb": 1.0,
                "to_bb": 1.0,
                "all_in": False,
                "stack_bb": 37.0,
                "reason": None,
            },
        )
        started = p.encode_event(SAMPLE_EVENTS[0])
        self.assertEqual(started["participants"], [[0, "A", 40.0], [1, "B", 40.0], [2, "C", 40.0]])
        shown = p.encode_event(SAMPLE_EVENTS[7])
        self.assertEqual(shown["hand"], ["As", "Kd", "7h", "2c"])

    def test_bad_events(self):
        with self.assertRaises(ProtocolError):
            p.decode_event({"type": "nope"})
        with self.assertRaises(ProtocolError):
            p.decode_event({"type": "player_left", "seat": 1, "name": "A"})  # no stack_bb
        with self.assertRaises(ProtocolError):
            p.decode_event("player_left")

    def test_legal_actions(self):
        for legal in (LEGAL, dataclasses.replace(LEGAL, raises_left=None, can_check=True, call_amount=0.0)):
            self.assertEqual(p.decode_legal(through_json(p.encode_legal(legal))), legal)
        with self.assertRaises(ProtocolError):
            p.decode_legal({"can_check": True})

    def test_table_view(self):
        for view in (sample_view(), sample_view(hole=None)):
            self.assertEqual(p.decode_table_view(through_json(p.encode_table_view(view))), view)
        encoded = p.encode_table_view(sample_view())
        self.assertEqual(encoded["hole_cards"], ["As", "Kd", "7h", "2c"])
        self.assertEqual(encoded["seats"][3]["status"], "sitting_out")
        self.assertEqual(encoded["side_pots"][1], {"amount": 2.5, "eligible": [0, 1]})

    def test_hand_result(self):
        result = sample_result()
        encoded = p.encode_hand_result(result)
        self.assertEqual(
            encoded["seats"][1],
            {
                "seat": 1,
                "name": "B",
                "dealt_in": True,
                "net": 0.0,
                "stack": 40.0,
                "shown": ["Qs", "Qd", "3h", "4c"],
                "returned": 2.0,
            },
        )
        self.assertEqual(encoded["pots"][0]["low"], [[1, 3.0]])
        self.assertEqual(p.decode_hand_result(through_json(encoded)), result)

    def test_engine_output_round_trips(self):
        """Everything a real engine run hands to agents survives the wire."""
        agents = play_hands()
        a = agents[0]
        self.assertTrue(a.events and a.views and a.results and a.holes)
        for event in a.events:
            self.assertEqual(p.decode_event(through_json(p.encode_event(event))), event)
        for view in a.views:
            self.assertEqual(p.decode_table_view(through_json(p.encode_table_view(view))), view)
        for legal in a.legals:
            self.assertEqual(p.decode_legal(through_json(p.encode_legal(legal))), legal)
        for result in a.results:
            self.assertEqual(p.decode_hand_result(through_json(p.encode_hand_result(result))), result)
        self.assertTrue(any(r.showdown for r in a.results))


class ExactRoundTripTest(unittest.TestCase):
    """Wire protocol -> Round trip is exact: other projects (e.g. rl_env) rely on
    `decode(json.loads(json.dumps(encode(x)))) == x` for the actor's view and
    legal actions, at any chip unit."""

    # (chips_per_bb, stacks in chips): short and uneven stacks give all-ins,
    # side pots and BB amounts that are not multiples of 0.5.
    TABLES = (
        (6, (13, 25, 40, 100)),
        (2, (7, 15, 31, 60, 9)),
        (100, (133, 457, 1001)),
    )

    def test_actor_views_and_legal_actions_round_trip_exactly(self):
        decisions = []
        for chips_per_bb, stacks in self.TABLES:
            for seed in range(6):
                decisions += step_decisions(chips_per_bb, stacks, seed)
        self.assertTrue(decisions)
        for view, legal in decisions:
            view_back = p.decode_table_view(json.loads(json.dumps(p.encode_table_view(view))))
            legal_back = p.decode_legal(json.loads(json.dumps(p.encode_legal(legal))))
            self.assertEqual(view_back, view)
            self.assertEqual(legal_back, legal)
            self.assertIs(type(view_back.seats), type(view.seats))
            self.assertIs(type(view_back.side_pots), type(view.side_pots))
            self.assertIs(view_back.street, view.street)
            for got, want in zip(view_back.seats, view.seats):
                self.assertIs(got.status, want.status)
                self.assertEqual(got.stack, want.stack)
        # The hands covered what matters: side pots and odd BB amounts.
        self.assertTrue(any(len(v.side_pots) > 1 for v, _ in decisions), "no side pot was played")
        self.assertTrue(
            any((s.stack * 2) % 1 for v, _ in decisions for s in v.seats),
            "no stack that is not a multiple of 0.5 BB",
        )


class MessagesTest(unittest.TestCase):
    def round_trip(self, msg, **kwargs):
        encoded = p.encode_server_message(msg, "A", **kwargs)
        self.assertEqual(encoded["to"], "A")
        self.assertEqual(encoded["type"], msg[0])
        decoded = p.decode_server_message(p.decode_line(p.encode_line(encoded)))
        self.assertEqual(decoded, msg)
        return encoded

    def test_every_server_message(self):
        view = sample_view()
        hand = Hand.parse("AsKd7h2c")
        self.round_trip(("seated", view))
        hole = self.round_trip(("hole_cards", hand, view))
        self.assertEqual(hole["hand_number"], 2)
        self.assertEqual(hole["cards"], ["As", "Kd", "7h", "2c"])
        for event in SAMPLE_EVENTS:
            self.round_trip(("event", event))
        turn = self.round_trip(("turn", 7, view, LEGAL))
        self.assertEqual(turn["seq"], 7)
        self.assertIsNone(turn["timeout"])  # no limit given
        hand_end = self.round_trip(("hand_end", 8, 0, sample_result()))
        self.assertEqual((hand_end["seq"], hand_end["seat"]), (8, 0))
        self.round_trip(("sat_out",))
        self.round_trip(("returned",))
        left = self.round_trip(("left", 39.5))
        self.assertEqual(left["final_stack_bb"], 39.5)
        self.round_trip(("close",))
        covered = {"seated", "hole_cards", "event", "turn", "hand_end", "sat_out", "returned", "left", "close"}
        self.assertEqual(covered, set(p.SERVER_MESSAGE_TYPES))

    def test_turn_carries_the_timeout(self):
        turn = self.round_trip(("turn", 7, sample_view(), LEGAL), turn_timeout=180)
        self.assertEqual(turn["timeout"], 180.0)
        self.assertIsInstance(turn["timeout"], float)
        # only turns carry it
        seated = p.encode_server_message(("seated", sample_view()), "A", turn_timeout=180)
        self.assertNotIn("timeout", seated)

    def test_joined_is_not_a_callback_message(self):
        with self.assertRaises(ProtocolError):
            p.encode_server_message(("joined", JoinReply(True)), "A")
        with self.assertRaises(ProtocolError):
            p.decode_server_message({"type": "joined"})

    def test_join_response(self):
        for reply in (
            JoinReply(True, queued=False, name="A", seat=1),
            JoinReply(True, queued=True, name="D", seat=None),
            JoinReply(False, name="A", error="name 'A' is already taken"),
        ):
            encoded = p.encode_join_response(reply)
            self.assertEqual(encoded["type"], "response")
            self.assertEqual(encoded["ok"], reply.accepted)
            self.assertNotIn("table", encoded)
            self.assertEqual(p.decode_join_response(through_json(encoded)), reply)
            self.assertIsNone(p.decode_join_settings(through_json(encoded)))

    def test_join_response_with_table_settings(self):
        reply = JoinReply(True, queued=False, name="A", seat=0)
        encoded = through_json(p.encode_join_response(reply, SETTINGS))
        self.assertEqual(encoded["table"], p.encode_settings(SETTINGS))
        self.assertEqual(p.decode_join_response(encoded), reply)
        self.assertEqual(p.decode_join_settings(encoded), SETTINGS)
        # a rejected join has no table, even if one is given
        rejected = p.encode_join_response(JoinReply(False, name="A", error="full"), SETTINGS)
        self.assertNotIn("table", rejected)

    def test_responses(self):
        self.assertEqual(p.ok_response(), {"type": "response", "ok": True})
        self.assertEqual(
            p.error_response("bad"), {"type": "response", "ok": False, "error": "bad"}
        )


class TableSettingsTest(unittest.TestCase):
    def test_wire_format(self):
        self.assertEqual(
            p.encode_settings(SETTINGS),
            {
                "small_blind_bb": 0.5,
                "big_blind_bb": 1.0,
                "small_bet_bb": 1.0,
                "big_bet_bb": 2.0,
                "max_seats": 10,
                "max_buy_in_bb": 100.0,
                "action_timeout": 180.0,
                "grace_period": 15.0,
                "hand_pause": 3.0,
            },
        )

    def test_round_trip_with_and_without_max_buy_in(self):
        no_cap = dataclasses.replace(SETTINGS, max_buy_in_bb=None, action_timeout=None)
        for settings in (SETTINGS, no_cap):
            with self.subTest(settings=settings):
                self.assertEqual(p.decode_settings(through_json(p.encode_settings(settings))), settings)
        self.assertIsNone(p.encode_settings(no_cap)["max_buy_in_bb"])

    def test_ints_are_sent_as_floats(self):
        settings = dataclasses.replace(SETTINGS, max_buy_in_bb=100, grace_period=15, hand_pause=0)
        encoded = p.encode_settings(settings)
        self.assertIsInstance(encoded["max_buy_in_bb"], float)
        self.assertIsInstance(encoded["hand_pause"], float)

    def test_bad_settings(self):
        good = p.encode_settings(SETTINGS)
        for bad in (
            "table",
            {k: v for k, v in good.items() if k != "big_blind_bb"},
            {**good, "max_seats": "ten"},
            {**good, "small_blind_bb": None},
            {**good, "action_timeout": "soon"},
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ProtocolError):
                    p.decode_settings(bad)


class InfoMessagesTest(unittest.TestCase):
    def test_acting(self):
        acting = p.Acting(hand_number=3, seat=2, name="C", timeout=180)
        encoded = through_json(p.encode_acting(acting, "A"))
        self.assertEqual(
            encoded,
            {"type": "acting", "to": "A", "hand_number": 3, "seat": 2, "name": "C", "timeout": 180.0},
        )
        self.assertEqual(p.decode_acting(encoded), acting)
        no_limit = p.Acting(3, 2, "C", None)
        self.assertEqual(p.decode_acting(through_json(p.encode_acting(no_limit, "A"))), no_limit)

    def test_lobby(self):
        for status in (
            p.LobbyStatus("waiting", 1, 2, None),
            p.LobbyStatus("starting", 3, 2, 14.5),
            p.LobbyStatus("next_hand", 2, 2, 3.0),
        ):
            with self.subTest(status=status):
                encoded = through_json(p.encode_lobby(status, "A"))
                self.assertEqual(encoded["type"], "lobby")
                self.assertEqual(encoded["to"], "A")
                self.assertEqual(p.decode_lobby(encoded), status)
        waiting = p.encode_lobby(p.LobbyStatus("waiting", 1, 2, None), "A")
        self.assertIsNone(waiting["starts_in"])
        self.assertEqual(set(p.LOBBY_STATES), {"waiting", "starting", "next_hand"})

    def test_bad_info_messages(self):
        with self.assertRaises(ProtocolError):
            p.encode_lobby(p.LobbyStatus("playing", 2, 2, None), "A")
        for bad in (
            {"type": "lobby", "state": "playing", "active": 1, "needed": 2, "starts_in": None},
            {"type": "lobby", "state": "waiting", "active": 1, "needed": 2},
            {"type": "lobby", "state": "waiting", "active": "1", "needed": 2, "starts_in": None},
            {"type": "acting", "state": "waiting", "active": 1, "needed": 2, "starts_in": None},
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ProtocolError):
                    p.decode_lobby(bad)
        for bad in (
            {"type": "acting", "hand_number": 1, "seat": 0, "name": "A"},
            {"type": "acting", "hand_number": 1, "seat": "0", "name": "A", "timeout": 1.0},
            {"type": "lobby", "hand_number": 1, "seat": 0, "name": "A", "timeout": 1.0},
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(ProtocolError):
                    p.decode_acting(bad)

    def test_info_messages_are_not_engine_messages(self):
        self.assertEqual(set(p.INFO_MESSAGE_TYPES), {"acting", "lobby"})
        self.assertFalse(set(p.INFO_MESSAGE_TYPES) & set(p.SERVER_MESSAGE_TYPES))
        with self.assertRaises(ProtocolError):
            p.decode_server_message(p.encode_lobby(p.LobbyStatus("waiting", 1, 2, None), "A"))


class RequestsTest(unittest.TestCase):
    def test_every_request_round_trips(self):
        requests = [
            p.join_request("A", 40.0, 5001),
            p.join_request("A", 40, 5001, seat=2),
            p.action_request("A", 3, "raise"),
            p.action_request("A", 3, Action.call()),
            p.action_request("A", 3, Leave()),
            p.hand_end_request("A", 4),
            p.hand_end_request("A", 4, rebuy_bb=10.0, sit_in=True, leave=False),
            p.rebuy_request("A", 20.0),
            p.sit_in_request("A"),
            p.leave_request("A"),
        ]
        self.assertEqual({r["type"] for r in requests}, set(p.REQUEST_TYPES))
        for request in requests:
            with self.subTest(request=request):
                self.assertEqual(p.validate_request(p.decode_line(p.encode_line(request))), request)
        self.assertEqual(p.action_request("A", 3, Action.call())["action"], "call")

    def test_optional_fields_get_defaults(self):
        self.assertEqual(
            p.validate_request({"type": "join", "name": "A", "stack_bb": 40, "callback_port": 9000}),
            p.join_request("A", 40, 9000),
        )
        self.assertEqual(
            p.validate_request({"type": "hand_end", "name": "A", "seq": 2}),
            p.hand_end_request("A", 2),
        )

    def test_hand_end_reply_fields_match_engine(self):
        req = p.validate_request(p.hand_end_request("A", 4, rebuy_bb=5.0, leave=True))
        reply = HandEndReply(req["rebuy_bb"], req["sit_in"], req["leave"])
        self.assertEqual(reply, HandEndReply(5.0, False, True))

    def test_malformed_requests(self):
        bad = [
            "join",
            {"type": "bogus", "name": "A"},
            {"type": "leave"},
            {"type": "leave", "name": ""},
            {"type": "leave", "name": 3},
            {"type": "join", "name": "A", "callback_port": 9000},
            {"type": "join", "name": "A", "stack_bb": "40", "callback_port": 9000},
            {"type": "join", "name": "A", "stack_bb": True, "callback_port": 9000},
            {"type": "join", "name": "A", "stack_bb": float("nan"), "callback_port": 9000},
            {"type": "join", "name": "A", "stack_bb": 40, "callback_port": "x"},
            {"type": "join", "name": "A", "stack_bb": 40, "callback_port": 0},
            {"type": "join", "name": "A", "stack_bb": 40, "callback_port": 70000},
            {"type": "join", "name": "A", "stack_bb": 40, "callback_port": 9000, "seat": 1.5},
            {"type": "action", "name": "A", "seq": 1, "action": "bet"},
            {"type": "action", "name": "A", "seq": 1},
            {"type": "action", "name": "A", "seq": "1", "action": "call"},
            {"type": "hand_end", "name": "A"},
            {"type": "hand_end", "name": "A", "seq": 1, "sit_in": "yes"},
            {"type": "hand_end", "name": "A", "seq": 1, "rebuy_bb": "5"},
            {"type": "rebuy", "name": "A"},
            {"type": "rebuy", "name": "A", "amount_bb": None},
        ]
        for msg in bad:
            with self.subTest(msg=msg):
                with self.assertRaises(ProtocolError):
                    p.validate_request(msg)


if __name__ == "__main__":
    unittest.main()
