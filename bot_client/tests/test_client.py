"""BotClient against a fake ServerConnection (records requests, returns scripted
responses), an inbox fed with messages encoded by cli_server.protocol, and a
fake clock."""

import logging
import queue
import unittest

from cli_client.net import DISCONNECTED, ConnectionLost
from cli_server.protocol import LobbyStatus, TableSettings, encode_lobby, encode_server_message
from engine import (
    ActionType,
    Board,
    Card,
    Hand,
    HandResult,
    HandStarted,
    LegalActions,
    PlayerActed,
    PlayerRebought,
    PlayerStatus,
    SeatView,
    Street,
    TableView,
)

from bot_client.client import STOP, BotClient, fallback_action, is_stale_turn_error

NAME = "Bot"
LOGGER = "bot_client.test"
STALE = f"turn 3 is not the pending turn of {NAME} (4)"
ILLEGAL = "raise is not legal now (legal: fold, call, leave)"


class FakeConn:
    def __init__(self):
        self.requests = []
        self.responses = []  # queued responses (default: ok)
        self.lost = False
        self.closed = False

    def request(self, msg):
        if self.lost:
            raise ConnectionLost("gone")
        self.requests.append(msg)
        if self.responses:
            return self.responses.pop(0)
        return {"type": "response", "ok": True}

    def close(self):
        self.closed = True

    def of_type(self, kind):
        return [r for r in self.requests if r["type"] == kind]


class FixedPlayer:
    label = "fixed"

    def __init__(self, action=ActionType.CALL, raises=False):
        self.action = action
        self.raises = raises
        self.calls = 0

    def decide(self, view, legal):
        self.calls += 1
        if self.raises:
            raise RuntimeError("bot bug")
        return self.action


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def view(seat=1, stack=99.0, hand=1, names=("Alice", NAME)):
    seats = tuple(
        SeatView(
            i, n, stack if n == NAME else 100.0, 0.0, 0.0, False, False, True, PlayerStatus.ACTIVE
        )
        for i, n in enumerate(names)
    )
    return TableView(
        hand_number=hand,
        seat=seat,
        hole_cards=None,
        street=Street.PREFLOP,
        board=Board.empty(),
        pot=1.5,
        side_pots=(),
        current_bet=1.0,
        bet_unit=1.0,
        raises=1,
        button=0,
        sb_seat=0,
        bb_seat=1,
        seats=seats,
        history=(),
    )


FACING_BET = LegalActions(False, 0.5, True, 2.0, 1.0, None)
CAN_CHECK = LegalActions(True, 0.0, True, 1.0, 1.0, None)
CHECK_ONLY = LegalActions(True, 0.0, False, 1.0, 1.0, 0)


def turn(seq, legal=FACING_BET):
    return encode_server_message(("turn", seq, view(), legal), NAME, turn_timeout=5.0)


def result(hand=1, net=-1.0, stack=99.0, dealt_in=True, seat=1):
    return HandResult(
        hand_number=hand,
        button=0,
        sb_seat=0,
        bb_seat=1,
        board=Board.empty(),
        showdown=False,
        shown={},
        pots=(),
        net={0: -net, seat: net},
        stacks={0: 100.0 - net, seat: stack},
        history=(),
        names={0: "Alice", seat: NAME},
        dealt_in={0: True, seat: dealt_in},
    )


def hand_end(seq, **kwargs):
    seat = kwargs.get("seat", 1)
    return encode_server_message(("hand_end", seq, seat, result(**kwargs)), NAME)


def event(e):
    return encode_server_message(("event", e), NAME)


def started(hand, stack=100.0):
    return event(HandStarted(0, 0, 1, ((0, "Alice", 100.0), (1, NAME, stack)), hand_no=hand, seq=1))


def rebought(amount, stack, hand=None):
    return event(PlayerRebought(1, NAME, amount, stack, hand_no=hand, seq=2))


def lobby(state="waiting", active=1, needed=2):
    return encode_lobby(LobbyStatus(state, active, needed, None), NAME)


LEFT = {"type": "left", "to": NAME, "final_stack_bb": 99.0}
CLOSE = {"type": "close", "to": NAME}


class ClientTest(unittest.TestCase):
    def make(self, player=None, **kwargs):
        self.conn = FakeConn()
        self.inbox = queue.Queue()
        self.clock = Clock()
        opts = dict(
            stack_bb=100.0,
            rebuy="topup",
            max_hands=None,
            settings=None,
            queued=False,
            clock=self.clock,
            leave_timeout=5.0,
            poll=0.01,
        )
        opts.update(kwargs)
        self.client = BotClient(
            NAME, player or FixedPlayer(), self.conn, self.inbox, logger=logging.getLogger(LOGGER), **opts
        )
        return self.client

    def feed(self, *msgs):
        code = None
        for m in msgs:
            code = self.client.step(m)
        return code

    # --- turns ---------------------------------------------------------------------

    def test_turn_is_answered_with_its_seq_and_the_decision(self):
        c = self.make(FixedPlayer(ActionType.RAISE))
        self.feed(turn(7))
        self.assertEqual(
            self.conn.requests, [{"type": "action", "name": NAME, "seq": 7, "action": "raise"}]
        )
        self.assertEqual(c.turns_answered, 1)
        self.assertEqual((c.rejected_actions, c.fallbacks, c.problems), (0, 0, []))

    def test_stale_seq_rejection_is_not_retried(self):
        c = self.make()
        self.conn.responses = [{"ok": False, "error": STALE}]
        with self.assertLogs(LOGGER, level="WARNING"):
            self.feed(turn(3))
        self.assertEqual(len(self.conn.of_type("action")), 1)
        self.assertEqual((c.rejected_actions, c.fallbacks), (1, 0))

    def test_illegal_rejection_falls_back_to_check_or_call_once(self):
        c = self.make(FixedPlayer(ActionType.RAISE))
        self.conn.responses = [{"ok": False, "error": ILLEGAL}]
        self.feed(turn(4))
        actions = self.conn.of_type("action")
        self.assertEqual([(a["seq"], a["action"]) for a in actions], [(4, "raise"), (4, "call")])
        self.assertEqual((c.rejected_actions, c.fallbacks, c.turns_answered), (1, 1, 1))

    def test_fallback_rejected_too_gives_up(self):
        c = self.make(FixedPlayer(ActionType.RAISE))
        self.conn.responses = [{"ok": False, "error": ILLEGAL}, {"ok": False, "error": ILLEGAL}]
        self.feed(turn(4))
        self.assertEqual(len(self.conn.of_type("action")), 2)
        self.assertEqual(c.turns_answered, 0)
        self.assertIsNone(c.exit_code)

    def test_raising_player_falls_back_without_contacting_the_server_first(self):
        c = self.make(FixedPlayer(raises=True))
        with self.assertLogs(LOGGER, level="ERROR"):
            self.feed(turn(1, CAN_CHECK), turn(2, FACING_BET))
        self.assertEqual([a["action"] for a in self.conn.requests], ["check", "call"])
        self.assertEqual(len(c.problems), 2)

    def test_decision_the_legal_actions_forbid_is_replaced(self):
        c = self.make(FixedPlayer(ActionType.RAISE))
        with self.assertLogs(LOGGER, level="ERROR"):
            self.feed(turn(1, CHECK_ONLY))
        self.assertEqual([a["action"] for a in self.conn.requests], ["check"])
        self.assertEqual(c.fallbacks, 1)

    def test_undecodable_turn_is_not_answered(self):
        c = self.make()
        with self.assertLogs(LOGGER, level="ERROR"):
            self.feed({"type": "turn", "to": NAME, "seq": 1, "view": {}, "legal": {}})
        self.assertEqual(self.conn.requests, [])
        self.assertEqual(len(c.problems), 1)
        self.assertIsNone(c.exit_code)

    def test_stale_error_wording(self):
        self.assertTrue(is_stale_turn_error(STALE))
        self.assertTrue(is_stale_turn_error(f"{NAME} has no pending turn"))
        self.assertFalse(is_stale_turn_error(ILLEGAL))
        self.assertFalse(is_stale_turn_error(None))
        self.assertIs(fallback_action(CAN_CHECK), ActionType.CHECK)
        self.assertIs(fallback_action(FACING_BET), ActionType.CALL)

    # --- hand end ------------------------------------------------------------------

    def test_hand_end_is_answered_once_with_its_seq_and_the_policy(self):
        c = self.make()
        with self.assertLogs(LOGGER, level="INFO") as logs:
            self.feed(hand_end(9, hand=1, net=-2.0, stack=98.0))
        self.assertEqual(
            self.conn.requests,
            [
                {
                    "type": "hand_end",
                    "name": NAME,
                    "seq": 9,
                    "rebuy_bb": 2.0,
                    "sit_in": False,
                    "leave": False,
                }
            ],
        )
        self.assertEqual(c.hands_played, 1)
        self.assertEqual(c.stats.net_chips, -200)
        self.assertEqual(c.hand_ends_answered, 1)
        self.assertIn(
            "hand 1: net -2.00 BB, stack 98.00 BB, rebuy 2.00 BB", "\n".join(logs.output)
        )

    def test_not_dealt_in_does_not_count(self):
        c = self.make()
        self.feed(hand_end(2, hand=1, net=0.0, stack=100.0, dealt_in=False))
        self.assertEqual(c.hands_played, 0)
        self.assertEqual(c.stats.hands, 0)
        self.assertEqual(len(self.conn.of_type("hand_end")), 1)

    def test_seat_found_by_name(self):
        c = self.make()
        msg = encode_server_message(("hand_end", 3, 0, result(net=1.0, stack=101.0)), NAME)
        self.feed(msg)  # the message's seat 0 is Alice's: the name is looked up
        self.assertEqual(c.seat, 1)
        self.assertEqual(c.stack, 101.0)
        self.assertIsNone(self.conn.requests[0]["rebuy_bb"])

    def test_rebuy_capped_by_max_buy_in(self):
        settings = TableSettings(0.5, 1.0, 1.0, 2.0, 10, 60.0, 5.0, 0.05, 0.0)
        self.make(stack_bb=60.0, settings=settings)
        self.feed(hand_end(1, net=-30.0, stack=30.0))
        self.assertEqual(self.conn.requests[0]["rebuy_bb"], 30.0)

    def test_hands_limit_leaves(self):
        c = self.make(max_hands=2)
        self.feed(hand_end(1, hand=1, stack=99.0), rebought(1.0, 100.0, hand=1))
        self.assertFalse(self.conn.requests[-1]["leave"])
        with self.assertLogs(LOGGER, level="INFO") as logs:
            self.feed(hand_end(2, hand=2, stack=0.0))
        self.assertIn("leaving: 2 hands played", "\n".join(logs.output))
        last = self.conn.requests[-1]
        self.assertEqual((last["leave"], last["rebuy_bb"], last["sit_in"]), (True, None, False))
        self.assertEqual(c.status, "leaving")
        self.feed({"type": "left", "to": NAME, "final_stack_bb": 0.0})
        self.assertEqual(c.status, "left")
        self.assertEqual(self.feed(CLOSE), 0)

    def test_never_leaves_when_busted(self):
        c = self.make(rebuy="never")
        self.feed(hand_end(1, stack=50.0))
        self.assertIsNone(self.conn.requests[-1]["rebuy_bb"])
        self.feed(hand_end(2, hand=2, stack=0.0))
        self.assertTrue(self.conn.requests[-1]["leave"])
        self.assertEqual(c.status, "leaving")

    def test_sitting_out_asks_to_sit_in(self):
        c = self.make()
        with self.assertLogs(LOGGER, level="WARNING"):
            self.feed({"type": "sat_out", "to": NAME})
        self.assertEqual((c.status, c.sat_out_count), ("sitting_out", 1))
        self.assertEqual(self.conn.requests, [])
        self.feed(hand_end(5, net=0.0, stack=100.0, dealt_in=False))
        self.assertTrue(self.conn.requests[-1]["sit_in"])
        self.feed({"type": "returned", "to": NAME})
        self.assertEqual(c.status, "seated")

    def test_rejected_hand_end_is_logged(self):
        c = self.make()
        self.conn.responses = [{"ok": False, "error": "hand_end 1 is not the pending hand_end"}]
        with self.assertLogs(LOGGER, level="WARNING"):
            self.feed(hand_end(1))
        self.assertEqual(c.rejected_hand_ends, 1)
        self.assertIsNone(c.exit_code)

    # --- rebuy outcome ---------------------------------------------------------------

    def test_applied_rebuy(self):
        c = self.make()
        self.feed(hand_end(1, hand=1, net=-2.0, stack=98.0), rebought(2.0, 100.0, hand=1))
        self.assertEqual(c.stack, 100.0)
        self.assertEqual(c.stats.rebought_chips, 200)
        self.assertEqual(c.stats.rebuys, [(1, 200)])
        with self.assertNoLogs(LOGGER, level="WARNING"):
            self.feed(started(2))

    def test_missing_rebuy_is_reported_and_asked_again(self):
        c = self.make()
        self.feed(hand_end(1, hand=1, net=-2.0, stack=98.0))
        with self.assertLogs(LOGGER, level="WARNING") as logs:
            self.feed(started(2, stack=98.0))
        self.assertIn("rebuy of 2.00 BB was not applied", "\n".join(logs.output))
        self.assertEqual(c.stack, 98.0)
        self.feed(hand_end(2, hand=2, net=0.0, stack=98.0))
        self.assertEqual([r["rebuy_bb"] for r in self.conn.of_type("hand_end")], [2.0, 2.0])
        self.assertEqual(c.stats.rebought_chips, 0)

    def test_missing_rebuy_reported_at_next_hand_end(self):
        self.make()
        self.feed(hand_end(1, hand=1, net=-2.0, stack=98.0))
        with self.assertLogs(LOGGER, level="WARNING") as logs:
            self.feed(hand_end(2, hand=2, net=0.0, stack=98.0, dealt_in=False))
        self.assertIn("was not applied", "\n".join(logs.output))

    def test_other_players_rebuys_are_ignored(self):
        c = self.make()
        self.feed(event(PlayerRebought(0, "Alice", 5.0, 100.0, seq=3)))
        self.assertEqual(c.stats.rebought_chips, 0)
        self.assertEqual(c.stack, 100.0)

    # --- between hands -----------------------------------------------------------------

    def test_lobby_retry_when_busted(self):
        c = self.make()
        c.stack = 0.0
        self.feed(lobby("waiting", 1, 2))
        self.assertEqual(self.conn.requests, [{"type": "rebuy", "name": NAME, "amount_bb": 100.0}])
        self.feed(lobby("waiting", 1, 2))  # same status: no new attempt
        self.assertEqual(len(self.conn.requests), 1)
        self.conn.responses = [{"ok": False, "error": "nope"}]
        with self.assertLogs(LOGGER, level="WARNING"):
            self.feed(lobby("starting", 2, 2))  # changed: one more attempt
        self.assertEqual(len(self.conn.of_type("rebuy")), 2)

    def test_lobby_retry_when_sitting_out(self):
        self.make()
        self.feed({"type": "sat_out", "to": NAME})
        self.feed(lobby())
        self.assertEqual(self.conn.requests, [{"type": "sit_in", "name": NAME}])
        self.feed(lobby())
        self.assertEqual(len(self.conn.requests), 1)
        self.feed(started(3))
        self.feed(lobby())  # a hand happened since: one more attempt
        self.assertEqual(len(self.conn.requests), 2)

    def test_lobby_needs_nothing_when_fine(self):
        self.make()
        self.feed(lobby(), lobby("starting", 2, 2))
        self.assertEqual(self.conn.requests, [])
        c = self.make(rebuy="never")
        c.stack = 0.0
        self.feed(lobby())
        self.assertEqual(self.conn.requests, [])

    # --- other messages ------------------------------------------------------------------

    def test_messages_that_need_no_request(self):
        c = self.make(queued=True)
        self.assertEqual(c.status, "waiting")
        v = view(seat=1, stack=100.0)
        cards = Hand([Card.parse(x) for x in ("As", "Ks", "Qs", "Js")])
        msgs = [
            encode_server_message(("seated", v), NAME),
            encode_server_message(("hole_cards", cards, v), NAME),
            event(
                PlayerActed(
                    1, NAME, Street.PREFLOP, ActionType.CALL, 0.5, 1.0, False, 99.0, hand_no=1, seq=4
                )
            ),
            {"type": "acting", "to": NAME, "hand_number": 1, "seat": 0, "name": "Alice", "timeout": 5.0},
            lobby("starting", 2, 2),
            {"type": "future_thing", "to": NAME, "x": 1},
            {"type": "event", "to": NAME, "event": {"type": "future_event", "hand_no": 1, "seq": 9}},
            {"type": "returned", "to": NAME, "extra": {"a": 1}},
        ]
        self.feed(*msgs)
        self.assertEqual(self.conn.requests, [])
        self.assertEqual(c.status, "seated")
        self.assertEqual(c.stack, 99.0)
        self.assertEqual(c.seat, 1)
        self.assertEqual(c.problems, [])
        self.assertIsNone(c.exit_code)

    def test_malformed_event_is_skipped(self):
        c = self.make()
        with self.assertLogs(LOGGER, level="WARNING"):
            self.feed({"type": "event", "to": NAME, "event": {"type": "player_acted", "seat": 1}})
        self.assertEqual(len(c.problems), 1)
        self.assertIsNone(c.exit_code)
        self.feed(turn(1))
        self.assertEqual(len(self.conn.requests), 1)

    def test_unknown_fields_are_ignored(self):
        c = self.make()
        msg = event(PlayerRebought(1, NAME, 2.0, 77.0, seq=3))
        msg["event"]["new_field"] = True
        msg["also_new"] = 1
        self.feed(msg)
        self.assertEqual(c.stack, 77.0)
        self.assertEqual(c.problems, [])

    # --- end conditions --------------------------------------------------------------------

    def test_left_then_close(self):
        c = self.make()
        self.assertIsNone(self.feed({"type": "left", "to": NAME, "final_stack_bb": 104.5}))
        self.assertEqual(c.final_stack, 104.5)
        self.assertEqual(self.feed(CLOSE), 0)

    def test_left_then_disconnected(self):
        self.make()
        self.feed(LEFT)
        self.assertEqual(self.feed(DISCONNECTED), 0)

    def test_left_alone_ends_after_a_second(self):
        c = self.make()
        self.feed(LEFT)
        self.clock.t += 0.5
        self.assertIsNone(self.feed(None))
        self.clock.t += 0.6
        self.assertEqual(self.feed(None), 0)
        self.assertEqual(c.exit_code, 0)

    def test_close_alone(self):
        self.make()
        self.assertEqual(self.feed(CLOSE), 0)

    def test_disconnected_without_left(self):
        self.make()
        with self.assertLogs(LOGGER, level="ERROR") as logs:
            self.assertEqual(self.feed(DISCONNECTED), 1)
        self.assertIn("connection to the server lost", "\n".join(logs.output))

    def test_connection_lost_on_a_request(self):
        self.make()
        self.conn.lost = True
        with self.assertLogs(LOGGER, level="ERROR") as logs:
            self.assertEqual(self.feed(turn(1)), 1)
        self.assertIn("connection to the server lost", "\n".join(logs.output))

    def test_run_returns_the_exit_status(self):
        c = self.make()
        for m in (turn(1), hand_end(2), LEFT, CLOSE):
            self.inbox.put(m)
        self.assertEqual(c.run(), 0)
        self.assertEqual([r["type"] for r in self.conn.requests], ["action", "hand_end"])

    # --- user stop -------------------------------------------------------------------------------

    def test_stop_leaves_and_waits_for_left(self):
        c = self.make(FixedPlayer(ActionType.CHECK))
        c.request_stop()
        self.assertIsNone(self.feed(self.inbox.get_nowait()))
        self.assertEqual(self.conn.requests, [{"type": "leave", "name": NAME}])
        self.assertTrue(c.stop_requested)
        self.assertEqual(c.status, "leaving")
        self.feed(turn(5, CAN_CHECK))  # still answered
        self.assertEqual(self.conn.requests[-1]["action"], "check")
        self.feed(hand_end(6, stack=50.0))
        self.assertEqual(
            self.conn.requests[-1],
            {"type": "hand_end", "name": NAME, "seq": 6, "rebuy_bb": None, "sit_in": False, "leave": True},
        )
        self.feed({"type": "left", "to": NAME, "final_stack_bb": 50.0})
        self.assertEqual(self.feed(CLOSE), 0)

    def test_stop_times_out(self):
        c = self.make()
        c.request_stop()
        self.feed(self.inbox.get_nowait())
        self.clock.t += 4.9
        self.assertIsNone(self.feed(None))
        self.clock.t += 0.2
        with self.assertNoLogs(LOGGER, level="ERROR"):
            self.assertEqual(self.feed(None), 0)
        self.assertEqual(c.exit_code, 0)

    def test_stop_then_disconnect_after_accepted_leave(self):
        c = self.make()
        c.request_stop()
        self.feed(self.inbox.get_nowait())
        self.assertEqual(self.feed(DISCONNECTED), 0)

    def test_second_stop_exits_at_once(self):
        c = self.make()
        c.request_stop()
        c.request_stop()
        self.feed(self.inbox.get_nowait())
        self.assertEqual(self.feed(self.inbox.get_nowait()), 0)
        self.assertEqual(len(self.conn.of_type("leave")), 1)

    def test_stop_with_the_server_gone(self):
        c = self.make()
        self.conn.lost = True
        c.request_stop()
        with self.assertLogs(LOGGER, level="ERROR"):
            self.assertEqual(self.feed(self.inbox.get_nowait()), 1)

    def test_run_with_stop(self):
        c = self.make()
        c.request_stop()
        self.inbox.put({"type": "left", "to": NAME, "final_stack_bb": 100.0})
        self.inbox.put(CLOSE)
        self.assertEqual(c.run(), 0)
        self.assertEqual(c.status, "left")

    # --- misc ----------------------------------------------------------------------------------

    def test_summary(self):
        c = self.make()
        self.feed(hand_end(1, hand=1, net=1.5, stack=101.5))
        self.assertEqual(
            c.summary(),
            "Bot (fixed): 1 hands, net +1.50 BB = +75.00 big bets/100 hands, "
            "rebought 0.00 BB, final stack 101.50 BB",
        )

    def test_close(self):
        class Listener:
            closed = False

            def close(self):
                self.closed = True

        listener = Listener()
        c = self.make(listener=listener)
        c.close()
        self.assertTrue(listener.closed)
        self.assertTrue(self.conn.closed)

    def test_stop_marker(self):
        self.assertEqual(STOP, ("stop",))


if __name__ == "__main__":
    unittest.main()
