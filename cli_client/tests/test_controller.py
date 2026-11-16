import unittest

from entities import Street
from engine import ActionType, HandStarted, PlayerActed, PlayerStatus

from cli_client.controller import PRESET_CALL_ANY, PRESET_CHECK_FOLD, Client, ClientSettings
from cli_client.net import DISCONNECTED, ConnectionLost

try:
    from cli_client_tests.tests import helpers as h
except ImportError:  # run directly from the tests directory
    import helpers as h

ME = h.ME


def three_seats(alice_stack=100.0, **alice):
    return [
        h.seat_view(0, "Alice", alice_stack, **alice),
        h.seat_view(1, "Bob", 50.0, 0.5, 0.5),
        h.seat_view(2, "Carol", 20.0, 1.0, 1.0),
    ]


def preflop_view(**kw):
    defaults = dict(hole="AhKd7s2c", pot=1.5, current_bet=1.0, button=0, sb=1, bb=2)
    defaults.update(kw)
    return h.table_view(0, three_seats(), **defaults)


class ControllerTestCase(unittest.TestCase):
    def make(self, responder=None, settings=None, table=h.SETTINGS, queued=False):
        self.clock = h.FakeClock()
        self.conn = h.FakeConnection(responder)
        self.inbox = h.new_inbox()
        self.client = Client(
            ME,
            self.conn,
            self.inbox,
            clock=self.clock,
            settings=settings or ClientSettings(stack_bb=100.0, hand_end_timeout=20.0, exit_delay=1.0),
            table=table,
            queued=queued,
        )
        return self.client

    def setUp(self):
        self.make()

    def feed(self, *msgs):
        for m in msgs:
            self.inbox.put(m)
        self.client.drain()

    def keys(self, *keys):
        for k in keys:
            self.client.handle_key(k)

    def start_hand(self, hand_no=1):
        self.feed(
            h.seated_msg(h.table_view(0, three_seats(in_hand=False), hand_number=hand_no - 1, pot=0, current_bet=0)),
            h.event_msg(HandStarted(0, 1, 2, ((0, "Alice", 100.0), (1, "Bob", 50.0), (2, "Carol", 20.0)), hand_no=hand_no)),
            h.hole_cards_msg("AhKd7s2c", preflop_view(hand_number=hand_no)),
        )

    def turn(self, seq=7, legal=None, view=None, timeout=30.0):
        self.feed(h.turn_msg(seq, view or preflop_view(), legal or h.legal(), timeout))

    def labels(self):
        return [i.label for i in self.client.menu]

    def actions(self):
        return [(r["seq"], r["action"]) for r in self.conn.of_type("action")]


class MenuTest(ControllerTestCase):
    def test_call_and_raise_menu(self):
        self.start_hand()
        self.turn(legal=h.legal(can_check=False, call_amount=1.0, can_raise=True, raise_to=2.0))
        self.assertEqual(self.client.menu_kind, "action")
        self.assertEqual(self.labels(), ["Fold", "Call 1", "Raise to 2", "Leave table"])
        self.assertTrue(self.client.bell)
        self.assertEqual(self.client.countdown(), 30.0)
        self.assertEqual(self.client.state.acting, (0, self.clock.now + 30.0))

    def test_check_and_bet_menu(self):
        self.start_hand()
        view = preflop_view(street=Street.FLOP, board="Ah7d2c", current_bet=0.0)
        self.turn(legal=h.legal(can_check=True, call_amount=0.0, raise_to=1.0), view=view)
        self.assertEqual(self.labels(), ["Fold", "Check", "Bet 1", "Leave table"])

    def test_no_raise_and_all_in_labels(self):
        self.start_hand()
        seats = three_seats(alice_stack=1.5)
        view = h.table_view(0, seats, hole="AhKd7s2c", current_bet=4.0, sb=1, bb=2)
        self.turn(legal=h.legal(call_amount=1.5, can_raise=False), view=view)
        self.assertEqual(self.labels(), ["Fold", "Call 1.5 (all-in)", "Leave table"])
        view = h.table_view(0, three_seats(alice_stack=3.0), hole="AhKd7s2c", current_bet=1.0, sb=1, bb=2)
        self.turn(seq=8, legal=h.legal(call_amount=1.0, can_raise=True, raise_to=3.0), view=view)
        self.assertEqual(self.labels()[2], "Raise to 3 (all-in)")

    def test_number_keys(self):
        self.start_hand()
        self.turn(seq=7)
        self.keys("3")
        self.assertEqual(self.actions(), [(7, "raise")])
        self.assertEqual(self.client.menu, [])
        self.assertIsNone(self.client.pending_turn)

    def test_arrow_keys_and_enter(self):
        self.start_hand()
        self.turn(seq=9)
        self.assertEqual(self.client.selected, 1)  # call / check is preselected
        self.keys("down", "enter")
        self.assertEqual(self.actions(), [(9, "raise")])

    def test_hotkeys(self):
        self.start_hand()
        self.turn(seq=1)
        self.keys("c")
        self.turn(seq=2)
        self.keys("r")
        self.turn(seq=3, legal=h.legal(can_check=False))
        self.keys("f")  # no check possible: no confirmation
        self.assertEqual(self.actions(), [(1, "call"), (2, "raise"), (3, "fold")])

    def test_fold_when_check_is_possible_asks(self):
        self.start_hand()
        self.turn(seq=4, legal=h.legal(can_check=True, call_amount=0.0))
        self.keys("f")
        self.assertEqual(self.actions(), [])
        self.assertIn("Fold instead of checking", self.client.ui_view().prompt)
        self.keys("n")
        self.assertEqual(self.actions(), [])
        self.assertEqual(self.client.menu_kind, "action")
        self.keys("1", "y")
        self.assertEqual(self.actions(), [(4, "fold")])

    def test_illegal_action_keeps_the_menu(self):
        self.make(lambda m: {"type": "response", "ok": False, "error": "raise is not legal now (legal: fold, call)"})
        self.start_hand()
        self.turn(seq=5)
        self.keys("r")
        self.assertEqual(self.client.menu_kind, "action")
        self.assertIsNotNone(self.client.pending_turn)
        self.assertIn("not legal", self.client.error)

    def test_stale_turn_clears_the_menu(self):
        self.make(lambda m: {"type": "response", "ok": False, "error": "Alice has no pending turn"})
        self.start_hand()
        self.turn(seq=5)
        self.keys("c")
        self.assertIsNone(self.client.pending_turn)
        self.assertEqual(self.client.menu, [])
        self.assertIn("Too late", self.client.error)

    def test_timeout_fold_event_clears_the_turn(self):
        self.start_hand()
        self.turn(seq=5)
        self.feed(h.event_msg(PlayerActed(0, "Alice", Street.PREFLOP, ActionType.FOLD, 0.0, 0.0, False, 100.0, "timeout", hand_no=1)))
        self.assertIsNone(self.client.pending_turn)
        self.assertEqual(self.client.menu, [])
        self.assertIn("sit out", self.client.status)


class PresetTest(ControllerTestCase):
    def test_check_fold_folds_facing_a_bet(self):
        self.start_hand()
        self.keys("x")
        self.assertEqual(self.client.preset, PRESET_CHECK_FOLD)
        self.assertEqual(self.client.ui_view().preset, "Check/Fold")
        self.turn(seq=3, legal=h.legal(can_check=False, call_amount=1.0))
        self.assertEqual(self.actions(), [(3, "fold")])
        self.assertIsNone(self.client.preset)
        self.assertEqual(self.client.menu, [])

    def test_check_fold_checks(self):
        self.start_hand()
        self.keys("F2")
        self.turn(seq=3, legal=h.legal(can_check=True, call_amount=0.0))
        self.assertEqual(self.actions(), [(3, "check")])

    def test_call_any(self):
        self.start_hand()
        self.keys("a")
        self.assertEqual(self.client.preset, PRESET_CALL_ANY)
        self.turn(seq=3, legal=h.legal(can_check=False, call_amount=2.0))
        self.keys("F3")
        self.turn(seq=4, legal=h.legal(can_check=True, call_amount=0.0))
        self.assertEqual(self.actions(), [(3, "call"), (4, "check")])

    def test_toggle_off_and_not_in_hand(self):
        self.start_hand()
        self.keys("x", "x")
        self.assertIsNone(self.client.preset)
        self.make()
        self.keys("x")
        self.assertIsNone(self.client.preset)
        self.assertIn("in a hand", self.client.error)

    def test_cleared_at_hand_end(self):
        self.start_hand()
        self.keys("a")
        self.feed(h.hand_end_msg(11, 0, h.hand_result({0: "Alice", 1: "Bob", 2: "Carol"}, {0: 100, 1: 51, 2: 19})))
        self.assertIsNone(self.client.preset)


class HandEndTest(ControllerTestCase):
    def result(self, alice_stack):
        return h.hand_result({0: "Alice", 1: "Bob", 2: "Carol"}, {0: alice_stack, 1: 60.0, 2: 20.0})

    def hand_end_requests(self):
        return self.conn.of_type("hand_end")

    def test_answered_at_once(self):
        self.start_hand()
        self.feed(h.hand_end_msg(12, 0, self.result(90.0)))
        self.assertEqual(
            self.hand_end_requests(),
            [{"type": "hand_end", "name": ME, "seq": 12, "rebuy_bb": None, "sit_in": False, "leave": False}],
        )
        self.assertIsNone(self.client.pending_hand_end)
        self.assertIsNone(self.client.menu_kind)

    def test_busted_menu_rebuy(self):
        self.start_hand()
        self.feed(h.hand_end_msg(12, 0, self.result(0.0)))
        self.assertEqual(self.client.menu_kind, "hand_end")
        self.assertEqual(self.labels(), ["Rebuy 100 BB", "Sit out", "Leave table"])
        self.assertEqual(self.hand_end_requests(), [])
        self.assertEqual(self.client.countdown(), 20.0)
        self.keys("1")
        self.assertEqual(self.hand_end_requests()[0]["rebuy_bb"], 100.0)
        self.assertIsNone(self.client.menu_kind)

    def test_busted_menu_edit_amount_capped(self):
        table = h.SETTINGS.__class__(**{**h.SETTINGS.__dict__, "max_buy_in_bb": 60.0})
        self.make(table=table)
        self.start_hand()
        self.feed(h.hand_end_msg(12, 0, self.result(0.0)))
        self.assertEqual(self.labels()[0], "Rebuy 60 BB")
        self.keys("e", "backspace", "backspace", "4", "0", "enter")
        self.assertEqual(self.labels()[0], "Rebuy 40 BB")
        self.keys("1")
        self.assertEqual(self.hand_end_requests()[0]["rebuy_bb"], 40.0)

    def test_busted_sit_out_and_leave(self):
        self.start_hand()
        self.feed(h.hand_end_msg(12, 0, self.result(0.0)))
        self.keys("2")
        self.assertEqual(self.hand_end_requests()[-1], {"type": "hand_end", "name": ME, "seq": 12, "rebuy_bb": None, "sit_in": False, "leave": False})
        self.feed(h.hand_end_msg(13, 0, self.result(0.0)))
        self.keys("3")
        self.assertTrue(self.hand_end_requests()[-1]["leave"])
        self.assertTrue(self.client.leaving)

    def test_timeout_sends_the_default(self):
        self.start_hand()
        self.feed(h.hand_end_msg(12, 0, self.result(0.0)))
        self.client.tick(self.clock.advance(10))
        self.assertEqual(self.hand_end_requests(), [])
        self.client.tick(self.clock.advance(10.5))
        self.assertEqual(self.hand_end_requests()[0]["rebuy_bb"], None)
        self.assertFalse(self.hand_end_requests()[0]["sit_in"])
        self.assertIsNone(self.client.menu_kind)

    def test_sitting_out_menu(self):
        self.start_hand()
        self.feed(h.simple_msg("sat_out"))
        self.assertTrue(self.client.state.sitting_out)
        self.feed(h.hand_end_msg(12, 0, self.result(80.0)))
        self.assertEqual(self.labels(), ["Sit in", "Keep sitting out", "Leave table"])
        self.keys("1")
        self.assertTrue(self.hand_end_requests()[0]["sit_in"])
        self.feed(h.simple_msg("returned"))
        self.assertFalse(self.client.state.sitting_out)

    def test_sitting_out_timeout_keeps_sitting_out(self):
        self.start_hand()
        self.feed(h.simple_msg("sat_out"), h.hand_end_msg(12, 0, self.result(80.0)))
        self.client.tick(self.clock.advance(21))
        self.assertFalse(self.hand_end_requests()[0]["sit_in"])

    def test_queued_top_up_goes_into_the_reply(self):
        self.start_hand()
        self.keys("t")
        self.assertIn("Top up", self.client.ui_view().prompt)
        self.keys("enter")  # default: back to --stack (100 - 100 = 0) -> empty
        self.keys("t", "2", "5", "enter")
        self.assertEqual(self.client.queued_topup, 25.0)
        self.assertEqual(self.conn.of_type("rebuy"), [])
        self.feed(h.hand_end_msg(12, 0, self.result(70.0)))
        self.assertEqual(self.hand_end_requests()[0]["rebuy_bb"], 25.0)
        self.assertIsNone(self.client.queued_topup)

    def test_top_up_between_hands_is_sent_at_once(self):
        self.feed(h.seated_msg(h.table_view(0, three_seats(alice_stack=60.0, in_hand=False), hand_number=0, pot=0, current_bet=0)))
        self.keys("t")
        self.assertEqual(self.client.input.text, "40")
        self.keys("enter")
        self.assertEqual(self.conn.of_type("rebuy"), [{"type": "rebuy", "name": ME, "amount_bb": 40.0}])

    def test_queued_leave_goes_into_the_reply(self):
        self.start_hand()
        self.client.queued_leave = True
        self.feed(h.hand_end_msg(12, 0, self.result(0.0)))
        self.assertTrue(self.hand_end_requests()[0]["leave"])
        self.assertIsNone(self.client.menu_kind)

    def test_rejected_sit_in_is_offered_again(self):
        self.make(lambda m: {"type": "response", "ok": False, "error": "cannot sit in during a hand"} if m["type"] == "sit_in" else {"type": "response", "ok": True})
        self.start_hand()
        self.feed(h.simple_msg("sat_out"))
        self.keys("s")
        self.assertEqual(self.client.deferred, "sit_in")
        self.feed(h.hand_end_msg(12, 0, self.result(80.0)))
        self.assertEqual(self.client.menu[self.client.selected].kind, "sit_in")


class LeaveAndCloseTest(ControllerTestCase):
    def test_leave_between_hands(self):
        self.feed(h.seated_msg(h.table_view(0, three_seats(in_hand=False), hand_number=0, pot=0, current_bet=0)))
        self.keys("l")
        self.assertEqual(self.conn.of_type("leave"), [{"type": "leave", "name": ME}])
        self.assertTrue(self.client.leaving)
        self.feed(h.left_msg(100.0), h.simple_msg("close"), DISCONNECTED)
        self.assertTrue(self.client.state.left)
        self.assertFalse(self.client.done)
        self.client.tick(self.clock.advance(1.5))
        self.assertTrue(self.client.done)
        self.assertEqual(self.client.exit_code, 0)
        self.assertEqual(self.client.final_stack, 100.0)
        self.assertIn("final stack 100 BB", self.client.summary())

    def test_leave_in_hand_asks_first(self):
        self.start_hand()
        self.keys("q")
        self.assertEqual(self.conn.of_type("leave"), [])
        self.keys("y")
        self.assertEqual(len(self.conn.of_type("leave")), 1)
        self.assertIn("after this hand", self.client.ui_view().status)

    def test_leave_during_my_turn_answers_the_turn(self):
        self.start_hand()
        self.turn(seq=6)
        self.keys("l", "y")
        self.assertEqual(self.actions(), [(6, "leave")])
        self.assertEqual(self.conn.of_type("leave"), [])
        self.assertTrue(self.client.leaving)

    def test_second_ctrl_c_quits(self):
        self.start_hand()
        self.keys("ctrl-c")
        self.assertFalse(self.client.done)
        self.assertIsNotNone(self.client.confirm)
        self.keys("ctrl-c")
        self.assertTrue(self.client.done)

    def test_close_without_left(self):
        self.start_hand()
        self.feed(h.simple_msg("close"))
        self.assertTrue(self.client.server_closed)
        self.assertIn("Server closed the table", self.client.ui_view().status)
        self.assertFalse(self.client.done)
        self.keys("x")
        self.assertTrue(self.client.done)
        self.assertEqual(self.client.exit_code, 1)

    def test_connection_lost(self):
        self.make(lambda m: ConnectionLost("boom"))
        self.start_hand()
        self.turn(seq=3)
        self.keys("c")
        self.assertTrue(self.client.server_closed)
        self.assertIn("Connection to server lost", self.client.status)
        self.assertEqual(self.client.exit_code, 1)


class InfoMessagesTest(ControllerTestCase):
    def test_acting_highlights_the_seat(self):
        self.start_hand()
        self.feed(h.acting_msg(1, 1, "Bob", 30.0))
        self.assertEqual(self.client.state.acting, (1, self.clock.now + 30.0))
        self.feed(h.event_msg(PlayerActed(1, "Bob", Street.PREFLOP, ActionType.CALL, 0.5, 1.0, False, 49.0, hand_no=1)))
        self.assertIsNone(self.client.state.acting)

    def test_lobby_text(self):
        self.feed(h.seated_msg(h.table_view(0, [h.seat_view(0, "Alice", 100, in_hand=False)], hand_number=0, pot=0, current_bet=0)))
        self.assertEqual(self.client.lobby_text(), "Waiting for players")
        self.feed(h.lobby_msg("waiting", 1))
        self.assertEqual(self.client.lobby_text(), "Waiting for players (1/2)")
        self.feed(h.lobby_msg("starting", 2, starts_in=12.0))
        self.assertEqual(self.client.lobby_text(), "Game starts in 12 s")
        self.client.tick(self.clock.advance(5))
        self.assertEqual(self.client.lobby_text(), "Game starts in 7 s")

    def test_queued_lobby_text(self):
        self.make(queued=True)
        self.assertEqual(self.client.lobby_text(), "Waiting for the current hand to finish")
        self.feed(h.seated_msg(h.table_view(1, [h.seat_view(0, "Bob", 100, in_hand=False), h.seat_view(1, "Alice", 100, in_hand=False)], hand_number=3, pot=0, current_bet=0)))
        self.assertFalse(self.client.state.queued)

    def test_unknown_messages_ignored(self):
        self.feed({"type": "something_new", "to": ME}, {"type": "event", "to": ME, "event": {"type": "brand_new"}})
        self.feed("garbage")
        self.assertFalse(self.client.server_closed)

    def test_hands_played_counted(self):
        self.start_hand()
        self.assertEqual(self.client.hands_played, 1)
        self.assertEqual(self.client.state.me.status, PlayerStatus.ACTIVE)


if __name__ == "__main__":
    unittest.main()
