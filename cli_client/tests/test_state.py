import unittest

from entities import Board, Card, Hand, Street
from engine import (
    ActionType,
    BetReturned,
    BlindPosted,
    BoardDealt,
    HandShown,
    HandStarted,
    PlayerActed,
    PlayerJoined,
    PlayerLeft,
    PlayerRebought,
    PlayerReturned,
    PlayerSatOut,
    PlayerStatus,
    PotAwarded,
)

from cli_server.protocol import decode_event, encode_event

from cli_client.state import TableState

try:
    from cli_client_tests.tests.helpers import SETTINGS, hand_result, seat_view, table_view
except ImportError:  # run directly from the tests directory
    from helpers import SETTINGS, hand_result, seat_view, table_view


def wire(event):
    """Through the wire format, as the client receives it."""
    return decode_event(encode_event(event))


def acted(seat, name, action, amount, to, stack, street=Street.PREFLOP, all_in=False, reason=None, hand_no=1):
    return PlayerActed(seat, name, street, action, amount, to, all_in, stack, reason, hand_no=hand_no)


class StateTestCase(unittest.TestCase):
    def setUp(self):
        self.state = TableState("Alice", SETTINGS)
        # Seated between hands: Alice (0), Bob (1), Carol (2).
        view = table_view(
            0,
            [seat_view(0, "Alice", 100, in_hand=False), seat_view(1, "Bob", 50, in_hand=False), seat_view(2, "Carol", 20, in_hand=False)],
            hand_number=0,
            pot=0,
            current_bet=0,
            button=0,
            sb=None,
            bb=None,
        )
        self.state.apply_view(view, hand_running=False)

    def apply(self, *events):
        for e in events:
            self.assertTrue(self.state.apply_event(wire(e)))

    def start_hand(self, hand_no=1):
        self.apply(
            HandStarted(0, 1, 2, ((0, "Alice", 100.0), (1, "Bob", 50.0), (2, "Carol", 20.0)), hand_no=hand_no),
            BlindPosted(1, "Bob", "SB", 0.5, False, 49.5, hand_no=hand_no),
            BlindPosted(2, "Carol", "BB", 1.0, False, 19.0, hand_no=hand_no),
        )

    def stacks(self):
        return {s.name: s.stack for s in self.state.seats}

    def log_texts(self):
        return [line.text for line in self.state.log]


class HandFlowTest(StateTestCase):
    def test_blinds_actions_and_pot(self):
        self.start_hand()
        s = self.state
        self.assertTrue(s.hand_running)
        self.assertEqual(s.hand_number, 1)
        self.assertEqual((s.button, s.sb_seat, s.bb_seat), (0, 1, 2))
        self.assertEqual(s.pot, 1.5)
        self.assertEqual(s.current_bet, 1.0)
        self.assertEqual(s.find("Bob").street_bet, 0.5)
        self.assertEqual(s.find("Carol").last_action, "BB 1")
        self.apply(
            acted(0, "Alice", ActionType.RAISE, 2.0, 2.0, 98.0),
            acted(1, "Bob", ActionType.CALL, 1.5, 2.0, 48.0),
            acted(2, "Carol", ActionType.FOLD, 0.0, 1.0, 19.0),
        )
        self.assertEqual(self.stacks(), {"Alice": 98.0, "Bob": 48.0, "Carol": 19.0})
        self.assertEqual(s.pot, 5.0)
        self.assertEqual(s.find("Alice").last_action, "Raise to 2")
        self.assertEqual(s.find("Bob").last_action, "Call 1.5")
        self.assertTrue(s.find("Carol").folded)
        self.assertEqual(s.find("Carol").last_action, "Fold")
        texts = self.log_texts()
        self.assertIn("── Hand #1 ──", texts)
        self.assertIn("Bob posts SB 0.5", texts)
        self.assertIn("Carol posts BB 1", texts)
        self.assertIn("Alice raises to 2", texts)
        self.assertIn("Bob calls 1.5", texts)
        self.assertIn("Carol folds", texts)
        mine = [line for line in s.log if line.text == "Alice raises to 2"][0]
        self.assertTrue(mine.mine)

        board = Board.parse("Ah7d2c")
        self.apply(BoardDealt(Street.FLOP, board.flop, board, hand_no=1))
        self.assertEqual(s.street, Street.FLOP)
        self.assertEqual(s.board, board)
        self.assertEqual(s.find("Alice").street_bet, 0.0)
        self.assertEqual(s.find("Alice").last_action, "")
        self.assertEqual(s.find("Carol").last_action, "Fold")
        self.assertEqual(s.current_bet, 0.0)
        flop_line = s.log[-1]
        self.assertEqual(flop_line.text, "Flop: [A♥][7♦][2♣]")
        self.assertEqual([st for _, st in flop_line.segments[1:]], ["hearts", "diamonds", "clubs"])

        self.apply(
            acted(1, "Bob", ActionType.CHECK, 0.0, 0.0, 48.0, Street.FLOP),
            acted(0, "Alice", ActionType.RAISE, 1.0, 1.0, 97.0, Street.FLOP),
        )
        self.assertEqual(s.find("Alice").last_action, "Bet 1")
        self.assertIn("Bob checks", self.log_texts())
        self.assertIn("Alice bets 1", self.log_texts())
        self.apply(acted(1, "Bob", ActionType.FOLD, 0.0, 0.0, 48.0, Street.FLOP))
        self.assertEqual(s.pot, 6.0)
        self.apply(
            BetReturned(0, "Alice", 1.0, 98.0, hand_no=1),
            PotAwarded(0, "Alice", 0, "whole", 5.0, 103.0, hand_no=1),
        )
        self.assertEqual(s.pot, 0.0)
        self.assertEqual(s.find("Alice").stack, 103.0)
        self.assertEqual(s.find("Alice").won, 5.0)
        self.assertIn("1 returned to Alice", self.log_texts())
        self.assertIn("Alice wins 5 (uncontested)", self.log_texts())

    def test_showdown_and_all_in(self):
        self.start_hand()
        self.apply(
            acted(0, "Alice", ActionType.FOLD, 0.0, 0.0, 100.0),
            acted(1, "Bob", ActionType.RAISE, 1.5, 2.0, 48.0),
            acted(2, "Carol", ActionType.CALL, 1.0, 2.0, 18.0, all_in=False),
        )
        board = Board.parse("2h7h9hJcQd")
        self.apply(BoardDealt(Street.RIVER, (board.river,), board, hand_no=1))
        self.apply(acted(2, "Carol", ActionType.RAISE, 18.0, 18.0, 0.0, Street.RIVER, all_in=True))
        self.assertIn("Carol bets 18 (all-in)", self.log_texts())
        self.assertTrue(self.state.find("Carol").all_in)
        hand = Hand.parse("AhKh2c3d")
        self.apply(HandShown(1, "Bob", hand, 48.0, hand_no=1))
        self.assertEqual(self.state.find("Bob").shown, hand)
        self.assertIn("Bob shows [A♥][K♥][2♣][3♦] – Flush, A high; no low", self.log_texts())
        self.apply(
            PotAwarded(1, "Bob", 0, "high", 3.0, 51.0, hand_no=1),
            PotAwarded(2, "Carol", 0, "low", 2.0, 2.0, hand_no=1),
        )
        self.assertIn("Bob wins 3 (high)", self.log_texts())
        self.assertIn("Carol wins 2 (low)", self.log_texts())

    def test_forced_folds(self):
        self.start_hand()
        self.apply(acted(0, "Alice", ActionType.FOLD, 0.0, 0.0, 100.0, reason="timeout"))
        self.apply(acted(1, "Bob", ActionType.FOLD, 0.0, 0.5, 49.5, reason="leave"))
        self.assertEqual(self.state.find("Alice").last_action, "Fold (timeout)")
        self.assertEqual(self.state.find("Bob").last_action, "Fold (left)")
        self.assertIn("Alice folds (timeout)", self.log_texts())

    def test_stacks_follow_events(self):
        self.start_hand()
        self.apply(acted(0, "Alice", ActionType.CALL, 1.0, 1.0, 99.0))
        for s in self.state.seats:
            self.assertEqual(s.stack, {"Alice": 99.0, "Bob": 49.5, "Carol": 19.0}[s.name])

    def test_pot_matches_the_next_view(self):
        self.start_hand()
        self.apply(
            acted(0, "Alice", ActionType.RAISE, 2.0, 2.0, 98.0),
            acted(1, "Bob", ActionType.CALL, 1.5, 2.0, 48.0),
        )
        pot_before = self.state.pot
        view = table_view(
            2,
            [
                seat_view(0, "Alice", 98, 2, 2),
                seat_view(1, "Bob", 48, 2, 2),
                seat_view(2, "Carol", 19, 1, 1),
            ],
            pot=5.0,
            current_bet=2.0,
            button=0,
            sb=1,
            bb=2,
        )
        self.assertEqual(pot_before, view.pot)
        self.state.apply_view(view, hand_running=True)
        self.assertEqual(self.state.pot, 5.0)
        # Per-hand display survives the resync.
        self.assertEqual(self.state.find("Alice").last_action, "Raise to 2")

    def test_unknown_event_ignored(self):
        class Strange:
            pass

        self.assertFalse(self.state.apply_event(Strange()))


class SeatingTest(StateTestCase):
    def test_join_and_leave_renumber_by_name(self):
        self.apply(PlayerJoined(1, "Dave", 40.0))
        self.assertEqual([s.name for s in self.state.seats], ["Alice", "Dave", "Bob", "Carol"])
        self.assertEqual(self.state.my_seat, 0)
        self.assertIn("Dave joins the table (40 BB)", self.log_texts())
        self.apply(PlayerLeft(0, "Alice", 100.0))
        self.assertEqual([s.name for s in self.state.seats], ["Dave", "Bob", "Carol"])
        self.assertIsNone(self.state.my_seat)
        self.apply(PlayerJoined(3, "Erin", 100.0))
        self.assertEqual(self.state.seat_of("Erin"), 3)
        self.assertEqual(self.state.seat_of("Bob"), 1)

    def test_join_shifts_the_button(self):
        self.state.button = 2
        self.apply(PlayerJoined(1, "Dave", 40.0))
        self.assertEqual(self.state.button, 3)
        self.assertEqual(self.state.seat_name(self.state.button), "Carol")
        self.apply(PlayerLeft(0, "Alice", 100.0))
        self.assertEqual(self.state.seat_name(self.state.button), "Carol")

    def test_own_join_after_seated_view_is_not_duplicated(self):
        self.apply(PlayerJoined(0, "Alice", 100.0))
        self.assertEqual([s.name for s in self.state.seats], ["Alice", "Bob", "Carol"])

    def test_status_events(self):
        self.apply(PlayerSatOut(0, "Alice", 100.0))
        self.assertTrue(self.state.sitting_out)
        self.assertEqual(self.state.me.status, PlayerStatus.SITTING_OUT)
        self.apply(PlayerRebought(0, "Alice", 20.0, 120.0), PlayerReturned(0, "Alice", 120.0))
        self.assertFalse(self.state.sitting_out)
        self.assertEqual(self.state.me.stack, 120.0)
        self.assertIn("Alice sits out", self.log_texts())
        self.assertIn("Alice rebuys 20", self.log_texts())
        self.assertIn("Alice is back", self.log_texts())


class ResyncTest(StateTestCase):
    def test_apply_view_replaces_drifted_model(self):
        self.start_hand()
        self.state.find("Bob").stack = 1234.0  # drift
        self.state.seats.append(self.state.seats[0].__class__("Ghost", 1.0))
        view = table_view(
            0,
            [seat_view(0, "Alice", 100), seat_view(1, "Bob", 49.5, 0.5, 0.5), seat_view(2, "Carol", 19, 1, 1)],
            hole="AhKd7s2c",
            pot=1.5,
            button=0,
            sb=1,
            bb=2,
        )
        self.state.apply_view(view, hand_running=True)
        self.assertEqual([s.name for s in self.state.seats], ["Alice", "Bob", "Carol"])
        self.assertEqual(self.state.find("Bob").stack, 49.5)
        self.assertEqual(self.state.my_cards, Hand.parse("AhKd7s2c"))

    def test_apply_result_reconciles(self):
        self.start_hand()
        self.state.find("Carol").stack = 999.0  # drift
        result = hand_result(
            {0: "Alice", 1: "Bob", 2: "Carol"},
            {0: 101.0, 1: 49.5, 2: 19.0},
            board="2h7h9hJcQd",
            shown={0: "AhKh2c3d"},
            sb=1,
            bb=2,
        )
        self.state.apply_result(result)
        self.assertFalse(self.state.hand_running)
        self.assertEqual(self.stacks(), {"Alice": 101.0, "Bob": 49.5, "Carol": 19.0})
        self.assertEqual(self.state.find("Alice").shown, Hand.parse("AhKh2c3d"))
        self.assertEqual(self.state.board, Board.parse("2h7h9hJcQd"))
        self.assertEqual(self.state.pot, 0.0)

    def test_hand_started_resets_the_hand(self):
        self.start_hand(1)
        self.state.find("Alice").won = 3.0
        self.state.my_cards = Hand.parse("AhKd7s2c")
        self.start_hand(2)
        self.assertEqual(self.state.hand_number, 2)
        self.assertEqual(self.state.find("Alice").won, 0.0)
        self.assertIsNone(self.state.my_cards)
        self.assertEqual(self.state.pot, 1.5)

    def test_log_is_capped(self):
        for i in range(600):
            self.state.add_log(f"line {i}")
        self.assertEqual(len(self.state.log), 500)
        self.assertEqual(self.state.log[-1].text, "line 599")


if __name__ == "__main__":
    unittest.main()
