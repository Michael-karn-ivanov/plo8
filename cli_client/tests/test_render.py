import unittest

from entities import Board, Hand, Street
from engine import PlayerStatus

from cli_client.controller import MenuItem, UIView
from cli_client.render import MIN_HEIGHT, MIN_WIDTH, SEAT_H, SEAT_W, render, seat_positions
from cli_client.state import SeatState, TableState

try:
    from cli_client_tests.tests.helpers import SETTINGS
except ImportError:  # run directly from the tests directory
    from helpers import SETTINGS

W, H = 130, 40
TABLE_CENTRE_X = 49  # table area is x 1..96 at width 130 (log column on the right)


def fixed_state(my_name="Alice", names=("Alice", "Bob", "Carol", "Dave")):
    state = TableState(my_name, SETTINGS)
    stacks = {"Alice": 37.5, "Bob": 42.5, "Carol": 18.0, "Dave": 100.0, "Erin": 9.0, "Fay": 12.0}
    state.seats = [SeatState(n, stacks.get(n, 50.0), in_hand=True) for n in names]
    state.hand_number = 12
    state.hand_running = True
    state.street = Street.TURN
    state.board = Board.parse("Ah7d2cKs")
    state.button, state.sb_seat, state.bb_seat = 1, 2, 3
    state.pot = 9.5
    state.my_cards = Hand.parse("As2dKh3c")
    if "Carol" in names:
        carol = state.find("Carol")
        carol.folded = True
        carol.last_action = "Fold"
    if "Bob" in names:
        state.find("Bob").street_bet = 2.0
        state.find("Bob").last_action = "Raise to 2"
    state.add_log("Bob posts SB 0.5")
    state.add_log("Dave raises to 2")
    return state


def ui(**kw):
    defaults = dict(now=100.0)
    defaults.update(kw)
    return UIView(**defaults)


def box_of(frame, title):
    """(x, y) of the top-left corner of the seat box whose title starts with `title`."""
    for x, y in frame.find_all(title):
        if x > 0 and frame.cells[y][x - 1][0] in "┌+":
            return x - 1, y
    raise AssertionError(f"no seat box for {title!r}")


class RenderTest(unittest.TestCase):
    def test_table_contents(self):
        state = fixed_state()
        frame = render(state, ui(), W, H)
        text = frame.text()
        for name in ("You (Alice)", "Bob", "Carol", "Dave"):
            self.assertIn(name, text)
        for stack in ("37.5 BB", "42.5 BB", "18 BB", "100 BB"):
            self.assertIn(stack, text)
        self.assertIn("Omaha Hi-Lo 8/b", text)
        self.assertIn("Limit 1/2 (blinds 0.5/1)", text)
        self.assertIn("Hand #12", text)
        self.assertIn("Turn", text)
        self.assertIn("Pot: 9.5 BB", text)
        self.assertIn("Raise to 2", text)
        self.assertIn("●2", text)
        # Board: four cards in suit colours and one empty slot.
        self.assertIn("[A♥][7♦][2♣][K♠][  ]", text)
        self.assertEqual(frame.styles_of("[A♥]"), {"hearts"})
        self.assertEqual(frame.styles_of("[7♦]"), {"diamonds"})
        self.assertEqual(frame.styles_of("[K♠]"), {"spades"})
        self.assertEqual(frame.styles_of("[2♣]"), {"clubs"})
        # My cards face up (in the seat box and my panel), others face down.
        self.assertIn("A♠ 2♦ K♥ 3♣", text)
        self.assertIn("Your hand: [A♠][2♦][K♥][3♣]", text)
        self.assertEqual(len(frame.find_all("▒▒ ▒▒ ▒▒ ▒▒")), 2)  # Bob and Dave; Carol folded
        # Log lines.
        self.assertIn("Bob posts SB 0.5", text)
        self.assertIn("Dave raises to 2", text)

    def test_badges(self):
        state = fixed_state()
        state.find("Dave").all_in = True
        frame = render(state, ui(), W, H)
        bx, by = box_of(frame, "Bob")
        self.assertIn("D", frame.row_text(by + 1)[bx : bx + SEAT_W])
        cx, cy = box_of(frame, "Carol")
        self.assertIn("SB", frame.row_text(cy + 1)[cx : cx + SEAT_W])
        dx, dy = box_of(frame, "Dave")
        self.assertIn("BB", frame.row_text(dy + 1)[dx : dx + SEAT_W])
        self.assertIn("ALL-IN", frame.row_text(dy + 2)[dx : dx + SEAT_W])
        state.find("Dave").all_in = False
        state.find("Dave").status = PlayerStatus.SITTING_OUT
        state.find("Bob").leaving = True
        text = render(state, ui(), W, H).text()
        self.assertIn("SITTING OUT", text)
        self.assertIn("LEAVING", text)

    def test_menu_and_panel(self):
        state = fixed_state()
        menu = [
            MenuItem("Fold", "fold"),
            MenuItem("Call 2", "call"),
            MenuItem("Raise to 4", "raise"),
            MenuItem("Leave table", "leave"),
        ]
        frame = render(state, ui(menu=menu, menu_title="Your turn", selected=1, countdown=24.0, error="oops"), W, H)
        text = frame.text()
        for label in ("[1] Fold", "[2] Call 2", "[3] Raise to 4", "[4] Leave table"):
            self.assertIn(label, text)
        self.assertEqual(frame.styles_of("[2] Call 2"), {"menu_selected"})
        self.assertEqual(frame.styles_of("[1] Fold"), {"menu"})
        self.assertIn("Time: 0:24", text)
        self.assertIn("oops", text)
        frame = render(state, ui(preset="Check/Fold"), W, H)
        self.assertIn("Preset: Check/Fold", frame.text())
        frame = render(state, ui(prompt="Fold instead of checking? [y/N]"), W, H)
        self.assertIn("Fold instead of checking? [y/N]", frame.text())

    def test_my_seat_is_bottom_centre_for_every_index(self):
        names = ("Alice", "Bob", "Carol", "Dave", "Erin", "Fay")
        for i in range(len(names)):
            order = names[i:] + names[:i]
            state = fixed_state(names=order)
            frame = render(state, ui(), W, H)
            x, y = box_of(frame, "You (Alice)")
            others = [box_of(frame, n)[1] for n in names if n != "Alice"]
            self.assertGreater(y, max(others), f"my seat is not the lowest (index {state.my_seat})")
            self.assertLessEqual(abs(x + SEAT_W / 2 - TABLE_CENTRE_X), 1)
            positions = seat_positions(len(order), state.my_seat, 0, 0, 99, 29)
            px, py = positions[state.my_seat]
            self.assertAlmostEqual(px, 50.0)
            self.assertAlmostEqual(py, 30 / 2 + (30 - SEAT_H) / 2)

    def test_acting_and_winner_highlight(self):
        state = fixed_state()
        state.acting = (3, 110.0)  # Dave, 10 s left
        frame = render(state, ui(now=100.0), W, H)
        dx, dy = box_of(frame, "Dave")
        self.assertEqual(frame.style_at(dx, dy), "seat_acting")
        self.assertIn("10s", frame.row_text(dy + 2)[dx : dx + SEAT_W])
        state.acting = None
        state.hand_running = False
        state.find("Bob").won = 3.5
        frame = render(state, ui(), W, H)
        bx, by = box_of(frame, "Bob")
        self.assertEqual(frame.style_at(bx, by), "winner")
        self.assertIn("+3.5", frame.text())
        cx, cy = box_of(frame, "Carol")
        self.assertEqual(frame.style_at(cx, cy), "folded")
        ax, ay = box_of(frame, "You (Alice)")
        self.assertEqual(frame.style_at(ax, ay), "seat_me")

    def test_lobby_text_and_ascii(self):
        state = TableState("Alice", None)
        state.seats = [SeatState("Alice", 100.0)]
        frame = render(state, ui(lobby_text="Waiting for players (1/2)", ascii=True), MIN_WIDTH, MIN_HEIGHT)
        text = frame.text()
        self.assertIn("Waiting for players (1/2)", text)
        self.assertIn("[  ][  ][  ][  ][  ]", text)
        self.assertNotIn("─", text)
        self.assertIn("1/10 seats", text)  # engine defaults without settings

    def test_too_small(self):
        frame = render(fixed_state(), ui(), 80, 24)
        self.assertIn("Please enlarge the terminal to at least 100×30", frame.text())
        self.assertIsNone(frame.find("Bob"))

    def test_log_scroll(self):
        state = fixed_state()
        for i in range(100):
            state.add_log(f"line {i}")
        text = render(state, ui(), W, H).text()
        self.assertIn("line 99", text)
        text = render(state, ui(log_scroll=50), W, H).text()
        self.assertNotIn("line 99", text)
        self.assertIn("line 49", text)


if __name__ == "__main__":
    unittest.main()
