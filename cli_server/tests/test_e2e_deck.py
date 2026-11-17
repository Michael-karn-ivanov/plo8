import random
import unittest

from engine import Environment, Player
from entities import Card

from cli_server.e2e.case import parse_case
from cli_server.e2e.deck import StackedDeck, deal_script, engine_seats, join_offset, seating

try:
    from cli_server_tests.tests.e2e_helpers import fold_case_data, sample_data
except ImportError:  # run directly from the tests directory
    from e2e_helpers import fold_case_data, sample_data


def cards(*names):
    return [Card.parse(n) for n in names]


class CallingAgent:
    """Checks if possible, else calls; records its hole cards."""

    def __init__(self):
        self.holes = []

    def on_turn(self, player, view, legal):
        if legal.can_check:
            player.check()
        else:
            player.call()

    def on_hand_end(self, player, result):
        pass

    def on_hole_cards(self, player, hand, view):
        self.holes.append([str(c) for c in hand])


class SeatingTest(unittest.TestCase):
    def test_heads_up_button_gets_engine_seat_0(self):
        case = parse_case(sample_data())
        self.assertEqual([p.id for p in seating(case)], ["SB", "BB"])
        self.assertEqual(engine_seats(case), {"SB": 0, "BB": 1})

    def test_seats_with_gaps_and_a_button_that_is_not_the_lowest_seat(self):
        case = parse_case(fold_case_data())  # seats A=1, B=4 (button), C=6
        self.assertEqual([p.id for p in seating(case)], ["B", "C", "A"])
        self.assertEqual(engine_seats(case), {"B": 0, "C": 1, "A": 2})

    def test_button_on_the_lowest_seat(self):
        data = fold_case_data()
        data["button"] = "A"
        data["streets"][0]["actions"][0]["player"] = "B"  # SB: first after the button
        data["streets"][0]["actions"][1]["player"] = "C"  # BB
        data["streets"][0]["actions"][2]["player"] = "A"
        data["streets"][0]["actions"][3]["player"] = "B"
        data["pots"][0].update(eligible=["C"], hi={"winners": ["C"], "amount": 2}, awards={"C": 2})
        data["result"] = {
            "ended_by": "fold",
            "net": {"A": 0, "B": -1, "C": 1},
            "final_stacks": {"A": 40, "B": 49, "C": 61},
        }
        case = parse_case(data)
        self.assertEqual(engine_seats(case), {"A": 0, "B": 1, "C": 2})

    def test_join_offsets(self):
        self.assertEqual([join_offset(k) for k in range(5)], [None, None, 2, 3, 4])

    def test_joins_land_on_the_planned_engine_seats(self):
        case = parse_case(fold_case_data())
        env = Environment(chips_per_bb=2)
        for k, p in enumerate(seating(case)):
            env.join(p.stack / 2, seat=join_offset(k), name=p.id)
        self.assertEqual([p.name for p in env.players], ["B", "C", "A"])
        self.assertEqual(env.button, 0)
        self.assertEqual([p.chips for p in env.players], [50, 60, 40])


class ScriptTest(unittest.TestCase):
    def test_heads_up_script(self):
        case = parse_case(sample_data())
        # Left of the button first (BB), the button (SB) last; then the board.
        self.assertEqual(
            deal_script(case),
            cards(
                "7c", "Ah", "6h", "Ad", "3s", "Kc", "4d", "Qc",  # holes, round robin
                "9c", "8c", "2d", "5s", "Jh",  # flop, turn, river
            ),
        )

    def test_unknown_cards_are_none(self):
        case = parse_case(fold_case_data())  # engine seats B, C, A; B unknown; no board
        script = deal_script(case)
        self.assertEqual(len(script), 4 * 3 + 5)
        c, a = cards("Kd", "Qd", "Jd", "Td"), cards("As", "2s", "3s", "4s")
        for r in range(4):
            self.assertEqual(script[3 * r : 3 * r + 3], [c[r], a[r], None])
        self.assertEqual(script[12:], [None] * 5)


class StackedDeckTest(unittest.TestCase):
    def test_top_of_the_deck_is_the_script_with_filler(self):
        script = cards("Ah", "Kd") + [None, None] + cards("2c") + [None]
        deck = StackedDeck(script, random.Random(3))
        deck.reset()
        deck.shuffle()
        top = deck.remaining()[: len(script)]
        self.assertEqual(len(deck), 52)
        for want, got in zip(script, top):
            if want is not None:
                self.assertEqual(got, want)
        filler = [top[i] for i, c in enumerate(script) if c is None]
        self.assertEqual(len(set(filler)), 3)
        self.assertFalse(set(filler) & {c for c in script if c is not None})
        self.assertEqual(len(set(deck.remaining())), 52)

        again = StackedDeck(script, random.Random(3))
        again.reset()
        again.shuffle()
        self.assertEqual(again.remaining()[: len(script)], top)  # same seed, same filler

    def test_invalid_scripts(self):
        with self.assertRaises(ValueError):
            StackedDeck(cards("Ah", "Ah"))
        with self.assertRaises(ValueError):
            StackedDeck([None] * 53)
        with self.assertRaises(TypeError):
            StackedDeck(["Ah"])

    def test_dealer_deals_the_case_cards_every_hand(self):
        case = parse_case(sample_data())
        deck = StackedDeck(deal_script(case), random.Random(0))
        agents = {"SB": CallingAgent(), "BB": CallingAgent()}
        env = Environment(
            [Player("SB", 50.0, agents["SB"]), Player("BB", 50.0, agents["BB"])],
            button=0,
            chips_per_bb=2,
            deck=deck,
        )
        result = env.start_round()
        self.assertEqual(agents["SB"].holes, [["Ah", "Ad", "Kc", "Qc"]])
        self.assertEqual(agents["BB"].holes, [["7c", "6h", "3s", "4d"]])
        self.assertEqual([str(c) for c in result.board.cards], ["9c", "8c", "2d", "5s", "Jh"])
        self.assertEqual(deck.shuffles, 1)

        # The next hand reshuffles and restacks: the same cards, dealt from the
        # left of the new button (seat 1 now).
        result = env.start_round()
        self.assertEqual(deck.shuffles, 2)
        self.assertEqual(agents["SB"].holes[1], ["7c", "6h", "3s", "4d"])
        self.assertEqual(agents["BB"].holes[1], ["Ah", "Ad", "Kc", "Qc"])
        self.assertEqual([str(c) for c in result.board.cards], ["9c", "8c", "2d", "5s", "Jh"])


if __name__ == "__main__":
    unittest.main()
