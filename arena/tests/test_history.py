"""Hand histories: a showdown with a side pot and a split low, and a fold-out
hand with a returned bet."""

import unittest

from arena import format_hand
from rl_env import BET_RAISE, CHECK_CALL, FOLD, HandSpec, Table

try:
    from . import helpers
except ImportError:  # run as a top-level module
    import helpers  # type: ignore[no-redef]

CC, BR = CHECK_CALL, BET_RAISE
NAMES = ("alice", "bob", "carol")


def play(holes, board_cards, stacks, actions):
    deck = helpers.StackedDeck(helpers.deal_script(holes, board_cards, 0))
    table = Table()
    table.start_hand(3, stacks, 0, 0, deck=deck)
    for a in actions:
        table.act(a)
    assert table.hand_over
    spec = HandSpec(3, tuple(stacks), 0, 0)
    return format_hand(7, 2, spec, NAMES, table.finished_hole_cards(), table.result)


class HistoryTest(unittest.TestCase):
    def test_showdown_side_pot_split_low(self):
        # bob (SB) has 3 BB and is all-in on the flop; alice and carol split the low.
        holes = ["As 4d Jc Jd", "Ks Kd 9c 9d", "Ah 4c Tc Th"]
        actions = [
            BR, CC, CC,  # preflop: alice raises to 2, bob and carol call
            BR, CC, CC,  # flop: bob bets 1 all-in, carol and alice call
            BR, CC,  # turn: carol bets 2, alice calls
            CC, CC,  # river: checks
        ]
        text = play(holes, "2c 3d 7s Kh Qd", (100.0, 3.0, 100.0), actions)
        lines = text.splitlines()
        self.assertEqual(
            lines[0], "Deal 7, rotation 2 - 3 players, button seat 0 (amounts in BB)"
        )
        self.assertIn("alice", lines[1])
        self.assertIn("100.00", lines[1])
        self.assertIn("As 4d Jc Jd", lines[1])
        self.assertIn("bob", lines[2])
        self.assertIn("3.00", lines[2])
        self.assertIn("Ks Kd 9c 9d", lines[2])
        self.assertIn("Ah 4c Tc Th", lines[3])
        self.assertIn("Blinds: seat 1 bob SB 0.50; seat 2 carol BB 1.00", text)
        self.assertIn(
            "Preflop: seat 0 alice raises to 2.00; seat 1 bob calls 1.50; seat 2 carol calls 1.00",
            text,
        )
        self.assertIn("Flop 2c 3d 7s: seat 1 bob raises to 1.00 all-in;", text)
        self.assertIn("Turn Kh (2c 3d 7s Kh): seat 2 carol raises to 2.00; seat 0 alice calls 2.00", text)
        self.assertIn("River Qd (2c 3d 7s Kh Qd): seat 2 carol checks; seat 0 alice checks", text)
        self.assertIn("Shows: seat 1 bob Ks Kd 9c 9d; seat 2 carol Ah 4c Tc Th; seat 0 alice", text)
        self.assertIn("Pot 1 9.00: high seat 1 bob 4.50;", text)
        self.assertIn("low seat 0 alice 2.25", text)
        self.assertIn("low seat 2 carol 2.25", text)
        self.assertIn("Pot 2 4.00: high seat 0 alice 2.00;", text)
        self.assertIn("seat 1 bob +1.50 (+0.75 big bets)", text)
        self.assertIn("seat 0 alice +0.25", text)
        self.assertIn("seat 2 carol -1.75", text)
        self.assertNotIn("s0", text.replace("seat 0", ""))

    def test_fold_out_is_uncontested_and_returns_the_bet(self):
        holes = ["As 4d Jc Jd", "Ks Kd 9c 9d", "Ah 4c Tc Th"]
        text = play(holes, "2c 3d 7s Kh Qd", (100.0, 100.0, 100.0), [BR, FOLD, FOLD])
        self.assertIn("Preflop: seat 0 alice raises to 2.00; seat 1 bob folds; seat 2 carol folds", text)
        self.assertIn("Returned: seat 0 alice 1.00", text)
        self.assertIn("Pot 1 2.50: seat 0 alice wins uncontested 2.50", text)
        self.assertNotIn("Flop", text)
        self.assertNotIn("Shows:", text)
        self.assertIn("seat 0 alice +1.50 (+0.75 big bets)", text)
        self.assertIn("seat 1 bob -0.50", text)
        self.assertIn("seat 2 carol -1.00", text)
        # the evaluator shows every seat's cards, folded ones included
        self.assertIn("Ks Kd 9c 9d", text)


if __name__ == "__main__":
    unittest.main()
