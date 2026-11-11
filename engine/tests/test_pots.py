import unittest
from dataclasses import dataclass

from engine.pots import Pot, build_pots, split_pot


@dataclass
class Seat:
    seat: int
    total_chips: int
    folded: bool = False


# Rank stand-ins: high tuples compare higher-is-better, lows lower-is-better.
STRONG = (8, 9)
MEDIUM = (5, 10)
WEAK = (1, 13, 11, 9, 5, 2)
NUT_LOW = (5, 4, 3, 2, 1)
OK_LOW = (8, 7, 4, 3, 2)


class BuildPotsTest(unittest.TestCase):
    def test_single_pot(self):
        pots = build_pots([Seat(0, 200), Seat(1, 200), Seat(2, 200)])
        self.assertEqual(pots, [Pot(600, (0, 1, 2))])

    def test_side_pots_multiple_all_ins_and_folded(self):
        players = [Seat(0, 100), Seat(1, 300), Seat(2, 500), Seat(3, 200, folded=True)]
        pots = build_pots(players)
        self.assertEqual(
            pots,
            [Pot(400, (0, 1, 2)), Pot(500, (1, 2)), Pot(200, (2,))],
        )
        self.assertEqual(sum(p.amount for p in pots), 1100)
        self.assertTrue(pots[-1].uncontested)  # uncalled bet of seat 2

    def test_three_all_in_levels(self):
        pots = build_pots([Seat(0, 50), Seat(1, 150), Seat(2, 400), Seat(3, 400)])
        self.assertEqual(
            pots, [Pot(200, (0, 1, 2, 3)), Pot(300, (1, 2, 3)), Pot(500, (2, 3))]
        )

    def test_folded_contribution_stays_in_pot(self):
        # Blinds folded to the big blind: everything is uncontested for seat 2.
        pots = build_pots([Seat(0, 0, folded=True), Seat(1, 50, folded=True), Seat(2, 100)])
        self.assertEqual(pots, [Pot(150, (2,))])

    def test_folded_player_above_live_players(self):
        pots = build_pots([Seat(0, 300, folded=True), Seat(1, 200), Seat(2, 200)])
        self.assertEqual(pots, [Pot(700, (1, 2))])


class SplitPotTest(unittest.TestCase):
    def test_hi_lo_split(self):
        pot = Pot(400, (0, 1, 2))
        hands = {0: (STRONG, None), 1: (MEDIUM, OK_LOW), 2: (WEAK, NUT_LOW)}
        split = split_pot(pot, hands, [1, 2, 0])
        self.assertEqual(split.high, {0: 200})
        self.assertEqual(split.low, {2: 200})

    def test_no_low_high_takes_all(self):
        pot = Pot(400, (0, 1))
        split = split_pot(pot, {0: (STRONG, None), 1: (MEDIUM, None)}, [1, 0])
        self.assertEqual(split.high, {0: 400})
        self.assertEqual(split.low, {})

    def test_scoop(self):
        pot = Pot(400, (0, 1))
        split = split_pot(pot, {0: (STRONG, NUT_LOW), 1: (MEDIUM, OK_LOW)}, [1, 0])
        self.assertEqual(split.high, {0: 200})
        self.assertEqual(split.low, {0: 200})

    def test_quartering(self):
        pot = Pot(400, (0, 1, 2))
        hands = {0: (STRONG, NUT_LOW), 1: (MEDIUM, NUT_LOW), 2: (WEAK, None)}
        split = split_pot(pot, hands, [1, 2, 0])
        self.assertEqual(split.high, {0: 200})
        self.assertEqual(split.low, {1: 100, 0: 100})
        total = {s: split.high.get(s, 0) + split.low.get(s, 0) for s in (0, 1, 2)}
        self.assertEqual(total, {0: 300, 1: 100, 2: 0})

    def test_odd_chip_goes_to_high_half(self):
        pot = Pot(401, (0, 1))
        split = split_pot(pot, {0: (STRONG, None), 1: (MEDIUM, NUT_LOW)}, [1, 0])
        self.assertEqual(split.high, {0: 201})
        self.assertEqual(split.low, {1: 200})

    def test_odd_chip_in_tie_goes_left_of_button(self):
        # Button is seat 0: order starts at seat 1.
        pot = Pot(401, (0, 1, 2))
        hands = {0: (STRONG, None), 1: (WEAK, NUT_LOW), 2: (STRONG, None)}
        split = split_pot(pot, hands, [1, 2, 0])
        self.assertEqual(split.high, {2: 101, 0: 100})
        self.assertEqual(split.low, {1: 200})

    def test_ignores_ineligible_hands(self):
        pot = Pot(200, (1, 2))
        hands = {0: (STRONG, NUT_LOW), 1: (MEDIUM, None), 2: (WEAK, None)}
        split = split_pot(pot, hands, [1, 2, 0])
        self.assertEqual(split.high, {1: 200})
        self.assertEqual(split.low, {})


if __name__ == "__main__":
    unittest.main()
