"""RuleBot: preflop score, nut-low table, postflop class, decisions at known
positions (stacked decks)."""

import unittest

from arena import (
    NUT_LOW_PAIRS,
    RULE_PARAMS,
    RuleBot,
    decide,
    nut_low_pairs,
    postflop_class,
    preflop_score,
)
from rl_env import BET_RAISE, CHECK_CALL, FOLD, DecisionBatch

try:
    from . import helpers
except ImportError:  # run as a top-level module
    import helpers  # type: ignore[no-redef]

hand = helpers.hand
board = helpers.board
CC, BR = CHECK_CALL, BET_RAISE


def fs(*values):
    return frozenset(values)


def decision(holes, board_cards, actions):
    """(view, legal) at the decision after `actions` (3-handed, button seat 0:
    preflop order 0, 1, 2; postflop order 1, 2, 0)."""
    table = helpers.table_at_decision(holes, board_cards, actions, button=0)
    return table.view(), table.legal


class PreflopScoreTest(unittest.TestCase):
    def test_worked_examples(self):
        self.assertEqual(preflop_score(hand("As 2s Ah 3h")), 30 + 6 + 20 + 10 + 6)
        self.assertEqual(preflop_score(hand("9c 9d 9h Ks")), -20 - 18)
        self.assertGreaterEqual(preflop_score(hand("As 2s Ah 3h")), RULE_PARAMS.raise_at)
        self.assertLess(preflop_score(hand("9c 9d 9h Ks")), RULE_PARAMS.call_at)

    def test_low_pairs_order(self):
        a2 = preflop_score(hand("As 2d 7c 9h"))
        a3 = preflop_score(hand("As 3d 7c 9h"))
        t3 = preflop_score(hand("2s 3d 7c 9h"))
        self.assertEqual((a2, a3, t3), (33, 20, 18))
        self.assertGreater(a2, a3)
        self.assertGreater(a3, t3)

    def test_trips_and_quads(self):
        self.assertEqual(preflop_score(hand("Kc Kd Kh 5s")), -20)
        self.assertEqual(preflop_score(hand("Kc Kd Kh Ks")), -40)

    def test_middle_cards_only_without_ace_or_big_cards(self):
        self.assertEqual(preflop_score(hand("Tc 6d 5h 4s")), 6 - 3)  # connected 4-5-6, lone ten
        self.assertEqual(preflop_score(hand("Tc Jd 5h 4s")), 4 + 6)  # no penalty: J
        self.assertEqual(preflop_score(hand("Tc Ad 5h 4s")), 4 + 3)  # no penalty: ace

    def test_suits(self):
        # A-K-Q-J: big three 10, connected 3x3 = 9; suited ace (As Ks) 10, other (Qh Jh) 4
        self.assertEqual(preflop_score(hand("As Ks Qh Jh")), 10 + 9 + 10 + 4)
        self.assertEqual(preflop_score(hand("As Kd Qh Jc")), 10 + 9)


class NutLowTableTest(unittest.TestCase):
    def test_size(self):
        self.assertEqual(len(NUT_LOW_PAIRS), 56 + 70 + 56)

    def test_spot_checks(self):
        self.assertEqual(NUT_LOW_PAIRS[fs(2, 3, 7)], {fs(1, 4)})
        self.assertEqual(NUT_LOW_PAIRS[fs(1, 2, 3)], {fs(4, 5)})
        self.assertEqual(nut_low_pairs([2, 7]), frozenset())  # fewer than 3 low ranks
        # A-2 is counterfeited on an A-2-7 board; 3-4 is the nut
        self.assertNotIn(fs(1, 2), NUT_LOW_PAIRS[fs(1, 2, 7)])
        self.assertEqual(NUT_LOW_PAIRS[fs(1, 2, 7)], {fs(3, 4)})
        # 4 low board ranks: a hole card may pair a board card that is not used
        self.assertIn(fs(1, 5), NUT_LOW_PAIRS[fs(1, 2, 3, 4)])
        self.assertIn(fs(2, 5), NUT_LOW_PAIRS[fs(1, 2, 3, 4)])


class PostflopClassTest(unittest.TestCase):
    def test_nut_low(self):
        c = postflop_class(hand("As 4d Kc Qh"), board("2c 3d 7s"))
        self.assertEqual(c.high_cat, 0)
        self.assertTrue(c.has_low)
        self.assertTrue(c.nut_low)
        self.assertFalse(c.nut_low_draw)
        self.assertFalse(c.nut_flush_draw)
        c = postflop_class(hand("As 5d Kc Qh"), board("2c 3d 7s"))
        self.assertTrue(c.has_low)
        self.assertFalse(c.nut_low)

    def test_nut_low_draw(self):
        c = postflop_class(hand("As 3h 9c 9d"), board("2c 7d Ks"))  # a 4 gives 7-4-3-2-A
        self.assertFalse(c.has_low)
        self.assertTrue(c.nut_low_draw)
        c = postflop_class(hand("As 3h 9c 9d"), board("2c 7d Ks Qh"))
        self.assertTrue(c.nut_low_draw)
        c = postflop_class(hand("As 3h 9c 9d"), board("2c 7d Ks Qh Js"))  # river: no draws
        self.assertFalse(c.nut_low_draw)
        c = postflop_class(hand("8s 6h 9c 9d"), board("2c 7d Ks"))
        self.assertFalse(c.nut_low_draw)

    def test_nut_flush_draw(self):
        self.assertTrue(postflop_class(hand("Ah 5h 9d Tc"), board("Kh 7h 2c")).nut_flush_draw)
        self.assertFalse(postflop_class(hand("Qh 5h 9d Tc"), board("Kh 7h 2c")).nut_flush_draw)
        # ace on the board: the king is the nut card
        self.assertTrue(postflop_class(hand("Kh 5h 9d 9c"), board("Ah 7h 2c")).nut_flush_draw)
        # one hole card of the suit is not a draw (exactly two hole cards play)
        self.assertFalse(postflop_class(hand("Ah 5d 9d Tc"), board("Kh 7h 2c")).nut_flush_draw)
        # three of the suit on board: not "exactly two"
        self.assertFalse(postflop_class(hand("Ah 5h 9d Tc"), board("Kh 7h 2h")).nut_flush_draw)
        self.assertFalse(
            postflop_class(hand("Ah 5h 9d Tc"), board("Kh 7h 2c 3s 4s")).nut_flush_draw
        )

    def test_high_categories(self):
        self.assertEqual(postflop_class(hand("Kc Kd 9s 8h"), board("Ks 5d 2h")).high_cat, 3)
        self.assertEqual(postflop_class(hand("Kd Jd 6c 7c"), board("Kc 9d 4s 2h Jc")).high_cat, 2)


# Hole cards for the preflop positions (3-handed, button 0).
AA23 = "As 2s Ah 3h"
NINES = "9c 9d 9h Ks"
MEDIUM = "As 3d 8c Kh"  # 20 + 6 + 4 + 3 = 33
OTHER_A = "Kc Kd Qc Qd"
OTHER_B = "7c 7d 8h 8d"


class PreflopDecisionTest(unittest.TestCase):
    def test_raises_aa23_double_suited(self):
        view, legal = decision([AA23, OTHER_A, OTHER_B], "Js Ts 9s 4c 5c", [])
        self.assertEqual(view.raises, 1)
        self.assertEqual(decide(view, legal), BR)

    def test_folds_nines_to_a_raise(self):
        view, legal = decision([AA23, NINES, OTHER_B], "Js Ts 4h 4c 5c", [BR])
        self.assertFalse(legal.can_check)
        self.assertEqual(decide(view, legal), FOLD)

    def test_folds_nines_to_the_big_blind(self):
        view, legal = decision([NINES, OTHER_A, OTHER_B], "Js Ts 4h 4c 5c", [])
        self.assertEqual(decide(view, legal), FOLD)

    def test_checks_nines_when_free(self):
        view, legal = decision([AA23, OTHER_A, NINES], "Js Ts 4h 4c 5c", [CC, CC])
        self.assertTrue(legal.can_check)
        self.assertEqual(decide(view, legal), CC)

    def test_medium_calls_one_bet_folds_to_a_raise(self):
        self.assertEqual(preflop_score(hand(MEDIUM)), 33)
        view, legal = decision([MEDIUM, OTHER_A, OTHER_B], "Js Ts 4h 4c 5c", [])
        self.assertEqual(decide(view, legal), CC)
        view, legal = decision([OTHER_A, MEDIUM, OTHER_B], "Js Ts 4h 4c 5c", [BR])
        self.assertEqual(view.raises, 2)
        self.assertEqual(decide(view, legal), FOLD)


PREFLOP_LIMPED = [CC, CC, CC]
CHECKED_STREET = [CC, CC, CC]
TO_RIVER = PREFLOP_LIMPED + CHECKED_STREET + CHECKED_STREET


class PostflopDecisionTest(unittest.TestCase):
    def test_bets_the_nut_low_on_the_flop(self):
        holes = ["Th Tc 6s 6d", "As 4d Kc Qh", "8h 8s 5c 5h"]
        view, legal = decision(holes, "2c 3d 7s Jd 9h", PREFLOP_LIMPED)
        self.assertEqual(view.seat, 1)
        self.assertEqual(view.raises, 0)
        self.assertEqual(decide(view, legal), BR)

    def test_bets_a_set_on_the_flop(self):
        holes = ["Th Tc 6s 6d", "Kc Kd 9s 8h", "7h 7s 4c 4h"]
        view, legal = decision(holes, "Ks 5d 2h Jc Qd", PREFLOP_LIMPED)
        self.assertEqual(decide(view, legal), BR)

    def test_calls_a_nut_flush_draw_folds_without_it(self):
        holes = ["6c 6d 3c 3d", "Kd Ks 8c 8d", "Ah 5h 9d Tc"]
        view, legal = decision(holes, "Kh 7h 2c Js Qs", PREFLOP_LIMPED + [BR])
        self.assertEqual(view.seat, 2)
        self.assertFalse(legal.can_check)
        self.assertEqual(decide(view, legal), CC)
        holes[2] = "Qh 5h 9d Tc"
        view, legal = decision(holes, "Kh 7h 2c Js Ad", PREFLOP_LIMPED + [BR])
        self.assertEqual(decide(view, legal), FOLD)

    def test_raises_a_scoop_on_the_river(self):
        holes = ["Tc Td 6s 6h", "8c 8d 7s 7h", "As 5d Qh Jh"]
        board_cards = "2c 3d 4h Ks 9c"
        view, legal = decision(holes, board_cards, TO_RIVER + [BR])
        self.assertEqual(view.seat, 2)
        self.assertEqual(decide(view, legal), BR)
        # facing three bets: the scoop still raises ...
        view, legal = decision(holes, board_cards, TO_RIVER + [BR, BR, BR, CC])
        self.assertEqual((view.seat, view.raises), (2, 3))
        self.assertTrue(legal.can_raise)
        self.assertEqual(decide(view, legal), BR)
        # ... a one-sided strong hand only calls
        holes[2] = "Kc Kd Qh Jh"
        view, legal = decision(holes, board_cards, TO_RIVER + [BR, BR, BR, CC])
        self.assertEqual(decide(view, legal), CC)

    def test_river_two_pair_calls_one_bet_folds_to_a_raise(self):
        board_cards = "Kc 9d 4s 2h Jc"
        holes = ["Th Ts 8h 8s", "Qs Qh 3c 3h", "Kd Jd 6c 7c"]
        view, legal = decision(holes, board_cards, TO_RIVER + [BR])
        self.assertEqual((view.seat, view.raises), (2, 1))
        self.assertEqual(decide(view, legal), CC)
        holes = ["Kd Jd 6c 7c", "Qs Qh 3c 3h", "Th Ts 8h 8s"]
        view, legal = decision(holes, board_cards, TO_RIVER + [BR, BR])
        self.assertEqual((view.seat, view.raises), (0, 2))
        self.assertEqual(decide(view, legal), FOLD)

    def test_river_air_folds_to_a_bet_checks_when_free(self):
        board_cards = "Kc 9d 4s 2h Jc"
        holes = ["8h 8s 7d 7h", "Qs Qh 3c 3h", "6d 6h Ts Th"]
        view, legal = decision(holes, board_cards, TO_RIVER + [BR])
        self.assertEqual(decide(view, legal), FOLD)
        view, legal = decision(holes, board_cards, TO_RIVER)
        self.assertEqual(view.seat, 1)
        self.assertTrue(legal.can_check)
        self.assertEqual(decide(view, legal), CC)

    def test_rulebot_act_matches_decide(self):
        holes = ["Th Tc 6s 6d", "As 4d Kc Qh", "8h 8s 5c 5h"]
        view, legal = decision(holes, "2c 3d 7s Jd 9h", PREFLOP_LIMPED)
        out = RuleBot().act(DecisionBatch.single(view, legal))
        self.assertEqual(out.tolist(), [decide(view, legal)])

    def test_rulebot_needs_views(self):
        holes = ["Th Tc 6s 6d", "As 4d Kc Qh", "8h 8s 5c 5h"]
        view, legal = decision(holes, "2c 3d 7s Jd 9h", PREFLOP_LIMPED)
        batch = DecisionBatch.single(view, legal)
        batch.views = None
        with self.assertRaises(ValueError):
            RuleBot().act(batch)


if __name__ == "__main__":
    unittest.main()
