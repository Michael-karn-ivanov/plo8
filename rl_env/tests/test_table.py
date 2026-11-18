"""Table: start / act / end of a hand, validation, rejected actions, rewards."""

import random
import unittest

import numpy as np

from engine import ActionType
from rl_env import (
    BET_RAISE,
    CHECK_CALL,
    FOLD,
    IllegalMaskedActionError,
    NoHandError,
    Table,
    action_mask,
    to_engine,
)

try:
    from . import helpers
except ImportError:  # run as a top-level module
    import helpers  # type: ignore[no-redef]


def state(table):
    env = table.env
    return (
        env.decision,
        len(env.history),
        tuple(p.chips for p in env.players),
        tuple(p.total_chips for p in env.players),
    )


class HandTest(unittest.TestCase):
    def test_hands_at_every_size(self):
        rng = random.Random(1)
        for n in range(2, 11):
            for _ in range(30):
                with self.subTest(n=n):
                    table = Table()
                    spec = helpers.random_spec(rng, n)

                    def check(t):
                        self.assertFalse(t.hand_over)
                        self.assertTrue(0 <= t.seat_to_act < n)
                        self.assertEqual(t.mask, action_mask(t.legal))
                        self.assertEqual(t.mask_array().tolist(), list(t.mask))

                    helpers.play_random(table, spec, rng, check)
                    self.assertTrue(table.hand_over)
                    self.assertFalse(table.decision_pending)
                    self.assertIsNotNone(table.result)
                    self.assertEqual(table.net_bb.dtype, np.float64)
                    self.assertEqual(table.rewards.dtype, np.float32)
                    self.assertEqual(table.net_bb.shape, (n,))
                    for seat in range(n):
                        self.assertEqual(table.net_bb[seat], table.result.net[seat])
                    np.testing.assert_array_equal(
                        table.rewards, (table.net_bb / 2).astype(np.float32)
                    )
                    self.assertEqual(sum(int(round(x * 100)) for x in table.net_bb), 0)
                    with self.assertRaises(NoHandError):
                        table.act(CHECK_CALL)

    def test_first_decision(self):
        table = Table()
        self.assertTrue(table.start_hand(3, [100.0] * 3, 0, 7))
        # SB seat 1, BB seat 2; the button (seat 0) acts first preflop.
        self.assertEqual(table.seat_to_act, 0)
        self.assertEqual(table.mask, (True, True, True))
        self.assertEqual(table.legal.call_amount, 1.0)
        self.assertEqual(len(table.hole_cards()), 4)
        self.assertEqual(table.view().seat, 0)

    def test_no_decision_hand_is_over_at_once(self):
        # Heads-up: the button posts 0.5 (all-in), the BB 1.0 (all-in).
        for stacks, button in (([0.5, 1.0], 0), ([0.75, 0.5], 1)):
            with self.subTest(stacks=stacks):
                table = Table()
                self.assertFalse(table.start_hand(2, stacks, button, 3))
                self.assertTrue(table.hand_over)
                self.assertFalse(table.decision_pending)
                self.assertEqual(len(table.result.board), 5)
                self.assertTrue(table.result.showdown)
                self.assertEqual(sum(int(round(x * 100)) for x in table.net_bb), 0)
                np.testing.assert_array_equal(
                    table.rewards, (table.net_bb / 2).astype(np.float32)
                )

    def test_validation(self):
        table = Table()
        bad = [
            (1, [100.0], 0, 1),
            (11, [100.0] * 11, 0, 1),
            (3, [100.0] * 2, 0, 1),
            (2, [100.0, 0.0], 0, 1),
            (2, [100.0, -5.0], 0, 1),
            (2, [100.0, 0.005], 0, 1),  # half a chip at 100 chips/BB
            (2, [100.0, 1.234], 0, 1),
            (2, [100.0, float("nan")], 0, 1),
            (2, [100.0, "100"], 0, 1),
            (2, [100.0, 100.0], 2, 1),
            (2, [100.0, 100.0], -1, 1),
            (2, [100.0, 100.0], 0, 1.5),
            (2, [100.0, 100.0], 0, "7"),
            (2, [100.0, 100.0], 0, True),
            (True, [100.0, 100.0], 0, 1),
        ]
        for args in bad:
            with self.subTest(args=args), self.assertRaises(ValueError):
                table.start_hand(*args)
        for cpb in (0, 1, 3, 101, 2.0, True):
            with self.subTest(cpb=cpb), self.assertRaises(ValueError):
                Table(cpb)
        # Exact at other chip units, numpy numbers accepted.
        self.assertTrue(Table(1000).start_hand(2, [1.234, 2.0], 0, 1))
        self.assertTrue(
            Table().start_hand(2, [np.float32(50.0), np.int64(3)], np.int64(1), np.int64(9))
        )

    def test_allowed_actions_accepted_masked_rejected_state_unchanged(self):
        rng = random.Random(2)
        seen_fold_masked = seen_raise_masked = 0
        for _ in range(400):
            table = Table()
            spec = helpers.random_spec(rng)
            table.start_hand(spec.n_seats, spec.stacks_bb, spec.button, spec.deck_seed)
            while table.decision_pending:
                mask = table.mask
                for a in range(3):
                    if not mask[a]:
                        before = state(table)
                        with self.assertRaises(IllegalMaskedActionError):
                            table.act(a)
                        self.assertEqual(state(table), before)
                        if a == FOLD:
                            seen_fold_masked += 1
                        else:
                            seen_raise_masked += 1
                    else:
                        self.assertTrue(table.legal.allows(to_engine(a, table.legal)))
                for bad in (3, -1, None):
                    before = state(table)
                    with self.assertRaises(IllegalMaskedActionError):
                        table.act(bad)
                    self.assertEqual(state(table), before)
                table.act(helpers.random_action(rng, mask))  # accepted by the engine
        self.assertGreater(seen_fold_masked, 0)
        self.assertGreater(seen_raise_masked, 0)

    def test_act_engine_fold_when_checking_is_free(self):
        table = Table()
        table.start_hand(2, [100.0, 100.0], 0, 5)
        table.act(CHECK_CALL)  # the button / SB calls; the BB may check
        self.assertTrue(table.legal.can_check)
        with self.assertRaises(IllegalMaskedActionError):
            table.act(FOLD)
        table.act_engine(ActionType.FOLD)
        self.assertTrue(table.hand_over)
        self.assertEqual(table.net_bb.tolist(), [1.0, -1.0])
        self.assertEqual(table.rewards.tolist(), [0.5, -0.5])

    def test_rewards_in_big_bets(self):
        table = Table()
        table.start_hand(2, [100.0, 100.0], 1, 11)
        self.assertEqual(table.seat_to_act, 1)  # heads-up the button / SB acts first
        table.act(BET_RAISE)  # raises to 2
        self.assertTrue(table.mask[FOLD])
        table.act(FOLD)
        self.assertTrue(table.hand_over)
        self.assertEqual(table.net_bb.tolist(), [-1.0, 1.0])
        self.assertEqual(table.rewards.tolist(), [-0.5, 0.5])

    def test_same_seed_different_stacks_same_cards_relative_to_the_button(self):
        rng = random.Random(3)
        for _ in range(40):
            n = rng.randint(2, 10)
            seed = rng.getrandbits(63)
            b1, b2 = rng.randrange(n), rng.randrange(n)
            t1, t2 = Table(), Table()
            t1.start_hand(n, [100.0] * n, b1, seed)
            t2.start_hand(n, helpers.random_stacks(rng, n), b2, seed)
            c1, c2 = t1.privileged_hole_cards(), t2.privileged_hole_cards()
            for i in range(n):
                self.assertEqual(c1[helpers.seat_of(n, b1, i)], c2[helpers.seat_of(n, b2, i)])
            helpers.passive(t1)
            helpers.passive(t2)
            self.assertEqual(t1.result.board, t2.result.board)
            self.assertEqual(len(t1.result.board), 5)


if __name__ == "__main__":
    unittest.main()
