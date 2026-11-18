"""Critic input: relative slot order, ascending indices, NO_CARD, never the
acting player's cards; expand_critic."""

import random
import unittest

import numpy as np

from rl_env import NO_CARD, PRIV_DIM, Table, critic_cards, expand_critic

try:
    from . import helpers
except ImportError:  # run as a top-level module
    import helpers  # type: ignore[no-redef]


class CriticCardsTest(unittest.TestCase):
    def test_at_every_decision(self):
        rng = random.Random(60)
        decisions = 0
        for k in range(600):
            n = 2 + k % 9
            spec = helpers.random_spec(rng, n)

            def check(t):
                nonlocal decisions
                decisions += 1
                seat = t.seat_to_act
                cc = critic_cards(t, seat)
                self.assertEqual(cc.shape, (9, 4))
                self.assertEqual(cc.dtype, np.int8)
                cards = t.privileged_hole_cards()
                own = {c.index for c in t.hole_cards()}
                obs = t.observation()
                for k_slot in range(1, 10):
                    row = cc[k_slot - 1].tolist()
                    if k_slot < n:
                        other = (seat + k_slot) % n
                        self.assertEqual(row, sorted(c.index for c in cards[other]))
                        self.assertEqual(row, sorted(row))
                        self.assertFalse(own & set(row))
                        # Same slot order as the observation: slot k's button flag.
                        flag = obs[254 + 15 * (k_slot - 1) + 14]
                        self.assertEqual(flag == 1.0, other == spec.button)
                    else:
                        self.assertEqual(row, [NO_CARD] * 4)
                # Into a caller's buffer.
                buf = np.zeros((9, 4), dtype=np.int8)
                critic_cards(t, seat, buf)
                np.testing.assert_array_equal(buf, cc)

            helpers.play_random(Table(), spec, rng, check)
        self.assertGreater(decisions, 1000)

    def test_seat_not_dealt_in(self):
        table = Table()
        table.start_hand(2, [100.0, 100.0], 0, 1)
        with self.assertRaises(ValueError):
            critic_cards(table, 5)


class ExpandTest(unittest.TestCase):
    def test_expand(self):
        rng = random.Random(61)
        batch = np.full((5, 9, 4), NO_CARD, dtype=np.int8)
        for b in range(5):
            dealt = b + 1  # slots with cards
            deck = list(range(52))
            rng.shuffle(deck)
            for s in range(dealt):
                batch[b, s] = sorted(deck[4 * s : 4 * s + 4])
        out = expand_critic(batch)
        self.assertEqual(out.shape, (5, PRIV_DIM))
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(PRIV_DIM, 468)
        for b in range(5):
            for s in range(9):
                block = out[b, 52 * s : 52 * s + 52]
                if s < b + 1:
                    self.assertEqual(int(block.sum()), 4)
                    for idx in batch[b, s]:
                        self.assertEqual(block[int(idx)], 1.0)
                else:
                    self.assertEqual(int(block.sum()), 0)
        self.assertEqual(set(np.unique(out).tolist()) <= {0.0, 1.0}, True)
        np.testing.assert_array_equal(expand_critic(batch[0]), out[:1])
        with self.assertRaises(ValueError):
            expand_critic(np.zeros((2, 8, 4), dtype=np.int8))
        with self.assertRaises(ValueError):
            expand_critic(np.full((1, 9, 4), 53, dtype=np.int8))


if __name__ == "__main__":
    unittest.main()
