"""Observation: layout, hand-written situations, shape / range, and the
training path equal to `encode_view(view, legal)` byte for byte."""

import random
import unittest

import numpy as np

from entities import Card
from rl_env import (
    BET_RAISE,
    CHECK_CALL,
    ENCODER_VERSION,
    OBS_DIM,
    OBS_LAYOUT,
    Table,
    encode_view,
)
from rl_env.encoder import OBS_BLOCKS

try:
    from . import helpers
except ImportError:  # run as a top-level module
    import helpers  # type: ignore[no-redef]


def cards(text):
    return [Card.parse(text[i : i + 2]) for i in range(0, len(text), 2)]


# Three players, button seat 0, SB seat 1, BB seat 2.
SEAT0 = cards("KcKh7d6d")
SEAT1 = cards("As2sKdQd")
SEAT2 = cards("TsThJcJd")
FLOP = cards("3c4h8c")
TURN_RIVER = cards("9h5s")


def three_handed_table():
    # Participants from the left of the button: seat 1, seat 2, seat 0.
    deck = helpers.StackedDeck(helpers.deal_script([SEAT1, SEAT2, SEAT0], FLOP + TURN_RIVER))
    table = Table()
    assert table.start_hand(3, [100.0, 100.0, 100.0], 0, 0, deck=deck)
    return table


def pot(x_bb):
    return x_bb * (0.5 / 25.0)


def street(x_bb):
    return x_bb * (0.5 / 2.0)


def stack(x_bb):
    return x_bb * (0.5 / 50.0)


def vector(values):
    out = np.zeros(OBS_DIM, dtype=np.float32)
    for col, v in values.items():
        out[col] = v
    return out


class LayoutTest(unittest.TestCase):
    def test_blocks_are_contiguous(self):
        self.assertEqual(ENCODER_VERSION, 1)
        self.assertEqual(OBS_DIM, 554)
        offset = 0
        for name, off, size in OBS_LAYOUT:
            self.assertEqual(off, offset, name)
            self.assertGreater(size, 0)
            offset += size
        self.assertEqual(offset, OBS_DIM)
        self.assertEqual(len({name for name, _, _ in OBS_LAYOUT}), len(OBS_LAYOUT))
        self.assertEqual(OBS_BLOCKS["other_seats"], (254, 135))
        self.assertEqual(OBS_BLOCKS["recent_actions"], (389, 144))
        self.assertEqual(OBS_BLOCKS["hints"], (533, 21))


class HandWrittenTest(unittest.TestCase):
    def test_first_decision_preflop(self):
        table = three_handed_table()
        self.assertEqual(table.seat_to_act, 0)
        obs = table.observation()
        want = {c.index: 1.0 for c in SEAT0}
        want.update(
            {
                208: 1.0,  # preflop
                212 + 1: 1.0,  # 3 players dealt in
                221 + 0: 1.0,  # the button
                231: pot(1.5),
                232: street(1.0),
                233: 1.0 / (1.5 + 1.0),
                234 + 1: 1.0,  # one bet (the big blind)
                241: 0.3,
                242: 0.3,
                243: stack(100.0),
                244: 1.0,  # can raise
                247: 3 / 4,  # raises left
                # slot 1 = seat 1 (SB)
                254: 1.0,
                255: 1.0,
                257: pot(0.5),
                258: street(0.5),
                265: 1.0,  # acts after the button preflop
                267: stack(99.5),
                # slot 2 = seat 2 (BB)
                269: 1.0,
                270: 1.0,
                272: pot(1.0),
                273: street(1.0),
                280: 1.0,
                282: stack(99.0),
                # hints: two distinct low ranks (7, 6) in the hole cards
                533 + 15 + 2: 1.0,
            }
        )
        np.testing.assert_array_equal(obs, vector(want))

    def test_second_decision_after_a_raise(self):
        table = three_handed_table()
        table.act(BET_RAISE)  # seat 0 raises to 2
        self.assertEqual(table.seat_to_act, 1)
        obs = table.observation()
        self.assertEqual(obs[222], 1.0)  # position 1: SB
        self.assertAlmostEqual(float(obs[231]), pot(3.5), places=6)
        self.assertAlmostEqual(float(obs[232]), street(1.5), places=6)
        self.assertEqual(obs[236], 1.0)  # two bets this street
        self.assertEqual(obs[553], 1.0)  # A-2 in the hole
        # slot 2 = seat 0 (the raiser, the button)
        self.assertEqual(obs[269 + 5], 0.25)  # one raise preflop
        self.assertEqual(obs[269 + 9], 1.0)  # last aggressor on this street
        self.assertEqual(obs[269 + 11], 0.0)  # acted before the SB preflop
        self.assertEqual(obs[269 + 12], 1.0)  # has acted
        self.assertEqual(obs[269 + 14], 1.0)  # button
        # recent action 0: seat 0 (relative slot 2) raised preflop
        self.assertEqual(obs[389], 1.0)
        self.assertEqual(obs[389 + 1 + 2], 1.0)
        self.assertEqual(obs[389 + 11 + 2], 1.0)
        self.assertEqual(obs[389 + 14 + 0], 1.0)
        self.assertTrue(np.all(obs[389 + 18 :] [: 144 - 18] == 0))

    def test_flop_decision(self):
        table = three_handed_table()
        table.act(BET_RAISE)  # seat 0 raises
        table.act(CHECK_CALL)  # seat 1 calls
        table.act(CHECK_CALL)  # seat 2 calls; the flop is dealt
        self.assertEqual(table.seat_to_act, 1)
        self.assertTrue(table.legal.can_check)
        obs = table.observation()
        want = {c.index: 1.0 for c in SEAT1}
        want.update({52 + c.index: 1.0 for c in FLOP})
        want.update(
            {
                209: 1.0,  # flop
                213: 1.0,  # 3 players
                222: 1.0,  # position 1
                231: pot(6.0),
                234: 1.0,  # no bet yet on the flop
                239: pot(2.0),
                241: 0.3,
                242: 0.3,
                243: stack(98.0),
                244: 1.0,
                245: 1.0,
                247: 1.0,  # 4 raises left
                # slot 1 = seat 2 (BB)
                254: 1.0,
                255: 1.0,
                257: pot(2.0),
                265: 1.0,  # acts after seat 1 on the flop
                267: stack(98.0),
                # slot 2 = seat 0 (button, preflop raiser)
                269: 1.0,
                270: 1.0,
                272: pot(2.0),
                274: 0.25,
                279: 1.0,  # last aggressor before this street
                280: 1.0,  # acts after seat 1 on the flop
                282: stack(98.0),
                283: 1.0,
                # recent 0: seat 2 called (slot 1), preflop
                389: 1.0,
                389 + 1 + 1: 1.0,
                389 + 12: 1.0,
                389 + 14: 1.0,
                # recent 1: seat 1 called (slot 0), preflop
                407: 1.0,
                407 + 1: 1.0,
                407 + 12: 1.0,
                407 + 14: 1.0,
                # recent 2: seat 0 raised (slot 2), preflop
                425: 1.0,
                425 + 3: 1.0,
                425 + 13: 1.0,
                425 + 14: 1.0,
                # hints: high card, low 8-4-3-2-A (index 21 of 56), 2 low hole ranks, A-2
                533 + 0: 1.0,
                533 + 9: 1.0,
                533 + 10 + 3: 1.0,
                533 + 14: 1.0 - 21 / 55,
                533 + 15 + 2: 1.0,
                533 + 20: 1.0,
            }
        )
        np.testing.assert_array_equal(obs, vector(want))
        np.testing.assert_array_equal(obs, encode_view(table.view(), table.legal))

    def test_heads_up_cap_lifted(self):
        table = Table()
        table.start_hand(2, [100.0, 100.0], 0, 3)
        obs = table.observation()
        self.assertEqual(obs[212], 1.0)  # 2 players
        self.assertEqual(obs[246], 1.0)  # cap lifted
        self.assertEqual(obs[247], 0.0)
        self.assertTrue(np.all(obs[254 + 15 :389] == 0))  # slots 2..9 empty


class RandomHandsTest(unittest.TestCase):
    def test_range_and_training_path_equals_view_path(self):
        rng = random.Random(20)
        decisions = 0
        for k in range(2500):
            spec = helpers.random_spec(rng, 2 + k % 9)
            table = Table()

            def check(t):
                nonlocal decisions
                decisions += 1
                obs = t.observation()
                self.assertEqual(obs.shape, (OBS_DIM,))
                self.assertEqual(obs.dtype, np.float32)
                self.assertTrue(np.all(np.isfinite(obs)))
                self.assertTrue(np.all(obs >= -1.0) and np.all(obs <= 2.0))
                view_obs = encode_view(t.view(), t.legal)
                self.assertEqual(obs.tobytes(), view_obs.tobytes())
                # Into a caller's buffer (zeroed first), and asked twice.
                buf = np.full(OBS_DIM, 7.0, dtype=np.float32)
                t.observation(buf)
                self.assertEqual(buf.tobytes(), view_obs.tobytes())
                self.assertEqual(t.observation().tobytes(), view_obs.tobytes())

            helpers.play_random(table, spec, rng, check)
        self.assertGreater(decisions, 5000)


if __name__ == "__main__":
    unittest.main()
