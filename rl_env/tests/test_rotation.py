"""Rotation: the same situation dealt and played from different physical seats
(same deck seed, same button-relative stacks and actions, the button at every
seat) gives identical observation sequences."""

import random
import unittest

from rl_env import Table, encode_view

try:
    from . import helpers
except ImportError:  # run as a top-level module
    import helpers  # type: ignore[no-redef]


def sequence(spec, actions):
    """Per decision: (seat offset from the button, mask, training obs, view obs)."""
    out = []

    def grab(t):
        out.append(
            (
                (t.seat_to_act - spec.button) % spec.n_seats,
                t.mask,
                t.observation().tobytes(),
                encode_view(t.view(), t.legal).tobytes(),
            )
        )

    helpers.replay(Table(), spec, actions, grab)
    return out


class RotationTest(unittest.TestCase):
    def test_button_at_every_seat(self):
        rng = random.Random(40)
        compared = 0
        for k in range(300):
            n = 2 + k % 9
            rel_stacks = helpers.random_stacks(rng, n)  # by offset from the button
            seed = rng.getrandbits(63)

            def spec_for(button):
                stacks = tuple(rel_stacks[(seat - button) % n] for seat in range(n))
                return helpers.HandSpec(n, stacks, button, seed)

            first = spec_for(0)
            actions = helpers.play_random(Table(), first, rng)
            reference = sequence(first, actions)
            for button in range(1, n):
                got = sequence(spec_for(button), actions)
                self.assertEqual(got, reference, (n, button))
                compared += len(got)
        self.assertGreater(compared, 1000)


if __name__ == "__main__":
    unittest.main()
