"""Server path: the observation from the engine's view and legal actions equals
the one from the same view and legal actions after a cli_server protocol round
trip (encode, JSON text, decode)."""

import json
import random
import unittest

from cli_server.protocol import decode_legal, decode_table_view, encode_legal, encode_table_view
from rl_env import DecisionBatch, Table, encode_view

try:
    from . import helpers
except ImportError:  # run as a top-level module
    import helpers  # type: ignore[no-redef]


def round_trip(view, legal):
    msg = {"type": "turn", "view": encode_table_view(view), "legal": encode_legal(legal)}
    d = json.loads(json.dumps(msg, separators=(",", ":"), allow_nan=False))
    return decode_table_view(d["view"]), decode_legal(d["legal"])


class ServerPathTest(unittest.TestCase):
    def test_round_trip_gives_the_same_observation(self):
        rng = random.Random(50)
        decisions = 0
        for k in range(2000):
            spec = helpers.random_spec(rng, 2 + k % 9)

            def check(t):
                nonlocal decisions
                decisions += 1
                view, legal = t.view(), t.legal
                local = encode_view(view, legal)
                view2, legal2 = round_trip(view, legal)
                self.assertEqual(legal2, legal)
                remote = encode_view(view2, legal2)
                self.assertEqual(local.tobytes(), remote.tobytes())
                self.assertEqual(local.tobytes(), t.observation().tobytes())
                # A bot answering a decoded `turn` gets the same single-row batch.
                single = DecisionBatch.single(view2, legal2)
                self.assertEqual(single.obs[0].tobytes(), local.tobytes())
                self.assertEqual(single.mask[0].tolist(), list(t.mask))

            helpers.play_random(Table(), spec, rng, check)
        self.assertGreater(decisions, 5000)


if __name__ == "__main__":
    unittest.main()
