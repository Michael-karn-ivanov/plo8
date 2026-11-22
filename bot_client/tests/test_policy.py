"""hand_end_reply: the rebuy / sit-in / leave reply (table-driven)."""

import unittest

from bot_client.policy import HandEndChoice, hand_end_reply


def reply(stack, policy="topup", target=100.0, max_buy_in=None, sitting_out=False,
          hands_played=1, max_hands=None, stopping=False):
    return hand_end_reply(
        stack=stack,
        sitting_out=sitting_out,
        target=target,
        max_buy_in=max_buy_in,
        policy=policy,
        hands_played=hands_played,
        max_hands=max_hands,
        stopping=stopping,
    )


class PolicyTest(unittest.TestCase):
    def test_table(self):
        cases = [
            # (kwargs, rebuy_bb, sit_in, leave)
            (dict(stack=98.0), 2.0, False, False),
            (dict(stack=100.0), None, False, False),
            (dict(stack=130.5), None, False, False),
            (dict(stack=0.0), 100.0, False, False),
            (dict(stack=97.123), 2.88, False, False),  # rounded to cents
            (dict(stack=99.999), None, False, False),  # rounds to 0.00
            (dict(stack=40.0, max_buy_in=60.0), 20.0, False, False),  # capped
            (dict(stack=70.0, max_buy_in=60.0), None, False, False),  # above the cap
            (dict(stack=60.0, max_buy_in=60.0), None, False, False),
            (dict(stack=40.0, policy="bust"), None, False, False),
            (dict(stack=0.0, policy="bust"), 100.0, False, False),
            (dict(stack=0.0, policy="bust", max_buy_in=50.0), 50.0, False, False),
            (dict(stack=40.0, policy="never"), None, False, False),
            (dict(stack=0.0, policy="never"), None, False, True),
            (dict(stack=50.0, sitting_out=True), 50.0, True, False),
            (dict(stack=0.0, sitting_out=True), 100.0, True, False),
            (dict(stack=50.0, sitting_out=True, policy="never"), None, True, False),
            (dict(stack=0.0, hands_played=5, max_hands=5), None, False, True),
            (dict(stack=50.0, hands_played=5, max_hands=5), None, False, True),
            (dict(stack=50.0, hands_played=4, max_hands=5), 50.0, False, False),
            (dict(stack=50.0, stopping=True, sitting_out=True), None, False, True),
        ]
        for kwargs, rebuy, sit_in, leave in cases:
            with self.subTest(**kwargs):
                r = reply(**kwargs)
                self.assertIsInstance(r, HandEndChoice)
                self.assertEqual(r.rebuy_bb, rebuy)
                self.assertEqual(r.sit_in, sit_in)
                self.assertEqual(r.leave, leave)
                self.assertEqual(bool(r.reason), leave)

    def test_reasons(self):
        self.assertEqual(reply(50.0, stopping=True).reason, "stopping")
        self.assertEqual(reply(50.0, hands_played=200, max_hands=200).reason, "200 hands played")
        self.assertEqual(reply(0.0, policy="never").reason, "busted, --rebuy never")

    def test_unknown_policy(self):
        with self.assertRaises(ValueError):
            reply(50.0, policy="sometimes")


if __name__ == "__main__":
    unittest.main()
