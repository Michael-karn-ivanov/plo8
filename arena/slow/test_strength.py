"""Acceptance: RuleBot beats RandomBot and CallingStation 6-handed (one RuleBot
against five copies), 95% interval excluding 0, at a fixed seed.

ARENA_SLOW_DEALS lowers the deal count for local runs (default 1,000 deals =
6,000 hands per match). If a run fails, tune RULE_PARAMS, not the seed.
"""

import os
import unittest

from arena import play_match

DEALS = int(os.environ.get("ARENA_SLOW_DEALS", "1000"))
SEED = 0


class StrengthTest(unittest.TestCase):
    def check_beats(self, opponent):
        res = play_match(["rule"] + [opponent] * 5, DEALS, SEED)
        st = res.stats["rule"]
        print(
            f"\nrule vs 5 x {opponent}: {st.bb_per_100:+.2f} +- {st.ci95:.2f} bb/100 "
            f"({st.hands} hands, {res.hands_per_second:,.0f} hands/s)"
        )
        self.assertEqual(st.hands, 6 * DEALS)
        self.assertGreater(st.bb_per_100, 0.0)
        self.assertGreater(st.bb_per_100 - st.ci95, 0.0)

    def test_rule_beats_random(self):
        self.check_beats("random")

    def test_rule_beats_calling_station(self):
        self.check_beats("calling")


if __name__ == "__main__":
    unittest.main()
