"""HandTracker: the public state equals the engine view's at every decision;
incremental feeding equals feeding the whole history at once."""

import random
import unittest

from rl_env import HandTracker, Table

try:
    from . import helpers
except ImportError:  # run as a top-level module
    import helpers  # type: ignore[no-redef]

PLACES = 9


class TrackerTest(unittest.TestCase):
    def check_decision(self, table: Table) -> None:
        tr = table.tracker
        view = table.view()
        self.assertTrue(tr.started)
        self.assertEqual(tr.street, view.street.value)
        self.assertAlmostEqual(tr.pot, view.pot, places=PLACES)
        self.assertEqual(tr.seats, tuple(s.seat for s in view.seats if s.in_hand))
        self.assertEqual(tr.seats[tr.button], view.button)
        self.assertEqual(tr.seats[tr.sb], view.sb_seat)
        self.assertEqual(tr.seats[tr.bb], view.bb_seat)
        self.assertAlmostEqual(tr.current_bet, view.current_bet, places=PLACES)
        self.assertEqual(tr.board.cards, view.board.cards)
        self.assertGreaterEqual(tr.street_bets, view.raises)
        for i, seat in enumerate(tr.seats):
            sv = view.seats[seat]
            self.assertAlmostEqual(tr.stack[i], sv.stack, places=PLACES)
            self.assertAlmostEqual(tr.street_bet[i], sv.street_bet, places=PLACES)
            self.assertAlmostEqual(tr.total[i], sv.total_bet, places=PLACES)
            self.assertEqual(tr.folded[i], sv.folded)
            self.assertEqual(tr.all_in[i], sv.all_in)
        # The same state from the whole history at once.
        whole = HandTracker.from_events(view.history)
        self.assertEqual(whole.snapshot(), tr.snapshot())
        self.assertEqual(tr.n_events, len(view.history))

    def test_tracker_equals_view_at_every_decision(self):
        rng = random.Random(10)
        decisions = 0
        for k in range(1500):
            spec = helpers.random_spec(rng, 2 + k % 9)
            table = Table()

            def check(t):
                nonlocal decisions
                decisions += 1
                self.check_decision(t)

            helpers.play_random(table, spec, rng, check)
        self.assertGreater(decisions, 3000)

    def test_events_one_by_one(self):
        rng = random.Random(11)
        for _ in range(100):
            table = Table()
            spec = helpers.random_spec(rng)
            helpers.play_random(table, spec, rng)
            events = table.result.history
            one = HandTracker()
            for ev in events:
                one.feed(ev)
            self.assertEqual(one.snapshot(), HandTracker.from_events(events).snapshot())

    def test_ignores_settlement_events(self):
        table = Table()
        table.start_hand(2, [100.0, 100.0], 0, 1)
        helpers.passive(table)
        events = table.result.history
        last_decision_events = [e for e in events if type(e).__name__ not in (
            "HandShown", "BetReturned", "PotAwarded")]
        self.assertEqual(
            HandTracker.from_events(events).snapshot(),
            HandTracker.from_events(last_decision_events).snapshot(),
        )

    def test_second_hand_started_is_rejected(self):
        table = Table()
        table.start_hand(2, [100.0, 100.0], 0, 1)
        events = list(table.history)
        tr = HandTracker.from_events(events)
        with self.assertRaises(ValueError):
            tr.feed(events[0])


if __name__ == "__main__":
    unittest.main()
