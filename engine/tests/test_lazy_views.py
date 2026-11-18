"""Views are built only when they are consumed (Dealer.table_view is the only
builder): counted by wrapping Dealer.table_view."""

import contextlib
import unittest

from engine import ActionType, Dealer, Environment, Player


@contextlib.contextmanager
def counting_views():
    counter = {"n": 0}
    original = Dealer.table_view

    def table_view(self, p):
        counter["n"] += 1
        return original(self, p)

    Dealer.table_view = table_view
    try:
        yield counter
    finally:
        Dealer.table_view = original


class EventsOnly:
    """Has on_event / on_hand_end but no on_hole_cards (and is never asked to act
    in step mode)."""

    def __init__(self):
        self.events = 0

    def on_turn(self, player, view, legal):  # pragma: no cover - step mode only
        raise AssertionError("on_turn in step mode")

    def on_event(self, player, event):
        self.events += 1

    def on_hand_end(self, player, result):
        pass


class HoleCardsAgent(EventsOnly):
    def __init__(self):
        super().__init__()
        self.hands = 0

    def on_hole_cards(self, player, hand, view):
        self.hands += 1


class TurnCounter:
    """Callback-mode agent without optional callbacks; counts its turns."""

    def __init__(self):
        self.turns = 0

    def on_turn(self, player, view, legal):
        self.turns += 1
        if legal.can_check:
            player.check()
        else:
            player.call()

    def on_hand_end(self, player, result):
        pass


def passive(decision):
    return ActionType.CHECK if decision.legal.can_check else ActionType.CALL


def play_step(env, hands, views_per_decision=0, reads=False):
    decisions = 0
    for _ in range(hands):
        d = env.start_hand()
        while d is not None:
            decisions += 1
            if reads:  # the view-free reads
                env.history[:]
                env.hole_cards()
                env.privileged_hole_cards()
            for _ in range(views_per_decision):
                env.view()
            d = env.step(passive(d))
    return decisions


def table(agents, seed=3):
    players = [Player(f"p{i}", 100, a) for i, a in enumerate(agents)]
    return Environment(players, seed=seed), players


class StepModeViewsTest(unittest.TestCase):
    def test_no_views_without_requests_or_callbacks(self):
        with counting_views() as count:
            env, _ = table([None] * 6)
            decisions = play_step(env, 10)
        self.assertGreater(decisions, 0)
        self.assertEqual(count["n"], 0)

    def test_no_views_for_agents_without_on_hole_cards(self):
        agents = [EventsOnly() for _ in range(4)]
        with counting_views() as count:
            env, _ = table(agents)
            play_step(env, 10)
        self.assertTrue(all(a.events for a in agents))
        self.assertEqual(count["n"], 0)

    def test_no_views_for_view_free_reads(self):
        agents = [EventsOnly(), None, EventsOnly(), None, None]
        with counting_views() as count:
            env, _ = table(agents)
            decisions = play_step(env, 10, reads=True)
        self.assertGreater(decisions, 0)
        self.assertEqual(count["n"], 0)

    def test_one_view_per_requested_decision(self):
        with counting_views() as count:
            env, _ = table([None] * 4)
            decisions = play_step(env, 5, views_per_decision=3)  # cached after the first
        self.assertEqual(count["n"], decisions)

    def test_one_view_per_requested_decision_with_reads(self):
        with counting_views() as count:
            env, _ = table([None] * 4)
            decisions = play_step(env, 5, views_per_decision=2, reads=True)
        self.assertEqual(count["n"], decisions)

    def test_one_view_per_on_hole_cards_agent_per_hand(self):
        agents = [HoleCardsAgent(), None, HoleCardsAgent(), EventsOnly(), None]
        with counting_views() as count:
            env, _ = table(agents)
            play_step(env, 7)
        self.assertEqual([a.hands for a in (agents[0], agents[2])], [7, 7])
        self.assertEqual(count["n"], 2 * 7)


class CallbackModeViewsTest(unittest.TestCase):
    def test_one_view_per_turn_of_a_real_agent_none_for_the_default_agent(self):
        agents = [TurnCounter(), None, TurnCounter(), None]
        with counting_views() as count:
            env, _ = table(agents)
            env.run(6)
        turns = agents[0].turns + agents[2].turns
        self.assertGreater(turns, 0)
        self.assertEqual(count["n"], turns)

    def test_default_agents_only(self):
        with counting_views() as count:
            env, _ = table([None] * 5)
            env.run(6)
        self.assertEqual(count["n"], 0)

    def test_on_seated_view_only_for_agents_that_have_it(self):
        class Seated:
            def __init__(self):
                self.views = []

            def on_seated(self, player, view):
                self.views.append(view)

        with counting_views() as count:
            env, _ = table([None, None])
            env.join(10, name="a")  # no agent
            env.join(10, name="b", agent=TurnCounter())  # no on_seated
            self.assertEqual(count["n"], 0)
            seated = Seated()
            env.join(10, name="c", agent=seated)
        self.assertEqual(count["n"], 1)
        self.assertEqual(len(seated.views), 1)


if __name__ == "__main__":
    unittest.main()
