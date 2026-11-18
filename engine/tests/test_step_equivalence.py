"""Callback mode and step mode are interchangeable: for the same deck, seating,
stacks, button and actions they give identical events (as received through
on_event, seq and hand_no included), identical legal actions at every decision,
identical hand results and identical final stacks."""

import random
import unittest

from engine import Action, Deck, Environment, Player


def draw_stack(rng):
    """Stacks from short (< 1 BB) to deep, so that all-ins and side pots happen."""
    kind = rng.random()
    if kind < 0.15:
        return round(rng.uniform(0.05, 0.99), 2)
    if kind < 0.35:
        return round(rng.uniform(1.0, 3.0), 2)
    if kind < 0.65:
        return round(rng.uniform(3.0, 15.0), 2)
    return round(rng.uniform(15.0, 200.0), 2)


class Agent:
    """Records the events it receives. In callback mode it also records each
    decision it is offered and picks the action with the shared RNG; in step mode
    the test does that with the same RNG. Rebuys deterministically when busted."""

    def __init__(self, rng, decisions, rebuy=False):
        self.rng = rng
        self.decisions = decisions
        self.rebuy = rebuy
        self.events = []
        self.hand_ends = []

    def on_turn(self, player, view, legal):
        self.decisions.append((view.hand_number, player.seat, legal))
        player.act(Action.of(self.rng.choice(legal.actions)))

    def on_event(self, player, event):
        self.events.append(event)

    def on_hand_end(self, player, result):
        self.hand_ends.append(result.hand_number)
        # Busted players rebuy after odd hands only, so some of them sit out.
        if self.rebuy and player.chips == 0 and result.hand_number % 2 == 1:
            player.rebuy(10)


def session(seed, step_mode, hands=1, churn=False, views=False, rebuy=False):
    """Play `hands` hands at a table set up from `seed`; returns everything the
    two modes must agree on."""
    setup = random.Random(seed)
    n = setup.randint(2, 10)
    stacks = [draw_stack(setup) for _ in range(n)]
    button = setup.randrange(n)
    rng = random.Random(seed * 7919 + 1)
    decisions = []
    agents = [Agent(rng, decisions, rebuy) for _ in range(n)]
    players = [Player(f"p{i}", s, a) for i, (s, a) in enumerate(zip(stacks, agents))]
    env = Environment(players, button=button, deck=Deck(random.Random(seed)))
    everyone = list(players)
    results = []
    for h in range(hands):
        if churn and h == hands // 2:
            # Between hands: one player leaves and a new one joins (as big blind).
            if len(env.players) > 2:
                env.players[setup.randrange(len(env.players))].leave()
            joiner = Agent(rng, decisions, rebuy)
            agents.append(joiner)
            env.join(round(setup.uniform(1, 50), 2), name=f"j{h}", agent=joiner)
            everyone.append(next(p for p in env.players if p.name == f"j{h}"))
        if not env.can_start():
            break
        if step_mode:
            d = env.start_hand()
            while d is not None:
                decisions.append((d.hand_number, d.seat, d.legal))
                if views:
                    view = env.view()
                    assert view.seat == d.seat and view.hole_cards == d.player.hole_cards
                    assert env.view() is view
                d = env.step(rng.choice(d.legal.actions))
            results.append(env.last_result)
        else:
            results.append(env.start_round())
    return {
        "decisions": decisions,
        "events": [a.events for a in agents],
        "hand_ends": [a.hand_ends for a in agents],
        "results": results,
        "stacks": [(p.name, p.chips, p.status) for p in everyone],
        "button": env.button,
        "seq": env._seq,
    }


class StepEquivalenceTest(unittest.TestCase):
    def assert_same(self, a, b, seed):
        for key in a:
            self.assertEqual(a[key], b[key], f"seed {seed}: {key} differ")

    def test_single_hands(self):
        sizes = set()
        all_in_hands = side_pot_hands = 0
        for seed in range(400):
            cb = session(seed, step_mode=False)
            st = session(seed, step_mode=True)
            self.assert_same(cb, st, seed)
            self.assertTrue(cb["results"], seed)
            result = cb["results"][0]
            sizes.add(len(result.dealt_in))
            if any(getattr(e, "all_in", False) for e in result.history):
                all_in_hands += 1
            if len(result.pots) > 1:
                side_pot_hands += 1
        # The seeds cover every table size, all-ins and multi-way side pots.
        self.assertEqual(sizes, set(range(2, 11)))
        self.assertGreater(all_in_hands, 20)
        self.assertGreater(side_pot_hands, 5)

    def test_sessions_with_busts_rebuys_leave_and_join(self):
        for seed in range(1000, 1060):
            cb = session(seed, step_mode=False, hands=10, churn=True, rebuy=True)
            st = session(seed, step_mode=True, hands=10, churn=True, rebuy=True)
            self.assert_same(cb, st, seed)
            self.assertTrue(cb["results"], seed)

    def test_views_have_no_effect(self):
        for seed in range(2000, 2060):
            without = session(seed, step_mode=True, hands=4, rebuy=True)
            with_views = session(seed, step_mode=True, hands=4, rebuy=True, views=True)
            self.assert_same(without, with_views, seed)
            cb = session(seed, step_mode=False, hands=4, rebuy=True)
            self.assert_same(cb, with_views, seed)


if __name__ == "__main__":
    unittest.main()
