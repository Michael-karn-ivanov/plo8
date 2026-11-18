"""Step mode: Environment.start_hand() / step() / decision / view() / last_result."""

import random
import unittest

from entities import Card, Deck
from engine import (
    Action,
    ActionType,
    BlindPosted,
    BoardDealt,
    Decision,
    Environment,
    HandInProgressError,
    HandShown,
    HandStarted,
    IllegalActionError,
    NoDecisionError,
    NotEnoughPlayersError,
    Player,
    PlayerActed,
    PlayerStatus,
    PotAwarded,
    Street,
)


class Watcher:
    """Records every notification; counts on_turn calls (never expected in step
    mode). Optionally rebuys to `rebuy_to` in on_hand_end, or leaves when it
    gets its hole cards."""

    def __init__(self, rebuy_to=None, leave_on_hole_cards=False):
        self.rebuy_to = rebuy_to
        self.leave_on_hole_cards = leave_on_hole_cards
        self.turns = 0
        self.hole = {}
        self.events = []
        self.hand_ends = []

    def on_turn(self, player, view, legal):
        self.turns += 1
        player.fold()

    def on_hole_cards(self, player, hand, view):
        self.hole[view.hand_number] = (hand, view)
        if self.leave_on_hole_cards:
            player.leave()

    def on_event(self, player, event):
        self.events.append(event)

    def on_hand_end(self, player, result):
        self.hand_ends.append(result.hand_number)
        if self.rebuy_to is not None and player.stack < self.rebuy_to:
            player.rebuy(self.rebuy_to - player.stack)


class FirstViewAgent:
    """Callback mode: records the view and legal actions of its first turn."""

    def __init__(self, store):
        self.store = store

    def on_turn(self, player, view, legal):
        self.store.setdefault("view", view)
        self.store.setdefault("legal", legal)
        if legal.can_check:
            player.check()
        else:
            player.call()

    def on_hand_end(self, player, result):
        pass


def make_env(stacks, agents=None, button=0, seed=1, deck=None):
    agents = agents if agents is not None else [Watcher() for _ in stacks]
    players = [Player(f"p{i}", s, a) for i, (s, a) in enumerate(zip(stacks, agents))]
    return Environment(players, button=button, seed=seed, deck=deck), players, agents


def passive(decision):
    return ActionType.CHECK if decision.legal.can_check else ActionType.CALL


def finish(env, decision, choose=passive):
    while decision is not None:
        decision = env.step(choose(decision))
    return env.last_result


def last_action(env):
    return [e for e in env.dealer.history if isinstance(e, PlayerActed)][-1]


def snapshot(env):
    dealer = env.dealer
    return (
        env.decision,
        tuple(dealer.history),
        env._seq,
        tuple(
            (p.chips, p.street_chips, p.total_chips, p.folded, p.all_in, p.has_acted, p.status)
            for p in dealer.players
        ),
        dealer.deck.remaining(),
        dealer.board,
        dealer._current_bet,
        dealer._bets,
        dealer.pending,
    )


class StartHandTest(unittest.TestCase):
    def test_first_decision_left_of_big_blind(self):
        env, players, agents = make_env([100] * 3)
        d = env.start_hand()
        self.assertIsInstance(d, Decision)
        self.assertEqual((d.hand_number, d.seat, d.name), (1, 0, "p0"))
        self.assertIs(d.player, players[0])
        self.assertEqual(d.legal, env.dealer.legal_actions(players[0]))
        self.assertEqual(d.legal.actions, (ActionType.FOLD, ActionType.CALL, ActionType.RAISE))
        self.assertIs(env.decision, d)
        self.assertTrue(env.hand_in_progress)
        # The blinds and the deal are done; every participant has its cards.
        history = env.dealer.history
        self.assertIsInstance(history[0], HandStarted)
        self.assertEqual([type(e) for e in history[1:3]], [BlindPosted, BlindPosted])
        self.assertEqual(len(history), 3)
        for p, a in zip(players, agents):
            self.assertEqual(len(p.hole_cards), 4)
            self.assertEqual(a.hole[1][0], p.hole_cards)
            self.assertEqual(a.hole[1][1].hole_cards, p.hole_cards)
        self.assertEqual(len(env.deck), 52 - 12)

    def test_heads_up_button_acts_first(self):
        env, players, _ = make_env([100, 100], button=1)
        d = env.start_hand()
        self.assertEqual(d.seat, 1)  # the button is the small blind
        self.assertEqual(d.legal, env.dealer.legal_actions(players[1]))
        self.assertEqual(d.legal.call_amount, 0.5)
        d = env.step(ActionType.CALL)
        self.assertEqual(d.seat, 0)  # the big blind's option
        self.assertTrue(d.legal.can_check)
        d = env.step(ActionType.CHECK)
        self.assertEqual(d.seat, 0)  # postflop the big blind acts first
        self.assertEqual(env.dealer.board.street, Street.FLOP)

    def test_on_turn_is_never_called(self):
        env, _, agents = make_env([100] * 4)
        for _ in range(5):
            finish(env, env.start_hand())
        self.assertEqual(sum(a.turns for a in agents), 0)
        self.assertEqual([a.hand_ends for a in agents], [[1, 2, 3, 4, 5]] * 4)

    def test_player_actions_are_rejected_in_step_mode(self):
        env, players, _ = make_env([100] * 3)
        d = env.start_hand()
        for act in (d.player.fold, d.player.call, d.player.raise_):
            with self.assertRaises(IllegalActionError):
                act()
        self.assertIs(env.decision, d)


class ViewTest(unittest.TestCase):
    def test_view_shows_only_own_cards_and_equals_on_turn_view(self):
        store = {}
        cb_env, _, _ = make_env([100] * 3, agents=[FirstViewAgent(store), None, None], seed=4)
        cb_env.start_round()

        env, players, _ = make_env([100] * 3, agents=[None, None, None], seed=4)
        d = env.start_hand()
        view = env.view()
        self.assertEqual(view, store["view"])
        self.assertEqual(d.legal, store["legal"])
        self.assertEqual(view.seat, d.seat)
        self.assertEqual(view.hole_cards, players[0].hole_cards)
        self.assertFalse(any(hasattr(s, "hole_cards") for s in view.seats))
        self.assertIs(env.view(), view)  # cached
        self.assertIs(env.decision, d)
        d2 = env.step(ActionType.CALL)
        self.assertIs(env.decision, d2)
        view2 = env.view()
        self.assertIsNot(view2, view)
        self.assertEqual(view2.seat, d2.seat)
        self.assertEqual(view2.hole_cards, players[d2.seat].hole_cards)

    def test_view_without_decision(self):
        env, _, _ = make_env([100] * 2)
        with self.assertRaises(NoDecisionError):
            env.view()
        finish(env, env.start_hand())
        with self.assertRaises(NoDecisionError):
            env.view()
        self.assertIsNone(env.decision)


class StepTest(unittest.TestCase):
    def test_each_action_type(self):
        for t in (ActionType.FOLD, ActionType.CALL, ActionType.RAISE):
            for as_action in (False, True):
                env, players, _ = make_env([100] * 3)
                env.start_hand()
                d = env.step(Action.of(t) if as_action else t)
                acted = env.dealer.history[-1]
                self.assertIsInstance(acted, PlayerActed)
                self.assertEqual((acted.seat, acted.action, acted.reason), (0, t, None))
                self.assertEqual(d.seat, 1)
        # CHECK: the big blind's option after everybody calls.
        env, players, _ = make_env([100] * 3)
        env.start_hand()
        env.step(ActionType.CALL)
        d = env.step(ActionType.CALL)
        self.assertEqual(d.seat, 2)
        self.assertTrue(d.legal.can_check)
        d = env.step(Action.check())
        acted = last_action(env)
        self.assertEqual((acted.seat, acted.action), (2, ActionType.CHECK))
        self.assertEqual(env.dealer.board.street, Street.FLOP)
        self.assertIsInstance(env.dealer.history[-1], BoardDealt)

    def test_illegal_actions_leave_the_state_unchanged(self):
        env, players, agents = make_env([100] * 3)
        d = env.start_hand()
        before = snapshot(env)
        events_before = [len(a.events) for a in agents]
        for bad in (ActionType.CHECK, Action(ActionType.CHECK), "call", None, 3, Action("raise")):
            with self.assertRaises(IllegalActionError, msg=repr(bad)):
                env.step(bad)
            self.assertEqual(snapshot(env), before)
            self.assertIs(env.decision, d)
        self.assertEqual([len(a.events) for a in agents], events_before)
        self.assertTrue(all(p.status is PlayerStatus.ACTIVE for p in players))
        # A legal action afterwards proceeds normally.
        d2 = env.step(ActionType.CALL)
        self.assertEqual(d2.seat, 1)
        self.assertEqual(env.dealer.history[-1].action, ActionType.CALL)
        self.assertEqual(env._seq, before[2] + 1)

    def test_illegal_raise_when_capped(self):
        env, players, _ = make_env([100] * 3)
        d = env.start_hand()
        while d.legal.can_raise:
            d = env.step(ActionType.RAISE)
        self.assertEqual(d.legal.raises_left, 0)
        before = snapshot(env)
        with self.assertRaises(IllegalActionError):
            env.step(ActionType.RAISE)
        self.assertEqual(snapshot(env), before)

    def test_step_without_decision(self):
        env, _, _ = make_env([100] * 3)
        with self.assertRaises(NoDecisionError):
            env.step(ActionType.FOLD)
        result = finish(env, env.start_hand())
        self.assertIsNotNone(result)
        with self.assertRaises(NoDecisionError):
            env.step(ActionType.FOLD)
        self.assertTrue(issubclass(NoDecisionError, RuntimeError))

    def test_rebuy_and_sit_in_rejected_while_decision_pending(self):
        env, players, _ = make_env([100] * 3)
        env.start_hand()
        for p in players:
            with self.assertRaises(IllegalActionError):
                p.rebuy(5)
            with self.assertRaises(IllegalActionError):
                p.sit_in()
        self.assertEqual(sum(p.rebuy_chips for p in players), 0)
        finish(env, env.decision)
        players[0].rebuy(5)  # between hands it works
        self.assertEqual(players[0].rebuy_chips, 500)


class NoDecisionHandTest(unittest.TestCase):
    def test_heads_up_both_all_in_on_the_blinds(self):
        env, players, agents = make_env([0.5, 1], button=0)
        self.assertIsNone(env.start_hand())
        result = env.last_result
        self.assertIsNotNone(result)
        self.assertIsNone(env.decision)
        self.assertFalse(env.hand_in_progress)
        self.assertTrue(result.showdown)
        self.assertEqual(len(result.board), 5)
        self.assertEqual(sum(isinstance(e, HandShown) for e in result.history), 2)
        self.assertTrue(any(isinstance(e, PotAwarded) for e in result.history))
        self.assertFalse(any(isinstance(e, PlayerActed) for e in result.history))
        self.assertEqual(
            [e.street for e in result.history if isinstance(e, BoardDealt)],
            [Street.FLOP, Street.TURN, Street.RIVER],
        )
        self.assertEqual([a.hand_ends for a in agents], [[1], [1]])
        self.assertEqual(env.button, 1)  # moved (seat 1 keeps at least its returned 0.5)
        self.assertEqual(sum(p.chips for p in players), 150)
        self.assertEqual(sum(round(v * 100) for v in result.net.values()), 0)

    def test_big_blind_covers_all_in_small_blind(self):
        env, players, _ = make_env([0.5, 50], button=0)
        self.assertIsNone(env.start_hand())
        result = env.last_result
        self.assertTrue(result.showdown)
        self.assertEqual(result.returned, {1: 0.5})
        self.assertEqual(sum(p.chips for p in players), 5050)

    def test_three_players_all_in_after_the_blinds(self):
        # The button leaves as soon as it sees its cards (forced fold, no
        # decision); the blinds are both all-in.
        agents = [Watcher(leave_on_hole_cards=True), Watcher(), Watcher()]
        env, players, _ = make_env([100, 0.5, 1], agents=agents, button=0)
        self.assertIsNone(env.start_hand())
        result = env.last_result
        self.assertTrue(result.showdown)
        self.assertEqual(set(result.shown), {1, 2})
        self.assertEqual(len(result.board), 5)
        folds = [e for e in result.history if isinstance(e, PlayerActed)]
        self.assertEqual([(e.seat, e.action, e.reason) for e in folds], [(0, ActionType.FOLD, "leave")])
        self.assertFalse(env.hand_in_progress)
        self.assertIs(players[0].status, PlayerStatus.LEFT)
        self.assertEqual(agents[0].hand_ends, [])
        self.assertEqual([agents[1].hand_ends, agents[2].hand_ends], [[1], [1]])
        self.assertEqual(len(env.players), 2)
        self.assertEqual(sum(p.chips for p in players), 10150)


class HandInProgressTest(unittest.TestCase):
    def test_starting_another_hand_is_rejected(self):
        env, _, _ = make_env([100] * 3)
        d = env.start_hand()
        before = snapshot(env)
        with self.assertRaises(HandInProgressError):
            env.start_hand()
        with self.assertRaises(HandInProgressError):
            env.start_round()
        with self.assertRaises(HandInProgressError):
            env.run(1)
        self.assertEqual(snapshot(env), before)
        self.assertIs(env.decision, d)
        self.assertTrue(issubclass(HandInProgressError, RuntimeError))
        finish(env, d)
        env.start_round()  # fine once the hand is over

    def test_not_enough_players(self):
        env, _, _ = make_env([100])
        with self.assertRaises(NotEnoughPlayersError):
            env.start_hand()
        self.assertFalse(env.hand_in_progress)
        self.assertIsNone(env.last_result)


class EndOfHandTest(unittest.TestCase):
    def test_end_of_hand_processing(self):
        # Button p0, SB p1, BB p2; p3 acts first.
        agents = [Watcher(rebuy_to=100), Watcher(), Watcher(), Watcher()]
        env, players, _ = make_env([100] * 4, agents=agents, seed=6)
        d = env.start_hand()
        self.assertIsNone(env.last_result)
        self.assertEqual(d.seat, 3)
        joiner = Watcher()
        env.join(40, name="late", agent=joiner)  # queued
        late = env.waiting[0]
        d = env.step(ActionType.RAISE)
        self.assertEqual(d.seat, 0)
        players[1].leave()  # the SB: folded at its turn, removed after the hand
        result = finish(env, d)
        self.assertIs(env.last_result, result)
        self.assertEqual(result.hand_number, 1)
        self.assertFalse(env.hand_in_progress)
        sb_folds = [e for e in result.history if isinstance(e, PlayerActed) and e.seat == 1]
        self.assertEqual([(e.action, e.reason) for e in sb_folds], [(ActionType.FOLD, "leave")])
        # Stacks updated; p0 topped up to 100 in its rebuy window.
        self.assertEqual(players[0].stack, max(100.0, result.stacks[0]))
        self.assertEqual(players[2].stack, result.stacks[2])
        self.assertEqual(players[3].stack, result.stacks[3])
        # The leaver was removed, the joiner seated, the button moved past the leaver.
        self.assertIs(players[1].status, PlayerStatus.LEFT)
        self.assertEqual(agents[1].hand_ends, [])
        self.assertEqual([agents[i].hand_ends for i in (0, 2, 3)], [[1], [1], [1]])
        self.assertIs(late.status, PlayerStatus.ACTIVE)
        self.assertEqual(joiner.hand_ends, [])
        self.assertEqual(len(env.players), 4)
        self.assertIs(env.players[env.button], players[2])
        self.assertEqual(sum(a.turns for a in agents + [joiner]), 0)

        second = finish(env, env.start_hand())
        self.assertEqual(second.hand_number, 2)
        self.assertTrue(second.dealt_in[late.seat])
        self.assertEqual(second.bb_seat, late.seat)
        self.assertEqual(joiner.hand_ends, [2])
        self.assertIs(env.last_result, second)


class DeterminismTest(unittest.TestCase):
    def _play(self, deck_factory, seed=None, hands=6):
        rng = random.Random(1234)
        agents = [Watcher(rebuy_to=20) for _ in range(5)]
        players = [Player(f"p{i}", 20, a) for i, a in enumerate(agents)]
        env = Environment(players, button=2, seed=seed, deck=deck_factory())
        results = []
        for _ in range(hands):
            d = env.start_hand()
            while d is not None:
                d = env.step(rng.choice(d.legal.actions))
            results.append(env.last_result)
        return [a.events for a in agents], results, [p.chips for p in players]

    def test_same_seed_same_hands(self):
        a = self._play(lambda: None, seed=77)
        b = self._play(lambda: None, seed=77)
        self.assertEqual(a, b)
        c = self._play(lambda: None, seed=78)
        self.assertNotEqual(a[1], c[1])

    def test_same_stacked_deck_same_hands(self):
        script = random.Random(5).sample([Card.from_index(i) for i in range(52)], 30)

        class Stacked(Deck):
            def __init__(self):
                super().__init__(random.Random(random.randrange(1 << 30)))

            def shuffle(self):
                super().shuffle()
                self.put_on_top(script)

        a = self._play(Stacked)
        b = self._play(Stacked)
        self.assertEqual(a, b)
        first_hand = a[1][0]
        self.assertTrue(set(first_hand.board) <= set(script))


if __name__ == "__main__":
    unittest.main()
