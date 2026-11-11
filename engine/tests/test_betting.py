import unittest
from collections import Counter

from engine import Environment, EventType, Player, Street


class Scripted:
    """Plays codes f/k/c/r (fold/check/call/raise) in order, then a default style.

    Records (street, legal) for each of its turns; optionally appends
    (street, seat) to a shared log.
    """

    def __init__(self, script="", aggressive=False, log=None):
        self.script = list(script)
        self.aggressive = aggressive
        self.log = log
        self.turns = []

    def on_turn(self, player, view, legal):
        self.turns.append((view.street, legal))
        if self.log is not None:
            self.log.append((view.street, player.seat))
        if self.script:
            code = self.script.pop(0)
        elif self.aggressive and legal.can_raise:
            code = "r"
        else:
            code = "k" if legal.can_check else "c"
        {"f": player.fold, "k": player.check, "c": player.call, "r": player.raise_}[code]()

    def on_hand_end(self, player, result):
        pass

    def legals(self, street):
        return [legal for s, legal in self.turns if s is street]


def make_env(stacks, agents, button=0, seed=1):
    players = [Player(f"p{i}", s, a) for i, (s, a) in enumerate(zip(stacks, agents))]
    return Environment(players, button=button, seed=seed), players


def raise_counts(result):
    return Counter(e.street for e in result.history if e.type is EventType.RAISE)


class BetSizeTest(unittest.TestCase):
    def test_small_bet_preflop_flop_big_bet_turn_river(self):
        agents = [Scripted() for _ in range(3)]
        env, _ = make_env([100] * 3, agents)
        env.start_round()
        units = {}
        for agent in agents:
            for street, legal in agent.turns:
                units.setdefault(street, set()).add(legal.bet_unit)
        self.assertEqual(
            units,
            {Street.PREFLOP: {1.0}, Street.FLOP: {1.0}, Street.TURN: {2.0}, Street.RIVER: {2.0}},
        )
        # Seat 0 is first to act preflop (left of the BB): call 1, raise to 2.
        first = agents[0].legals(Street.PREFLOP)[0]
        self.assertFalse(first.can_check)
        self.assertEqual(first.call_amount, 1.0)
        self.assertTrue(first.can_raise)
        self.assertEqual(first.raise_to, 2.0)
        self.assertEqual(first.raises_left, 3)
        # The small blind completes for half a bet.
        self.assertEqual(agents[1].legals(Street.PREFLOP)[0].call_amount, 0.5)
        # Postflop the first bet is one unit.
        flop = agents[1].legals(Street.FLOP)[0]
        self.assertTrue(flop.can_check)
        self.assertEqual(flop.raise_to, 1.0)
        self.assertEqual(flop.raises_left, 4)
        self.assertEqual(agents[1].legals(Street.TURN)[0].raise_to, 2.0)

    def test_bet_amounts_applied(self):
        # Seat 1: call preflop, check flop, bet the turn; everyone else calls.
        agents = [Scripted(), Scripted("ckr"), Scripted()]
        env, players = make_env([100] * 3, agents)
        result = env.start_round()
        turn_raises = [
            e for e in result.history if e.type is EventType.RAISE and e.street is Street.TURN
        ]
        self.assertEqual(len(turn_raises), 1)
        self.assertEqual(turn_raises[0].seat, 1)
        self.assertEqual(turn_raises[0].amount, 2.0)
        self.assertEqual(turn_raises[0].to, 2.0)
        self.assertEqual(sum(p.amount for p in result.pots), 9.0)  # 3 x (1 + 2)


class CapTest(unittest.TestCase):
    def test_four_bet_cap_multiway(self):
        agents = [Scripted(aggressive=True) for _ in range(3)]
        env, players = make_env([100] * 3, agents)
        result = env.start_round()
        counts = raise_counts(result)
        # Preflop the big blind is the first bet, so 3 raises; postflop bet + 3 raises.
        self.assertEqual(counts[Street.PREFLOP], 3)
        self.assertEqual(counts[Street.FLOP], 4)
        self.assertEqual(counts[Street.TURN], 4)
        self.assertEqual(counts[Street.RIVER], 4)
        # 4 small bets on preflop/flop, 4 big bets on turn/river: 24 BB each.
        self.assertEqual(sum(p.amount for p in result.pots), 72.0)
        capped = [
            legal
            for agent in agents
            for _, legal in agent.turns
            if legal.raises_left == 0
        ]
        self.assertTrue(capped)
        self.assertTrue(all(not legal.can_raise for legal in capped))
        self.assertEqual(sum(p.chips for p in players), 30000)

    def test_cap_lifted_heads_up(self):
        agents = [Scripted(aggressive=True), Scripted(aggressive=True)]
        env, players = make_env([10, 10], agents)
        result = env.start_round()
        self.assertEqual(raise_counts(result)[Street.PREFLOP], 9)  # well over the cap
        self.assertTrue(all(legal.raises_left is None for a in agents for _, legal in a.turns))
        self.assertTrue(result.showdown)
        self.assertEqual(len(result.board), 5)
        self.assertEqual(sum(p.amount for p in result.pots), 20.0)
        self.assertEqual(sum(p.chips for p in players), 2000)


class AllInRaiseTest(unittest.TestCase):
    def _play(self, short_stack):
        # Preflop all limp. Flop: seat 1 bets, seat 2 raises all-in, seat 0 calls,
        # then seat 1 acts again facing the all-in raise.
        agents = [Scripted("cc"), Scripted("crc"), Scripted("kr")]
        env, players = make_env([100, 100, short_stack], agents)
        env.start_round()
        return agents

    def test_all_in_less_than_half_bet_does_not_reopen(self):
        agents = self._play(2.4)  # 1.4 BB left on the flop: raise of 0.4
        seat0 = agents[0].legals(Street.FLOP)[0]
        self.assertTrue(seat0.can_raise)  # had not acted yet
        self.assertAlmostEqual(seat0.call_amount, 1.4)
        self.assertAlmostEqual(seat0.raise_to, 2.4)
        seat1 = agents[1].legals(Street.FLOP)
        self.assertEqual(len(seat1), 2)
        self.assertFalse(seat1[1].can_raise)
        self.assertAlmostEqual(seat1[1].call_amount, 0.4)

    def test_all_in_half_bet_or_more_reopens(self):
        agents = self._play(2.5)  # 1.5 BB left on the flop: raise of 0.5
        seat1 = agents[1].legals(Street.FLOP)
        self.assertEqual(len(seat1), 2)
        self.assertTrue(seat1[1].can_raise)
        self.assertAlmostEqual(seat1[1].call_amount, 0.5)
        self.assertAlmostEqual(seat1[1].raise_to, 2.5)


class HeadsUpOrderTest(unittest.TestCase):
    def test_blinds_and_action_order(self):
        log = []
        agents = [Scripted(log=log), Scripted(log=log)]
        env, players = make_env([100, 100], agents, button=0)
        result = env.start_round()
        self.assertEqual((result.sb_seat, result.bb_seat), (0, 1))
        self.assertEqual(
            log,
            [
                (Street.PREFLOP, 0),
                (Street.PREFLOP, 1),
                (Street.FLOP, 1),
                (Street.FLOP, 0),
                (Street.TURN, 1),
                (Street.TURN, 0),
                (Street.RIVER, 1),
                (Street.RIVER, 0),
            ],
        )
        self.assertEqual(agents[0].legals(Street.PREFLOP)[0].call_amount, 0.5)
        # The big blind gets the option to raise when the button just calls.
        bb_option = agents[1].legals(Street.PREFLOP)[0]
        self.assertTrue(bb_option.can_check)
        self.assertTrue(bb_option.can_raise)

    def test_three_handed_blinds(self):
        agents = [Scripted() for _ in range(3)]
        env, _ = make_env([100] * 3, agents, button=2)
        result = env.start_round()
        self.assertEqual((result.sb_seat, result.bb_seat), (0, 1))


class IllegalActionTest(unittest.TestCase):
    def test_illegal_action_rejected_and_falls_back(self):
        errors = []

        class CheckFacingBet:
            def on_turn(self, player, view, legal):
                try:
                    player.check()  # illegal: facing the big blind
                except Exception as e:  # noqa: BLE001
                    errors.append(e)
                player.call()
                try:
                    player.call()  # only one action per turn
                except Exception as e:  # noqa: BLE001
                    errors.append(e)

            def on_hand_end(self, player, result):
                pass

        agents = [CheckFacingBet(), Scripted(), Scripted()]
        env, _ = make_env([100] * 3, agents)
        result = env.start_round()
        self.assertGreaterEqual(len(errors), 2)
        self.assertEqual(type(errors[0]).__name__, "IllegalActionError")
        player_actions = {EventType.FOLD, EventType.CHECK, EventType.CALL, EventType.RAISE}
        calls = [
            e
            for e in result.history
            if e.seat == 0 and e.street is Street.PREFLOP and e.type in player_actions
        ]
        self.assertEqual(calls[0].type, EventType.CALL)

    def test_no_action_checks_or_folds(self):
        class DoNothing:
            def on_turn(self, player, view, legal):
                pass

            def on_hand_end(self, player, result):
                pass

        env, players = make_env([100] * 3, [DoNothing(), DoNothing(), DoNothing()])
        result = env.start_round()
        # Seat 0 and the SB fold to the big blind.
        self.assertFalse(result.showdown)
        self.assertEqual(result.net, {0: 0.0, 1: -0.5, 2: 0.5})


if __name__ == "__main__":
    unittest.main()
