import unittest

from engine import Environment, IllegalActionError, Player, Street


class Passive:
    def on_turn(self, player, view, legal):
        if legal.can_check:
            player.check()
        else:
            player.call()

    def on_hand_end(self, player, result):
        pass


class Aggressive(Passive):
    def on_turn(self, player, view, legal):
        if legal.can_raise:
            player.raise_()
        else:
            super().on_turn(player, view, legal)


class Observer(Passive):
    """Records the table state at the first turn of every hand (shared between seats)."""

    def __init__(self, snapshots):
        self.env = None
        self.snapshots = snapshots

    def on_turn(self, player, view, legal):
        if view.hand_number not in self.snapshots:
            self.snapshots[view.hand_number] = {
                "seat": player.seat,
                "deck": len(self.env.dealer.deck),
                "street": view.street,
                "board": len(view.board),
                "pot": view.pot,
                "bets": {s.seat: s.total_bet for s in view.seats},
                "hole": view.hole_cards,
                "others_hidden": not hasattr(view.seats[0], "hole_cards"),
            }
        super().on_turn(player, view, legal)


class Rebuyer(Passive):
    def __init__(self, target=100.0):
        self.target = target
        self.calls = 0
        self.in_hand_error = None

    def on_turn(self, player, view, legal):
        try:
            player.rebuy(5)
        except IllegalActionError as e:
            self.in_hand_error = e
        super().on_turn(player, view, legal)

    def on_hand_end(self, player, result):
        self.calls += 1
        if player.stack < self.target:
            player.rebuy(self.target - player.stack)


class StartRoundTest(unittest.TestCase):
    def test_start_round_resets_and_posts_blinds(self):
        snapshots = {}
        observers = [Observer(snapshots) for _ in range(3)]
        players = [Player(n, 100, o) for n, o in zip("abc", observers)]
        env = Environment(players, button=0, seed=2)
        for o in observers:
            o.env = env
        first = env.start_round()
        second = env.start_round()
        self.assertEqual((first.hand_number, second.hand_number), (1, 2))
        self.assertEqual(len(first.board), 5)  # the previous hand had a full board

        # Hand 1: button 0, blinds on seats 1 and 2, seat 0 acts first.
        snap = snapshots[1]
        self.assertEqual(snap["seat"], 0)
        self.assertEqual(snap["deck"], 52 - 12)
        self.assertEqual(snap["street"], Street.PREFLOP)
        self.assertEqual(snap["board"], 0)
        self.assertEqual(snap["pot"], 1.5)
        self.assertEqual(snap["bets"], {0: 0.0, 1: 0.5, 2: 1.0})
        self.assertIsNotNone(snap["hole"])
        self.assertTrue(snap["others_hidden"])

        # Hand 2: the button moved to seat 1, so blinds are seats 2 and 0, and seat 1
        # acts first. Fresh deck, empty board, new hole cards.
        snap = snapshots[2]
        self.assertEqual(snap["seat"], 1)
        self.assertEqual(snap["deck"], 52 - 12)
        self.assertEqual(snap["street"], Street.PREFLOP)
        self.assertEqual(snap["board"], 0)
        self.assertEqual(snap["pot"], 1.5)
        self.assertEqual(snap["bets"], {0: 1.0, 1: 0.0, 2: 0.5})
        self.assertEqual(snap["hole"], players[1].hole_cards)
        for p in players:
            self.assertEqual(len(p.hole_cards), 4)

    def test_requires_two_to_ten_players(self):
        with self.assertRaises(ValueError):
            Environment([Player("a", 100)])
        with self.assertRaises(ValueError):
            Environment([Player(str(i), 100) for i in range(11)])
        Environment([Player(str(i), 100) for i in range(10)])

    def test_needs_two_players_with_chips(self):
        env = Environment([Player("a", 100), Player("b", 0)])
        self.assertFalse(env.can_start())
        with self.assertRaises(RuntimeError):
            env.start_round()
        self.assertEqual(env.run(5), [])

    def test_chips_per_bb(self):
        p = Player("a", 100)
        env = Environment([p, Player("b", 100)], chips_per_bb=10)
        self.assertEqual(p.stack, 100.0)
        self.assertEqual(p.chips, 1000)
        result = env.start_round()
        self.assertAlmostEqual(sum(result.stacks.values()), 200.0)


class RebuyTest(unittest.TestCase):
    def test_rebuy_callback_after_every_hand(self):
        agents = [Rebuyer(), Rebuyer(), Rebuyer()]
        players = [Player(f"p{i}", 100, a) for i, a in enumerate(agents)]
        env = Environment(players, seed=4)
        results = env.run(5)
        self.assertEqual(len(results), 5)
        self.assertEqual([a.calls for a in agents], [5, 5, 5])
        for p in players:
            self.assertGreaterEqual(p.stack, 100.0)
        total_rebuys = sum(p.rebuy_chips for p in players)
        self.assertEqual(sum(p.chips for p in players), 30000 + total_rebuys)

    def test_rebuy_not_allowed_during_hand(self):
        agent = Rebuyer()
        env = Environment([Player("a", 100, agent), Player("b", 100, Passive())])
        env.start_round()
        self.assertIsInstance(agent.in_hand_error, IllegalActionError)

    def test_rebuy_increases_stack(self):
        p = Player("a", 100)
        Environment([p, Player("b", 100)])
        p.rebuy(10)
        self.assertEqual(p.stack, 110.0)
        self.assertEqual(p.rebought, 10.0)
        with self.assertRaises(ValueError):
            p.rebuy(0)
        with self.assertRaises(ValueError):
            p.rebuy(-5)

    def test_max_buy_in(self):
        p = Player("a", 100)
        Environment([p, Player("b", 100)], max_buy_in=150)
        with self.assertRaises(ValueError):
            p.rebuy(60)
        p.rebuy(50)
        self.assertEqual(p.stack, 150.0)

    def test_busted_player_rebuys_and_plays_again(self):
        a = Player("a", 0, Passive())
        b = Player("b", 100, Passive())
        c = Player("c", 100, Passive())
        env = Environment([a, b, c])
        env.start_round()
        self.assertFalse(a.in_hand)
        self.assertIsNone(a.hole_cards)
        a.rebuy(20)
        env.start_round()
        self.assertTrue(a.in_hand)
        self.assertIsNotNone(a.hole_cards)


class RunTest(unittest.TestCase):
    def test_run_stops_when_one_player_has_chips(self):
        # Heads-up with no cap: both players get all-in every hand until one busts.
        players = [Player("a", 5, Aggressive()), Player("b", 5, Aggressive())]
        env = Environment(players, seed=9)
        results = env.run(500)
        self.assertLess(len(results), 500)
        self.assertFalse(env.can_start())
        self.assertEqual(len([p for p in players if p.chips > 0]), 1)
        self.assertEqual(sum(p.chips for p in players), 1000)


if __name__ == "__main__":
    unittest.main()
