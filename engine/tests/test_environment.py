import unittest

from engine import (
    DEFAULT_ACTION_TIMEOUT,
    Environment,
    IllegalActionError,
    NotEnoughPlayersError,
    Player,
    PlayerStatus,
    Street,
    TableFullError,
)


class Passive:
    def on_turn(self, player, view, legal):
        if legal.can_check:
            player.check()
        else:
            player.call()

    def on_hand_end(self, player, result):
        pass


class Idle(Passive):
    """Does nothing on its turns (and so sits out after the hand)."""

    def on_turn(self, player, view, legal):
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
        self.assertEqual(first.dealt_in, {0: True, 1: True, 2: True})

    def test_seat_limits(self):
        # A table may start with fewer than two players (others join later), but a
        # hand needs at least two.
        env = Environment([Player("a", 100)])
        self.assertFalse(env.can_start())
        with self.assertRaises(NotEnoughPlayersError):
            env.start_round()
        self.assertTrue(issubclass(NotEnoughPlayersError, RuntimeError))
        empty = Environment()
        self.assertEqual(len(empty.players), 0)
        self.assertIsNone(empty.button)
        with self.assertRaises(NotEnoughPlayersError):
            empty.start_round()
        with self.assertRaises(TableFullError):
            Environment([Player(str(i), 100) for i in range(11)])
        with self.assertRaises(TableFullError):
            Environment([Player(str(i), 100) for i in range(4)], max_seats=3)
        with self.assertRaises(ValueError):
            Environment(max_seats=11)
        with self.assertRaises(ValueError):
            Environment(max_seats=1)
        Environment([Player(str(i), 100) for i in range(10)])

    def test_initial_players_need_a_positive_stack(self):
        with self.assertRaises(ValueError):
            Environment([Player("a", 100), Player("b", 0)])

    def test_initial_players_are_active(self):
        players = [Player("a", 100), Player("b", 100)]
        env = Environment(players)
        self.assertTrue(all(p.status is PlayerStatus.ACTIVE for p in players))
        self.assertEqual(env.active_players, players)

    def test_needs_two_active_players(self):
        # Seat 0 does not act in hand 1 and sits out: one active player is left.
        players = [Player("a", 100, Idle()), Player("b", 100, Passive())]
        env = Environment(players)
        env.start_round()
        self.assertIs(players[0].status, PlayerStatus.SITTING_OUT)
        self.assertGreater(players[0].chips, 0)  # sitting-out players do not count
        self.assertFalse(env.can_start())
        with self.assertRaises(NotEnoughPlayersError):
            env.start_round()
        self.assertEqual(env.run(5), [])

    def test_chips_per_bb(self):
        p = Player("a", 100)
        env = Environment([p, Player("b", 100)], chips_per_bb=10)
        self.assertEqual(p.stack, 100.0)
        self.assertEqual(p.chips, 1000)
        result = env.start_round()
        self.assertAlmostEqual(sum(result.stacks.values()), 200.0)

    def test_action_timeout(self):
        self.assertEqual(Environment().action_timeout, DEFAULT_ACTION_TIMEOUT)
        self.assertEqual(DEFAULT_ACTION_TIMEOUT, 600.0)
        self.assertEqual(Environment(action_timeout=2.5).action_timeout, 2.5)
        for bad in (0, -1, "10", True):
            with self.assertRaises(ValueError):
                Environment(action_timeout=bad)


class NameTest(unittest.TestCase):
    def test_default_names(self):
        players = [Player(None, 100), Player("x", 100), Player(stack=100)]
        env = Environment(players)
        self.assertEqual([p.name for p in env.players], ["Player1", "x", "Player2"])
        result = env.start_round()
        self.assertEqual(result.names, {0: "Player1", 1: "x", 2: "Player2"})

    def test_default_name_skips_taken(self):
        env = Environment([Player("Player1", 100), Player(None, 100)])
        self.assertEqual(env.players[1].name, "Player2")

    def test_duplicate_names_rejected(self):
        with self.assertRaises(ValueError):
            Environment([Player("a", 100), Player("a", 100)])

    def test_invalid_name(self):
        with self.assertRaises(ValueError):
            Player("", 100)

    def test_views_show_names(self):
        seen = []

        class Look(Passive):
            def on_turn(self, player, view, legal):
                seen.append(tuple(s.name for s in view.seats))
                super().on_turn(player, view, legal)

        env = Environment([Player("ann", 100, Look()), Player("bob", 100, Look())])
        env.start_round()
        self.assertTrue(seen)
        self.assertTrue(all(names == ("ann", "bob") for names in seen))


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

    def test_rebuy_not_allowed_during_betting(self):
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

    def test_left_player_cannot_rebuy(self):
        p = Player("a", 100)
        env = Environment([p, Player("b", 100), Player("c", 100)])
        p.leave()
        with self.assertRaises(IllegalActionError):
            p.rebuy(10)
        with self.assertRaises(IllegalActionError):
            p.sit_in()
        self.assertEqual(len(env.players), 2)


class RunTest(unittest.TestCase):
    def test_run_stops_when_one_player_has_chips(self):
        # Heads-up with no cap: both players get all-in every hand until one busts.
        players = [Player("a", 5, Aggressive()), Player("b", 5, Aggressive())]
        env = Environment(players, seed=9)
        results = env.run(500)
        self.assertLess(len(results), 500)
        self.assertFalse(env.can_start())
        self.assertEqual(len([p for p in players if p.chips > 0]), 1)
        busted = [p for p in players if p.chips == 0][0]
        self.assertIs(busted.status, PlayerStatus.SITTING_OUT)
        self.assertEqual(sum(p.chips for p in players), 1000)


if __name__ == "__main__":
    unittest.main()
