import random
import unittest

from entities import Board, Deck, Hand
from engine import (
    ActionType,
    BlindPosted,
    Environment,
    IllegalActionError,
    NotEnoughPlayersError,
    Player,
    PlayerActed,
    PlayerReturned,
    PlayerSatOut,
    PlayerStatus,
)


class Bot:
    """Checks/calls. Does nothing on its turns in `idle_hands`; rebuys
    `rebuy_at[hand]` and calls sit_in() for hands in `sit_in_at` in on_hand_end."""

    def __init__(self, idle_hands=(), rebuy_at=None, sit_in_at=()):
        self.idle_hands = set(idle_hands)
        self.rebuy_at = dict(rebuy_at or {})
        self.sit_in_at = set(sit_in_at)
        self.turns = []  # hand numbers of the turns
        self.views = []
        self.hand_ends = []  # (hand_number, dealt_in)
        self.sat_out = 0
        self.returned = 0
        self.events = []
        self.errors = []

    def on_turn(self, player, view, legal):
        self.turns.append(view.hand_number)
        self.views.append(view)
        if view.hand_number in self.idle_hands:
            return
        if legal.can_check:
            player.check()
        else:
            player.call()

    def on_hand_end(self, player, result):
        self.hand_ends.append((result.hand_number, result.dealt_in[player.seat]))
        amount = self.rebuy_at.get(result.hand_number)
        if amount:
            player.rebuy(amount)
        if result.hand_number in self.sit_in_at:
            player.sit_in()

    def on_sit_out(self, player):
        self.sat_out += 1

    def on_return(self, player):
        self.returned += 1

    def on_event(self, player, event):
        self.events.append(event)


class StackedDeck(Deck):
    """A deck whose top cards are fixed after every shuffle."""

    def __init__(self, top):
        super().__init__(random.Random(0))
        self.top = list(top)
        self.queue = []

    def shuffle(self):
        super().shuffle()
        self.remove(self.top)
        self.queue = list(self.top)

    def deal(self, n=1):
        out = []
        while self.queue and len(out) < n:
            out.append(self.queue.pop(0))
        if len(out) < n:
            out += super().deal(n - len(out))
        return out


def stacked(hands_in_deal_order, board):
    hands = [Hand.parse(h) for h in hands_in_deal_order]
    top = [h.cards[r] for r in range(4) for h in hands]
    return StackedDeck(top + list(Board.parse(board)))


def bust_env(agent2):
    """Hand 1 (button 0): p2 posts a 1 BB big blind all-in and loses it.

    Cards are dealt starting with seat 1: p1 9c9s5c5s, p2 KhKdQcQh,
    p0 AcAs3d4d. On 2s7d8h Tc Ts p0 scoops (aces up and the only low).
    """
    deck = stacked(["9c9s5c5s", "KhKdQcQh", "AcAs3d4d"], "2s7d8h Tc Ts")
    agents = [Bot(), Bot(), agent2]
    players = [Player("p0", 100, agents[0]), Player("p1", 100, agents[1]), Player("p2", 1, agent2)]
    return Environment(players, deck=deck, seed=1), players, agents


def events_of(agent, event_type):
    return [e for e in agent.events if isinstance(e, event_type)]


class BustedTest(unittest.TestCase):
    def test_busted_player_sits_out(self):
        env, players, agents = bust_env(Bot())
        p2 = players[2]
        first = env.start_round()
        self.assertEqual(first.stacks[2], 0.0)
        self.assertEqual(first.dealt_in, {0: True, 1: True, 2: True})
        self.assertIs(p2.status, PlayerStatus.SITTING_OUT)
        self.assertEqual(agents[2].sat_out, 1)
        sat = events_of(agents[0], PlayerSatOut)
        self.assertEqual([(e.name, e.seat) for e in sat], [("p2", 2)])
        self.assertEqual(env.button, 1)
        self.assertEqual(env.active_players, players[:2])

        second = env.start_round()
        # Not dealt, no blinds, no turns, not in the pots; still in the result.
        self.assertEqual(second.dealt_in, {0: True, 1: True, 2: False})
        self.assertEqual(second.net[2], 0.0)
        self.assertEqual(second.names[2], "p2")
        self.assertEqual(second.button, 1)
        self.assertNotIn(2, (second.sb_seat, second.bb_seat))
        self.assertIsNone(p2.hole_cards)
        self.assertFalse(any(getattr(e, "seat", None) == 2 for e in second.history))
        self.assertTrue(all(2 not in pot.eligible for pot in second.pots))
        self.assertNotIn(2, agents[2].turns)
        view = [v for v in agents[0].views if v.hand_number == 2][0]
        self.assertIs(view.seats[2].status, PlayerStatus.SITTING_OUT)
        self.assertTrue(view.seats[2].sitting_out)
        self.assertFalse(view.seats[2].in_hand)

        third = env.start_round()
        self.assertEqual(third.button, 0)  # the button skipped seat 2
        self.assertEqual(agents[2].hand_ends, [(1, True), (2, False), (3, False)])
        self.assertEqual(agents[2].sat_out, 1)  # notified once
        # While sitting out it still receives the public events of later hands.
        self.assertTrue(any(e.hand_no == 3 for e in agents[2].events))
        self.assertEqual(
            [e.seq for e in agents[2].events if e.hand_no == 3],
            [e.seq for e in agents[0].events if e.hand_no == 3],
        )
        self.assertEqual(sum(p.chips for p in players), 20100)

    def test_rebuy_in_hand_end_returns_player(self):
        env, players, agents = bust_env(Bot(rebuy_at={2: 50}))
        p2 = players[2]
        env.start_round()
        self.assertIs(p2.status, PlayerStatus.SITTING_OUT)
        env.start_round()
        self.assertIs(p2.status, PlayerStatus.ACTIVE)
        self.assertEqual(p2.stack, 50.0)
        self.assertEqual(agents[2].returned, 1)
        back = events_of(agents[0], PlayerReturned)
        self.assertEqual([(e.name, e.seat) for e in back], [("p2", 2)])

        third = env.start_round()
        self.assertTrue(third.dealt_in[2])
        self.assertIsNotNone(p2.hole_cards)
        self.assertIn(3, agents[2].turns)
        posts = [e.seat for e in third.history if isinstance(e, BlindPosted)]
        self.assertEqual(posts, [third.sb_seat, third.bb_seat])  # no extra blinds
        self.assertEqual(agents[2].hand_ends, [(1, True), (2, False), (3, True)])

    def test_rebuy_between_hands_returns_player(self):
        env, players, agents = bust_env(Bot())
        p2 = players[2]
        env.start_round()
        self.assertIs(p2.status, PlayerStatus.SITTING_OUT)
        p2.rebuy(20)
        self.assertIs(p2.status, PlayerStatus.ACTIVE)
        self.assertEqual(agents[2].returned, 1)
        result = env.start_round()
        self.assertTrue(result.dealt_in[2])
        self.assertIsNotNone(p2.hole_cards)

    def test_immediate_rebuy_never_sits_out(self):
        env, players, agents = bust_env(Bot(rebuy_at={1: 100}))
        p2 = players[2]
        env.start_round()
        self.assertIs(p2.status, PlayerStatus.ACTIVE)
        self.assertEqual(p2.stack, 100.0)
        self.assertEqual(agents[2].sat_out, 0)
        self.assertEqual(events_of(agents[0], PlayerSatOut), [])
        result = env.start_round()
        self.assertTrue(result.dealt_in[2])

    def test_busted_player_cannot_sit_in_without_chips(self):
        env, players, _ = bust_env(Bot())
        env.start_round()
        with self.assertRaises(IllegalActionError):
            players[2].sit_in()
        self.assertIs(players[2].status, PlayerStatus.SITTING_OUT)

    def test_sitting_out_player_cannot_rebuy_during_betting(self):
        errors = []

        class RebuyOther(Bot):
            def on_turn(self, player, view, legal):
                if view.hand_number == 2:
                    try:
                        sitting_out.rebuy(10)
                    except IllegalActionError as e:
                        errors.append(e)
                super().on_turn(player, view, legal)

        deck = stacked(["9c9s5c5s", "KhKdQcQh", "AcAs3d4d"], "2s7d8h Tc Ts")
        players = [Player("p0", 100, RebuyOther()), Player("p1", 100, Bot()), Player("p2", 1, Bot())]
        sitting_out = players[2]
        env = Environment(players, deck=deck, seed=1)
        env.start_round()
        env.start_round()
        self.assertTrue(errors)
        self.assertIs(sitting_out.status, PlayerStatus.SITTING_OUT)
        self.assertEqual(sitting_out.chips, 0)


class NoActionTest(unittest.TestCase):
    def test_idle_agent_folds_and_sits_out_until_sit_in(self):
        agents = [Bot(), Bot(), Bot(idle_hands={1, 2, 3, 4}, sit_in_at={2})]
        players = [Player(f"p{i}", 100, a) for i, a in enumerate(agents)]
        env = Environment(players, seed=1)
        p2 = players[2]

        # Hand 1: the big blind could check, but does nothing -> folded.
        first = env.start_round()
        folds = [
            e
            for e in first.history
            if isinstance(e, PlayerActed) and e.seat == 2 and e.action is ActionType.FOLD
        ]
        self.assertEqual(len(folds), 1)
        self.assertEqual(folds[0].reason, "timeout")
        self.assertEqual(first.net[2], -1.0)
        self.assertIs(p2.status, PlayerStatus.SITTING_OUT)
        self.assertEqual(p2.stack, 99.0)  # keeps its chips
        self.assertEqual(agents[2].sat_out, 1)

        # Hand 2: not dealt in, still gets on_hand_end and sits in there.
        second = env.start_round()
        self.assertFalse(second.dealt_in[2])
        self.assertIs(p2.status, PlayerStatus.ACTIVE)
        self.assertEqual(agents[2].returned, 1)

        # Hand 3: dealt in again (and idle again, so it sits out again).
        third = env.start_round()
        self.assertTrue(third.dealt_in[2])
        self.assertIs(p2.status, PlayerStatus.SITTING_OUT)
        self.assertEqual(agents[2].turns.count(2), 0)
        self.assertEqual(set(agents[2].turns), {1, 3})

        # Without sit_in it stays out: no more turns, so no more timeouts.
        env.run(3)
        self.assertEqual(set(agents[2].turns), {1, 3})
        self.assertEqual([h for h, _ in agents[2].hand_ends], [1, 2, 3, 4, 5, 6])
        self.assertEqual(agents[2].sat_out, 2)

    def test_sit_in_between_hands(self):
        agents = [Bot(), Bot(), Bot(idle_hands={1})]
        players = [Player(f"p{i}", 100, a) for i, a in enumerate(agents)]
        env = Environment(players, seed=1)
        env.start_round()
        self.assertIs(players[2].status, PlayerStatus.SITTING_OUT)
        players[2].sit_in()
        self.assertIs(players[2].status, PlayerStatus.ACTIVE)
        self.assertEqual(agents[2].returned, 1)
        players[2].sit_in()  # no-op for an active player
        self.assertEqual(agents[2].returned, 1)
        result = env.start_round()
        self.assertTrue(result.dealt_in[2])

    def test_sit_in_not_allowed_during_betting(self):
        errors = []

        class SitInOnTurn(Bot):
            def on_turn(self, player, view, legal):
                try:
                    player.sit_in()
                except IllegalActionError as e:
                    errors.append(e)
                super().on_turn(player, view, legal)

        env = Environment([Player("a", 100, SitInOnTurn()), Player("b", 100, Bot())])
        env.start_round()
        self.assertTrue(errors)

    def test_waiting_player_cannot_sit_in(self):
        p = Player("w", 10)
        with self.assertRaises(IllegalActionError):
            p.sit_in()


class RunTest(unittest.TestCase):
    def test_run_stops_with_fewer_than_two_active_players(self):
        # Heads-up: the button (SB) acts first preflop, does nothing and sits out.
        players = [Player("a", 100, Bot(idle_hands={1})), Player("b", 100, Bot())]
        env = Environment(players, seed=1)
        results = env.run(5)
        self.assertEqual(len(results), 1)
        self.assertIs(players[0].status, PlayerStatus.SITTING_OUT)
        self.assertFalse(env.can_start())
        with self.assertRaises(NotEnoughPlayersError):
            env.start_round()

    def test_run_continues_after_rebuy(self):
        env, players, _ = bust_env(Bot(rebuy_at={2: 30}))
        results = env.run(3)
        self.assertEqual(len(results), 3)
        self.assertEqual([r.dealt_in[2] for r in results], [True, False, True])


if __name__ == "__main__":
    unittest.main()
