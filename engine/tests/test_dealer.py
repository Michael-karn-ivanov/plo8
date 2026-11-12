import random
import unittest

from entities import Board, Card, Deck, Hand
from engine import Environment, EventType, Player, PlayerStatus, Street


class Scripted:
    """Plays codes f/k/c/r in order, then checks/calls (or raises if aggressive).

    Hand numbers in `idle_hands` are played by doing nothing on every turn.
    """

    def __init__(self, script="", aggressive=False, idle_hands=()):
        self.script = list(script)
        self.aggressive = aggressive
        self.idle_hands = set(idle_hands)
        self.turns = 0

    def on_turn(self, player, view, legal):
        self.turns += 1
        if view.hand_number in self.idle_hands:
            return
        if self.script:
            code = self.script.pop(0)
        elif self.aggressive and legal.can_raise:
            code = "r"
        else:
            code = "k" if legal.can_check else "c"
        {"f": player.fold, "k": player.check, "c": player.call, "r": player.raise_}[code]()

    def on_hand_end(self, player, result):
        pass


class RandomAgent:
    def __init__(self, rng, rebuy_below=3.0, rebuy=20.0):
        self.rng = rng
        self.rebuy_below = rebuy_below
        self.rebuy_amount = rebuy

    def on_turn(self, player, view, legal):
        options = [player.check] if legal.can_check else [player.fold, player.call, player.call]
        if legal.can_raise:
            options.append(player.raise_)
        self.rng.choice(options)()

    def on_hand_end(self, player, result):
        if player.stack < self.rebuy_below:
            player.rebuy(self.rebuy_amount)


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
    """Top of deck for hole cards dealt one at a time, then the board."""
    hands = [Hand.parse(h) for h in hands_in_deal_order]
    top = [h.cards[r] for r in range(4) for h in hands]
    return StackedDeck(top + list(Board.parse(board)))


def make_env(stacks, agents, button=0, seed=1, deck=None):
    players = [Player(f"p{i}", s, a) for i, (s, a) in enumerate(zip(stacks, agents))]
    return Environment(players, button=button, seed=seed, deck=deck), players


class EarlyEndTest(unittest.TestCase):
    def test_all_fold_to_big_blind(self):
        agents = [Scripted("f"), Scripted("f"), Scripted()]
        env, players = make_env([100] * 3, agents)
        result = env.start_round()
        self.assertFalse(result.showdown)
        self.assertEqual(result.shown, {})
        self.assertEqual(len(result.board), 0)  # remaining streets skipped
        self.assertEqual(agents[2].turns, 0)
        self.assertEqual(result.net, {0: 0.0, 1: -0.5, 2: 0.5})
        self.assertEqual([p.stack for p in players], [100.0, 99.5, 100.5])
        self.assertEqual(len(result.pots), 1)
        self.assertTrue(result.pots[0].uncontested)
        self.assertEqual(result.pots[0].high, ((2, 1.5),))
        self.assertFalse(any(e.type is EventType.SHOW for e in result.history))
        # Folding is a valid action: nobody sits out.
        self.assertTrue(all(p.status is PlayerStatus.ACTIVE for p in players))

    def test_fold_on_flop(self):
        # Seat 1 bets the flop, everybody else folds: no showdown, no turn.
        agents = [Scripted("cf"), Scripted("cr"), Scripted("kf")]
        env, players = make_env([100] * 3, agents)
        result = env.start_round()
        self.assertFalse(result.showdown)
        self.assertEqual(result.board.street, Street.FLOP)
        self.assertEqual(result.net, {0: -1.0, 1: 2.0, 2: -1.0})


class AllInTest(unittest.TestCase):
    def test_all_in_run_out(self):
        agents = [Scripted(aggressive=True), Scripted(aggressive=True)]
        env, players = make_env([5, 5], agents)
        result = env.start_round()
        self.assertTrue(result.showdown)
        self.assertEqual(len(result.board), 5)
        self.assertEqual(set(result.shown), {0, 1})
        self.assertEqual(sum(p.chips for p in players), 1000)
        self.assertAlmostEqual(sum(result.net.values()), 0.0)

    def test_side_pots(self):
        # Seat 0 all-in for 2 preflop, seat 1 all-in for 5 on the flop, seat 2 covers.
        agents = [Scripted(aggressive=True) for _ in range(3)]
        env, players = make_env([2, 5, 10], agents)
        result = env.start_round()
        self.assertTrue(result.showdown)
        self.assertEqual(len(result.board), 5)
        self.assertEqual([(p.amount, p.eligible) for p in result.pots], [(6.0, (0, 1, 2)), (6.0, (1, 2))])
        self.assertEqual(sum(p.chips for p in players), 1700)
        for pot in result.pots:
            self.assertAlmostEqual(sum(a for _, a in pot.high + pot.low), pot.amount)

    def test_short_big_blind_all_in(self):
        agents = [Scripted(), Scripted(), Scripted()]
        env, players = make_env([100, 100, 0.3], agents)
        result = env.start_round()
        posts = [e for e in result.history if e.type is EventType.POST_BIG_BLIND]
        self.assertEqual(posts[0].amount, 0.3)
        self.assertTrue(posts[0].all_in)
        # Callers still had to match the full big blind.
        calls = [e for e in result.history if e.type is EventType.CALL and e.seat == 0]
        self.assertEqual(calls[0].amount, 1.0)
        self.assertEqual(sum(p.chips for p in players), 20030)


class ShowdownTest(unittest.TestCase):
    def test_hi_lo_split_with_stacked_deck(self):
        # Heads-up, button 0: cards are dealt to seat 1 first.
        deck = stacked(["QhQdTcTs", "Ac2dKsKd"], "4s5d7c Jh Qs")
        env, players = make_env([100, 100], [Scripted(), Scripted()], deck=deck)
        result = env.start_round()
        self.assertEqual(players[0].hole_cards, Hand.parse("Ac2dKsKd"))
        self.assertEqual(players[1].hole_cards, Hand.parse("QhQdTcTs"))
        self.assertEqual(result.board, Board.parse("4s5d7c Jh Qs"))
        self.assertTrue(result.showdown)
        self.assertEqual(result.shown, {0: players[0].hole_cards, 1: players[1].hole_cards})
        self.assertEqual(len(result.pots), 1)
        pot = result.pots[0]
        self.assertEqual(pot.amount, 2.0)
        self.assertEqual(pot.high, ((1, 1.0),))  # trip queens
        self.assertEqual(pot.low, ((0, 1.0),))  # 7-5-4-2-A
        self.assertEqual(result.net, {0: 0.0, 1: 0.0})

    def test_scoop_with_stacked_deck(self):
        deck = stacked(["Ah2dQcJc", "KsKdTcTs"], "3h4h5h 9s 9d")
        env, players = make_env([100, 100], [Scripted(), Scripted()], deck=deck)
        result = env.start_round()
        # Seat 1 has A2 for the wheel (straight) and the nut low; seat 0 KK/TT two pair.
        pot = result.pots[0]
        self.assertEqual(pot.high, ((1, 1.0),))
        self.assertEqual(pot.low, ((1, 1.0),))
        self.assertEqual(result.net, {0: -1.0, 1: 1.0})


class DeckUseTest(unittest.TestCase):
    def test_no_card_dealt_twice(self):
        env, players = make_env([100] * 6, [Scripted() for _ in range(6)], seed=3)
        for _ in range(5):
            result = env.start_round()
            dealt = [c for p in players for c in p.hole_cards] + list(result.board)
            self.assertEqual(len(dealt), 29)
            self.assertEqual(len(set(dealt)), 29)

    def test_same_seed_same_hand(self):
        def play(seed):
            env, players = make_env([100] * 4, [Scripted() for _ in range(4)], seed=seed)
            result = env.start_round()
            return result.board, [p.hole_cards for p in players]

        self.assertEqual(play(123), play(123))
        self.assertNotEqual(play(123), play(124))


class ConservationTest(unittest.TestCase):
    def test_chips_conserved_across_hands(self):
        rng = random.Random(5)
        players = [Player(f"p{i}", 10, RandomAgent(rng)) for i in range(6)]
        env = Environment(players, seed=11)
        initial = sum(p.chips for p in players)
        showdowns = 0
        for _ in range(200):
            result = env.start_round()
            showdowns += result.showdown
            self.assertAlmostEqual(sum(result.net.values()), 0.0)
            rebuys = sum(p.rebuy_chips for p in players)
            self.assertEqual(sum(p.chips for p in players), initial + rebuys)
            self.assertTrue(all(p.chips >= 0 for p in players))
        self.assertGreater(showdowns, 0)

    def test_chips_conserved_with_joins_and_leaves(self):
        rng = random.Random(8)
        players = [Player(f"p{i}", 10, RandomAgent(rng)) for i in range(4)]
        env = Environment(players, seed=2)
        everyone = list(players)
        bought_in = sum(p.chips for p in players)
        for hand in range(60):
            if hand % 10 == 3 and len(env.players) > 2:
                env.players[0].leave()
            if hand % 10 == 7:
                name = f"j{hand}"
                env.join(10, name=name, agent=RandomAgent(rng))
                everyone.append(next(p for p in env.players if p.name == name))
                bought_in += 1000
            if not env.can_start():
                break
            env.start_round()
            rebuys = sum(p.rebuy_chips for p in everyone)
            # Chips at the table plus chips taken away by leavers.
            self.assertEqual(sum(p.chips for p in everyone), bought_in + rebuys)
        self.assertTrue(any(p.status is PlayerStatus.LEFT for p in everyone))


class ButtonTest(unittest.TestCase):
    def test_button_moves(self):
        env, _ = make_env([100] * 3, [Scripted() for _ in range(3)])
        buttons = [env.start_round().button for _ in range(4)]
        self.assertEqual(buttons, [0, 1, 2, 0])
        self.assertEqual(env.button, 1)

    def test_sitting_out_player_is_skipped(self):
        # Seat 1 (the SB of hand 1) does not act and sits out from hand 2.
        agents = [Scripted(), Scripted(idle_hands={1}), Scripted()]
        env, players = make_env([100] * 3, agents)
        env.start_round()
        self.assertIs(players[1].status, PlayerStatus.SITTING_OUT)
        self.assertEqual(env.button, 2)  # seat 1 skipped
        result = env.start_round()
        self.assertEqual(result.button, 2)
        self.assertEqual((result.sb_seat, result.bb_seat), (2, 0))  # heads-up
        self.assertIsNone(players[1].hole_cards)
        self.assertFalse(players[1].in_hand)
        self.assertEqual(env.button, 0)  # seat 1 skipped again

    def test_button_on_sitting_out_seat_moves_to_next_active(self):
        # Hand 1: seat 1 holds nothing special; it sits out and the button skips it.
        agents = [Scripted(), Scripted(idle_hands={1}), Scripted()]
        env, players = make_env([100] * 3, agents, button=0)
        env.start_round()
        # Force the button onto the sitting-out seat (e.g. after table changes).
        env.dealer.button = 1
        result = env.start_round()
        self.assertEqual(result.button, 2)


if __name__ == "__main__":
    unittest.main()
