"""Replaying hands: a caller-supplied deck and chip unit (see plan.md, Dealer ->
Deck use and deal order, and Units / chips)."""

import contextlib
import random
import unittest

from entities import Board, Card, Deck, Hand
from engine import (
    Action,
    ActionType,
    BlindPosted,
    BoardDealt,
    Environment,
    HandStarted,
    Player,
    PlayerActed,
    PlayerStatus,
    PotAwarded,
    Street,
)

ALL = [Card.from_index(i) for i in range(52)]


def random_script(seed, k=52):
    """A fixed, seed-dependent order of `k` distinct cards."""
    return random.Random(seed).sample(ALL, k)


class Scripted:
    """Plays codes f/k/c/r in order, then checks/calls. Does nothing on its turns
    in `idle_hands` (and so sits out after that hand)."""

    def __init__(self, script="", idle_hands=()):
        self.script = list(script)
        self.idle_hands = set(idle_hands)

    def on_turn(self, player, view, legal):
        if view.hand_number in self.idle_hands:
            return
        if self.script:
            code = self.script.pop(0)
        else:
            code = "k" if legal.can_check else "c"
        {"f": player.fold, "k": player.check, "c": player.call, "r": player.raise_}[code]()

    def on_hand_end(self, player, result):
        pass


class Aggressive(Scripted):
    def on_turn(self, player, view, legal):
        if legal.can_raise:
            player.raise_()
        else:
            super().on_turn(player, view, legal)


class Recorder(Scripted):
    """Checks/calls and records every public event it receives."""

    def __init__(self, script=""):
        super().__init__(script)
        self.events = []

    def on_event(self, player, event):
        self.events.append(event)


class StackedDeck(Deck):
    """Stacked deck as a replaying caller builds it: `shuffle()` shuffles, then
    puts the next script on top. A None script (or no script left) leaves the
    shuffled deck as is."""

    def __init__(self, scripts, seed=0):
        super().__init__(random.Random(seed))
        self.scripts = list(scripts)
        self.shuffles = 0

    def shuffle(self):
        super().shuffle()
        self.shuffles += 1
        script = self.scripts.pop(0) if self.scripts else None
        if script:
            self.put_on_top(script)


class RecordingDeck(Deck):
    """Records the public Deck methods the dealer calls (not the nested ones)."""

    def __init__(self, seed=0):
        super().__init__(random.Random(seed))
        self.calls = []
        self._depth = 0

    @contextlib.contextmanager
    def _call(self, name):
        if self._depth == 0:
            self.calls.append(name)
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1

    def reset(self):
        with self._call("reset"):
            return super().reset()

    def shuffle(self):
        with self._call("shuffle"):
            return super().shuffle()

    def deal(self, n=1):
        with self._call("deal"):
            return super().deal(n)

    def deal_one(self):
        with self._call("deal_one"):
            return super().deal_one()

    def deal_hand(self):
        with self._call("deal_hand"):
            return super().deal_hand()

    def remove(self, cards):
        with self._call("remove"):
            return super().remove(cards)

    def put_on_top(self, cards):
        with self._call("put_on_top"):
            return super().put_on_top(cards)

    def remaining(self):
        with self._call("remaining"):
            return super().remaining()


class RecordingStackedDeck(RecordingDeck):
    """A recording deck whose `shuffle()` stacks `script` on top every hand."""

    def __init__(self, script, seed=0):
        super().__init__(seed)
        self.script = list(script)

    def shuffle(self):
        with self._call("shuffle"):
            Deck.shuffle(self)
            self.put_on_top(self.script)


def make_env(agents, stacks=None, button=0, deck=None, chips_per_bb=100):
    stacks = stacks or [100] * len(agents)
    players = [Player(f"p{i}", s, a) for i, (s, a) in enumerate(zip(stacks, agents))]
    env = Environment(players, button=button, deck=deck, chips_per_bb=chips_per_bb, seed=99)
    return env, players


def expected_deal(script, result, n_seats):
    """Hole cards per seat and the full board from the deal-order formula: with n
    participants counted from the left of the button, deck card i < 4n goes to
    participant i % n as hole card i // n; then flop (3), turn, river."""
    button = result.button
    parts = sorted(
        (s for s, dealt in result.dealt_in.items() if dealt),
        key=lambda s: (s - button - 1) % n_seats,
    )
    n = len(parts)
    hands = {seat: Hand(script[k * n + j] for k in range(4)) for j, seat in enumerate(parts)}
    b = script[4 * n : 4 * n + 5]
    board = Board(flop=tuple(b[:3]), turn=b[3], river=b[4])
    return hands, board


def policy(seat, legal):
    """A fixed, rng-free choice used in both modes: seat 1 raises whenever it
    can, seat 3 folds to any bet, the others check / call."""
    if seat == 1 and legal.can_raise:
        return ActionType.RAISE
    if legal.can_check:
        return ActionType.CHECK
    if seat == 3:
        return ActionType.FOLD
    return ActionType.CALL


class PolicyAgent(Recorder):
    """Callback-mode player of `policy`."""

    def on_turn(self, player, view, legal):
        player.act(Action.of(policy(player.seat, legal)))


class DeckContractTest(unittest.TestCase):
    def test_reset_then_shuffle_once_per_hand_then_only_deal_one(self):
        deck = RecordingDeck()
        snapshots = []

        class Watcher(Scripted):
            def on_event(self, player, event):
                if isinstance(event, HandStarted) and player.seat == 0:
                    snapshots.append(list(deck.calls))

        env, players = make_env([Watcher(), Scripted(), Scripted()], deck=deck)
        self.assertIs(env.deck, deck)
        self.assertIs(env.dealer.deck, deck)

        start = 0
        for hand in range(3):
            result = env.start_round()
            self.assertIs(env.dealer.deck, deck)  # same object for every hand
            calls = deck.calls[start:]
            self.assertEqual(calls[:2], ["reset", "shuffle"])
            self.assertEqual(set(calls[2:]), {"deal_one"})
            self.assertEqual(len(calls) - 2, 4 * 3 + len(result.board))
            # The deck was prepared before HandStarted, and nothing dealt yet.
            self.assertEqual(snapshots[hand][start:], ["reset", "shuffle"])
            start = len(deck.calls)
        self.assertEqual(deck.calls.count("reset"), 3)
        self.assertEqual(deck.calls.count("shuffle"), 3)

    def test_step_mode_uses_the_deck_the_same_way(self):
        deck = RecordingDeck()
        env, players = make_env([None, None, None], deck=deck)
        start = 0
        for hand in range(3):
            d = env.start_hand()
            self.assertEqual(deck.calls[start : start + 2], ["reset", "shuffle"])
            while d is not None:
                d = env.step(ActionType.CHECK if d.legal.can_check else ActionType.CALL)
            result = env.last_result
            self.assertIs(env.dealer.deck, deck)
            calls = deck.calls[start:]
            self.assertEqual(calls[:2], ["reset", "shuffle"])
            self.assertEqual(set(calls[2:]), {"deal_one"})
            self.assertEqual(len(calls) - 2, 4 * 3 + len(result.board))
            start = len(deck.calls)

    def test_seed_not_used_with_supplied_deck(self):
        def play(seed):
            deck = StackedDeck([None], seed=7)
            players = [Player(f"p{i}", 100, Scripted()) for i in range(3)]
            env = Environment(players, deck=deck, seed=seed)
            result = env.start_round()
            return result.board, [p.hole_cards for p in players]

        self.assertEqual(play(1), play(2))


class StackedDeckTest(unittest.TestCase):
    def assert_dealt(self, env, players, script, result):
        hands, board = expected_deal(script, result, len(env.players))
        for p in players:
            if result.dealt_in[p.seat]:
                self.assertEqual(p.hole_cards, hands[p.seat], f"seat {p.seat}")
            else:
                self.assertIsNone(p.hole_cards)
        self.assertEqual(result.board, board)
        self.assertEqual(result.board.flop, tuple(board.flop))

    def test_heads_up_bb_gets_the_first_card(self):
        script = random_script(1)
        deck = StackedDeck([script])
        env, players = make_env([Scripted(), Scripted()], button=1, deck=deck)
        result = env.start_round()
        self.assertEqual((result.button, result.sb_seat, result.bb_seat), (1, 1, 0))
        # BB (seat 0, left of the button) gets card 0, the button card 1.
        self.assertEqual(players[0].hole_cards.cards[0], script[0])
        self.assertEqual(players[1].hole_cards.cards[0], script[1])
        self.assertEqual(players[0].hole_cards, Hand(script[0:8:2]))
        self.assertEqual(players[1].hole_cards, Hand(script[1:8:2]))
        self.assertEqual(result.board, Board(tuple(script[8:11]), script[11], script[12]))
        self.assert_dealt(env, players, script, result)

    def test_three_players_button_not_on_seat_zero(self):
        script = random_script(2)
        env, players = make_env([Scripted() for _ in range(3)], button=2, deck=StackedDeck([script]))
        result = env.start_round()
        self.assertEqual(result.button, 2)
        # Order from the left of the button: seats 0, 1, 2.
        self.assertEqual(players[0].hole_cards, Hand(script[0:12:3]))
        self.assertEqual(players[2].hole_cards, Hand(script[2:12:3]))
        self.assert_dealt(env, players, script, result)

    def test_six_players(self):
        script = random_script(3)
        env, players = make_env([Scripted() for _ in range(6)], button=3, deck=StackedDeck([script]))
        result = env.start_round()
        self.assertTrue(result.showdown)
        # Seat 4 is first left of the button, the button (seat 3) is last.
        self.assertEqual(players[4].hole_cards.cards[0], script[0])
        self.assertEqual(players[3].hole_cards.cards[0], script[5])
        self.assertEqual(players[3].hole_cards.cards[3], script[23])
        self.assertEqual(list(result.board), script[24:29])
        self.assert_dealt(env, players, script, result)

    def test_sitting_out_seat_is_skipped(self):
        # Hand 1 (unstacked): seat 1 does not act and sits out. Hand 2 is stacked.
        script = random_script(4)
        deck = StackedDeck([None, script])
        agents = [Scripted(), Scripted(idle_hands={1}), Scripted(), Scripted()]
        env, players = make_env(agents, button=0, deck=deck)
        env.start_round()
        self.assertIs(players[1].status, PlayerStatus.SITTING_OUT)
        result = env.start_round()
        self.assertFalse(result.dealt_in[1])
        self.assertEqual(result.button, 2)
        # Participants from the left of the button: seats 3, 0, 2 (seat 1 skipped).
        self.assertEqual(players[3].hole_cards, Hand(script[0:12:3]))
        self.assertEqual(players[0].hole_cards, Hand(script[1:12:3]))
        self.assertEqual(players[2].hole_cards, Hand(script[2:12:3]))
        self.assertEqual(list(result.board), script[12:17])
        self.assert_dealt(env, players, script, result)

    def test_all_in_run_out(self):
        script = random_script(5)
        env, players = make_env(
            [Aggressive(), Aggressive()], stacks=[2, 2], deck=StackedDeck([script])
        )
        result = env.start_round()
        self.assertTrue(all(p.all_in for p in players))
        dealt = [e for e in result.history if isinstance(e, BoardDealt)]
        self.assertEqual([e.street for e in dealt], [Street.FLOP, Street.TURN, Street.RIVER])
        self.assertEqual(dealt[0].cards, tuple(script[8:11]))
        self.assertEqual(dealt[1].cards, (script[11],))
        self.assertEqual(dealt[2].cards, (script[12],))
        self.assert_dealt(env, players, script, result)

    def test_hand_ending_preflop_deals_no_board(self):
        script = random_script(6)
        deck = StackedDeck([script])
        env, players = make_env([Scripted("f"), Scripted("f"), Scripted()], deck=deck)
        result = env.start_round()
        self.assertEqual(len(result.board), 0)
        self.assertFalse(any(isinstance(e, BoardDealt) for e in result.history))
        # The would-be board is still on top of the deck.
        self.assertEqual(len(deck), 52 - 12)
        self.assertEqual(list(deck.remaining()[:5]), script[12:17])
        self.assertEqual(players[1].hole_cards, Hand(script[0:12:3]))

    def test_each_hand_deals_its_own_script(self):
        scripts = [random_script(10 + k) for k in range(3)]
        deck = StackedDeck(scripts)
        env, players = make_env([Scripted() for _ in range(3)], deck=deck)
        for k in range(3):
            result = env.start_round()
            self.assertEqual(result.button, k)
            self.assert_dealt(env, players, scripts[k], result)
        self.assertEqual(deck.shuffles, 3)

    def test_step_mode_deals_the_script(self):
        script = random_script(7)
        env, players = make_env([None] * 6, button=3, deck=StackedDeck([script]))
        d = env.start_hand()
        while d is not None:
            d = env.step(ActionType.CHECK if d.legal.can_check else ActionType.CALL)
        result = env.last_result
        self.assertTrue(result.showdown)
        self.assert_dealt(env, players, script, result)


# Pot of 7 chips at chips_per_bb=2: seat 1 (SB) posts 1 chip and folds, the other
# three put in 2 each and check down. Seat 3 has quad kings (high, no low); seats
# 2 and 0 tie for the low (wheel). Order from the left of the button: 1, 2, 3, 0.
ODD_HANDS = {0: "As2sQcJd", 1: "6c7c8c9h", 2: "Ah2hQhJs", 3: "KcKh9s9d"}
ODD_BOARD = "3h4d5c Kd Ks"


def odd_chip_script():
    order = [1, 2, 3, 0]
    hands = {s: Hand.parse(h).cards for s, h in ODD_HANDS.items()}
    return [hands[s][r] for r in range(4) for s in order] + list(Board.parse(ODD_BOARD))


class ChipUnitTest(unittest.TestCase):
    def test_blinds_and_bet_sizes_scale(self):
        units = []

        class Raiser(Scripted):
            """Raises once per street when allowed, otherwise checks/calls."""

            def __init__(self):
                super().__init__()
                self.raised = set()

            def on_turn(self, player, view, legal):
                units.append((view.street, legal.bet_unit))
                if legal.can_raise and view.street not in self.raised:
                    self.raised.add(view.street)
                    player.raise_()
                else:
                    super().on_turn(player, view, legal)

        env, players = make_env([Raiser(), Raiser()], stacks=[20, 20], chips_per_bb=2)
        self.assertEqual(env.chips_per_bb, 2)
        self.assertEqual((env.dealer.small_blind, env.dealer.big_blind), (1, 2))
        self.assertEqual([p.chips for p in players], [40, 40])
        result = env.start_round()

        blinds = [(e.blind, e.amount_bb) for e in result.history if isinstance(e, BlindPosted)]
        self.assertEqual(blinds, [("SB", 0.5), ("BB", 1.0)])
        raises = [
            (e.street, e.to_bb)
            for e in result.history
            if isinstance(e, PlayerActed) and e.action is ActionType.RAISE
        ]
        self.assertEqual(
            raises,
            [
                (Street.PREFLOP, 2.0),
                (Street.PREFLOP, 3.0),
                (Street.FLOP, 1.0),
                (Street.FLOP, 2.0),
                (Street.TURN, 2.0),
                (Street.TURN, 4.0),
                (Street.RIVER, 2.0),
                (Street.RIVER, 4.0),
            ],
        )
        # In chips: a small bet is 2 chips, a big bet 4.
        self.assertEqual({env.dealer.to_chips(u) for s, u in units if s in (Street.PREFLOP, Street.FLOP)}, {2})
        self.assertEqual({env.dealer.to_chips(u) for s, u in units if s in (Street.TURN, Street.RIVER)}, {4})
        self.assertEqual(sum(p.chips for p in players), 80)
        for p in players:
            self.assertEqual(result.stacks[p.seat], p.chips / 2)

    def check_odd_chips(self, chips_per_bb, high, low_first, low_second):
        deck = StackedDeck([odd_chip_script()])
        agents = [Scripted(), Scripted("f"), Scripted(), Scripted()]
        env, players = make_env(agents, stacks=[10] * 4, deck=deck, chips_per_bb=chips_per_bb)
        result = env.start_round()
        self.assertEqual(result.board, Board.parse(ODD_BOARD))
        self.assertEqual(players[3].hole_cards, Hand.parse(ODD_HANDS[3]))
        self.assertTrue(result.showdown)

        cpb = chips_per_bb
        pot_chips = cpb // 2 + 3 * cpb
        self.assertEqual(high + low_first + low_second, pot_chips)
        self.assertEqual(len(result.pots), 1)
        pot = result.pots[0]
        self.assertEqual(pot.amount, pot_chips / cpb)
        self.assertEqual(pot.high, ((3, high / cpb),))
        self.assertEqual(pot.low, ((2, low_first / cpb), (0, low_second / cpb)))
        awards = [
            (e.seat, e.share, env.dealer.to_chips(e.amount_bb))
            for e in result.history
            if isinstance(e, PotAwarded)
        ]
        self.assertEqual(awards, [(3, "high", high), (2, "low", low_first), (0, "low", low_second)])

        start = 10 * cpb
        expected_chips = {
            0: start - cpb + low_second,
            1: start - cpb // 2,
            2: start - cpb + low_first,
            3: start - cpb + high,
        }
        self.assertEqual({p.seat: p.chips for p in players}, expected_chips)
        for p in players:
            self.assertEqual(result.stacks[p.seat], p.chips / cpb)
            self.assertEqual(p.stack, p.chips / cpb)
        for e in result.history:
            if hasattr(e, "stack_bb"):
                self.assertEqual(e.stack_bb * cpb, round(e.stack_bb * cpb))

    def test_odd_chips_with_two_chips_per_bb(self):
        # Pot 7: odd chip of the halving to high (4 / 3); low 3 split 2 / 1, the
        # odd chip to seat 2 (first left of the button among the low winners).
        self.check_odd_chips(2, high=4, low_first=2, low_second=1)

    def test_odd_chips_with_ten_chips_per_bb(self):
        # Pot 35: high 18, low 17 split 9 / 8.
        self.check_odd_chips(10, high=18, low_first=9, low_second=8)


class ReplayDeterminismTest(unittest.TestCase):
    def test_same_script_same_hand(self):
        script = random_script(42)

        def play():
            deck = StackedDeck([script], seed=random.randrange(1 << 30))
            recorder = Recorder("crc")
            agents = [recorder, Scripted("rcf"), Scripted("cc"), Scripted("f")]
            env, _ = make_env(agents, stacks=[30, 25, 40, 10], button=1, deck=deck, chips_per_bb=2)
            result = env.start_round()
            return recorder.events, result

        events_a, result_a = play()
        events_b, result_b = play()
        self.assertTrue(events_a)
        self.assertEqual(events_a, events_b)
        self.assertEqual(result_a, result_b)

    def _replay(self, script, step_mode):
        """One scripted hand played with `policy`; returns (events, result, deck calls)."""
        deck = RecordingStackedDeck(script, seed=random.randrange(1 << 30))
        recorder = PolicyAgent()
        agents = [recorder, PolicyAgent(), PolicyAgent(), PolicyAgent()]
        env, _ = make_env(agents, stacks=[30, 25, 40, 10], button=1, deck=deck, chips_per_bb=2)
        if step_mode:
            d = env.start_hand()
            while d is not None:
                d = env.step(policy(d.seat, d.legal))
            result = env.last_result
        else:
            result = env.start_round()
        return recorder.events, result, list(deck.calls)

    def test_same_script_same_hand_in_step_mode(self):
        script = random_script(43)
        events_a, result_a, calls_a = self._replay(script, step_mode=True)
        events_b, result_b, calls_b = self._replay(script, step_mode=True)
        self.assertTrue(events_a)
        self.assertEqual(events_a, events_b)
        self.assertEqual(result_a, result_b)
        self.assertEqual(calls_a, calls_b)
        # ... and the same as the callback-mode replay, deck calls included.
        events_c, result_c, calls_c = self._replay(script, step_mode=False)
        self.assertEqual(events_a, events_c)
        self.assertEqual(result_a, result_c)
        self.assertEqual(calls_a, calls_c)
        self.assertEqual(calls_a[:2], ["reset", "shuffle"])
        self.assertEqual(set(calls_a[2:]), {"deal_one"})
        self.assertTrue(
            any(isinstance(e, PlayerActed) and e.action is ActionType.RAISE for e in events_a)
        )


class DirectPlayersReplayTest(unittest.TestCase):
    """A recorded hand replayed as `//rl_env` does: players created directly and
    passed to `Environment(players=...)`, chip unit = the recording's big blind,
    starting stacks that are not whole at 100 chips/BB."""

    START_CHIPS = {0: 12345, 1: 6789, 2: 20001, 3: 15003}

    def recorded_final(self, cpb, high, low_first, low_second):
        """Final chips of the odd-chip hand (seat 1 folds the SB, the others check
        it down; seat 3 wins the high, seats 2 and 0 split the low)."""
        s = self.START_CHIPS
        return {
            0: s[0] - cpb + low_second,
            1: s[1] - cpb // 2,
            2: s[2] - cpb + low_first,
            3: s[3] - cpb + high,
        }

    def replay(self, cpb, step_mode):
        stacks = [self.START_CHIPS[s] / cpb for s in range(4)]
        if step_mode:
            agents = [None] * 4
        else:
            agents = [Scripted(), Scripted("f"), Scripted(), Scripted()]
        players = [Player(f"p{i}", st, a) for i, (st, a) in enumerate(zip(stacks, agents))]
        env = Environment(players, deck=StackedDeck([odd_chip_script()]), chips_per_bb=cpb)
        self.assertEqual({p.seat: p.chips for p in players}, self.START_CHIPS)
        if step_mode:
            d = env.start_hand()
            while d is not None:
                if d.seat == 1:
                    action = ActionType.FOLD
                else:
                    action = ActionType.CHECK if d.legal.can_check else ActionType.CALL
                d = env.step(action)
            result = env.last_result
        else:
            result = env.start_round()
        self.assertTrue(result.showdown)
        started = result.history[0]
        self.assertIsInstance(started, HandStarted)
        self.assertEqual(
            [st for _, _, st in started.participants], [self.START_CHIPS[s] / cpb for s in range(4)]
        )
        return {p.seat: p.chips for p in players}

    def test_recorded_final_stacks_reproduced(self):
        # 1,000 chips/BB: pot 3,500 -> high 1,750, low 875 / 875.
        # 1,002 chips/BB: pot 3,507 -> high 1,754 (odd chip), low 877 / 876.
        for cpb, split in ((1000, (1750, 875, 875)), (1002, (1754, 877, 876))):
            expected = self.recorded_final(cpb, *split)
            for step_mode in (False, True):
                with self.subTest(chips_per_bb=cpb, step_mode=step_mode):
                    self.assertEqual(self.replay(cpb, step_mode), expected)


if __name__ == "__main__":
    unittest.main()
