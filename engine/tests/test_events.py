import random
import unittest

from entities import Board, Deck, Hand
from engine import (
    FOLD_LEAVE,
    FOLD_TIMEOUT,
    ActionType,
    BetReturned,
    BlindPosted,
    BoardDealt,
    Environment,
    HandShown,
    HandStarted,
    Player,
    PlayerActed,
    PlayerStatus,
    PotAwarded,
    Street,
)


class Recorder:
    """Plays codes f/k/c/r/l(eave) in order, then checks/calls; does nothing on
    its turns in `idle_hands`. Records everything it is told, in order."""

    def __init__(self, script="", idle_hands=()):
        self.script = list(script)
        self.idle_hands = set(idle_hands)
        self.log = []  # ("hole" | "turn" | "event", hand number)
        self.hole = {}  # hand number -> hand from on_hole_cards
        self.events = []
        self.history_checks = []  # (view.history, events of that hand seen so far)
        self.stack_checks = []  # (stack in the view, last event stack) per seat
        self.results = []

    def on_hole_cards(self, player, hand, view):
        self.hole[view.hand_number] = hand
        self.log.append(("hole", view.hand_number))
        self.hole_view = view

    def on_event(self, player, event):
        self.events.append(event)
        self.log.append(("event", event.hand_no))

    def on_turn(self, player, view, legal):
        self.log.append(("turn", view.hand_number))
        seen = tuple(e for e in self.events if e.hand_no == view.hand_number)
        self.history_checks.append((view.history, seen))
        last = {}
        for e in seen:
            if hasattr(e, "stack_bb"):
                last[e.seat] = e.stack_bb
        for seat, stack in last.items():
            self.stack_checks.append((view.seats[seat].stack, stack))
        if view.hand_number in self.idle_hands:
            return
        if self.script:
            code = self.script.pop(0)
        else:
            code = "k" if legal.can_check else "c"
        {
            "f": player.fold,
            "k": player.check,
            "c": player.call,
            "r": player.raise_,
            "l": player.leave,
        }[code]()

    def on_hand_end(self, player, result):
        self.results.append(result)


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


def table(agents, stacks=None, **kwargs):
    stacks = stacks or [100] * len(agents)
    players = [Player(f"p{i}", s, a) for i, (a, s) in enumerate(zip(agents, stacks))]
    return Environment(players, **kwargs), players


def hand_events(agent, hand_no):
    return [e for e in agent.events if e.hand_no == hand_no]


class HoleCardsTest(unittest.TestCase):
    def test_each_participant_gets_own_cards_before_first_turn(self):
        agents = [Recorder() for _ in range(3)]
        env, players = table(agents, seed=5)
        env.start_round()
        for p, a in zip(players, agents):
            self.assertEqual(a.hole[1], p.hole_cards)
            self.assertEqual(len(a.hole[1].cards), 4)
            # Only its own cards: the view carries no other hole cards.
            self.assertEqual(a.hole_view.hole_cards, p.hole_cards)
            entries = [x for x in a.log if x[1] == 1 and x[0] != "event"]
            self.assertEqual(entries[0], ("hole", 1))
            self.assertEqual(sum(1 for x in entries if x[0] == "hole"), 1)
            # Blinds were already posted when the cards arrived.
            hole_at = a.log.index(("hole", 1))
            before = [e for e in a.events[: sum(1 for x in a.log[:hole_at] if x[0] == "event")]]
            self.assertEqual(sum(isinstance(e, BlindPosted) for e in before), 2)
        all_cards = [c for p in players for c in p.hole_cards.cards]
        self.assertEqual(len(set(all_cards)), 12)

    def test_sitting_out_player_gets_no_hole_cards(self):
        agents = [Recorder(), Recorder(), Recorder(idle_hands={1})]
        env, players = table(agents, seed=1)
        env.start_round()
        self.assertIs(players[2].status, PlayerStatus.SITTING_OUT)
        env.start_round()
        self.assertIn(1, agents[2].hole)
        self.assertNotIn(2, agents[2].hole)
        # ... but it still sees the public events of hand 2.
        self.assertEqual(
            [e.seq for e in hand_events(agents[2], 2)],
            [e.seq for e in hand_events(agents[0], 2)],
        )
        self.assertTrue(hand_events(agents[2], 2))


class EventStreamTest(unittest.TestCase):
    def test_showdown_hand_event_sequence(self):
        # Deal starts at seat 1: p1 9c9s5c5s, p2 KhKdQcQh, p0 AcAs3d4d.
        deck = stacked(["9c9s5c5s", "KhKdQcQh", "AcAs3d4d"], "2s7d8h Tc Ts")
        agents = [Recorder() for _ in range(3)]
        env, players = table(agents, deck=deck, seed=1)
        result = env.start_round()

        events = hand_events(agents[0], 1)
        # Every seated player, actor included, sees the same events in order.
        for a in agents[1:]:
            self.assertEqual(hand_events(a, 1), events)
        self.assertEqual(tuple(events), result.history)

        self.assertIsInstance(events[0], HandStarted)
        self.assertEqual(events[0].button, result.button)
        self.assertEqual(
            [(e.blind, e.seat) for e in events[1:3]],
            [("SB", result.sb_seat), ("BB", result.bb_seat)],
        )
        seqs = [e.seq for e in events]
        self.assertEqual(seqs, sorted(set(seqs)))

        boards = [e for e in events if isinstance(e, BoardDealt)]
        self.assertEqual([e.street for e in boards], [Street.FLOP, Street.TURN, Street.RIVER])
        self.assertEqual([len(e.cards) for e in boards], [3, 1, 1])
        self.assertEqual(boards[-1].board, result.board)

        acted = [e for e in events if isinstance(e, PlayerActed)]
        self.assertTrue(all(e.reason is None for e in acted))
        self.assertTrue(all(e.action in (ActionType.CHECK, ActionType.CALL) for e in acted))

        shown = [e for e in events if isinstance(e, HandShown)]
        self.assertEqual({e.seat: e.hand for e in shown}, result.shown)
        awards = [e for e in events if isinstance(e, PotAwarded)]
        self.assertTrue(awards)
        # p0 scoops: a high and a low share.
        self.assertEqual({(e.seat, e.share) for e in awards}, {(0, "high"), (0, "low")})
        self.assertEqual(sum(e.amount_bb for e in awards), sum(result.winnings.values()))
        self.assertLess(events.index(shown[-1]), events.index(awards[0]))

    def test_stack_after_each_event(self):
        agents = [Recorder("r"), Recorder("c"), Recorder("cr")]
        env, players = table(agents, seed=9)
        start = {p.seat: p.stack for p in players}
        result = env.start_round()

        last = dict(start)
        for e in result.history:
            if isinstance(e, (BlindPosted, PlayerActed)):
                self.assertAlmostEqual(e.stack_bb, last[e.seat] - e.amount_bb)
            elif isinstance(e, (BetReturned, PotAwarded)):
                self.assertAlmostEqual(e.stack_bb, last[e.seat] + e.amount_bb)
            if hasattr(e, "stack_bb"):
                last[e.seat] = e.stack_bb
        self.assertEqual(last, result.stacks)
        # The stacks in each view agree with the events delivered before it.
        for a in agents:
            for view_stack, event_stack in a.stack_checks:
                self.assertAlmostEqual(view_stack, event_stack)
            self.assertTrue(a.stack_checks)

    def test_view_history_matches_delivered_events(self):
        agents = [Recorder() for _ in range(3)]
        env, _ = table(agents, seed=2)
        env.run(2)
        for a in agents:
            self.assertTrue(a.history_checks)
            for history, seen in a.history_checks:
                self.assertEqual(history, seen)

    def test_seq_increases_across_hands(self):
        agents = [Recorder() for _ in range(3)]
        env, _ = table(agents, seed=4)
        env.run(3)
        seqs = [e.seq for e in agents[0].events]
        self.assertEqual(seqs, sorted(set(seqs)))
        self.assertEqual({e.hand_no for e in agents[0].events}, {1, 2, 3})

    def test_fold_out_has_no_showdown_and_one_whole_award(self):
        # Seat 0 (UTG) raises, the blinds fold.
        agents = [Recorder("r"), Recorder("f"), Recorder("f")]
        env, _ = table(agents, seed=3)
        result = env.start_round()
        events = list(result.history)
        self.assertFalse(any(isinstance(e, HandShown) for e in events))
        self.assertFalse(any(isinstance(e, BoardDealt) for e in events))
        awards = [e for e in events if isinstance(e, PotAwarded)]
        self.assertEqual([(e.seat, e.share) for e in awards], [(0, "whole")])
        self.assertFalse(result.showdown)

    def test_all_in_run_out_deals_the_board(self):
        # Heads-up, both all-in from the blinds: the board is run out.
        agents = [Recorder(), Recorder()]
        env, _ = table(agents, stacks=[1, 1], seed=6)
        result = env.start_round()
        boards = [e for e in result.history if isinstance(e, BoardDealt)]
        self.assertEqual([e.street for e in boards], [Street.FLOP, Street.TURN, Street.RIVER])
        self.assertTrue(any(isinstance(e, BlindPosted) and e.all_in for e in result.history))


class ForcedFoldTest(unittest.TestCase):
    def _folds(self, result, seat):
        return [
            e
            for e in result.history
            if isinstance(e, PlayerActed) and e.seat == seat and e.action is ActionType.FOLD
        ]

    def test_timeout_fold_has_reason(self):
        agents = [Recorder(), Recorder(), Recorder(idle_hands={1})]
        env, _ = table(agents, seed=1)
        result = env.start_round()
        folds = self._folds(result, 2)
        self.assertEqual([e.reason for e in folds], [FOLD_TIMEOUT])

    def test_leave_fold_has_reason(self):
        agents = [Recorder("l"), Recorder(), Recorder()]
        env, _ = table(agents, seed=1)
        result = env.start_round()
        folds = self._folds(result, 0)
        self.assertEqual([e.reason for e in folds], [FOLD_LEAVE])


class FaultyObserverTest(unittest.TestCase):
    def test_raising_on_event_does_not_break_the_hand(self):
        class Faulty(Recorder):
            def on_event(self, player, event):
                super().on_event(player, event)
                raise RuntimeError("observer bug")

            def on_hole_cards(self, player, hand, view):
                super().on_hole_cards(player, hand, view)
                raise RuntimeError("observer bug")

        agents = [Faulty(), Recorder(), Recorder()]
        env, players = table(agents, seed=7)
        with self.assertLogs(level="ERROR"):
            results = env.run(2)
        self.assertEqual(len(results), 2)
        self.assertEqual(
            [e.seq for e in agents[0].events], [e.seq for e in agents[1].events]
        )
        self.assertEqual(sum(p.chips for p in players), 30000)


if __name__ == "__main__":
    unittest.main()
