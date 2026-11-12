import random
import unittest

from entities import Board, Deck, Hand
from engine import (
    Environment,
    EventType,
    IllegalActionError,
    Player,
    PlayerStatus,
    TableFullError,
)


class Bot:
    """Checks/calls and records callbacks. `on_turn_hook(bot, player, view)` runs
    first on every turn; it may return True to skip acting."""

    def __init__(self, on_turn_hook=None, idle_hands=(), on_hand_end_hook=None):
        self.on_turn_hook = on_turn_hook
        self.on_hand_end_hook = on_hand_end_hook
        self.idle_hands = set(idle_hands)
        self.turns = 0
        self.views = []
        self.hand_ends = []
        self.left = []  # final stacks reported by on_left
        self.events = []

    def on_turn(self, player, view, legal):
        self.turns += 1
        self.views.append(view)
        if self.on_turn_hook is not None and self.on_turn_hook(self, player, view):
            return
        if view.hand_number in self.idle_hands:
            return
        if legal.can_check:
            player.check()
        else:
            player.call()

    def on_hand_end(self, player, result):
        self.hand_ends.append(result.hand_number)
        if self.on_hand_end_hook is not None:
            self.on_hand_end_hook(self, player, result)

    def on_left(self, player, final_stack_bb):
        self.left.append(final_stack_bb)

    def on_event(self, event):
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


def make_env(n, agents=None, button=0, **kwargs):
    agents = agents if agents is not None else [Bot() for _ in range(n)]
    players = [Player(f"p{i}", 100, a) for i, a in enumerate(agents)]
    return Environment(players, button=button, seed=1, **kwargs), players, agents


def player_actions(result, seat):
    kinds = (EventType.FOLD, EventType.CHECK, EventType.CALL, EventType.RAISE)
    return [e.type for e in result.history if e.seat == seat and e.type in kinds]


class LeaveBetweenHandsTest(unittest.TestCase):
    def test_leave_empties_seat_and_renumbers(self):
        env, players, agents = make_env(4)
        p2 = players[2]
        p2.leave()
        self.assertIs(p2.status, PlayerStatus.LEFT)
        self.assertIsNone(p2.seat)
        self.assertEqual([p.name for p in env.players], ["p0", "p1", "p3"])
        self.assertEqual([p.seat for p in env.players], [0, 1, 2])
        self.assertEqual(env.button, 0)
        self.assertEqual(agents[2].left, [100.0])
        left = [e for e in agents[0].events if e.type is EventType.PLAYER_LEFT]
        self.assertEqual([(e.name, e.seat) for e in left], [("p2", 2)])
        self.assertEqual([e for e in agents[2].events if e.type is EventType.PLAYER_LEFT], [])
        result = env.start_round()
        self.assertEqual(set(result.names.values()), {"p0", "p1", "p3"})
        self.assertEqual(agents[2].hand_ends, [])

    def test_leave_before_button_keeps_button_player(self):
        env, players, _ = make_env(4, button=2)
        players[0].leave()
        self.assertIs(env.players[env.button], players[2])

    def test_leaving_button_passes_it_on(self):
        env, players, _ = make_env(3, button=1)
        players[1].leave()
        self.assertIs(env.players[env.button], players[2])
        result = env.start_round()
        self.assertEqual(result.button, players[2].seat)

    def test_leaving_button_skips_sitting_out_players(self):
        # Hand 1 (button 0): p2 does nothing and sits out; the button moves to p1.
        agents = [Bot(), Bot(), Bot(idle_hands={1}), Bot()]
        env, players, _ = make_env(4, agents=agents)
        env.start_round()
        self.assertIs(players[2].status, PlayerStatus.SITTING_OUT)
        self.assertIs(env.players[env.button], players[1])
        players[1].leave()
        self.assertIs(env.players[env.button], players[3])

    def test_last_player_leaves(self):
        env, players, _ = make_env(2)
        players[0].leave()
        players[1].leave()
        self.assertEqual(env.players, [])
        self.assertIsNone(env.button)
        env.join(10, name="a")
        env.join(10, name="b")
        self.assertEqual(env.button, 0)
        env.start_round()

    def test_sitting_out_player_leaves(self):
        agents = [Bot(), Bot(), Bot(idle_hands={1})]
        env, players, _ = make_env(3, agents=agents)
        env.start_round()
        self.assertIs(players[2].status, PlayerStatus.SITTING_OUT)
        players[2].leave()
        self.assertIs(players[2].status, PlayerStatus.LEFT)
        self.assertEqual(len(env.players), 2)
        self.assertEqual(agents[2].left, [99.0])

    def test_leave_is_idempotent_and_name_reusable(self):
        env, players, agents = make_env(3)
        players[1].leave()
        players[1].leave()
        env.leave(players[1])
        self.assertEqual(agents[1].left, [100.0])  # notified once
        env.join(50, name="p1")
        new = [p for p in env.players if p.name == "p1"][0]
        self.assertIsNot(new, players[1])
        self.assertIs(new.status, PlayerStatus.ACTIVE)

    def test_leaving_player_of_another_table_rejected(self):
        env, _, _ = make_env(2)
        _, others, _ = make_env(2)
        with self.assertRaises(ValueError):
            env.leave(others[0])


class LeaveDuringHandTest(unittest.TestCase):
    def test_leave_inside_on_turn_folds_and_removes_after_hand(self):
        seen = {}

        def leave_now(bot, player, view):
            if view.hand_number == 1:
                player.leave()
                try:
                    player.call()  # no action after leaving
                except IllegalActionError as e:
                    seen["error"] = e
                return True
            return False

        def look(bot, player, view):
            seen.setdefault("view", view)
            return False

        agents = [Bot(on_turn_hook=leave_now), Bot(on_turn_hook=look), Bot()]
        env, players, _ = make_env(3, agents=agents)
        p0 = players[0]
        result = env.start_round()
        self.assertEqual(player_actions(result, 0), [EventType.FOLD])
        self.assertIsInstance(seen["error"], IllegalActionError)
        # Still seated (and shown as leaving) for the rest of the hand.
        self.assertTrue(seen["view"].seats[0].leaving)
        self.assertIn(0, result.names)
        # Removed after the hand; no on_hand_end; the button skipped him.
        self.assertIs(p0.status, PlayerStatus.LEFT)
        self.assertEqual([p.name for p in env.players], ["p1", "p2"])
        self.assertEqual(agents[0].hand_ends, [])
        self.assertEqual(agents[0].left, [100.0])
        self.assertIs(env.players[env.button], players[1])
        # The leaver is not counted as having missed a turn.
        self.assertTrue(all(p.status is PlayerStatus.ACTIVE for p in env.players))

    def test_leave_requested_mid_hand_folds_at_next_turn(self):
        # p0 (first to act) makes p2 (the big blind) leave; p2 is never asked to act.
        status = {}

        def make_p2_leave(bot, player, view):
            if view.hand_number == 1:
                players[2].leave()
                status["during"] = players[2].status
                status["leaving"] = players[2].leave_requested
            return False

        agents = [Bot(on_turn_hook=make_p2_leave), Bot(), Bot()]
        env, players, _ = make_env(3, agents=agents)
        result = env.start_round()
        self.assertIs(status["during"], PlayerStatus.ACTIVE)
        self.assertTrue(status["leaving"])
        self.assertEqual(agents[2].turns, 0)
        self.assertEqual(player_actions(result, 2), [EventType.FOLD])
        self.assertEqual(result.net[2], -1.0)  # the posted blind stays in the pot
        self.assertIs(players[2].status, PlayerStatus.LEFT)
        self.assertEqual(agents[2].left, [99.0])
        self.assertEqual(len(env.players), 2)

    def test_all_in_leaver_is_paid_before_removal(self):
        # p2 is all-in on the big blind with the best hand (AcAs3d4d scoops), and
        # asks to leave during the hand: he stays in and is paid.
        deck = stacked(["9c9s5c5s", "AcAs3d4d", "KhKdQcQh"], "2s7d8h Tc Ts")
        holder = {}

        def make_p2_leave(bot, player, view):
            if "done" not in holder:
                holder["done"] = True
                holder["p2"].leave()
            return False

        agents = [Bot(on_turn_hook=make_p2_leave), Bot(), Bot()]
        players = [Player("p0", 100, agents[0]), Player("p1", 100, agents[1]), Player("p2", 1, agents[2])]
        holder["p2"] = players[2]
        env = Environment(players, deck=deck, seed=1)
        result = env.start_round()
        self.assertTrue(result.showdown)
        self.assertIn(2, result.shown)
        self.assertEqual(result.winnings.get(2), 3.0)
        self.assertEqual(result.stacks[2], 3.0)
        self.assertIs(players[2].status, PlayerStatus.LEFT)
        self.assertEqual(agents[2].left, [3.0])
        self.assertEqual(players[2].stack, 3.0)  # his chips leave with him
        self.assertEqual(sum(p.chips for p in env.players) + players[2].chips, 20100)

    def test_sitting_out_player_leaves_during_hand(self):
        holder = {}

        def leave_sitter(bot, player, view):
            if view.hand_number == 2 and "done" not in holder:
                holder["done"] = True
                holder["p2"].leave()
                holder["still_seated"] = len(holder["env"].players)
            return False

        agents = [Bot(on_turn_hook=leave_sitter), Bot(), Bot(idle_hands={1})]
        env, players, _ = make_env(3, agents=agents)
        holder.update(p2=players[2], env=env)
        env.start_round()
        self.assertIs(players[2].status, PlayerStatus.SITTING_OUT)
        env.start_round()
        self.assertEqual(holder["still_seated"], 3)
        self.assertIs(players[2].status, PlayerStatus.LEFT)
        self.assertEqual(agents[2].hand_ends, [1])  # none for hand 2: he was leaving
        self.assertEqual(len(env.players), 2)

    def test_leave_from_on_hand_end(self):
        def leave_at_end(bot, player, result):
            player.leave()

        agents = [Bot(), Bot(on_hand_end_hook=leave_at_end), Bot()]
        env, players, _ = make_env(3, agents=agents)
        env.start_round()
        self.assertIs(players[1].status, PlayerStatus.LEFT)
        self.assertEqual(len(env.players), 2)
        # The button would have gone to p1; it skips the leaver and lands on p2.
        self.assertIs(env.players[env.button], players[2])

    def test_button_move_skips_player_leaving_mid_hand(self):
        def make_p1_leave(bot, player, view):
            if view.hand_number == 1:
                players[1].leave()
            return False

        agents = [Bot(on_turn_hook=make_p1_leave), Bot(), Bot()]
        env, players, _ = make_env(3, agents=agents)
        env.start_round()
        self.assertIs(env.players[env.button], players[2])

    def test_waiting_player_leaves_the_queue(self):
        holder = {}

        def join_and_leave(bot, player, view):
            if "w" not in holder:
                env = holder["env"]
                agent = Bot()
                env.join(10, name="w", agent=agent)
                w = env.waiting[0]
                holder.update(w=w, agent=agent)
                w.leave()
                holder["waiting_after"] = list(env.waiting)
            return False

        agents = [Bot(on_turn_hook=join_and_leave), Bot()]
        env, players, _ = make_env(2, agents=agents)
        holder["env"] = env
        env.start_round()
        self.assertEqual(holder["waiting_after"], [])
        self.assertIs(holder["w"].status, PlayerStatus.LEFT)
        self.assertEqual(holder["agent"].left, [10.0])
        self.assertEqual(len(env.players), 2)
        env.join(10, name="w")  # the name is free again

    def test_leaver_keeps_seat_counted_until_hand_ends(self):
        holder = {}

        def churn(bot, player, view):
            if "done" not in holder:
                holder["done"] = True
                players[2].leave()
                try:
                    holder["env"].join(40, name="new")
                except TableFullError as e:
                    holder["error"] = e
            return False

        agents = [Bot(on_turn_hook=churn), Bot(), Bot()]
        env, players, _ = make_env(3, agents=agents, max_seats=3)
        holder["env"] = env
        env.start_round()
        self.assertIsInstance(holder["error"], TableFullError)
        self.assertEqual(len(env.players), 2)
        env.join(40, name="new")  # free after the hand

    def test_leaver_removed_before_queued_joiner_is_seated(self):
        holder = {}

        def churn(bot, player, view):
            if "done" not in holder:
                holder["done"] = True
                players[2].leave()
                holder["env"].join(40, name="new")
            return False

        agents = [Bot(on_turn_hook=churn), Bot(), Bot()]
        env, players, _ = make_env(3, agents=agents, max_seats=4)
        holder["env"] = env
        env.start_round()
        self.assertEqual(sorted(p.name for p in env.players), ["new", "p0", "p1"])
        self.assertEqual([p.seat for p in env.players], [0, 1, 2])
        result = env.start_round()
        self.assertTrue(all(result.dealt_in.values()))
        new = [p for p in env.players if p.name == "new"][0]
        self.assertEqual(result.bb_seat, new.seat)


if __name__ == "__main__":
    unittest.main()
