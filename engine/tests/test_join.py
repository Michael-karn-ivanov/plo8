import threading
import unittest

from engine import (
    Environment,
    EventType,
    IllegalActionError,
    Player,
    PlayerStatus,
    TableFullError,
)


class Passive:
    """Checks or calls; records what it was told. Hand numbers in `idle_hands`
    are played by doing nothing (so the player sits out afterwards)."""

    def __init__(self, idle_hands=()):
        self.idle_hands = set(idle_hands)
        self.turns = 0
        self.turn_hands = []
        self.hole_cards = {}
        self.hand_ends = []
        self.seated = []
        self.events = []
        self.log = []  # ("seated" | "turn", hand_number)
        self.player = None

    def on_turn(self, player, view, legal):
        self.turns += 1
        self.turn_hands.append(view.hand_number)
        self.hole_cards[view.hand_number] = view.hole_cards
        self.log.append(("turn", view.hand_number))
        if view.hand_number in self.idle_hands:
            return
        if legal.can_check:
            player.check()
        else:
            player.call()

    def on_hand_end(self, player, result):
        self.hand_ends.append(result.hand_number)

    def on_seated(self, player, view):
        self.player = player
        self.seated.append((player.name, view.seat, view))
        self.log.append(("seated", view.hand_number))

    def on_event(self, event):
        self.events.append(event)


def by_name(env, name):
    for p in env.players + env.waiting:
        if p.name == name:
            return p
    raise LookupError(name)


class JoinOnFirstTurn(Passive):
    """On its first turn of hand 1, makes the given players join the table."""

    def __init__(self, env_ref, joins):
        super().__init__()
        self.env_ref = env_ref
        self.joins = list(joins)  # (kwargs) for env.join
        self.joined = []
        self.views = []
        self.errors = []
        self.returns = []

    def on_turn(self, player, view, legal):
        self.views.append(view)
        if self.joins:
            env = self.env_ref[0]
            for kwargs in self.joins:
                self.returns.append(env.join(**kwargs))
                p = by_name(env, kwargs["name"])
                self.joined.append(p)
                try:
                    p.rebuy(5)  # waiting players cannot rebuy
                except IllegalActionError as e:
                    self.errors.append(e)
            self.joins = []
        super().on_turn(player, view, legal)


def make_env(n, button=0, seed=1, agents=None, **kwargs):
    agents = agents if agents is not None else [Passive() for _ in range(n)]
    players = [Player(f"p{i}", 100, a) for i, a in enumerate(agents)]
    return Environment(players, button=button, seed=seed, **kwargs), players, agents


def offset(env, player):
    return (player.seat - env.button) % len(env.players)


class JoinBetweenHandsTest(unittest.TestCase):
    def test_join_returns_none_and_agent_gets_player(self):
        env, _, _ = make_env(3)
        agent = Passive()
        self.assertIsNone(env.join(50, name="new", agent=agent))
        new = by_name(env, "new")
        self.assertIs(agent.player, new)
        self.assertIs(new.status, PlayerStatus.ACTIVE)
        result = env.start_round()
        # on_seated first, then a turn in the next hand, with 4 cards dealt.
        self.assertEqual(agent.log[0], ("seated", 0))
        self.assertIn(("turn", 1), agent.log)
        self.assertEqual(len(agent.hole_cards[1]), 4)
        self.assertTrue(result.dealt_in[new.seat])

    def test_default_seat_is_big_blind(self):
        env, players, _ = make_env(3)
        agent = Passive()
        env.join(50, name="new", agent=agent)
        new = by_name(env, "new")
        self.assertEqual(new.stack, 50.0)
        self.assertEqual(len(env.players), 4)
        self.assertEqual([p.seat for p in env.players], [0, 1, 2, 3])
        self.assertEqual(offset(env, new), 2)
        self.assertEqual(agent.seated[0][:2], ("new", new.seat))
        result = env.start_round()
        self.assertEqual(result.bb_seat, new.seat)
        self.assertEqual(result.names[new.seat], "new")
        self.assertIsNotNone(new.hole_cards)
        self.assertEqual(agent.hand_ends, [1])

    def test_default_seat_after_button_moved(self):
        env, players, _ = make_env(3)
        env.start_round()
        self.assertEqual(env.button, 1)
        env.join(50, name="new")
        new = by_name(env, "new")
        self.assertEqual(env.players[env.button], players[1])  # button stays put
        self.assertEqual(offset(env, new), 2)
        result = env.start_round()
        self.assertEqual(result.button, players[1].seat)
        self.assertEqual(result.bb_seat, new.seat)

    def test_default_seat_skips_sitting_out_seats(self):
        # p1 (left of the button) sits out after hand 1; p2 gets the button.
        agents = [Passive(), Passive(idle_hands={1}), Passive(), Passive()]
        env, players, _ = make_env(4, agents=agents)
        env.start_round()
        self.assertIs(players[1].status, PlayerStatus.SITTING_OUT)
        self.assertIs(env.players[env.button], players[2])
        env.join(50, name="new")
        new = by_name(env, "new")
        # Directly after p3, the next small blind, so the newcomer is the BB.
        order = [env.players[(env.button + k) % 5] for k in range(5)]
        self.assertEqual(order, [players[2], players[3], new, players[0], players[1]])
        result = env.start_round()
        self.assertEqual(result.sb_seat, players[3].seat)
        self.assertEqual(result.bb_seat, new.seat)

    def test_default_seat_with_sitting_out_seat_left_of_button(self):
        # Hand 1 (button 0): p2 sits out. Button moves to p1; p2 is left of it.
        agents = [Passive(), Passive(), Passive(idle_hands={1}), Passive()]
        env, players, _ = make_env(4, agents=agents)
        env.start_round()
        self.assertIs(env.players[env.button], players[1])
        env.join(50, name="new")
        new = by_name(env, "new")
        # The next SB is p3 (p2 is skipped), so the newcomer sits right after p3.
        self.assertEqual(env.players[(players[3].seat + 1) % 5], new)
        result = env.start_round()
        self.assertEqual((result.sb_seat, result.bb_seat), (players[3].seat, new.seat))

    def test_heads_up_join_posts_big_blind(self):
        env = Environment([Player("a", 100, Passive())])
        env.join(100, name="b", agent=Passive())
        new = by_name(env, "b")
        self.assertEqual(new.seat, 1)
        result = env.start_round()
        self.assertEqual((result.button, result.sb_seat, result.bb_seat), (0, 0, 1))

    def test_join_empty_table(self):
        env = Environment()
        self.assertIsNone(env.button)
        env.join(100, seat=5)  # seat is ignored at an empty table
        a = by_name(env, "Player1")
        self.assertEqual((a.seat, env.button), (0, 0))
        self.assertFalse(env.can_start())
        env.join(100, seat=1)  # ignored as well
        b = by_name(env, "Player2")
        self.assertEqual((a.seat, b.seat, env.button), (0, 1, 0))
        result = env.start_round()
        self.assertEqual((result.button, result.sb_seat, result.bb_seat), (0, 0, 1))

    def test_join_with_one_active_player_left(self):
        # p0 does not act in hand 1 and sits out: only p1 is active.
        agents = [Passive(idle_hands={1}), Passive()]
        env, players, _ = make_env(2, agents=agents)
        env.start_round()
        self.assertIs(players[0].status, PlayerStatus.SITTING_OUT)
        env.join(100, seat=2, name="c")  # seat ignored: no game is running
        c = by_name(env, "c")
        self.assertIs(env.players[env.button], players[1])
        self.assertEqual(c.seat, players[1].seat + 1)
        result = env.start_round()
        self.assertEqual(
            (result.button, result.sb_seat, result.bb_seat),
            (players[1].seat, players[1].seat, c.seat),
        )
        self.assertFalse(result.dealt_in[players[0].seat])

    def test_explicit_offsets(self):
        for button in range(4):
            for k in range(1, 5):
                env, players, _ = make_env(4, button=button)
                button_player = env.players[env.button]
                env.join(20, seat=k, name="new")
                new = by_name(env, "new")
                self.assertIs(env.players[env.button], button_player)
                self.assertEqual(offset(env, new), k, (button, k))
                # The others keep their clockwise order.
                n = len(env.players)
                order = [env.players[(env.button + i) % n] for i in range(n)]
                self.assertEqual([p for p in order if p is not new][0], button_player)
                others = [p for p in order if p is not new]
                self.assertEqual(
                    others, [players[(button + i) % 4] for i in range(4)]
                )

    def test_offset_one_is_small_blind(self):
        env, _, _ = make_env(3)
        env.join(20, seat=1, name="new")
        result = env.start_round()
        self.assertEqual(result.sb_seat, by_name(env, "new").seat)

    def test_join_event_sent_to_others(self):
        env, _, agents = make_env(2)
        env.join(20, name="new")
        joined = [e for e in agents[0].events if e.type is EventType.PLAYER_JOINED]
        self.assertEqual(len(joined), 1)
        self.assertEqual((joined[0].name, joined[0].seat), ("new", by_name(env, "new").seat))

    def test_seated_view(self):
        env, _, _ = make_env(3)
        env.start_round()
        agent = Passive()
        env.join(30, name="new", agent=agent)
        new = by_name(env, "new")
        _, seat, view = agent.seated[0]
        self.assertEqual(seat, new.seat)
        self.assertEqual(view.me.name, "new")
        self.assertEqual(view.me.stack, 30.0)
        self.assertIs(view.me.status, PlayerStatus.ACTIVE)
        self.assertIsNone(view.hole_cards)
        self.assertEqual(view.pot, 0.0)
        self.assertEqual(view.button, env.button)


class JoinDuringHandTest(unittest.TestCase):
    def test_join_from_on_turn_is_queued(self):
        env_ref = []
        late_agent = Passive()
        joiner = JoinOnFirstTurn(env_ref, [dict(stack_bb=50, name="late", agent=late_agent)])
        players = [Player("a", 100, joiner), Player("b", 100, Passive()), Player("c", 100, Passive())]
        env = Environment(players, seed=3)
        env_ref.append(env)

        first = env.start_round()
        late = joiner.joined[0]
        self.assertEqual(joiner.returns, [None])
        # Not part of hand 1.
        self.assertEqual(late_agent.turns, 0)
        self.assertEqual(late_agent.hand_ends, [])
        self.assertIsInstance(joiner.errors[0], IllegalActionError)
        self.assertEqual(late.rebuy_chips, 0)
        self.assertEqual(set(first.stacks), {0, 1, 2})
        self.assertNotIn("late", first.names.values())
        # Later views in hand 1 show the waiting player.
        self.assertEqual(joiner.views[0].waiting, ())
        self.assertIn("late", [n for v in joiner.views[1:] for n in v.waiting])
        # Seated after the hand, relative to the moved button, as big blind.
        self.assertIs(late.status, PlayerStatus.ACTIVE)
        self.assertIs(late_agent.player, late)
        self.assertEqual(env.waiting, [])
        self.assertEqual(env.button, 1)
        self.assertEqual(offset(env, late), 2)
        self.assertIsNone(late.hole_cards)
        self.assertEqual(len(late_agent.seated), 1)
        self.assertEqual(late.stack, 50.0)

        second = env.start_round()
        self.assertEqual(second.bb_seat, late.seat)
        self.assertEqual(second.names[late.seat], "late")
        self.assertIsNotNone(late.hole_cards)
        self.assertGreater(late_agent.turns, 0)
        # on_seated (after hand 1), then turns in hand 2 with 4 cards.
        self.assertEqual(late_agent.log[0], ("seated", 1))
        self.assertEqual(set(late_agent.turn_hands), {2})
        self.assertEqual(len(late_agent.hole_cards[2]), 4)
        self.assertEqual(late_agent.hand_ends, [2])
        self.assertEqual(sum(p.chips for p in env.players), 30000 + 5000)

    def test_queued_player_state_while_waiting(self):
        env_ref = []
        captured = {}

        class Inspect(JoinOnFirstTurn):
            def on_turn(self, player, view, legal):
                had_joins = bool(self.joins)
                super().on_turn(player, view, legal)
                if had_joins:
                    p = self.joined[0]
                    env = self.env_ref[0]
                    captured.update(
                        status=p.status,
                        seat=p.seat,
                        waiting=env.waiting,
                        seated=len(env.players),
                    )

        joiner = Inspect(env_ref, [dict(stack_bb=10, name="w")])
        env = Environment([Player("a", 100, joiner), Player("b", 100, Passive())])
        env_ref.append(env)
        env.start_round()
        self.assertIs(captured["status"], PlayerStatus.WAITING)
        self.assertIsNone(captured["seat"])
        self.assertEqual([p.name for p in captured["waiting"]], ["w"])
        self.assertEqual(captured["seated"], 2)
        self.assertEqual(len(env.players), 3)

    def test_multiple_queued_joins_fifo(self):
        env_ref = []
        order = []

        class Recorder(Passive):
            def on_seated(self, player, view):
                order.append(player.name)

        joins = [
            dict(stack_bb=10, name="x", agent=Recorder()),
            dict(stack_bb=10, name="y", agent=Recorder()),
            dict(stack_bb=10, name="z", seat=1, agent=Recorder()),
        ]
        joiner = JoinOnFirstTurn(env_ref, joins)
        players = [Player("a", 100, joiner), Player("b", 100, Passive()), Player("c", 100, Passive())]
        env = Environment(players, seed=1)
        env_ref.append(env)
        env.start_round()
        self.assertEqual(order, ["x", "y", "z"])
        x, y, z = joiner.joined
        self.assertEqual(len(env.players), 6)
        self.assertEqual(sorted(p.seat for p in env.players), list(range(6)))
        # Seated one after the other relative to the new button: x went to BB first,
        # then y took the BB position ahead of x; z sits left of the button (SB).
        self.assertEqual(env.players[env.button], players[1])
        self.assertEqual((offset(env, z), offset(env, y), offset(env, x)), (1, 3, 4))
        result = env.start_round()
        self.assertEqual(result.sb_seat, z.seat)

    def test_join_from_on_hand_end_is_queued_until_button_moved(self):
        env_ref = []
        captured = {}

        class JoinAtEnd(Passive):
            def on_hand_end(self, player, result):
                super().on_hand_end(player, result)
                if not captured:
                    env_ref[0].join(10, name="end")
                    p = by_name(env_ref[0], "end")
                    captured["status"] = p.status
                    captured["player"] = p

        players = [Player("a", 100, JoinAtEnd()), Player("b", 100, Passive()), Player("c", 100, Passive())]
        env = Environment(players)
        env_ref.append(env)
        env.start_round()
        self.assertIs(captured["status"], PlayerStatus.WAITING)
        late = captured["player"]
        self.assertIs(late.status, PlayerStatus.ACTIVE)
        self.assertEqual(env.players[env.button], players[1])
        self.assertEqual(offset(env, late), 2)

    def test_join_from_another_thread_during_hand(self):
        env_ref = []
        holder = {}

        def join_in_thread():
            env_ref[0].join(25, name="thr")
            holder["player"] = by_name(env_ref[0], "thr")

        class JoinViaThread(Passive):
            def on_turn(self, player, view, legal):
                if "player" not in holder:
                    t = threading.Thread(target=join_in_thread)
                    t.start()
                    t.join(5)
                    holder["status"] = holder["player"].status
                super().on_turn(player, view, legal)

        env = Environment([Player("a", 100, JoinViaThread()), Player("b", 100, Passive())])
        env_ref.append(env)
        env.start_round()
        self.assertIs(holder["status"], PlayerStatus.WAITING)
        self.assertIs(holder["player"].status, PlayerStatus.ACTIVE)
        self.assertEqual(len(env.players), 3)


class JoinValidationTest(unittest.TestCase):
    def test_invalid_stack(self):
        env, _, _ = make_env(2, max_buy_in=100)
        for bad in (0, -5, 0.001):
            with self.assertRaises(ValueError):
                env.join(bad)
        with self.assertRaises(ValueError):
            env.join(101)
        with self.assertRaises(TypeError):
            env.join("10")
        env.join(100)
        self.assertEqual(len(env.players), 3)

    def test_invalid_offset(self):
        env, _, _ = make_env(3)
        for bad in (0, -1, 4, True, 1.5):
            with self.assertRaises(ValueError):
                env.join(10, seat=bad)
        self.assertEqual(len(env.players), 3)
        env.join(10, seat=3)  # n = right of the button

    def test_duplicate_name(self):
        env, _, _ = make_env(2)
        with self.assertRaises(ValueError):
            env.join(10, name="p0")
        with self.assertRaises(ValueError):
            env.join(10, name="")

    def test_duplicate_name_of_waiting_player(self):
        env_ref = []
        errors = []

        class TwoJoins(Passive):
            def on_turn(self, player, view, legal):
                if not errors and not env_ref[0].waiting:
                    env_ref[0].join(10, name="dup")
                    try:
                        env_ref[0].join(10, name="dup")
                    except ValueError as e:
                        errors.append(e)
                super().on_turn(player, view, legal)

        env = Environment([Player("a", 100, TwoJoins()), Player("b", 100, Passive())])
        env_ref.append(env)
        env.start_round()
        self.assertEqual(len(errors), 1)
        self.assertEqual([p.name for p in env.players].count("dup"), 1)

    def test_default_names_for_joiners(self):
        env, _, _ = make_env(2)
        env.join(10)
        env.join(10)
        self.assertEqual([p.name for p in env.players][2:], ["Player2", "Player1"])
        self.assertEqual(sorted(p.name for p in env.players), ["Player1", "Player2", "p0", "p1"])

    def test_full_table(self):
        env, _, _ = make_env(3, max_seats=4)
        env.join(10)
        with self.assertRaises(TableFullError):
            env.join(10)
        self.assertTrue(issubclass(TableFullError, ValueError))

    def test_full_table_counts_sitting_out_players(self):
        agents = [Passive(), Passive(idle_hands={1}), Passive()]
        env, players, _ = make_env(3, agents=agents, max_seats=3)
        env.start_round()
        self.assertIs(players[1].status, PlayerStatus.SITTING_OUT)
        with self.assertRaises(TableFullError):
            env.join(10)
        players[1].leave()  # the seat is freed only by leaving
        env.join(10, name="new")
        self.assertEqual(len(env.players), 3)

    def test_full_table_counts_waiting_players(self):
        env_ref = []
        errors = []

        class FillUp(Passive):
            def on_turn(self, player, view, legal):
                env = env_ref[0]
                if not errors and not env.waiting:
                    env.join(10, name="w")
                    try:
                        env.join(10, name="w2")
                    except TableFullError as e:
                        errors.append(e)
                super().on_turn(player, view, legal)

        env = Environment(
            [Player("a", 100, FillUp()), Player("b", 100, Passive())], max_seats=3
        )
        env_ref.append(env)
        env.start_round()
        self.assertEqual(len(errors), 1)
        self.assertEqual(len(env.players), 3)


if __name__ == "__main__":
    unittest.main()
