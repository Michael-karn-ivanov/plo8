"""BotAgent: bots play complete hands in engine callback mode, never fold when
checking is free, and decide exactly as inside a TableBatch."""

import unittest

import numpy as np

from arena import BOT_NAMES, BotAgent, make_bot
from engine import ActionType, Environment, Player, PlayerActed
from rl_env import DecisionBatch, HandSampler, StackConfig, TableBatch, to_engine


class RecordingAgent(BotAgent):
    """BotAgent that records (can_check, engine action) of every decision."""

    def __init__(self, bot, **kwargs):
        super().__init__(bot, **kwargs)
        self.log = []

    def decide(self, view, legal):
        t = super().decide(view, legal)
        self.log.append((legal.can_check, t))
        return t


class CallbackModeTest(unittest.TestCase):
    HANDS_PER_OFFSET = 75  # x 4 seatings = 300 hands per table size

    def test_bots_play_complete_hands_at_every_size(self):
        for n in range(2, 11):
            hands = 0
            for k in range(4):
                with self.subTest(n=n, offset=k):
                    agents = [
                        RecordingAgent(make_bot(BOT_NAMES[(i + k) % 4], seed=i), rebuy_to_bb=100)
                        for i in range(n)
                    ]
                    players = [Player(f"p{i}", 100.0, agent=a) for i, a in enumerate(agents)]
                    env = Environment(players=players, seed=1000 * n + k, action_timeout=None)
                    results = env.run(self.HANDS_PER_OFFSET)
                    self.assertEqual(len(results), self.HANDS_PER_OFFSET)
                    hands += len(results)
                    for res in results:
                        self.assertEqual(sum(res.dealt_in.values()), n, "a player sat out")
                        for e in res.history:
                            if isinstance(e, PlayerActed):
                                self.assertIsNone(e.reason, "forced fold (missed turn)")
                    for p in players:
                        self.assertFalse(p.sitting_out)
                    for a in agents:
                        self.assertGreater(len(a.log), 0)
                        for can_check, t in a.log:
                            self.assertFalse(can_check and t is ActionType.FOLD)
            self.assertGreaterEqual(hands, 300)

    def test_rebuy_is_off_by_default(self):
        agent = BotAgent(make_bot("maniac"))
        self.assertIsNone(agent.rebuy_to_bb)
        players = [
            Player("a", 5.0, agent=agent),
            Player("b", 100.0, agent=BotAgent(make_bot("maniac"))),
        ]
        env = Environment(players=players, seed=3, action_timeout=None)
        env.run(30)
        self.assertEqual(players[0].rebought, 0.0)
        self.assertEqual(players[1].rebought, 0.0)

    def test_failed_rebuy_is_logged_and_ignored(self):
        agents = [BotAgent(make_bot("calling"), rebuy_to_bb=500) for _ in range(3)]
        players = [Player(f"p{i}", 100.0, agent=a) for i, a in enumerate(agents)]
        env = Environment(players=players, seed=4, action_timeout=None, max_buy_in=150)
        with self.assertLogs("arena.agent", level="WARNING"):
            results = env.run(5)
        self.assertGreaterEqual(len(results), 1)
        for p in players:
            self.assertEqual(p.rebought, 0.0)


class Comparing:
    """Inside a TableBatch: every row's action from the batch equals the twin's
    action for `DecisionBatch.single(view, legal)` of that row."""

    wants_views = True
    wants_obs = False

    def __init__(self, bot, twin):
        self.bot = bot
        self.twin = twin
        self.compared = 0

    def act(self, batch):
        out = np.asarray(self.bot.act(batch))
        for r in range(len(batch)):
            single = DecisionBatch.single(batch.views[r], batch.legal[r])
            np.testing.assert_array_equal(single.mask[0], batch.mask[r])
            a = int(self.twin.act(single)[0])
            assert a == int(out[r]), (a, int(out[r]))
            self.compared += 1
        return out


class SameDecisionsTest(unittest.TestCase):
    def test_batch_and_single_decisions_agree(self):
        for name in BOT_NAMES:
            for n in (2, 3, 6, 9, 10):
                with self.subTest(bot=name, n=n):
                    if name == "random":  # a twin with the same seed draws the same numbers
                        bot, twin = make_bot(name, 42), make_bot(name, 42)
                    else:  # stateless: the same object
                        bot = twin = make_bot(name)
                    cmp = Comparing(bot, twin)
                    batch = TableBatch(16, observations=False)
                    sampler = HandSampler(n, sizes=(n,), cfg=StackConfig())
                    batch.run({"c": cmp}, sampler, 30, lambda spec: ("c",) * spec.n_seats)
                    self.assertGreater(cmp.compared, 0)

    def test_agent_decision_equals_bot_on_single_batch(self):
        bot = make_bot("rule")
        agent = BotAgent(bot)
        seen = []

        class Spy:
            wants_views = True
            wants_obs = False

            def act(self, batch):
                seen.extend(zip(batch.views, batch.legal))
                return bot.act(batch)

        batch = TableBatch(8, observations=False)
        sampler = HandSampler(5, sizes=(6,), cfg=StackConfig())
        batch.run({"s": Spy()}, sampler, 20, lambda spec: ("s",) * spec.n_seats)
        self.assertGreater(len(seen), 0)
        for v, lg in seen:
            expected = to_engine(int(bot.act(DecisionBatch.single(v, lg))[0]), lg)
            self.assertIs(agent.decide(v, lg), expected)


if __name__ == "__main__":
    unittest.main()
