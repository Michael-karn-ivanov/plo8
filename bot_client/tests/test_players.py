"""Same decisions as arena: for views and legal actions from many random hands
(also after the protocol's JSON round trip), BotPlayer decides exactly as the
same bot behind BotAgent and inside a TableBatch."""

import json
import unittest

from arena import BOT_NAMES, BotAgent, make_bot
from cli_server.protocol import decode_legal, decode_table_view, encode_legal, encode_table_view
from engine import ActionType
from rl_env import HandSampler, StackConfig, TableBatch, to_engine

from bot_client.players import BotPlayer, make_player


class Recording:
    """Wraps a bot inside a TableBatch; stores (view, legal, action) per row."""

    wants_views = True
    wants_obs = False

    def __init__(self, bot):
        self.bot = bot
        self.rows = []

    def act(self, batch):
        out = self.bot.act(batch)
        actions = out[0] if isinstance(out, tuple) else out
        for r in range(len(batch)):
            self.rows.append((batch.views[r], batch.legal[r], int(actions[r])))
        return out


def wire(view, legal):
    """The view and legal actions as a client decodes them from a `turn`."""
    v = decode_table_view(json.loads(json.dumps(encode_table_view(view))))
    lg = decode_legal(json.loads(json.dumps(encode_legal(legal))))
    return v, lg


def record(name, n_seats, n_tables, hands, seed):
    rec = Recording(make_bot(name, seed))
    batch = TableBatch(n_tables, observations=False)
    sampler = HandSampler(seed * 100 + n_seats, sizes=(n_seats,), cfg=StackConfig())
    batch.run({"r": rec}, sampler, hands, lambda spec: ("r",) * spec.n_seats)
    return rec.rows


class SameDecisionsTest(unittest.TestCase):
    SIZES = (2, 3, 4, 6, 8, 10)

    def check_rows(self, rows, direct, decoded, agent):
        """`direct`, `decoded` and `agent` are fed the rows in order."""
        self.assertGreater(len(rows), 0)
        for view, legal, action in rows:
            expected = to_engine(action, legal)
            got = direct.decide(view, legal)
            v, lg = wire(view, legal)
            self.assertEqual(v, view)
            self.assertEqual(lg, legal)
            got_wire = decoded.decide(v, lg)
            got_agent = agent.decide(view, legal)
            self.assertIs(got, expected)
            self.assertIs(got_wire, expected)
            self.assertIs(got_agent, expected)
            self.assertTrue(legal.allows(got))
            self.assertFalse(legal.can_check and got is ActionType.FOLD)

    def test_deterministic_bots_on_multi_table_batches(self):
        for name in BOT_NAMES:
            if name == "random":
                continue
            for n in self.SIZES:
                with self.subTest(bot=name, n=n):
                    rows = record(name, n, 8, 25, seed=n)
                    player = BotPlayer(name)
                    self.check_rows(rows, player, player, BotAgent(make_bot(name)))

    def test_random_bot_with_equally_seeded_twins(self):
        # RandomBot draws in call order: one row per call (one table per batch)
        # and equally seeded twins fed the same rows in the same order.
        for n in self.SIZES:
            for seed in (0, 5):
                with self.subTest(n=n, seed=seed):
                    rows = record("random", n, 1, 15, seed=seed)
                    self.check_rows(
                        rows,
                        BotPlayer("random", seed),
                        BotPlayer("random", seed),
                        BotAgent(make_bot("random", seed)),
                    )

    def test_make_player(self):
        class Args:
            bot = "maniac"
            seed = 3

        player = make_player(Args)
        self.assertIsInstance(player, BotPlayer)
        self.assertEqual(player.label, "maniac")
        self.assertIsNone(player.agent.rebuy_to_bb)


if __name__ == "__main__":
    unittest.main()
