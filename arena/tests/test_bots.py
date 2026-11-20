"""Bots: every action allowed by the mask, behaviour of the simple bots,
RandomBot spread and reproducibility, the factory, views only when wanted."""

import contextlib
import types
import unittest
from collections import defaultdict
from unittest import mock

import numpy as np

import arena.bots as bots_module
from arena import BOT_NAMES, CallingStation, Maniac, RandomBot, RuleBot, make_bot
from rl_env import BET_RAISE, CHECK_CALL, FOLD, HandSampler, StackConfig, Table, TableBatch

try:
    from . import helpers
except ImportError:  # run as a top-level module
    import helpers  # type: ignore[no-redef]


@contextlib.contextmanager
def counting_views():
    counter = {"n": 0}
    original = Table.view

    def wrapper(self, *args, **kwargs):
        counter["n"] += 1
        return original(self, *args, **kwargs)

    Table.view = wrapper
    try:
        yield counter
    finally:
        Table.view = original


def run_alone(bot, n, hands, seed, tables=32):
    """Every seat played by `bot` (wrapped to check masks); StackConfig stacks."""
    checker = helpers.MaskChecking(bot)
    batch = TableBatch(tables, observations=False)
    sampler = HandSampler(seed, sizes=(n,), cfg=StackConfig())
    done = batch.run({"b": checker}, sampler, hands, lambda spec: ("b",) * spec.n_seats)
    assert done == hands
    return checker


def run_mixed(n, hands, seed, tables=32):
    """The four bots mixed at the table, seat i -> BOT_NAMES[(i + deck_seed) % 4]."""
    checkers = {name: helpers.MaskChecking(make_bot(name, seed)) for name in BOT_NAMES}
    batch = TableBatch(tables, observations=False)
    sampler = HandSampler(seed, sizes=(n,), cfg=StackConfig())

    def keys(spec):
        return tuple(BOT_NAMES[(i + spec.deck_seed) % 4] for i in range(spec.n_seats))

    done = batch.run(checkers, sampler, hands, keys)
    assert done == hands
    return checkers


class MaskTest(unittest.TestCase):
    def test_every_bot_alone_at_every_size(self):
        for name in BOT_NAMES:
            for n in range(2, 11):
                with self.subTest(bot=name, n=n):
                    checker = run_alone(make_bot(name, n), n, 40, seed=100 + n)
                    self.assertGreater(checker.decisions, 0)

    def test_bots_mixed_at_every_size(self):
        for n in range(2, 11):
            with self.subTest(n=n):
                checkers = run_mixed(n, 80, seed=200 + n)
                self.assertGreater(sum(c.decisions for c in checkers.values()), 0)

    def test_simple_bots_never_need_the_mask_fallback(self):
        original = bots_module._masked

        def strict(actions, mask):
            a = np.asarray(actions, dtype=np.int64)
            m = np.asarray(mask, dtype=bool)
            if a.shape[0]:
                self.assertTrue(m[np.arange(a.shape[0]), a].all(), "fallback was needed")
            return original(actions, mask)

        with mock.patch.object(bots_module, "_masked", strict):
            for name in ("random", "calling", "maniac"):
                for n in (2, 6, 10):
                    run_alone(make_bot(name, 3), n, 40, seed=300 + n)


class BehaviourTest(unittest.TestCase):
    def test_calling_station_only_checks_or_calls(self):
        for n in (2, 6, 9):
            c = run_alone(CallingStation(), n, 60, seed=n)
            self.assertEqual(c.counts[FOLD], 0)
            self.assertEqual(c.counts[BET_RAISE], 0)
            self.assertGreater(c.counts[CHECK_CALL], 0)

    def test_maniac_raises_whenever_allowed_and_never_folds(self):
        for n in (2, 6, 9):
            c = run_alone(Maniac(), n, 60, seed=n)
            self.assertEqual(c.counts[FOLD], 0)
            for mask, a in c.log:
                self.assertEqual(a, BET_RAISE if mask[BET_RAISE] else CHECK_CALL)
            self.assertTrue(any(not m[BET_RAISE] for m, _ in c.log))  # cap reached seen

    def test_random_bot_spreads_over_allowed_actions(self):
        c = run_alone(RandomBot(7), 6, 400, seed=11)
        by_mask = defaultdict(lambda: np.zeros(3, dtype=np.int64))
        for mask, a in c.log:
            self.assertTrue(mask[a])
            by_mask[mask][a] += 1
        checked = 0
        for mask, counts in by_mask.items():
            total = int(counts.sum())
            if total < 300:
                continue
            k = sum(mask)
            for a in range(3):
                if mask[a]:
                    self.assertAlmostEqual(counts[a] / total, 1.0 / k, delta=0.1, msg=(mask, counts))
                else:
                    self.assertEqual(counts[a], 0)
            checked += 1
        self.assertGreaterEqual(checked, 2)

    def test_random_bot_is_reproducible_from_its_seed(self):
        rng = np.random.default_rng(0)
        masks = rng.random((500, 3)) < 0.6
        masks[:, CHECK_CALL] = True
        batch = types.SimpleNamespace(mask=masks)
        a1 = RandomBot(5).act(batch)
        a2 = RandomBot(5).act(batch)
        a3 = RandomBot(6).act(batch)
        np.testing.assert_array_equal(a1, a2)
        self.assertFalse(np.array_equal(a1, a3))
        self.assertEqual(a1.dtype, np.int64)
        self.assertTrue(masks[np.arange(500), a1].all())


class FactoryTest(unittest.TestCase):
    def test_make_bot(self):
        expected = {
            "random": RandomBot,
            "calling": CallingStation,
            "maniac": Maniac,
            "rule": RuleBot,
        }
        self.assertEqual(set(BOT_NAMES), set(expected))
        for name in BOT_NAMES:
            bot = make_bot(name, seed=3)
            self.assertIsInstance(bot, expected[name])
            self.assertEqual(bot.name, name)
            self.assertFalse(bot.wants_obs)
        self.assertEqual(make_bot("random", 9).seed, 9)

    def test_unknown_name_raises_listing_the_names(self):
        with self.assertRaises(ValueError) as ctx:
            make_bot("shark")
        for name in BOT_NAMES:
            self.assertIn(name, str(ctx.exception))

    def test_views_only_for_bots_that_want_them(self):
        for name in ("random", "calling", "maniac"):
            bot = make_bot(name)
            self.assertFalse(bot.wants_views)
            with counting_views() as counter:
                c = run_alone(bot, 6, 30, seed=1)
            self.assertGreater(c.decisions, 0)
            self.assertEqual(counter["n"], 0, name)
            self.assertEqual(c.views_seen, 0)
        self.assertTrue(RuleBot.wants_views)
        with counting_views() as counter:
            c = run_alone(RuleBot(), 6, 30, seed=1)
        self.assertEqual(counter["n"], c.decisions)
        self.assertEqual(c.views_seen, c.decisions)


if __name__ == "__main__":
    unittest.main()
