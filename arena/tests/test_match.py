"""Duplicate matches: deal order and rotation seen from the arena side, mirror
matches, zero sum, determinism, combining, histories."""

import unittest
from collections import defaultdict
from unittest import mock

import numpy as np

import rl_env.batch as rl_batch
from arena import (
    BOT_NAMES,
    AccountingError,
    Entrant,
    RandomBot,
    combine,
    play_match,
    summarize,
)
from rl_env import StackConfig


def mixed_lineup(n):
    return [BOT_NAMES[(i * 3 + n) % 4] for i in range(n)]


def by_deal(result):
    groups = defaultdict(dict)
    for rec in result.hands:
        groups[rec.deal][rec.rotation] = rec
    return groups


class DealOrderAndRotationTest(unittest.TestCase):
    def test_same_cards_per_seat_in_every_rotation(self):
        for n in range(2, 11):
            with self.subTest(n=n):
                res = play_match(
                    mixed_lineup(n), 4, seed=n, stacks=StackConfig(), record_hands=True
                )
                self.assertEqual(len(res.hands), 4 * n)
                for d, rots in by_deal(res).items():
                    self.assertEqual(sorted(rots), list(range(n)))
                    first = rots[0]
                    for r, rec in rots.items():
                        self.assertEqual(rec.spec, first.spec)
                        self.assertEqual(rec.hole_cards, first.hole_cards)
                        self.assertEqual(len(set(rec.hole_cards)), n)
                    boards = [rec.board for rec in rots.values()]
                    for a in boards:
                        for b in boards:
                            short, long_ = sorted((a, b), key=len)
                            self.assertTrue(long_.startswith(short), (a, b))

    def test_every_position_holds_every_seat_once(self):
        for n in (2, 5, 9, 10):
            res = play_match(mixed_lineup(n), 3, seed=1, record_hands=True)
            for d, rots in by_deal(res).items():
                seats_of = defaultdict(list)
                for r, rec in rots.items():
                    self.assertEqual(sorted(rec.positions), list(range(n)))
                    for s, p in enumerate(rec.positions):
                        self.assertEqual(p, (s - r) % n)
                        self.assertEqual(rec.names[s], res.lineup[p])
                        seats_of[p].append(s)
                for p in range(n):
                    self.assertEqual(sorted(seats_of[p]), list(range(n)))


class MirrorTest(unittest.TestCase):
    def assert_all_zero(self, res):
        self.assertTrue((res.position_chips == 0).all(), res.position_chips)
        for st in res.stats.values():
            self.assertEqual(st.total_chips, 0)
            self.assertEqual(st.bb_per_100, 0.0)
            self.assertTrue((st.samples == 0).all())

    def test_deterministic_bots_against_copies_give_exactly_zero(self):
        for bot in ("calling", "maniac", "rule"):
            for n in (2, 6, 9):
                with self.subTest(bot=bot, n=n, names="same"):
                    res = play_match([bot] * n, 6, seed=n)
                    self.assertEqual(list(res.stats), [bot])
                    self.assert_all_zero(res)
                with self.subTest(bot=bot, n=n, names="distinct"):
                    lineup = [Entrant(f"{bot}{i}", bot) for i in range(n)]
                    res = play_match(lineup, 6, seed=n)
                    self.assertEqual(len(res.stats), n)
                    self.assert_all_zero(res)

    def test_mirror_with_training_stacks(self):
        for bot in ("maniac", "rule"):
            res = play_match([Entrant(f"x{i}", bot) for i in range(6)], 10, 7, stacks=StackConfig())
            self.assert_all_zero(res)

    def test_random_against_copies_is_near_zero(self):
        for n in (2, 6):
            with self.subTest(n=n):
                lineup = [Entrant("random-a", "random")] + ["random"] * (n - 1)
                res = play_match(lineup, 500, seed=0)
                st = res.stats["random-a"]
                self.assertEqual(st.hands, n * 500)
                self.assertLess(st.ci95, 100.0)
                self.assertLessEqual(abs(st.bb_per_100), st.ci95, (st.bb_per_100, st.ci95))


class AccountingTest(unittest.TestCase):
    def test_totals_sum_to_zero(self):
        for n in (2, 4, 7):
            res = play_match(mixed_lineup(n), 10, seed=n, stacks=StackConfig())
            self.assertEqual(sum(st.total_chips for st in res.stats.values()), 0)
            self.assertEqual(int(res.position_chips.sum()), 0)
            self.assertTrue((res.position_chips.sum(axis=0) == 0).all())

    def test_bad_net_raises(self):
        original = rl_batch.FinishedHand

        def bad(**kwargs):
            net = np.array(kwargs["net_bb"], dtype=np.float64)
            net[0] += 0.01
            kwargs["net_bb"] = net
            return original(**kwargs)

        with mock.patch.object(rl_batch, "FinishedHand", bad):
            with self.assertRaises(AccountingError):
                play_match(["calling", "calling"], 3, seed=0)

    def test_stats_units_and_positions(self):
        res = play_match(["rule", "random", "random"], 20, seed=5)
        self.assertEqual(res.stats["rule"].positions, (0,))
        self.assertEqual(res.stats["random"].positions, (1, 2))
        self.assertEqual(res.stats["rule"].hands, 60)
        self.assertEqual(res.stats["random"].hands, 120)
        for name, st in res.stats.items():
            chips = res.position_chips[list(st.positions)].sum(axis=0)
            s = summarize(chips, len(st.positions), 3)
            self.assertEqual(st.total_chips, s.total_chips)
            self.assertAlmostEqual(st.total_bb, st.total_chips / 200)
            self.assertAlmostEqual(st.bb_per_100, 100 * st.total_bb / st.hands)

    def test_validation(self):
        with self.assertRaises(ValueError):
            play_match(["calling"], 3, 0)
        with self.assertRaises(ValueError):
            play_match(["calling"] * 11, 3, 0)
        with self.assertRaises(ValueError):
            play_match(["calling"] * 3, 3, 0, table_size=4)
        with self.assertRaises(ValueError):
            play_match(["calling"] * 3, 0, 0)
        with self.assertRaises(ValueError):
            play_match(["calling"] * 3, 2, 0, stacks=[100.0, 100.0])
        with self.assertRaises(ValueError):
            play_match(["shark", "calling"], 2, 0)


class DeterminismTest(unittest.TestCase):
    LINEUP = ["rule", "random", "maniac", "random", "calling", "random"]

    def test_same_inputs_same_match(self):
        a = play_match(self.LINEUP, 12, seed=3, n_histories=5, stacks=StackConfig())
        b = play_match(self.LINEUP, 12, seed=3, n_histories=5, stacks=StackConfig())
        np.testing.assert_array_equal(a.position_chips, b.position_chips)
        self.assertEqual(a.histories, b.histories)
        self.assertEqual(list(a.stats), list(b.stats))
        for name in a.stats:
            sa, sb = a.stats[name], b.stats[name]
            self.assertEqual(sa.total_chips, sb.total_chips)
            self.assertEqual(sa.bb_per_100, sb.bb_per_100)
            np.testing.assert_array_equal(sa.samples, sb.samples)
        c = play_match(self.LINEUP, 12, seed=4, n_histories=5, stacks=StackConfig())
        self.assertFalse(np.array_equal(a.position_chips, c.position_chips))

    def test_factories_and_ready_controllers(self):
        lineup = [
            Entrant("fac", lambda seed: RandomBot(seed)),
            Entrant("ready", RandomBot(11)),
            Entrant("rule", "rule"),
        ]
        a = play_match(lineup, 8, seed=2)
        lineup2 = [
            Entrant("fac", lambda seed: RandomBot(seed)),
            Entrant("ready", RandomBot(11)),
            Entrant("rule", "rule"),
        ]
        b = play_match(lineup2, 8, seed=2)
        np.testing.assert_array_equal(a.position_chips, b.position_chips)


class CombineTest(unittest.TestCase):
    def test_combine_equals_recomputing_from_concatenated_chips(self):
        lineup = ["rule", "random", "random"]
        m1 = play_match(lineup, 5, seed=1, n_histories=2)
        m2 = play_match(lineup, 7, seed=2, n_histories=3)
        c = combine(m1, m2)
        self.assertEqual(c.n_deals, 12)
        chips = np.concatenate([m1.position_chips, m2.position_chips], axis=1)
        np.testing.assert_array_equal(c.position_chips, chips)
        self.assertEqual(c.histories, m1.histories + m2.histories)
        for name, st in c.stats.items():
            s = summarize(chips[list(st.positions)].sum(axis=0), len(st.positions), 3)
            self.assertEqual(st.total_chips, s.total_chips)
            self.assertEqual(st.hands, s.hands)
            self.assertAlmostEqual(st.bb_per_100, s.bb_per_100)
            self.assertAlmostEqual(st.ci95, s.ci95)
            np.testing.assert_array_equal(st.samples, s.samples)
        with self.assertRaises(ValueError):
            combine(m1, play_match(["rule", "calling", "random"], 2, seed=1))


class HistoriesTest(unittest.TestCase):
    def test_count_and_reproducibility(self):
        lineup = ["rule", "random", "calling", "maniac"]
        a = play_match(lineup, 10, seed=8, n_histories=5)
        self.assertEqual(len(a.histories), 5)
        for h in a.histories:
            self.assertTrue(h.startswith("Deal "), h)
            self.assertIn("4 players", h)
            self.assertIn("Net:", h)
        b = play_match(lineup, 10, seed=8, n_histories=5)
        self.assertEqual(a.histories, b.histories)
        self.assertEqual(play_match(lineup, 2, seed=8, n_histories=0).histories, ())
        self.assertEqual(len(play_match(lineup, 2, seed=8, n_histories=100).histories), 8)
        self.assertIsNone(a.hands)


if __name__ == "__main__":
    unittest.main()
