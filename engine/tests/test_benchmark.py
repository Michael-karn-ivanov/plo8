"""The benchmark workload as a library function (engine.benchmark)."""

import unittest

from engine.benchmark import BenchResult, format_report, run_benchmark

TINY = dict(tables=4, hands=50, warmup=10)


class RunBenchmarkTest(unittest.TestCase):
    def check(self, result, players, mode):
        self.assertIsInstance(result, BenchResult)
        self.assertEqual(result.players, players)
        self.assertEqual(result.hands, 50)
        self.assertEqual(result.tables, 4)
        self.assertEqual(result.mode, mode)
        self.assertGreater(result.decisions, 0)
        self.assertGreater(result.hands_per_s, 0)
        self.assertGreater(result.decisions_per_s, 0)
        self.assertGreaterEqual(result.seconds, 0)
        self.assertGreaterEqual(result.showdown_share, 0.0)
        self.assertLessEqual(result.showdown_share, 1.0)

    def test_step_mode(self):
        for players in (2, 6):
            with self.subTest(players=players):
                self.check(run_benchmark(players, **TINY), players, "step")

    def test_deterministic(self):
        for players in (2, 6):
            with self.subTest(players=players):
                a = run_benchmark(players, seed=3, **TINY)
                b = run_benchmark(players, seed=3, **TINY)
                self.assertEqual(
                    (a.hands, a.decisions, a.showdown_share),
                    (b.hands, b.decisions, b.showdown_share),
                )

    def test_views_and_callback_modes(self):
        self.check(run_benchmark(6, views=True, **TINY), 6, "step+views")
        self.check(run_benchmark(6, callback=True, **TINY), 6, "callback")
        a = run_benchmark(2, callback=True, seed=1, **TINY)
        b = run_benchmark(2, callback=True, seed=1, **TINY)
        self.assertEqual((a.decisions, a.showdown_share), (b.decisions, b.showdown_share))

    def test_views_do_not_change_the_workload(self):
        a = run_benchmark(6, seed=2, **TINY)
        b = run_benchmark(6, seed=2, views=True, **TINY)
        self.assertEqual((a.decisions, a.showdown_share), (b.decisions, b.showdown_share))

    def test_invalid_arguments(self):
        for kwargs in (
            dict(players=1),
            dict(players=11),
            dict(players=2, tables=0),
            dict(players=2, hands=0),
            dict(players=2, warmup=-1),
            dict(players=2, stack=0),
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    run_benchmark(**kwargs)


class FormatReportTest(unittest.TestCase):
    def test_one_line_per_result(self):
        results = [run_benchmark(2, **TINY), run_benchmark(6, callback=True, **TINY)]
        lines = format_report(results).splitlines()
        self.assertEqual(len(lines), 2)
        for line, r in zip(lines, results):
            self.assertIn(f"{r.hands_per_s:.0f}", line)
            self.assertIn(f"{r.decisions_per_s:.0f}", line)
            self.assertIn(r.mode, line)
        self.assertEqual(format_report([]), "")


if __name__ == "__main__":
    unittest.main()
