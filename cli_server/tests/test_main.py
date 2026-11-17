import contextlib
import io
import socket
import unittest
from unittest import mock

from engine import MAX_PLAYERS

from cli_server import main as cli

SAMPLE = "12_heads_up_three_raises_preflop_bb_scoops"


def parse_rejected(test, argv):
    """Parse `argv`, expecting an argparse usage error (exit 2); return stderr."""
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        with test.assertRaises(SystemExit) as cm:
            cli.parse_args(argv)
    test.assertEqual(cm.exception.code, 2)
    return stderr.getvalue()


class ArgumentsTest(unittest.TestCase):
    def test_defaults(self):
        args = cli.parse_args([])
        self.assertEqual(args.port, 8008)
        self.assertEqual(args.grace_period, 15.0)
        self.assertEqual(args.action_timeout, 180.0)  # spec, not the engine's 600 s
        self.assertEqual(args.hand_pause, 3.0)
        self.assertEqual(args.max_seats, MAX_PLAYERS)
        self.assertIsNone(args.max_buy_in)
        self.assertIsNone(args.seed)
        self.assertEqual(args.log_level, "INFO")
        self.assertFalse(args.e2e_test)
        self.assertIsNone(args.case)
        self.assertEqual(cli.env_kwargs(args)["action_timeout"], 180.0)

    def test_values(self):
        args = cli.parse_args(
            [
                "--port", "0",
                "--grace-period", "0.5",
                "--action-timeout", "30",
                "--hand-pause", "1.5",
                "--max-seats", "6",
                "--max-buy-in", "100",
                "--seed", "42",
                "--log-level", "debug",
            ]
        )
        self.assertEqual(args.port, 0)  # pick a free port
        self.assertEqual(args.grace_period, 0.5)
        self.assertEqual(args.hand_pause, 1.5)
        self.assertEqual(args.log_level, "DEBUG")
        self.assertEqual(
            cli.env_kwargs(args),
            {"action_timeout": 30.0, "max_seats": 6, "max_buy_in": 100.0, "seed": 42},
        )
        self.assertEqual(cli.parse_args(["--grace-period", "0"]).grace_period, 0.0)
        self.assertEqual(cli.parse_args(["--hand-pause", "0"]).hand_pause, 0.0)  # no pause

    def test_invalid_values_are_rejected(self):
        for argv in (
            ["--port", "-1"],
            ["--port", "65536"],
            ["--port", "http"],
            ["--grace-period", "-1"],
            ["--grace-period", "soon"],
            ["--grace-period", "inf"],
            ["--action-timeout", "0"],
            ["--action-timeout", "-5"],
            ["--action-timeout", "nan"],
            ["--hand-pause", "-0.5"],
            ["--hand-pause", "later"],
            ["--hand-pause", "inf"],
            ["--max-seats", "1"],
            ["--max-seats", "11"],
            ["--max-buy-in", "0"],
            ["--seed", "x"],
            ["--log-level", "LOUD"],
            ["--unknown"],
        ):
            with self.subTest(argv=argv):
                parse_rejected(self, argv)


class E2EArgumentsTest(unittest.TestCase):
    def test_e2e_defaults(self):
        args = cli.parse_args(["--e2e_test"])
        self.assertTrue(args.e2e_test)
        self.assertIsNone(args.case)  # every case
        self.assertEqual(args.action_timeout, 10.0)  # a stuck replay fails quickly
        self.assertEqual(args.seed, 0)  # reproducible filler cards
        self.assertTrue(cli.parse_args(["--e2e-test"]).e2e_test)  # alias

    def test_e2e_explicit_values_are_kept(self):
        args = cli.parse_args(["--e2e_test", "--action-timeout", "3", "--seed", "9"])
        self.assertEqual((args.action_timeout, args.seed), (3.0, 9))

    def test_case_with_or_without_json(self):
        a = cli.parse_args(["--e2e_test", "--case", SAMPLE])
        b = cli.parse_args(["--e2e_test", "--case", f"{SAMPLE}.json"])
        self.assertEqual(a.case, SAMPLE)
        self.assertEqual(b.case, SAMPLE)

    def test_case_without_e2e_test(self):
        err = parse_rejected(self, ["--case", SAMPLE])
        self.assertIn("--e2e_test", err)

    def test_unknown_case(self):
        err = parse_rejected(self, ["--e2e_test", "--case", "no_such_case"])
        self.assertIn("no_such_case", err)
        self.assertIn(SAMPLE, err)  # the available cases are listed


class MainTest(unittest.TestCase):
    def test_port_in_use_exits_with_error(self):
        taken = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.addCleanup(taken.close)
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        port = taken.getsockname()[1]
        with contextlib.redirect_stderr(io.StringIO()):
            code = cli.main(["--port", str(port), "--log-level", "CRITICAL"])
        self.assertEqual(code, 1)

    def test_e2e_mode_runs_the_cases(self):
        with mock.patch.object(cli, "run_cases", return_value=0) as run:
            code = cli.main(["--e2e_test", "--case", f"{SAMPLE}.json", "--log-level", "CRITICAL"])
        self.assertEqual(code, 0)
        run.assert_called_once_with([SAMPLE], seed=0, action_timeout=10.0)

        with mock.patch.object(cli, "run_cases", return_value=1) as run:
            code = cli.main(["--e2e_test", "--seed", "4", "--log-level", "CRITICAL"])
        self.assertEqual(code, 1)
        run.assert_called_once_with(None, seed=4, action_timeout=10.0)

    def test_e2e_mode_interrupted(self):
        with mock.patch.object(cli, "run_cases", side_effect=KeyboardInterrupt):
            code = cli.main(["--e2e_test", "--log-level", "CRITICAL"])
        self.assertEqual(code, 1)


if __name__ == "__main__":
    unittest.main()
