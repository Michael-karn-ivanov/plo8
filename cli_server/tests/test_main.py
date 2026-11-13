import contextlib
import io
import socket
import unittest

from engine import DEFAULT_ACTION_TIMEOUT, MAX_PLAYERS

from cli_server import main as cli


class ArgumentsTest(unittest.TestCase):
    def assert_rejected(self, argv):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                cli.parse_args(argv)
        self.assertNotEqual(cm.exception.code, 0)

    def test_defaults(self):
        args = cli.parse_args([])
        self.assertEqual(args.port, 8008)
        self.assertEqual(args.grace_period, 15.0)
        self.assertEqual(args.action_timeout, DEFAULT_ACTION_TIMEOUT)
        self.assertEqual(args.max_seats, MAX_PLAYERS)
        self.assertIsNone(args.max_buy_in)
        self.assertIsNone(args.seed)
        self.assertEqual(args.log_level, "INFO")

    def test_values(self):
        args = cli.parse_args(
            [
                "--port", "0",
                "--grace-period", "0.5",
                "--action-timeout", "30",
                "--max-seats", "6",
                "--max-buy-in", "100",
                "--seed", "42",
                "--log-level", "debug",
            ]
        )
        self.assertEqual(args.port, 0)  # pick a free port
        self.assertEqual(args.grace_period, 0.5)
        self.assertEqual(args.log_level, "DEBUG")
        self.assertEqual(
            cli.env_kwargs(args),
            {"action_timeout": 30.0, "max_seats": 6, "max_buy_in": 100.0, "seed": 42},
        )
        self.assertEqual(cli.parse_args(["--grace-period", "0"]).grace_period, 0.0)

    def test_invalid_values_are_rejected(self):
        for argv in (
            ["--port", "-1"],
            ["--port", "65536"],
            ["--port", "http"],
            ["--grace-period", "-1"],
            ["--grace-period", "soon"],
            ["--grace-period", "inf"],
            ["--action-timeout", "0"],
            ["--action-timeout", "nan"],
            ["--max-seats", "1"],
            ["--max-seats", "11"],
            ["--max-buy-in", "0"],
            ["--seed", "x"],
            ["--log-level", "LOUD"],
            ["--unknown"],
        ):
            with self.subTest(argv=argv):
                self.assert_rejected(argv)


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


if __name__ == "__main__":
    unittest.main()
