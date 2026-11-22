"""Command line: defaults, accepted bots, and invalid arguments rejected before
any socket is opened."""

import contextlib
import io
import unittest
from unittest import mock

from arena import BOT_NAMES

import bot_client.main as bc_main


def fail(*args, **kwargs):
    raise AssertionError("a socket was opened for invalid arguments")


class ArgsTest(unittest.TestCase):
    def test_defaults(self):
        args = bc_main.parse_args([])
        self.assertEqual(args.server_port, 8008)
        self.assertEqual(args.port, 0)
        self.assertEqual(args.stack, 100.0)
        self.assertEqual(args.bot, "rule")
        self.assertEqual(args.seed, 0)
        self.assertEqual(args.rebuy, "topup")
        self.assertIsNone(args.hands)
        self.assertEqual(args.name, "rule-bot")
        self.assertEqual(args.log_level, "info")

    def test_every_arena_bot_is_accepted(self):
        for name in BOT_NAMES:
            with self.subTest(bot=name):
                args = bc_main.parse_args(["--bot", name])
                self.assertEqual(args.bot, name)
                self.assertEqual(args.name, f"{name}-bot")

    def test_explicit_values(self):
        args = bc_main.parse_args(
            ["--server-port", "9000", "--port", "9101", "--name", " Bot ", "--stack", "2.5",
             "--bot", "random", "--seed", "7", "--rebuy", "never", "--hands", "3",
             "--log-level", "DEBUG"]
        )
        self.assertEqual(
            (args.server_port, args.port, args.name, args.stack, args.bot, args.seed,
             args.rebuy, args.hands, args.log_level),
            (9000, 9101, "Bot", 2.5, "random", 7, "never", 3, "debug"),
        )

    def run_main(self, argv):
        stderr = io.StringIO()
        with mock.patch.object(bc_main, "CallbackListener", fail), mock.patch.object(
            bc_main, "ServerConnection", fail
        ), contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as cm:
                bc_main.main(argv)
        return cm.exception.code, stderr.getvalue()

    def test_unknown_bot_lists_the_known_names(self):
        code, err = self.run_main(["--bot", "genius"])
        self.assertEqual(code, 2)
        for name in BOT_NAMES:
            self.assertIn(name, err)

    def test_invalid_arguments_exit_2_before_connecting(self):
        cases = [
            ["--port", "-1"],
            ["--port", "65536"],
            ["--port", "abc"],
            ["--server-port", "0"],
            ["--server-port", "70000"],
            ["--stack", "0"],
            ["--stack", "-5"],
            ["--stack", "nan"],
            ["--stack", "inf"],
            ["--hands", "0"],
            ["--hands", "-2"],
            ["--rebuy", "sometimes"],
            ["--name", "   "],
            ["--log-level", "loud"],
            ["--seed", "x"],
        ]
        for argv in cases:
            with self.subTest(argv=argv):
                code, err = self.run_main(argv)
                self.assertEqual(code, 2)
                self.assertIn("error", err)


if __name__ == "__main__":
    unittest.main()
