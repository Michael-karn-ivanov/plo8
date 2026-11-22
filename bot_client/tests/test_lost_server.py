"""The server dies mid-game: every bot client exits with status 1 and a log
line, without hanging."""

import os
import re
import subprocess
import sys
import threading
import unittest

try:
    from . import helpers
except ImportError:  # run as a top-level module
    import helpers  # type: ignore[no-redef]

PORT_LINE = re.compile(r"listening on [0-9.]+:(\d+);")


class ServerProcess:
    """`python -m cli_server.main` as a child process; its stderr is drained by a
    thread (the server logs every hand) and scanned for the port."""

    def __init__(self):
        env = dict(os.environ)
        # The child finds the same modules (inside the buck2 test too).
        env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "cli_server.main",
                "--port",
                "0",
                "--grace-period",
                "0.05",
                "--hand-pause",
                "0",
                "--action-timeout",
                "5",
                "--seed",
                "1",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=env,
        )
        self.port = None
        self.lines = []
        self.found = threading.Event()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        for raw in self.proc.stderr:
            line = raw.decode("utf-8", "replace")
            if len(self.lines) < 200:
                self.lines.append(line)
            if self.port is None:
                m = PORT_LINE.search(line)
                if m:
                    self.port = int(m.group(1))
                    self.found.set()
        self.found.set()

    def kill(self):
        if self.proc.poll() is None:
            self.proc.kill()  # SIGKILL
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


class LostServerTest(unittest.TestCase):
    def test_bots_exit_1_when_the_server_is_killed(self):
        server = ServerProcess()
        self.addCleanup(server.kill)
        server.found.wait(60)
        self.assertIsNotNone(server.port, "".join(server.lines))

        bots = [
            helpers.run_bot(helpers.bot_argv(server.port, name, "--bot", bot))
            for name, bot in (("A", "rule"), ("B", "calling"))
        ]
        for _thread, box, client in bots:
            self.assertIsNotNone(client, box)
        helpers.wait_for(
            lambda: all(c.hands_played >= 3 for _t, _b, c in bots), 60, "a few hands"
        )

        with self.assertLogs("bot_client", level="ERROR") as logs:
            server.kill()  # no `close`, no `left`
            for thread, _box, _client in bots:
                thread.join(15)
        for thread, box, _client in bots:
            self.assertFalse(thread.is_alive(), "a bot client is still running")
            self.assertEqual(box.get("code"), 1)
        lost = [line for line in logs.output if "connection to the server lost" in line]
        self.assertGreaterEqual(len(lost), 2, logs.output)


if __name__ == "__main__":
    unittest.main()
