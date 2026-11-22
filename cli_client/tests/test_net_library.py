"""The networking library `//cli_client:net` on its own (target
`net_library_test`, built against `:net` only).

Other projects (`//bot_client`) talk to the server with `cli_client.net` and
nothing else from cli_client. This test makes the same use of it, so a change
that makes `net.py` need the UI, the table model or `curses` fails here first.
"""

import importlib
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import unittest

import cli_client.net as net

HOST = "127.0.0.1"
OTHER_MODULES = (
    "cli_client.ui",
    "cli_client.controller",
    "cli_client.state",
    "cli_client.render",
    "cli_client.cards",
    "cli_client.main",
)


def modules_after_import(*modules):
    """`sys.modules` of a fresh child interpreter after importing `modules`."""
    code = (
        "import json, sys\n"
        f"for m in {list(modules)!r}:\n"
        "    __import__(m)\n"
        "print(json.dumps(sorted(sys.modules)))\n"
    )
    env = dict(os.environ)
    # The child finds the same modules (inside the buck2 test too).
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        timeout=60,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"importing {modules} failed:\n{proc.stderr.decode('utf-8', 'replace')}"
        )
    return set(json.loads(proc.stdout.decode("utf-8").strip().splitlines()[-1]))


class ImportIsolationTest(unittest.TestCase):
    def test_net_loads_no_curses_and_no_other_cli_client_module(self):
        # In a fresh interpreter, so the test runner's own imports do not count.
        loaded = modules_after_import("cli_client.net")
        self.assertIn("cli_client.net", loaded)
        self.assertNotIn("curses", loaded)
        self.assertNotIn("_curses", loaded)
        self.assertEqual(sorted(loaded & set(OTHER_MODULES)), [])

    def test_no_other_cli_client_module_is_loaded_here(self):
        self.assertEqual(sorted(m for m in OTHER_MODULES if m in sys.modules), [])

    def test_other_cli_client_modules_are_not_in_the_library(self):
        for name in OTHER_MODULES:
            with self.subTest(module=name):
                with self.assertRaises(ImportError):
                    importlib.import_module(name)

    def test_public_api(self):
        self.assertEqual(
            sorted(net.__all__),
            sorted(["HOST", "DISCONNECTED", "ConnectionLost", "ServerConnection", "CallbackListener"]),
        )
        for name in net.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(net, name))
        self.assertEqual(net.HOST, "127.0.0.1")
        self.assertEqual(net.DISCONNECTED, ("disconnected",))
        self.assertTrue(issubclass(net.ConnectionLost, Exception))


class _StubServer:
    """Answers each request with `ok: true`. A `join` also opens the callback
    connection to the request's `port` and sends a `seated` message, a message
    for another player, and then a `close`."""

    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((HOST, 0))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        self.callbacks = []
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        f = conn.makefile("rwb")
        try:
            for line in f:
                msg = json.loads(line)
                if msg.get("type") == "join":
                    cb = socket.create_connection((HOST, msg["port"]), timeout=5)
                    self.callbacks.append(cb)
                    name = msg["name"]
                    for m in (
                        {"type": "seated", "to": name, "seat": 0},
                        {"type": "seated", "to": "Somebody", "seat": 1},
                        {"type": "close", "to": name},
                    ):
                        cb.sendall(json.dumps(m).encode() + b"\n")
                f.write(json.dumps({"type": "response", "ok": True, "echo": msg}).encode() + b"\n")
                f.flush()
        except (OSError, ValueError):
            pass
        finally:
            conn.close()

    def close(self):
        self.sock.close()
        for cb in self.callbacks:
            cb.close()


class RoundTripTest(unittest.TestCase):
    def test_round_trip_with_a_plain_queue(self):
        server = _StubServer()
        self.addCleanup(server.close)
        inbox = queue.Queue()
        listener = net.CallbackListener("Bot1", inbox, port=0).start()
        self.addCleanup(listener.close)
        self.assertGreater(listener.port, 0)
        conn = net.ServerConnection(server.port, timeout=5)
        self.addCleanup(conn.close)

        response = conn.request({"type": "join", "name": "Bot1", "port": listener.port})
        self.assertTrue(response["ok"])
        self.assertEqual(response["echo"]["name"], "Bot1")

        # Raw dicts on the caller's queue; the other player's message is dropped.
        self.assertEqual(inbox.get(timeout=5), {"type": "seated", "to": "Bot1", "seat": 0})
        self.assertEqual(inbox.get(timeout=5), {"type": "close", "to": "Bot1"})
        self.assertEqual(listener.dropped, 1)

        for cb in server.callbacks:
            cb.close()
        self.assertEqual(inbox.get(timeout=5), net.DISCONNECTED)

        conn.close()
        with self.assertRaises(net.ConnectionLost):
            conn.request({"type": "leave", "name": "Bot1"})


if __name__ == "__main__":
    unittest.main()
