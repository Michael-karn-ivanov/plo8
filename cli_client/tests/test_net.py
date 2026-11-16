import json
import queue
import socket
import threading
import time
import unittest

from cli_client.net import DISCONNECTED, CallbackListener, ConnectionLost, ServerConnection

HOST = "127.0.0.1"


def get(inbox, timeout=5.0):
    return inbox.get(timeout=timeout)


class CallbackListenerTest(unittest.TestCase):
    def setUp(self):
        self.inbox = queue.Queue()
        self.listener = CallbackListener("Alice", self.inbox).start()
        self.addCleanup(self.listener.close)

    def connect(self):
        sock = socket.create_connection((HOST, self.listener.port), timeout=5)
        self.addCleanup(sock.close)
        return sock

    def test_delivers_in_order_and_drops_others(self):
        sock = self.connect()
        lines = [
            {"type": "event", "to": "Alice", "n": 1},
            {"type": "event", "to": "Bob", "n": 2},
            {"type": "turn", "to": "Alice", "n": 3},
            {"type": "lobby", "to": "Alice", "n": 4},
        ]
        sock.sendall(b"".join(json.dumps(m).encode() + b"\n" for m in lines))
        got = [get(self.inbox)["n"] for _ in range(3)]
        self.assertEqual(got, [1, 3, 4])
        self.assertEqual(self.listener.dropped, 1)

    def test_survives_malformed_lines(self):
        sock = self.connect()
        sock.sendall(b"not json\n{\"no_type\": 1}\n\n{\"type\": \"close\", \"to\": \"Alice\"}\n")
        self.assertEqual(get(self.inbox)["type"], "close")
        self.assertEqual(self.listener.malformed, 2)

    def test_reports_disconnect(self):
        sock = self.connect()
        sock.sendall(b'{"type": "seated", "to": "Alice"}\n')
        self.assertEqual(get(self.inbox)["type"], "seated")
        sock.close()
        self.assertEqual(get(self.inbox), DISCONNECTED)

    def test_new_connection_replaces_the_old_one(self):
        first = self.connect()
        first.sendall(b'{"type": "a", "to": "Alice"}\n')
        self.assertEqual(get(self.inbox)["type"], "a")
        second = self.connect()
        second.sendall(b'{"type": "b", "to": "Alice"}\n')
        self.assertEqual(get(self.inbox)["type"], "b")
        # The replaced connection ends without a disconnect marker.
        time.sleep(0.2)
        self.assertTrue(self.inbox.empty())


class _Server:
    """A tiny request server: `mode` "echo" answers each line, "close" closes
    every connection at once."""

    def __init__(self, mode="echo"):
        self.mode = mode
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((HOST, 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self.connections = 0
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            self.connections += 1
            if self.mode == "close":
                conn.close()
                continue
            threading.Thread(target=self._echo, args=(conn,), daemon=True).start()

    def _echo(self, conn):
        f = conn.makefile("rwb")
        try:
            for line in f:
                msg = json.loads(line)
                f.write(json.dumps({"type": "response", "ok": True, "echo": msg}).encode() + b"\n")
                f.flush()
        except (OSError, ValueError):
            pass
        finally:
            conn.close()

    def close(self):
        self.sock.close()


class ServerConnectionTest(unittest.TestCase):
    def test_pairs_requests_with_responses(self):
        server = _Server()
        self.addCleanup(server.close)
        conn = ServerConnection(server.port, timeout=5)
        self.addCleanup(conn.close)
        for i in range(5):
            response = conn.request({"type": "leave", "name": "Alice", "i": i})
            self.assertTrue(response["ok"])
            self.assertEqual(response["echo"]["i"], i)
        self.assertEqual(server.connections, 1)  # one persistent connection

    def test_closed_socket_raises_connection_lost(self):
        server = _Server(mode="close")
        self.addCleanup(server.close)
        conn = ServerConnection(server.port, timeout=5)
        self.addCleanup(conn.close)
        with self.assertRaises(ConnectionLost):
            conn.request({"type": "leave", "name": "Alice"})
        self.assertEqual(server.connections, 2)  # reconnected once

    def test_unreachable_server(self):
        s = socket.socket()
        s.bind((HOST, 0))
        port = s.getsockname()[1]
        s.close()
        conn = ServerConnection(port, timeout=2)
        with self.assertRaises(ConnectionLost):
            conn.request({"type": "leave", "name": "Alice"})


if __name__ == "__main__":
    unittest.main()
