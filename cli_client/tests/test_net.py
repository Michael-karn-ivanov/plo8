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

    def test_unknown_types_are_passed_through(self):
        sock = self.connect()
        sock.sendall(b'{"type": "something_new", "to": "Alice", "x": 1}\n')
        self.assertEqual(get(self.inbox), {"type": "something_new", "to": "Alice", "x": 1})

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

    def test_nothing_is_reported_after_close(self):
        sock = self.connect()
        sock.sendall(b'{"type": "seated", "to": "Alice"}\n')
        self.assertEqual(get(self.inbox)["type"], "seated")
        self.listener.close()
        self.listener.close()  # idempotent
        try:
            sock.sendall(b'{"type": "late", "to": "Alice"}\n')
        except OSError:
            pass
        sock.close()
        time.sleep(0.3)
        self.assertTrue(self.inbox.empty())

    def test_bind_error_is_raised_in_the_constructor(self):
        # self.listener is listening on its port, so a second bind fails.
        with self.assertRaises(OSError):
            other = CallbackListener("Bob", queue.Queue(), port=self.listener.port)
            other.close()

    def test_threads_are_daemons(self):
        self.connect()
        time.sleep(0.1)
        names = ("callback-accept", "callback-reader")
        threads = [t for t in threading.enumerate() if t.name in names]
        self.assertTrue(threads)
        for t in threads:
            self.assertTrue(t.daemon, t.name)


class _Server:
    """A tiny request server.

    `mode`:
    - "echo": answers each line;
    - "close": closes every connection at once;
    - "once": answers one line per connection, then closes it;
    - "bad_first": answers the first line of a connection with garbage, then echoes.
    """

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
            for n, line in enumerate(f):
                msg = json.loads(line)
                if self.mode == "bad_first" and n == 0:
                    f.write(b"this is not json\n")
                else:
                    f.write(json.dumps({"type": "response", "ok": True, "echo": msg}).encode() + b"\n")
                f.flush()
                if self.mode == "once":
                    break
        except (OSError, ValueError):
            pass
        finally:
            try:
                f.close()
            except OSError:
                pass
            conn.close()

    def close(self):
        self.sock.close()


class ServerConnectionTest(unittest.TestCase):
    def test_pairs_requests_with_responses(self):
        server = _Server()
        self.addCleanup(server.close)
        conn = ServerConnection(server.port, timeout=5)
        self.addCleanup(conn.close)
        self.assertEqual(server.connections, 0)  # opened lazily
        for i in range(5):
            response = conn.request({"type": "leave", "name": "Alice", "i": i})
            self.assertTrue(response["ok"])
            self.assertEqual(response["echo"]["i"], i)
        self.assertEqual(server.connections, 1)  # one persistent connection

    def test_reconnects_once_after_a_dropped_connection(self):
        server = _Server(mode="once")
        self.addCleanup(server.close)
        conn = ServerConnection(server.port, timeout=5)
        self.addCleanup(conn.close)
        for i in range(3):
            response = conn.request({"type": "leave", "name": "Alice", "i": i})
            self.assertTrue(response["ok"])
            self.assertEqual(response["echo"]["i"], i)
        self.assertEqual(server.connections, 3)

    def test_malformed_response_keeps_the_stream_in_sync(self):
        server = _Server(mode="bad_first")
        self.addCleanup(server.close)
        conn = ServerConnection(server.port, timeout=5)
        self.addCleanup(conn.close)
        bad = conn.request({"type": "leave", "name": "Alice", "i": 0})
        self.assertFalse(bad["ok"])
        self.assertTrue(bad["error"].startswith("bad response"))
        good = conn.request({"type": "leave", "name": "Alice", "i": 1})
        self.assertTrue(good["ok"])
        self.assertEqual(good["echo"]["i"], 1)
        self.assertEqual(server.connections, 1)

    def test_closed_socket_raises_connection_lost(self):
        server = _Server(mode="close")
        self.addCleanup(server.close)
        conn = ServerConnection(server.port, timeout=5)
        self.addCleanup(conn.close)
        with self.assertRaises(ConnectionLost):
            conn.request({"type": "leave", "name": "Alice"})
        self.assertEqual(server.connections, 2)  # reconnected once

    def test_request_after_close_raises_connection_lost(self):
        server = _Server()
        self.addCleanup(server.close)
        conn = ServerConnection(server.port, timeout=5)
        self.assertTrue(conn.request({"type": "leave", "name": "Alice"})["ok"])
        conn.close()
        conn.close()  # idempotent
        with self.assertRaises(ConnectionLost):
            conn.request({"type": "leave", "name": "Alice"})
        self.assertEqual(server.connections, 1)  # no reconnect after close()

    def test_requests_from_many_threads(self):
        server = _Server()
        self.addCleanup(server.close)
        conn = ServerConnection(server.port, timeout=5)
        self.addCleanup(conn.close)
        results = {}

        def worker(k):
            results[k] = [conn.request({"type": "leave", "k": k, "i": i})["echo"] for i in range(10)]

        threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        for k in range(4):
            self.assertEqual([(e["k"], e["i"]) for e in results[k]], [(k, i) for i in range(10)])
        self.assertEqual(server.connections, 1)

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
