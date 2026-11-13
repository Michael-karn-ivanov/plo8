import json
import socket
import threading
import time
import unittest

from entities import Board, Hand, Street
from engine import (
    Action,
    ActionType,
    HandEndReply,
    HandResult,
    HandStarted,
    JoinReply,
    LegalActions,
    Leave,
    PlayerActed,
    PlayerJoined,
    PotView,
    SeatView,
    TableView,
)

from cli_server.protocol import decode_server_message
from cli_server.session import ClientSession, SessionError

HOST = "127.0.0.1"

LEGAL = LegalActions(
    can_check=False, call_amount=0.5, can_raise=True, raise_to=2.0, bet_unit=1.0, raises_left=3
)


def acted(name, action=ActionType.CALL, reason=None):
    return PlayerActed(0, name, Street.PREFLOP, action, 0.5, 1.0, False, 39.0, reason, hand_no=1, seq=5)


def view(history=()):
    return TableView(
        hand_number=1,
        seat=0,
        hole_cards=Hand.parse("AsKd7h2c"),
        street=Street.PREFLOP,
        board=Board.empty(),
        pot=1.5,
        side_pots=(PotView(1.5, (0, 1)),),
        current_bet=1.0,
        bet_unit=1.0,
        raises=1,
        button=0,
        sb_seat=0,
        bb_seat=1,
        seats=(
            SeatView(0, "A", 39.5, 0.5, 0.5, False, False, True),
            SeatView(1, "B", 39.0, 1.0, 1.0, False, False, True),
        ),
        history=tuple(history),
    )


def result():
    return HandResult(
        hand_number=1,
        button=0,
        sb_seat=0,
        bb_seat=1,
        board=Board.empty(),
        showdown=False,
        shown={},
        pots=(),
        net={0: -0.5, 1: 0.5},
        stacks={0: 39.5, 1: 40.5},
        history=(),
        names={0: "A", 1: "B"},
        dealt_in={0: True, 1: True},
    )


class Peer:
    """The client end of a session's callback connection."""

    def __init__(self, conn):
        self.conn = conn
        self.conn.settimeout(5.0)
        self.file = conn.makefile("rb")

    def read(self):
        line = self.file.readline()
        return json.loads(line) if line else None

    def close(self):
        try:
            self.conn.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.conn.close()


class SessionTestCase(unittest.TestCase):
    def make_session(self, name="A", start=True):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind((HOST, 0))
        listener.listen(1)
        self.released = []
        session = ClientSession.connect(
            name, listener.getsockname()[1], on_close=self.released.append
        )
        conn, _ = listener.accept()
        listener.close()
        if start:
            session.start()
        peer = Peer(conn)
        self.addCleanup(peer.close)
        self.addCleanup(session.abort)
        return session, peer

    def recv_in_thread(self, session, timeout=5.0):
        """session.recv() with a deadline: returns ("ok", item) or ("error", exc)."""
        out = []

        def run():
            try:
                out.append(("ok", session.recv()))
            except Exception as e:  # noqa: BLE001
                out.append(("error", e))

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(timeout)
        self.assertFalse(t.is_alive(), "recv() did not return")
        return out[0]


class MessageMappingTest(SessionTestCase):
    def test_engine_messages_become_json_lines(self):
        session, peer = self.make_session()
        msgs = [
            ("seated", view()),
            ("hole_cards", Hand.parse("AsKd7h2c"), view()),
            ("event", acted("B")),
            ("turn", 1, view([acted("B")]), LEGAL),
            ("hand_end", 2, 0, result()),
            ("sat_out",),
            ("returned",),
            ("left", 39.5),
        ]
        for msg in msgs:
            session.send(msg)
        for msg in msgs:
            line = peer.read()
            self.assertEqual(line["type"], msg[0])
            self.assertEqual(line["to"], "A")
            self.assertEqual(decode_server_message(line), msg)

    def test_joined_is_captured_not_forwarded(self):
        session, peer = self.make_session()
        reply = JoinReply(True, queued=False, name="A", seat=0)
        session.send(("joined", reply))
        self.assertEqual(session.join_reply, reply)
        session.send(("sat_out",))
        self.assertEqual(peer.read()["type"], "sat_out")

    def test_feed_and_recv(self):
        session, _ = self.make_session()
        session.feed("hello")
        self.assertEqual(self.recv_in_thread(session), ("ok", "hello"))


class RepliesTest(SessionTestCase):
    def test_action_must_match_pending_turn(self):
        session, peer = self.make_session()
        with self.assertRaises(SessionError):
            session.submit_action(1, Action.call())  # no turn yet
        session.send(("turn", 1, view(), LEGAL))
        self.assertEqual(session.pending_turn, (1, LEGAL))
        with self.assertRaises(SessionError):
            session.submit_action(2, Action.call())  # wrong seq
        with self.assertRaisesRegex(SessionError, "not legal"):
            session.submit_action(1, Action.check())  # check is not legal facing a bet
        self.assertEqual(session.pending_turn, (1, LEGAL))  # not used up
        session.submit_action(1, Action.call())
        self.assertIsNone(session.pending_turn)
        self.assertEqual(self.recv_in_thread(session), ("ok", ("action", 1, Action.call())))
        with self.assertRaises(SessionError):
            session.submit_action(1, Action.call())  # duplicate

    def test_leave_is_always_a_legal_answer(self):
        session, _ = self.make_session()
        session.send(("turn", 3, view(), LEGAL))
        session.submit_action(3, Leave())
        self.assertEqual(self.recv_in_thread(session), ("ok", ("action", 3, Leave())))

    def test_timed_out_turn_is_cleared_by_own_fold(self):
        session, _ = self.make_session()
        session.send(("turn", 5, view(), LEGAL))
        session.send(("event", acted("B")))  # someone else: still pending
        self.assertIsNotNone(session.pending_turn)
        session.send(("event", acted("A", ActionType.FOLD, "timeout")))
        self.assertIsNone(session.pending_turn)
        with self.assertRaises(SessionError):
            session.submit_action(5, Action.fold())

    def test_hand_end_reply(self):
        session, _ = self.make_session()
        with self.assertRaises(SessionError):
            session.submit_hand_end(2, HandEndReply())
        session.send(("hand_end", 2, 0, result()))
        with self.assertRaises(SessionError):
            session.submit_hand_end(3, HandEndReply())
        session.submit_hand_end(2, HandEndReply(rebuy_bb=10.0))
        self.assertEqual(
            self.recv_in_thread(session), ("ok", ("hand_end", 2, HandEndReply(rebuy_bb=10.0)))
        )
        with self.assertRaises(SessionError):
            session.submit_hand_end(2, HandEndReply())

    def test_next_hand_clears_pending_hand_end(self):
        session, _ = self.make_session()
        session.send(("hand_end", 2, 0, result()))
        session.send(("event", HandStarted(0, 0, 1, ((0, "A", 39.5), (1, "B", 40.5)), hand_no=2, seq=9)))
        self.assertIsNone(session.pending_hand_end)

    def test_unsolicited_leave(self):
        session, _ = self.make_session()
        session.submit_leave()
        self.assertEqual(self.recv_in_thread(session), ("ok", Leave()))


class ChannelTest(SessionTestCase):
    def test_send_does_not_block_on_a_client_that_does_not_read(self):
        session, peer = self.make_session()
        events = [PlayerJoined(i % 10, f"player{i}", 40.0, seq=i) for i in range(60)]
        big = ("seated", view(history=events))  # several KB per line
        done = threading.Event()

        def flood():
            for _ in range(2000):  # far more than the socket buffers hold
                session.send(big)
            done.set()

        t = threading.Thread(target=flood, daemon=True)
        t.start()
        self.assertTrue(done.wait(30), "send() blocked on a client that does not read")
        # The client can still read in order afterwards.
        self.assertEqual(peer.read()["type"], "seated")

    def test_broken_callback_socket_ends_recv(self):
        session, peer = self.make_session()
        peer.close()
        kind, exc = self.recv_in_thread(session)
        self.assertEqual(kind, "error")
        self.assertIsInstance(exc, EOFError)
        self.assertTrue(session.broken)
        self.assertEqual(self.released, [session])
        with self.assertRaises(OSError):
            session.send(("sat_out",))
        # recv keeps failing instead of blocking
        self.assertEqual(self.recv_in_thread(session)[0], "error")

    def test_close_message_flushes_then_closes(self):
        session, peer = self.make_session()
        for i in range(3):
            session.send(("event", PlayerJoined(i, f"P{i}", 40.0, seq=i + 1)))
        session.send(("close",))
        self.assertEqual(self.released, [session])  # released before the client reads it
        lines = [peer.read() for _ in range(4)]
        self.assertEqual([m["type"] for m in lines], ["event", "event", "event", "close"])
        self.assertEqual([m["event"]["seq"] for m in lines[:3]], [1, 2, 3])
        self.assertIsNone(peer.read())  # then EOF
        self.assertTrue(session.wait_closed(5))
        self.assertFalse(session.broken)
        kind, exc = self.recv_in_thread(session)
        self.assertIsInstance(exc, EOFError)
        with self.assertRaises(OSError):
            session.send(("sat_out",))
        self.assertEqual(self.released, [session])  # only once

    def test_close_without_message(self):
        session, peer = self.make_session()
        session.send(("sat_out",))
        session.close()
        self.assertEqual(peer.read()["type"], "sat_out")
        self.assertIsNone(peer.read())
        self.assertTrue(session.wait_closed(5))
        self.assertEqual(self.released, [session])

    def test_close_before_start(self):
        session, peer = self.make_session(start=False)
        session.close()
        self.assertTrue(session.wait_closed(1))
        self.assertIsNone(peer.read())
        self.assertEqual(self.released, [session])
        self.assertEqual(self.recv_in_thread(session)[0], "error")


if __name__ == "__main__":
    unittest.main()
