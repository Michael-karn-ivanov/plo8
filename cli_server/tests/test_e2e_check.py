import unittest

from cli_server.e2e.case import parse_case
from cli_server.e2e.check import check, normalize_event
from cli_server.e2e.expect import build_expected

try:
    from cli_server_tests.tests.e2e_helpers import correct_runs, fold_case_data, sample_data
except ImportError:  # run directly from the tests directory
    from e2e_helpers import correct_runs, fold_case_data, sample_data


def index_of(msgs, pred, start=0):
    return next(i for i in range(start, len(msgs)) if pred(msgs[i]))


def is_event(kind, **fields):
    def pred(m):
        if m.get("type") != "event" or m["event"].get("type") != kind:
            return False
        return all(m["event"].get(k) == v for k, v in fields.items())

    return pred


class CheckTestCase(unittest.TestCase):
    def setUp(self):
        self.case = parse_case(sample_data())
        self.expected = build_expected(self.case)
        self.runs = correct_runs(self.case)

    def msgs(self, pid):
        return self.runs[pid].messages

    def mismatches(self):
        return check(self.expected, self.runs)

    def assert_reported(self, client, index, what):
        found = self.mismatches()
        self.assertTrue(
            any(m.client == client and m.index == index and what in m.what for m in found),
            f"no mismatch ({client}, #{index}, {what!r}) in:\n" + "\n".join(map(str, found)),
        )
        return found


class CorrectTranscriptTest(CheckTestCase):
    def test_correct_transcripts_pass(self):
        self.assertEqual(self.mismatches(), [])

    def test_fold_case_passes(self):
        case = parse_case(fold_case_data())
        self.assertEqual(check(build_expected(case), correct_runs(case)), [])

    def test_expected_stream_of_the_sample(self):
        x = self.expected
        self.assertEqual(x.seats, {"SB": 0, "BB": 1})
        self.assertEqual(
            x.events[0],
            {"type": "hand_started", "button": 0, "sb_seat": 0, "bb_seat": 1,
             "participants": [[0, "SB", 100], [1, "BB", 100]]},
        )
        self.assertEqual([e["type"] for e in x.events].count("player_acted"), 11)
        self.assertEqual([e["type"] for e in x.events].count("board_dealt"), 3)
        self.assertEqual(x.events[3]["action"], "raise")
        self.assertEqual((x.events[3]["amount"], x.events[3]["to"], x.events[3]["stack"]), (3, 4, 96))
        self.assertEqual(x.stacks_after_betting, {"SB": 86, "BB": 86})
        self.assertEqual(
            list(x.scripts["SB"]),
            [
                {"street": "preflop", "action": "raise"},
                {"street": "preflop", "action": "raise"},
                {"street": "flop", "action": "raise"},  # a bet
                {"street": "turn", "action": "check"},
                {"street": "river", "action": "call"},
            ],
        )
        self.assertTrue(x.showdown)

    def test_normalize_event(self):
        ev = {"type": "player_acted", "hand_no": 1, "seq": 5, "seat": 0, "name": "SB",
              "street": "preflop", "action": "raise", "amount_bb": 1.5, "to_bb": 2.0,
              "all_in": False, "stack_bb": 48.0, "reason": None}
        self.assertEqual(
            normalize_event(ev, 2),
            {"type": "player_acted", "seat": 0, "name": "SB", "street": "preflop",
             "action": "raise", "amount": 3, "to": 4, "all_in": False, "stack": 96, "reason": None},
        )


class MismatchTest(CheckTestCase):
    def test_changed_stack(self):
        msgs = self.msgs("BB")
        i = index_of(msgs, is_event("player_acted", name="SB", action="raise"))
        msgs[i]["event"]["stack_bb"] += 0.5
        found = self.assert_reported("BB", i, "player_acted")
        # The other client saw the original: the streams differ too.
        self.assertTrue(any(m.client == "BB" and "compared with" in m.what for m in found))

    def test_changed_stack_in_a_turn_view(self):
        msgs = self.msgs("SB")
        i = index_of(msgs, lambda m: m["type"] == "turn")
        msgs[i]["view"]["seats"][1]["stack"] = 50.0
        self.assert_reported("SB", i, "stacks in the turn view")

    def test_missing_acting(self):
        msgs = self.msgs("SB")
        i = index_of(msgs, lambda m: m["type"] == "acting")
        del msgs[i]
        self.assert_reported("SB", None, "acting messages")

    def test_acting_after_the_action(self):
        msgs = self.msgs("SB")
        i = index_of(msgs, lambda m: m["type"] == "acting")
        acting = msgs.pop(i)
        msgs.insert(i + 1, acting)  # after the actor's player_acted
        self.assert_reported("SB", i + 1, "position")

    def test_lobby_during_the_hand(self):
        msgs = self.msgs("SB")
        i = index_of(msgs, is_event("hand_started"))
        msgs.insert(i + 1, {"type": "lobby", "to": "SB", "state": "waiting", "active": 2, "needed": 2, "starts_in": None})
        self.assert_reported("SB", i + 1, "lobby message during the hand")

    def test_wrong_pot_split(self):
        msgs = self.msgs("SB")
        i = index_of(msgs, lambda m: m["type"] == "hand_end")
        pot = msgs[i]["result"]["pots"][0]
        pot["low"] = [[0, 7.0]]  # the SB gets the low half instead of the BB
        self.assert_reported("SB", i, "result pot 0")

    def test_wrong_award_events(self):
        for pid in ("SB", "BB"):
            msgs = self.msgs(pid)
            i = index_of(msgs, is_event("pot_awarded", share="low"))
            msgs[i]["event"].update(seat=0, name="SB")
        found = self.assert_reported("SB", None, "pot 0 awards per seat")
        self.assertTrue(any("winner seats" in m.what for m in found))

    def test_wrong_final_stacks(self):
        msgs = self.msgs("BB")
        i = index_of(msgs, lambda m: m["type"] == "hand_end")
        msgs[i]["result"]["seats"][1]["stack"] = 56.5
        found = self.assert_reported("BB", i, "result stacks")
        self.assertTrue(any("sum of the final stacks" in m.what for m in found))

    def test_another_players_hole_cards(self):
        msgs = self.msgs("SB")
        i = index_of(msgs, lambda m: m["type"] == "turn")
        msgs[i]["view"]["hole_cards"] = ["7c", "6h", "3s", "4d"]  # the BB's hand
        self.assert_reported("SB", i, "hole cards in the turn view")

    def test_wrong_own_hole_cards(self):
        msgs = self.msgs("BB")
        i = index_of(msgs, lambda m: m["type"] == "hole_cards")
        msgs[i]["cards"] = ["7c", "6h", "3s", "5d"]
        self.assert_reported("BB", i, "own hole cards")

    def test_hole_cards_after_the_first_turn(self):
        msgs = self.msgs("SB")
        i = index_of(msgs, lambda m: m["type"] == "hole_cards")
        hole = msgs.pop(i)
        j = index_of(msgs, lambda m: m["type"] == "turn")
        msgs.insert(j + 1, hole)
        self.assert_reported("SB", j + 1, "hole_cards before the first turn")

    def test_clients_see_different_event_streams(self):
        msgs = self.msgs("BB")
        i = index_of(msgs, is_event("board_dealt"))
        msgs[i]["event"]["seq"] += 100
        found = self.assert_reported("BB", None, "compared with SB's")
        self.assertEqual(len(found), 1)  # seq is not part of the per-client checks

    def test_unused_script_and_problems(self):
        t = self.runs["BB"].transcript
        t["unused_script"] = [{"street": "river", "action": "raise"}]
        t["problems"] = ["street mismatch: ..."]
        self.runs["BB"].exit_code = 1
        found = self.assert_reported("BB", None, "unused script")
        self.assertTrue(any(m.what == "client problem" for m in found))
        self.assertTrue(any(m.what == "client exit status" for m in found))

    def test_rejected_join(self):
        self.runs["SB"].joined = {"type": "response", "ok": False, "error": "nope"}
        self.assert_reported("SB", None, "join")

    def test_missing_transcript(self):
        self.runs["SB"].transcript = None
        self.assert_reported("SB", None, "client transcript")

    def test_hand_shown_after_a_fold(self):
        case = parse_case(fold_case_data())
        runs = correct_runs(case)
        for pid, run in runs.items():
            i = index_of(run.messages, is_event("bet_returned"))
            run.messages.insert(
                i,
                {"type": "event", "to": pid, "event": {
                    "type": "hand_shown", "hand_no": 1, "seq": 50, "seat": 2, "name": "A",
                    "hand": ["As", "2s", "3s", "4s"], "stack_bb": 19.0}},
            )
        found = check(build_expected(case), runs)
        self.assertTrue(any(m.what == "hands shown" for m in found), found)


if __name__ == "__main__":
    unittest.main()
