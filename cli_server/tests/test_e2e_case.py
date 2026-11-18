import copy
import json
import tempfile
import unittest
from pathlib import Path

from entities import Card

from cli_server.e2e.case import (
    Case,
    CaseAction,
    CaseError,
    CasePlayer,
    blind_ids,
    clockwise_ids,
    load_case,
    parse_case,
)
from cli_server.e2e.cases import (
    E2E_DIR,
    UnknownCaseError,
    available_cases,
    case_name,
    case_path,
    load_cases,
)

SAMPLE = "12_heads_up_three_raises_preflop_bb_scoops"


def sample_data():
    return json.loads((E2E_DIR / f"{SAMPLE}.json").read_text(encoding="utf-8"))


def cards(*names):
    return tuple(Card.parse(n) for n in names)


class LoadSampleTest(unittest.TestCase):
    def test_sample_case_loads(self):
        case = load_case(E2E_DIR / f"{SAMPLE}.json")
        self.assertIsInstance(case, Case)
        self.assertEqual(case.name, SAMPLE)
        self.assertEqual(case.hand_id, "2026-11-17-000012")
        self.assertEqual((case.game.sb, case.game.bb), (1, 2))
        self.assertEqual((case.game.small_bet, case.game.big_bet, case.game.cap_raises), (2, 4, 4))
        self.assertEqual(
            case.players,
            (
                CasePlayer("SB", 1, 100, cards("Ah", "Ad", "Kc", "Qc")),
                CasePlayer("BB", 2, 100, None),
            ),
        )
        self.assertEqual(case.button, "SB")
        self.assertEqual([s.street for s in case.streets], ["preflop", "flop", "turn", "river"])
        self.assertEqual(case.streets[1].board, cards("9c", "8c", "2d"))
        self.assertEqual(case.streets[0].actions[2], CaseAction("SB", "raise", 3, 4))
        self.assertEqual(case.streets[1].actions[0], CaseAction("BB", "check", 0, None))
        self.assertEqual(case.streets[1].actions[1].engine_action, "raise")  # a bet
        self.assertEqual(case.streets[3].pot_after, 28)
        # The BB's hole is only known from the showdown.
        self.assertEqual(case.hole_of("BB"), cards("7c", "6h", "3s", "4d"))
        self.assertEqual(len(case.pots), 1)
        pot = case.pots[0]
        self.assertEqual((pot.type, pot.amount, pot.eligible), ("main", 28, ("BB", "SB")))
        self.assertEqual((pot.hi.winners, pot.hi.amount), (("BB",), 14))
        self.assertEqual((pot.lo.winners, pot.lo.amount), (("BB",), 14))
        self.assertEqual(dict(pot.awards), {"BB": 28})
        self.assertEqual(case.result.ended_by, "showdown")
        self.assertEqual(dict(case.result.net), {"SB": -14, "BB": 14})
        self.assertEqual(dict(case.result.final_stacks), {"SB": 86, "BB": 114})
        self.assertEqual(clockwise_ids(case), ["SB", "BB"])
        self.assertEqual(blind_ids(case), ("SB", "BB"))  # heads-up: the button is the SB

    def test_every_case_file_is_valid(self):
        for path in sorted(E2E_DIR.glob("*.json")):
            with self.subTest(case=path.stem):
                load_case(path)


class CaseFilesTest(unittest.TestCase):
    """`cases.py`: finding and loading the case files."""

    def test_available_cases_lists_every_file_sorted(self):
        names = available_cases()
        self.assertIn(SAMPLE, names)
        self.assertEqual(names, sorted(names))
        self.assertEqual(names, sorted(p.stem for p in E2E_DIR.glob("*.json")))
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(available_cases(Path(tmp, "missing")), [])

    def test_case_name_and_path(self):
        self.assertEqual(case_name(SAMPLE), SAMPLE)
        self.assertEqual(case_name(f"{SAMPLE}.json"), SAMPLE)
        self.assertEqual(case_path(SAMPLE), E2E_DIR / f"{SAMPLE}.json")
        self.assertEqual(case_path(f"{SAMPLE}.json"), case_path(SAMPLE))

    def test_case_path_rejects_unknown_names_and_separators(self):
        for bad in ("no_such_case", "", ".json", f"../e2e_test_sets/{SAMPLE}", f"x/{SAMPLE}"):
            with self.subTest(name=bad):
                with self.assertRaises(UnknownCaseError) as cm:
                    case_path(bad)
                self.assertIsInstance(cm.exception, KeyError)
                self.assertIn(SAMPLE, str(cm.exception))  # lists the available cases

    def test_load_cases_loads_every_file(self):
        cases = load_cases()
        self.assertEqual([c.name for c in cases], available_cases())
        self.assertTrue(all(isinstance(c, Case) for c in cases))
        self.assertEqual([c.name for c in load_cases([f"{SAMPLE}.json"])], [SAMPLE])
        with self.assertRaises(UnknownCaseError):
            load_cases(["no_such_case"])

    def test_an_invalid_file_makes_load_cases_raise_naming_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "a_good.json").write_text(json.dumps(sample_data()), encoding="utf-8")
            bad = sample_data()
            bad["game"]["rake"] = 1
            Path(tmp, "b_bad.json").write_text(json.dumps(bad), encoding="utf-8")
            self.assertEqual(available_cases(tmp), ["a_good", "b_bad"])
            self.assertEqual([c.name for c in load_cases(["a_good"], tmp)], ["a_good"])
            with self.assertRaisesRegex(CaseError, r"b_bad\.json.*rake"):
                load_cases(directory=tmp)
            Path(tmp, "b_bad.json").write_text("{", encoding="utf-8")
            with self.assertRaisesRegex(CaseError, r"b_bad\.json"):
                load_cases(directory=tmp)


class CaseConventionsTest(unittest.TestCase):
    """Conventions of the recorded case files the loader accepts."""

    def load(self, name):
        return load_case(E2E_DIR / f"{name}.json")

    def test_numbered_side_pots(self):
        case = self.load("06_turn_all_in_short_stack_side_pot")
        self.assertEqual([p.type for p in case.pots], ["main", "side"])

    def test_uncalled_bet_kept_in_the_pot_is_returned_instead(self):
        # A walk: the case awards the BB the whole 3-chip pot; the engine gives the
        # uncalled chip back and awards 2.
        case = self.load("01_walk_all_fold_to_bb")
        self.assertEqual(len(case.pots), 1)
        pot = case.pots[0]
        self.assertEqual((pot.amount, pot.hi.amount, dict(pot.awards)), (2, 2, {"BB": 2}))
        self.assertEqual(dict(case.result.net)["BB"], 1)

    def test_listed_uncalled_bet_leaves_the_pots_unchanged(self):
        case = self.load("19_preflop_all_in_side_pots_uncalled_return")
        self.assertEqual(dict(case.result.uncalled_returned), {"BB": 3})
        self.assertEqual([p.amount for p in case.pots], [8, 16])

    def test_listed_uncalled_bet_must_match(self):
        data = json.loads(
            (E2E_DIR / "19_preflop_all_in_side_pots_uncalled_return.json").read_text("utf-8")
        )
        data["result"]["uncalled_returned"] = {"BB": 2}
        with self.assertRaisesRegex(CaseError, "uncalled_returned"):
            parse_case(data, "broken")

    def test_unknown_pot_type_is_rejected(self):
        data = json.loads(
            (E2E_DIR / "06_turn_all_in_short_stack_side_pot.json").read_text("utf-8")
        )
        data["pots"][1]["type"] = "extra"
        with self.assertRaisesRegex(CaseError, "side pot"):
            parse_case(data, "broken")


class InvalidCaseTest(unittest.TestCase):
    def assert_invalid(self, mutate, reason):
        data = sample_data()
        mutate(data)
        with self.assertRaisesRegex(CaseError, reason):
            parse_case(data, "broken")

    def test_wrong_game(self):
        def set_game(key, value):
            return lambda d: d["game"].__setitem__(key, value)

        self.assert_invalid(set_game("variant", "holdem"), "variant")
        self.assert_invalid(set_game("betting", "pot_limit"), "betting")
        self.assert_invalid(set_game("unit", "bb"), "unit")
        self.assert_invalid(set_game("rake", 1), "rake")
        self.assert_invalid(set_game("blinds", {"sb": 1, "bb": 3}), "blinds")
        self.assert_invalid(set_game("small_bet", 4), "small_bet")
        self.assert_invalid(set_game("big_bet", 6), "big_bet")
        self.assert_invalid(set_game("cap_raises", 3), "cap_raises")

    def test_duplicate_card(self):
        def mutate(d):
            d["streets"][1]["board"][0] = "Ah"  # SB holds it

        self.assert_invalid(mutate, "duplicate card Ah")

    def test_bad_card(self):
        def mutate(d):
            d["streets"][2]["board"][0] = "Zz"

        self.assert_invalid(mutate, "turn board")

    def test_unknown_player_in_action(self):
        def mutate(d):
            d["streets"][1]["actions"][0]["player"] = "CO"

        self.assert_invalid(mutate, "unknown player 'CO'")

    def test_showdown_player_with_unknown_hole(self):
        def mutate(d):
            d["showdown"][0]["hole"] = None  # BB's hole is not under players either

        self.assert_invalid(mutate, "unknown hole cards")

    def test_holes_in_players_and_showdown_differ(self):
        def mutate(d):
            d["players"][0]["hole"] = ["Ah", "Ad", "Kc", "Ks"]

        self.assert_invalid(mutate, "differ")

    def test_inconsistent_pot_after(self):
        def mutate(d):
            d["streets"][1]["pot_after"] = 22

        self.assert_invalid(mutate, "pot_after")

    def test_inconsistent_to(self):
        def mutate(d):
            d["streets"][0]["actions"][2]["to"] = 5

        self.assert_invalid(mutate, "'to' is 5")

    def test_inconsistent_net(self):
        def mutate(d):
            d["result"]["net"] = {"SB": -13, "BB": 13}
            d["result"]["final_stacks"] = {"SB": 87, "BB": 113}

        self.assert_invalid(mutate, "net of")

    def test_inconsistent_final_stacks(self):
        def mutate(d):
            d["result"]["final_stacks"]["SB"] = 87

        self.assert_invalid(mutate, "final_stacks")

    def test_awards_do_not_sum_to_the_pot(self):
        def mutate(d):
            d["pots"][0]["awards"] = {"BB": 27, "SB": 0}

        self.assert_invalid(mutate, "awards sum")

    def test_blinds_must_come_first_and_match_the_seating(self):
        def swap(d):
            first = d["streets"][0]["actions"]
            first[0], first[1] = first[1], first[0]

        self.assert_invalid(swap, "post_sb and post_bb")

        def wrong_sb(d):
            d["button"] = "BB"  # then the BB player would post the small blind

        self.assert_invalid(wrong_sb, "small blind")

    def test_bet_after_a_bet(self):
        def mutate(d):
            d["streets"][0]["actions"][2]["action"] = "bet"  # the big blind is the bet

        self.assert_invalid(mutate, "a bet only when nobody has bet")

    def test_player_ids_and_seats_are_unique(self):
        def same_seat(d):
            d["players"][1]["seat"] = 1

        self.assert_invalid(same_seat, "seats must be unique")

        def one_player(d):
            del d["players"][1]

        self.assert_invalid(one_player, "2-10 players")

    def test_mutations_do_not_leak(self):
        before = sample_data()
        data = copy.deepcopy(before)
        parse_case(data)
        self.assertEqual(data, before)


if __name__ == "__main__":
    unittest.main()
