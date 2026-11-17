import io
import json
import tempfile
import unittest
from pathlib import Path

from cli_server.e2e.case import parse_case
from cli_server.e2e.driver import (
    ERROR,
    PASS,
    E2E_DIR,
    UnknownCaseError,
    available_cases,
    case_path,
    run_case,
    run_cases,
)

try:
    from cli_server_tests.tests.e2e_helpers import SAMPLE, fold_case_data
except ImportError:  # run directly from the tests directory
    from e2e_helpers import SAMPLE, fold_case_data


class RecordedCasesTest(unittest.TestCase):
    def test_every_recorded_case_passes(self):
        """The recorded hands are part of `buck2 test`: server, wire protocol and
        engine together, with real client processes."""
        self.assertIn(SAMPLE, available_cases())
        out = io.StringIO()
        code = run_cases(None, seed=0, action_timeout=10, out=out)
        self.assertEqual(code, 0, out.getvalue())
        self.assertIn(f"PASS {SAMPLE}", out.getvalue())
        self.assertRegex(out.getvalue(), r"\d+ passed, 0 failed, 0 errors")

    def test_three_player_fold_case(self):
        case = parse_case(fold_case_data(), "fold")
        report = run_case(case, seed=1, action_timeout=10)
        self.assertEqual(report.status, PASS, "\n".join(map(str, report.mismatches)) + str(report.error))


class CaseSelectionTest(unittest.TestCase):
    def test_case_path(self):
        self.assertEqual(case_path(SAMPLE), E2E_DIR / f"{SAMPLE}.json")
        self.assertEqual(case_path(f"{SAMPLE}.json"), case_path(SAMPLE))
        with self.assertRaises(UnknownCaseError) as cm:
            case_path("no_such_case")
        self.assertIn(SAMPLE, str(cm.exception))  # lists the available cases

    def test_invalid_case_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = fold_case_data()
            data["game"]["rake"] = 1
            Path(tmp, "bad.json").write_text(json.dumps(data), encoding="utf-8")
            Path(tmp, "garbage.json").write_text("{", encoding="utf-8")
            out = io.StringIO()
            code = run_cases(None, out=out, directory=Path(tmp))
        self.assertEqual(code, 1)
        text = out.getvalue()
        self.assertIn(f"{ERROR} bad", text)
        self.assertIn("rake", text)
        self.assertIn(f"{ERROR} garbage", text)
        self.assertIn("0 passed, 0 failed, 2 errors", text)


if __name__ == "__main__":
    unittest.main()
