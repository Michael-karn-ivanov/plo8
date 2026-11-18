"""Run e2e cases: a real `GameServer` plus one scripted client process per player.

`run_case(case, seed, action_timeout)` plays one recorded hand on a fresh
server and checks the client transcripts; `run_cases(names, seed,
action_timeout)` runs the selected cases (all of them by default), prints a
report and returns the exit status.

Finding the case files (`E2E_DIR`, `available_cases`, `case_path`, ...) lives in
`cases.py` (library `:e2e`); the names are re-exported here for convenience.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import random
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Optional, Sequence

from .case import Case, CaseError, load_case
from .cases import E2E_DIR, UnknownCaseError, available_cases, case_name, case_path
from .check import ClientRun, Mismatch, check
from .deck import StackedDeck, deal_script, join_offset, seating
from .expect import build_expected, client_scripts

__all__ = [
    "E2E_DIR",
    "PASS",
    "FAIL",
    "ERROR",
    "CaseReport",
    "UnknownCaseError",
    "available_cases",
    "case_name",
    "case_path",
    "run_case",
    "run_case_file",
    "run_cases",
    "case_deadline",
]

_log = logging.getLogger("cli_server.e2e")

PASS, FAIL, ERROR = "PASS", "FAIL", "ERROR"
MAX_REPORTED = 20  # mismatches printed per case
JOIN_LINE_TIMEOUT = 30.0  # seconds a client process has to print its `joined` line
EXIT_TIMEOUT = 10.0  # seconds a client has to print its transcript and exit after `close`


@dataclass
class CaseReport:
    name: str
    description: str
    status: str  # PASS / FAIL / ERROR
    mismatches: list[Mismatch] = field(default_factory=list)
    error: Optional[str] = None
    seconds: float = 0.0


def case_deadline(action_timeout: float) -> float:
    """Overall time limit of one case, in seconds."""
    return 30.0 + 4 * action_timeout


# --- client processes ---------------------------------------------------------------------


class _ClientProcess:
    """One scripted client process; its stdout lines are read by a thread."""

    def __init__(self, config: dict) -> None:
        env = dict(os.environ)
        # The child finds the same modules (inside the buck2 binary / test too).
        env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
        self.name = config["name"]
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "cli_server.e2e.client"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        self.lines: "queue.Queue[Optional[dict]]" = queue.Queue()
        self.stderr_chunks: list[bytes] = []
        self._out = threading.Thread(target=self._read_stdout, daemon=True)
        self._err = threading.Thread(target=self._read_stderr, daemon=True)
        self._out.start()
        self._err.start()
        assert self.proc.stdin is not None
        self.proc.stdin.write((json.dumps(config) + "\n").encode("utf-8"))
        self.proc.stdin.close()
        self.run = ClientRun()

    def _read_stdout(self) -> None:
        assert self.proc.stdout is not None
        for raw in self.proc.stdout:
            try:
                msg = json.loads(raw)
            except ValueError:
                msg = {"type": "_garbage", "line": raw.decode("utf-8", "replace")}
            self.lines.put(msg)
        self.lines.put(None)

    def _read_stderr(self) -> None:
        assert self.proc.stderr is not None
        for raw in self.proc.stderr:
            self.stderr_chunks.append(raw)

    def next_line(self, kind: str, timeout: float) -> Optional[dict]:
        """The next stdout message of type `kind`; None on EOF or timeout."""
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            try:
                msg = self.lines.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                return None
            if msg is None:
                self.lines.put(None)  # keep later calls from blocking
                return None
            if msg.get("type") == kind:
                return msg

    def finish(self, timeout: float) -> None:
        """Collect the transcript and the exit status (kill after `timeout`)."""
        if self.run.transcript is None:
            t = self.next_line("transcript", timeout)
            if t is not None:
                self.run.transcript = t
                self.run.messages = list(t.get("messages") or [])
        try:
            self.run.exit_code = self.proc.wait(timeout=max(0.1, timeout))
        except subprocess.TimeoutExpired:
            self.kill()
        self._err.join(1.0)
        self.run.stderr = b"".join(self.stderr_chunks).decode("utf-8", "replace")

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            try:
                self.proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pass
        if self.run.exit_code is None:
            self.run.exit_code = self.proc.returncode


# --- one case -------------------------------------------------------------------------------


def run_case(case: Case, seed: int = 0, action_timeout: float = 10.0) -> CaseReport:
    """Play `case` on a fresh server with one client process per player, then
    check what every client received."""
    from cli_server.server import GameServer  # lazy: no import cycle with :cli_server_lib

    started = time.monotonic()
    deadline = started + case_deadline(action_timeout)
    report = CaseReport(case.name, case.description, PASS)
    order = seating(case)
    scripts = client_scripts(case)
    expected = build_expected(case)
    bb = case.game.bb

    server = GameServer(
        port=0,
        grace_period=0.0,
        hand_pause=0.0,
        env_kwargs={
            "action_timeout": action_timeout,
            "chips_per_bb": bb,
            "deck": StackedDeck(deal_script(case), random.Random(seed)),
            "seed": seed,
        },
        required_players=len(order),
        max_hands=1,
    )
    server_error: list[BaseException] = []

    def serve() -> None:
        try:
            server.serve_forever()
        except BaseException as e:  # noqa: BLE001
            server_error.append(e)
            _log.exception("the game loop of case %s failed", case.name)

    thread = threading.Thread(target=serve, name=f"e2e-server-{case.name}", daemon=True)
    thread.start()
    clients: list[_ClientProcess] = []
    errors: list[str] = []
    timed_out = False
    try:
        for k, p in enumerate(order):
            client = _ClientProcess(
                {
                    "port": server.port,
                    "name": p.id,
                    "stack_bb": p.stack / bb,
                    "seat": join_offset(k),
                    "action_timeout": action_timeout,
                    "script": scripts[p.id],
                }
            )
            clients.append(client)
            remaining = min(JOIN_LINE_TIMEOUT, deadline - time.monotonic())
            joined = client.next_line("joined", remaining)
            if joined is None:
                errors.append(f"client {p.id} did not report its join")
                break
            client.run.joined = joined.get("response")
            if not (client.run.joined or {}).get("ok"):
                _log.warning("join of %s failed: %s", p.id, client.run.joined)
                break  # the hand cannot start; the checker reports the join
        else:
            thread.join(max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                timed_out = True
    finally:
        server.shutdown()  # sends `close` to every session
        thread.join(5.0)
        for client in clients:
            client.finish(max(1.0, min(EXIT_TIMEOUT, deadline + EXIT_TIMEOUT - time.monotonic())))
        for client in clients:
            client.kill()

    runs = {c.name: c.run for c in clients}
    for c in clients:
        if c.run.transcript is None:
            tail = c.run.stderr.strip().splitlines()[-5:]
            errors.append(
                f"client {c.name} crashed (exit status {c.run.exit_code}): " + " | ".join(tail)
            )
    if server_error:
        errors.append(f"the server failed: {server_error[0]!r}")
    report.mismatches = check(expected, runs)
    report.seconds = time.monotonic() - started
    if errors:
        report.status = ERROR
        report.error = "; ".join(errors)
    elif timed_out:
        report.status = FAIL
        report.error = f"timed out after {case_deadline(action_timeout):g}s"
    elif report.mismatches:
        report.status = FAIL
    return report


def run_case_file(path: Path, seed: int = 0, action_timeout: float = 10.0) -> CaseReport:
    """Load and run one case file; an invalid file is an ERROR."""
    try:
        case = load_case(path)
    except CaseError as e:
        return CaseReport(Path(path).stem, "", ERROR, error=f"invalid case: {e}")
    try:
        return run_case(case, seed=seed, action_timeout=action_timeout)
    except KeyboardInterrupt:
        raise
    except Exception as e:  # noqa: BLE001
        _log.exception("case %s failed to run", case.name)
        return CaseReport(case.name, case.description, ERROR, error=f"{type(e).__name__}: {e}")


# --- all cases ---------------------------------------------------------------------------------


def format_report(report: CaseReport) -> list[str]:
    lines = [f"{report.status} {report.name}: {report.description}".rstrip(": ")]
    if report.error:
        lines.append(f"    {report.error}")
    for m in report.mismatches[:MAX_REPORTED]:
        lines.append(f"    {m}")
    if len(report.mismatches) > MAX_REPORTED:
        lines.append(f"    ... and {len(report.mismatches) - MAX_REPORTED} more")
    return lines


def run_cases(
    names: Optional[Sequence[str]] = None,
    seed: int = 0,
    action_timeout: float = 10.0,
    out: Optional[IO[str]] = None,
    directory: Optional[Path] = None,
) -> int:
    """Run the named cases (all of them when `names` is None), sequentially,
    each on its own server. Prints one line per case (and the mismatches of
    failed ones) and a summary. Returns 0 if every case passed, else 1."""
    out = sys.stdout if out is None else out
    d = E2E_DIR if directory is None else Path(directory)
    if names is None:
        paths = [d / f"{n}.json" for n in available_cases(d)]
    else:
        paths = [case_path(n, d) for n in names]
    reports: list[CaseReport] = []
    interrupted = False
    try:
        for path in paths:
            report = run_case_file(path, seed=seed, action_timeout=action_timeout)
            reports.append(report)
            for line in format_report(report):
                print(line, file=out)
            out.flush()
    except KeyboardInterrupt:
        interrupted = True
        print("interrupted", file=out)
    passed = sum(r.status == PASS for r in reports)
    failed = sum(r.status == FAIL for r in reports)
    errors = sum(r.status == ERROR for r in reports)
    print(f"{passed} passed, {failed} failed, {errors} errors", file=out)
    out.flush()
    if interrupted or not paths:
        return 1
    return 0 if passed == len(reports) else 1
