"""The scripted e2e client process: `python -m cli_server.e2e.client`.

A very simple client (not `cli_client`). Its config is one JSON line on stdin:

    {"port": <server port>, "name": ..., "stack_bb": ..., "seat": <offset or null>,
     "action_timeout": <seconds>, "script": [{"street": "preflop", "action": "raise"}, ...]}

It listens on a free callback port, joins, and prints
`{"type": "joined", "response": ...}` on stdout. Then it records every callback
message in arrival order, answers `turn` with the next scripted action and
`hand_end` with "no rebuy, no sit-in, no leave". On `close` (or EOF) it prints
`{"type": "transcript", "messages", "responses", "problems", "unused_script"}`
and exits 0. Any exception is reported in `problems`, the transcript is still
printed, and the exit status is 1.
"""

from __future__ import annotations

import json
import socket
import sys
import traceback
from typing import Any, Optional

from cli_server.protocol import (
    action_request,
    hand_end_request,
    join_request,
    read_message,
    write_message,
)

HOST = "127.0.0.1"


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def legal_allows(legal: dict, action: str) -> bool:
    if action == "fold":
        return True
    if action == "check":
        return bool(legal.get("can_check"))
    if action == "call":
        return (legal.get("call_amount") or 0) > 0
    if action == "raise":
        return bool(legal.get("can_raise"))
    return False


class ScriptedClient:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.name: str = config["name"]
        self.script: list[dict] = list(config.get("script") or [])
        self.messages: list[dict] = []
        self.responses: list[dict] = []
        self.problems: list[str] = []
        self._req: Optional[socket.socket] = None
        self._rfile: Any = None
        self._wfile: Any = None

    # --- requests ------------------------------------------------------------

    def request(self, msg: dict) -> dict:
        write_message(self._wfile, msg)
        response = read_message(self._rfile)
        if response is None:
            raise ConnectionError("the server closed the request connection")
        self.responses.append(response)
        return response

    # --- main ------------------------------------------------------------------

    def run(self) -> int:
        timeout = float(self.config.get("action_timeout") or 10.0)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            listener.bind((HOST, 0))
            listener.listen(1)
            listener.settimeout(max(30.0, 4 * timeout))
            callback_port = listener.getsockname()[1]

            self._req = socket.create_connection((HOST, int(self.config["port"])), timeout=30.0)
            self._req.settimeout(None)
            self._rfile = self._req.makefile("rb")
            self._wfile = self._req.makefile("wb")
            response = self.request(
                join_request(self.name, self.config["stack_bb"], callback_port, self.config.get("seat"))
            )
            _emit({"type": "joined", "response": response})
            if not response.get("ok"):
                self.problems.append(f"join rejected: {response.get('error')}")
                return 1

            conn, _ = listener.accept()  # the server connected during the join
            conn.settimeout(None)
            with conn, conn.makefile("rb") as callbacks:
                while True:
                    msg = read_message(callbacks)
                    if msg is None:
                        break
                    self.messages.append(msg)
                    kind = msg.get("type")
                    if kind == "turn":
                        self.on_turn(msg)
                    elif kind == "hand_end":
                        r = self.request(hand_end_request(self.name, msg["seq"]))
                        if not r.get("ok"):
                            self.problems.append(f"hand_end reply rejected: {r.get('error')}")
                    elif kind == "close":
                        break
            return 0
        finally:
            listener.close()
            if self._req is not None:
                try:
                    self._req.close()
                except OSError:
                    pass

    def on_turn(self, msg: dict) -> None:
        street = msg.get("view", {}).get("street")
        legal = msg.get("legal", {})
        action = "fold"
        if not self.script:
            self.problems.append(f"unexpected turn on the {street} (the script is used up)")
        else:
            step = self.script[0]
            if step["street"] != street:
                self.problems.append(
                    f"street mismatch: turn on the {street}, the script expects "
                    f"{step['action']} on the {step['street']}"
                )
            elif not legal_allows(legal, step["action"]):
                self.problems.append(f"not legal: {step['action']} on the {street} (legal: {legal})")
            else:
                action = step["action"]
            self.script.pop(0)
        r = self.request(action_request(self.name, msg["seq"], action))
        if not r.get("ok"):
            self.problems.append(f"action {action} rejected: {r.get('error')}")

    def transcript(self) -> dict:
        return {
            "type": "transcript",
            "name": self.name,
            "messages": self.messages,
            "responses": self.responses,
            "problems": self.problems,
            "unused_script": self.script,
        }


def main() -> int:
    line = sys.stdin.readline()
    try:
        config = json.loads(line)
    except ValueError as e:
        _emit({"type": "transcript", "messages": [], "responses": [], "problems": [f"bad config: {e}"], "unused_script": []})
        return 1
    client = ScriptedClient(config)
    try:
        code = client.run()
    except BaseException as e:  # noqa: BLE001 - always print the transcript
        client.problems.append(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        code = 1
    _emit(client.transcript())
    return code


if __name__ == "__main__":
    sys.exit(main())
