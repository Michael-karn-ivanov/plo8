"""bot_client's own modules import only the standard library, arena, rl_env,
engine, entities, cli_server.protocol and cli_client.net: no torch, no
onnxruntime, no curses UI, no server code."""

import ast
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import bot_client

ALLOWED_PACKAGES = ("arena", "rl_env", "engine", "entities")
ALLOWED_MODULES = ("cli_server.protocol", "cli_client.net")
FORBIDDEN = (
    "torch",
    "onnxruntime",
    "numpy",
    "curses",
    "cli_client.ui",
    "cli_client.controller",
    "cli_client.render",
    "cli_client.state",
    "cli_client.main",
    "cli_server.server",
    "cli_server.session",
    "cli_server.main",
)


def imported_modules(path):
    """Absolute module names imported by one source file (relative imports skipped)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            out.append(node.module)
    return out


def allowed(module):
    top = module.split(".")[0]
    if top in sys.stdlib_module_names or top == "__future__":
        return True
    if top in ALLOWED_PACKAGES:
        return True
    return any(module == m or module.startswith(m + ".") for m in ALLOWED_MODULES)


def forbidden(module):
    return any(module == m or module.startswith(m + ".") for m in FORBIDDEN)


class ImportsTest(unittest.TestCase):
    def sources(self):
        files = sorted(Path(bot_client.__file__).resolve().parent.glob("*.py"))
        self.assertGreaterEqual(len(files), 6, files)
        return files

    def test_every_import_is_allowed(self):
        for path in self.sources():
            for module in imported_modules(path):
                with self.subTest(file=path.name, module=module):
                    self.assertFalse(forbidden(module))
                    self.assertTrue(allowed(module))

    def test_main_loads_no_curses_and_no_torch(self):
        code = (
            "import json, sys\n"
            "import bot_client.main\n"
            "print(json.dumps(sorted(sys.modules)))\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8", "replace"))
        loaded = set(json.loads(proc.stdout.decode("utf-8").strip().splitlines()[-1]))
        self.assertIn("bot_client.main", loaded)
        for name in ("curses", "_curses", "torch", "onnxruntime", "cli_client.controller", "cli_client.ui", "cli_server.server"):
            with self.subTest(module=name):
                self.assertNotIn(name, loaded)


if __name__ == "__main__":
    unittest.main()
