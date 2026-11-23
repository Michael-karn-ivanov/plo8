"""Import boundary: rl_env and arena never import torch or trainer (statically
and at runtime); trainer imports only the standard library, numpy, torch and
the public packages of rl_env and arena (arena only from the rollout loop)."""

import ast
import importlib.util
import json
import os
import pkgutil
import subprocess
import sys
import unittest

import arena
import rl_env
import trainer

FORBIDDEN = {"torch", "trainer"}
TRAINER_ALLOWED = {"numpy", "torch", "rl_env", "arena", "trainer", "__future__"} | set(
    sys.stdlib_module_names
)
# The only trainer modules that may import arena (the baseline bots).
ARENA_USERS = {"trainer.lineup", "trainer.rollout"}


def modules_of(pkg) -> list[str]:
    names = [pkg.__name__]
    names += [m.name for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + ".")]
    return sorted(set(names))


def source_of(name: str) -> str:
    spec = importlib.util.find_spec(name)
    assert spec is not None, name
    src = None
    if spec.loader is not None and hasattr(spec.loader, "get_source"):
        src = spec.loader.get_source(name)
    if src is None:
        assert spec.origin and spec.origin.endswith(".py"), (name, spec.origin)
        with open(spec.origin, encoding="utf-8") as f:
            src = f.read()
    return src


def absolute_imports(src: str) -> set[str]:
    """Full module names of every absolute import (top-level or nested)."""
    mods = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            mods.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module)
    return mods


def roots(mods: set[str]) -> set[str]:
    return {m.split(".")[0] for m in mods}


class ImportsTest(unittest.TestCase):
    def test_static_no_torch_or_trainer(self):
        for pkg, expected in ((rl_env, "rl_env.encoder"), (arena, "arena.match")):
            names = modules_of(pkg)
            self.assertIn(expected, names)
            for name in names:
                bad = roots(absolute_imports(source_of(name))) & FORBIDDEN
                self.assertFalse(bad, f"{name} imports {sorted(bad)}")

    def test_runtime_no_torch_or_trainer(self):
        code = (
            "import importlib, json, pkgutil, sys\n"
            "import rl_env, arena\n"
            "for pkg in (rl_env, arena):\n"
            "    for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + '.'):\n"
            "        importlib.import_module(m.name)\n"
            "loaded = sorted(n for n in sys.modules if n.split('.')[0] in ('torch', 'trainer'))\n"
            "print(json.dumps({'loaded': loaded, 'rl_env': 'rl_env.encoder' in sys.modules}))\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        out = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertTrue(out["rl_env"])
        self.assertEqual(out["loaded"], [])

    def test_trainer_imports(self):
        names = modules_of(trainer)
        self.assertIn("trainer.nets", names)
        self.assertIn("trainer.rollout", names)
        for name in names:
            mods = absolute_imports(source_of(name))
            extra = roots(mods) - TRAINER_ALLOWED
            self.assertFalse(extra, f"{name} imports {sorted(extra)}")
            # Only the public packages of rl_env and arena, never their internals.
            internals = sorted(m for m in mods if m.startswith(("rl_env.", "arena.")))
            self.assertFalse(internals, f"{name} imports {internals}")
            if "arena" in roots(mods):
                self.assertIn(name, ARENA_USERS, f"{name} imports arena")


if __name__ == "__main__":
    unittest.main()
