"""Allowed imports of the arena package; bots never touch hidden information."""

import ast
import importlib
import inspect
import pkgutil
import sys
import unittest

import arena

HIDDEN = ("privileged_hole_cards", "finished_hole_cards", "hole_cards_of_finished", "critic")


def package_modules():
    """(name, source) of every module of the arena package."""
    out = [("arena", inspect.getsource(arena))]
    for info in pkgutil.iter_modules(arena.__path__):
        name = f"arena.{info.name}"
        out.append((name, inspect.getsource(importlib.import_module(name))))
    return out


def imports(source):
    """Full names of the absolute imports of a module."""
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


class ImportScanTest(unittest.TestCase):
    def test_only_stdlib_numpy_engine_entities_rl_env(self):
        allowed = set(sys.stdlib_module_names) | {
            "numpy",
            "engine",
            "entities",
            "rl_env",
            "__future__",
        }
        modules = package_modules()
        self.assertGreaterEqual(len(modules), 8)
        for name, source in modules:
            with self.subTest(module=name):
                roots = {i.split(".")[0] for i in imports(source)}
                self.assertEqual(sorted(roots - allowed), [])
                for banned in ("torch", "cli_server", "cli_client"):
                    self.assertNotIn(banned, roots)

    def test_bots_see_no_hidden_information(self):
        for name in ("arena.bots", "arena.rules"):
            source = inspect.getsource(importlib.import_module(name))
            with self.subTest(module=name):
                self.assertNotIn("rl_env.critic", imports(source))
                for word in HIDDEN:
                    self.assertNotIn(word, source)

    def test_only_match_reads_finished_hole_cards(self):
        for name, source in package_modules():
            if name == "arena.match":
                self.assertIn("finished_hole_cards", source)
                continue
            with self.subTest(module=name):
                self.assertNotIn("finished_hole_cards", source)
                self.assertNotIn("privileged_hole_cards", source)


if __name__ == "__main__":
    unittest.main()
