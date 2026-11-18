"""Actions: mask, mapping to the engine, rejected actions; allowed imports."""

import ast
import importlib
import inspect
import pkgutil
import sys
import unittest

import numpy as np

import rl_env
from engine import ActionType, LegalActions
from rl_env import (
    BET_RAISE,
    CHECK_CALL,
    FOLD,
    IllegalMaskedActionError,
    action_kind,
    action_mask,
    mask_array,
    to_engine,
)


def legal(can_check, call_amount, can_raise, raises_left=3):
    return LegalActions(
        can_check=can_check,
        call_amount=call_amount,
        can_raise=can_raise,
        raise_to=2.0,
        bet_unit=1.0,
        raises_left=raises_left,
    )


CHECK_FREE = legal(True, 0.0, True)
FACING_BET = legal(False, 1.0, True)
CAP_REACHED = legal(False, 1.0, False, raises_left=0)
SHORT_CALL_ONLY = legal(False, 0.4, False)  # can only call (all-in for less)
OTHERS_ALL_IN = legal(True, 0.0, False)  # nobody else can act: no raise
HEADS_UP = legal(False, 2.0, True, raises_left=None)


class MaskTest(unittest.TestCase):
    def test_shapes(self):
        self.assertEqual(action_mask(CHECK_FREE), (False, True, True))
        self.assertEqual(action_mask(FACING_BET), (True, True, True))
        self.assertEqual(action_mask(CAP_REACHED), (True, True, False))
        self.assertEqual(action_mask(SHORT_CALL_ONLY), (True, True, False))
        self.assertEqual(action_mask(OTHERS_ALL_IN), (False, True, False))
        self.assertEqual(action_mask(HEADS_UP), (True, True, True))

    def test_mask_array(self):
        m = mask_array(FACING_BET)
        self.assertEqual(m.dtype, np.bool_)
        self.assertEqual(m.shape, (3,))
        self.assertEqual(m.tolist(), [True, True, True])
        self.assertEqual(mask_array(CHECK_FREE).tolist(), [False, True, True])

    def test_neither_check_nor_call_is_an_engine_bug(self):
        with self.assertRaises(AssertionError):
            action_mask(legal(False, 0.0, True))


class ToEngineTest(unittest.TestCase):
    def test_mapping(self):
        self.assertIs(to_engine(CHECK_CALL, CHECK_FREE), ActionType.CHECK)
        self.assertIs(to_engine(BET_RAISE, CHECK_FREE), ActionType.RAISE)
        self.assertIs(to_engine(FOLD, FACING_BET), ActionType.FOLD)
        self.assertIs(to_engine(CHECK_CALL, FACING_BET), ActionType.CALL)
        self.assertIs(to_engine(BET_RAISE, FACING_BET), ActionType.RAISE)
        self.assertIs(to_engine(np.int64(1), SHORT_CALL_ONLY), ActionType.CALL)

    def test_every_allowed_action_is_legal_for_the_engine(self):
        for lg in (CHECK_FREE, FACING_BET, CAP_REACHED, SHORT_CALL_ONLY, OTHERS_ALL_IN, HEADS_UP):
            for a, allowed in enumerate(action_mask(lg)):
                if allowed:
                    self.assertTrue(lg.allows(to_engine(a, lg)), (lg, a))

    def test_masked_and_invalid_actions_raise(self):
        with self.assertRaises(IllegalMaskedActionError):
            to_engine(FOLD, CHECK_FREE)
        with self.assertRaises(IllegalMaskedActionError):
            to_engine(BET_RAISE, CAP_REACHED)
        with self.assertRaises(IllegalMaskedActionError):
            to_engine(BET_RAISE, SHORT_CALL_ONLY)
        for bad in (3, -1, 1.0, "1", None, True, np.bool_(True)):
            with self.subTest(bad=bad), self.assertRaises(IllegalMaskedActionError):
                to_engine(bad, FACING_BET)
        self.assertTrue(issubclass(IllegalMaskedActionError, ValueError))

    def test_action_kind(self):
        self.assertEqual(action_kind(ActionType.FOLD), FOLD)
        self.assertEqual(action_kind(ActionType.CHECK), CHECK_CALL)
        self.assertEqual(action_kind(ActionType.CALL), CHECK_CALL)
        self.assertEqual(action_kind(ActionType.RAISE), BET_RAISE)


def package_modules():
    """(name, source) of every module of the rl_env package."""
    out = [("rl_env", inspect.getsource(rl_env))]
    for info in pkgutil.iter_modules(rl_env.__path__):
        name = f"rl_env.{info.name}"
        out.append((name, inspect.getsource(importlib.import_module(name))))
    return out


def imported_roots(source):
    """Top-level names of the absolute imports of a module."""
    roots = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


class ImportScanTest(unittest.TestCase):
    def test_only_stdlib_numpy_engine_entities(self):
        allowed = set(sys.stdlib_module_names) | {"numpy", "engine", "entities", "__future__"}
        modules = package_modules()
        self.assertGreaterEqual(len(modules), 10)
        for name, source in modules:
            with self.subTest(module=name):
                roots = imported_roots(source)
                self.assertEqual(sorted(roots - allowed), [])
                self.assertNotIn("cli_server", roots)


if __name__ == "__main__":
    unittest.main()
