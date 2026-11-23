"""Learner checkpoints: exact resume, weights_only files written atomically, the
config on resume, and refused (mismatching or malformed) checkpoints."""

import copy
import dataclasses
import os
import tempfile
import unittest

import torch

from rl_env import ENCODER_VERSION
from trainer import Learner, NetConfigError

try:
    from . import helpers
    from .test_learner import no_time, params, same_tree
except ImportError:  # run as a top-level module
    import helpers  # type: ignore[no-redef]
    from test_learner import no_time, params, same_tree  # type: ignore[no-redef]


class CheckpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        worker = helpers.small_worker(21)
        w = helpers.small_learner(21).get_weights()
        cls.r1, _ = worker.collect(w, 2000)
        cls.r2, _ = worker.collect(w, 2000)

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.dir = self._dir.name
        self.path = os.path.join(self.dir, "learner.pt")

    def tearDown(self):
        self._dir.cleanup()

    def saved_learner(self):
        a = helpers.small_learner(22)
        a.update(self.r1)
        a.save(self.path)
        return a

    def assert_same_learner(self, a, b):
        self.assertEqual(a.config, b.config)
        self.assertEqual(a.seed, b.seed)
        self.assertEqual(a.n_updates, b.n_updates)
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(params(a), params(b))))
        self.assertTrue(same_tree(a.optimizer.state_dict(), b.optimizer.state_dict()))
        self.assertTrue(torch.equal(a._gen.get_state(), b._gen.get_state()))

    # --- resume ----------------------------------------------------------------

    def test_resume_exactly(self):
        a = self.saved_learner()
        b = Learner.load(self.path)
        self.assert_same_learner(a, b)
        sa, sb = a.update(self.r2), b.update(self.r2)
        self.assertEqual(no_time(sa), no_time(sb))
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(params(a), params(b))))

    def test_only_the_file_matters(self):
        a = self.saved_learner()
        helpers.small_learner(99).update(self.r2)  # unrelated work in between
        torch.manual_seed(4321)
        b = Learner.load(self.path)
        sa, sb = a.update(self.r2), b.update(self.r2)
        self.assertEqual(no_time(sa), no_time(sb))
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(params(a), params(b))))

    # --- the file --------------------------------------------------------------

    def test_file_format(self):
        a = self.saved_learner()
        ckpt = torch.load(self.path, weights_only=True)
        self.assertEqual(ckpt["format"], "trainer.learner/1")
        self.assertEqual(ckpt["encoder_version"], ENCODER_VERSION)
        self.assertEqual(ckpt["config"], dataclasses.asdict(a.config))
        self.assertEqual(ckpt["n_updates"], 1)
        self.assertEqual(os.listdir(self.dir), ["learner.pt"])
        # overwrite an existing checkpoint
        a.update(self.r2)
        a.save(self.path)
        self.assertEqual(os.listdir(self.dir), ["learner.pt"])
        self.assertEqual(Learner.load(self.path).n_updates, 2)

    # --- config on resume ------------------------------------------------------

    def test_config_on_resume(self):
        a = self.saved_learner()
        b = Learner.load(self.path, config=dataclasses.replace(a.config, lr=1e-3))
        self.assertEqual(b.config.lr, 1e-3)
        for group in b.optimizer.param_groups:
            self.assertEqual(group["lr"], 1e-3)
        for change in ({"hidden": 32}, {"layers": 3}):
            with self.assertRaises(NetConfigError, msg=str(change)):
                Learner.load(self.path, config=dataclasses.replace(a.config, **change))

    # --- refusals --------------------------------------------------------------

    def rewritten(self, edit):
        ckpt = torch.load(self.path, weights_only=True)
        ckpt = copy.deepcopy(ckpt)
        edit(ckpt)
        path = os.path.join(self.dir, "edited.pt")
        torch.save(ckpt, path)
        return path

    def test_refused(self):
        self.saved_learner()

        def top_encoder(c):
            c["encoder_version"] = ENCODER_VERSION + 1

        def net_encoder(c):
            c["policy"]["_extra_state"]["encoder_version"] = ENCODER_VERSION + 1

        def net_hidden(c):
            c["critic"]["_extra_state"]["hidden"] = 32

        def policy_shape(c):
            c["policy"]["head.weight"] = torch.zeros(3, 7)

        def no_optimizer(c):
            del c["optimizer"]

        def other_format(c):
            c["format"] = "trainer.learner/0"

        cases = [
            (top_encoder, NetConfigError),
            (net_encoder, NetConfigError),
            (net_hidden, NetConfigError),
            (policy_shape, NetConfigError),
            (no_optimizer, ValueError),
            (other_format, ValueError),
        ]
        for edit, err in cases:
            path = self.rewritten(edit)
            with self.assertRaises(err, msg=edit.__name__):
                Learner.load(path)


if __name__ == "__main__":
    unittest.main()
