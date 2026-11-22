"""Shapes and dtypes on real rl_env data, no copy of the inputs, defaults that
follow rl_env, and rejection of wrong inputs."""

import unittest

import numpy as np
import torch
from torch import nn

import rl_env
from rl_env import ENCODER_VERSION, N_ACTIONS, OBS_DIM, PRIV_DIM, HandSampler, TableBatch
from trainer import CriticNet, PolicyNet, critic_input, mlp

try:
    from . import helpers
except ImportError:  # run as a top-level module
    import helpers  # type: ignore[no-redef]


def linears(net):
    return [m for m in net.modules() if isinstance(m, nn.Linear)]


class ShapesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.recs = helpers.records()
        cls.policy = PolicyNet()
        cls.critic = CriticNet()

    def test_enough_rows(self):
        self.assertGreaterEqual(len(self.recs), 512)
        # Mixed table sizes: rows from 2- and 10-player hands at least.
        self.assertGreater(len(np.unique(self.recs.hand_id)), 100)

    def test_batch_sizes_from_records(self):
        r = self.recs
        x_all = critic_input(r.obs, r.critic)
        for b in (1, 7, 512):
            obs = torch.from_numpy(np.ascontiguousarray(r.obs[:b]))
            mask = torch.from_numpy(np.ascontiguousarray(r.mask[:b]))
            logits = self.policy(obs, mask)
            self.assertEqual(tuple(logits.shape), (b, N_ACTIONS))
            self.assertEqual(logits.dtype, torch.float32)
            x = torch.from_numpy(np.ascontiguousarray(x_all[:b]))
            values = self.critic(x)
            self.assertEqual(tuple(values.shape), (b,))
            self.assertEqual(values.dtype, torch.float32)

    def test_empty_batch(self):
        obs = torch.zeros((0, OBS_DIM), dtype=torch.float32)
        mask = torch.zeros((0, N_ACTIONS), dtype=torch.bool)
        self.assertEqual(tuple(self.policy(obs, mask).shape), (0, N_ACTIONS))
        x = torch.zeros((0, OBS_DIM + PRIV_DIM), dtype=torch.float32)
        self.assertEqual(tuple(self.critic(x).shape), (0,))

    def test_decision_batch_directly(self):
        tb = TableBatch(16)
        sampler = HandSampler(3)
        for t in range(16):
            spec = sampler.sample()
            tb.start_hand(t, spec, ["a"] * spec.n_seats)
        batches = tb.pending()
        self.assertTrue(batches)
        for batch in batches:
            logits = self.policy(torch.from_numpy(batch.obs), torch.from_numpy(batch.mask))
            self.assertEqual(tuple(logits.shape), (len(batch), N_ACTIONS))

    def test_no_copy(self):
        r = self.recs
        x = critic_input(r.obs, r.critic)
        for arr in (r.obs, r.mask, x):
            self.assertEqual(torch.from_numpy(arr).data_ptr(), arr.ctypes.data)
        self.assertTrue(x.flags["C_CONTIGUOUS"])
        logits = self.policy(torch.from_numpy(r.obs), torch.from_numpy(r.mask))
        self.assertEqual(tuple(logits.shape), (len(r), N_ACTIONS))
        values = self.critic(torch.from_numpy(x))
        self.assertEqual(tuple(values.shape), (len(r),))

    def test_defaults_follow_rl_env(self):
        p, c = PolicyNet(), CriticNet()
        self.assertEqual(p.obs_dim, OBS_DIM)
        self.assertEqual(p.n_actions, N_ACTIONS)
        self.assertEqual(c.in_dim, OBS_DIM + PRIV_DIM)
        self.assertEqual(p.encoder_version, ENCODER_VERSION)
        self.assertEqual(c.encoder_version, ENCODER_VERSION)
        self.assertEqual(
            [(m.in_features, m.out_features) for m in linears(p)],
            [(OBS_DIM, 512), (512, 512), (512, 512), (512, N_ACTIONS)],
        )
        self.assertEqual(
            [(m.in_features, m.out_features) for m in linears(c)],
            [(OBS_DIM + PRIV_DIM, 512), (512, 512), (512, 512), (512, 1)],
        )
        # No dropout or normalisation layers.
        for net in (p, c):
            kinds = {type(m) for m in net.modules()}
            self.assertLessEqual(kinds, {type(net), nn.Sequential, nn.Linear, nn.ReLU})

    def test_custom_hidden_layers(self):
        p = PolicyNet(hidden=64, layers=2)
        self.assertEqual(
            [(m.in_features, m.out_features) for m in linears(p)],
            [(OBS_DIM, 64), (64, 64), (64, N_ACTIONS)],
        )
        c = CriticNet(hidden=32, layers=1)
        self.assertEqual(
            [(m.in_features, m.out_features) for m in linears(c)],
            [(OBS_DIM + PRIV_DIM, 32), (32, 1)],
        )
        body = mlp(10, 8, 4)
        self.assertEqual(len(linears(body)), 4)
        self.assertEqual(sum(isinstance(m, nn.ReLU) for m in body), 4)
        for bad in ((10, 8, 0), (10, 0, 2), (0, 8, 2)):
            with self.assertRaises(ValueError):
                mlp(*bad)

    def test_policy_rejects_wrong_inputs(self):
        b = 5
        obs = torch.from_numpy(np.ascontiguousarray(self.recs.obs[:b]))
        mask = torch.from_numpy(np.ascontiguousarray(self.recs.mask[:b]))
        x = torch.from_numpy(critic_input(self.recs.obs[:b], self.recs.critic[:b]))
        with self.assertRaises(ValueError):
            self.policy(x, mask)  # critic-sized input
        with self.assertRaises(ValueError):
            self.policy(obs[0], mask[0])  # 1-D
        with self.assertRaises(TypeError):
            self.policy(obs.double(), mask)
        with self.assertRaises(TypeError):
            self.policy(obs, mask.to(torch.uint8))
        with self.assertRaises(TypeError):
            self.policy(obs, mask.long())
        with self.assertRaises(ValueError):
            self.policy(obs, mask[:, :2])
        with self.assertRaises(ValueError):
            self.policy(obs, mask[:-1])
        with self.assertRaises(TypeError):
            self.policy(self.recs.obs[:b], self.recs.mask[:b])  # numpy, not tensors

    def test_critic_rejects_wrong_inputs(self):
        obs = torch.from_numpy(np.ascontiguousarray(self.recs.obs[:5]))
        with self.assertRaises(ValueError):
            self.critic(obs)
        x = torch.from_numpy(critic_input(self.recs.obs[:5], self.recs.critic[:5]))
        with self.assertRaises(TypeError):
            self.critic(x.double())

    def test_critic_input(self):
        r = self.recs
        x = critic_input(r.obs, r.critic)
        expected = np.concatenate([r.obs, rl_env.expand_critic(r.critic)], 1)
        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(x.shape, (len(r), OBS_DIM + PRIV_DIM))
        np.testing.assert_array_equal(x, expected)
        with self.assertRaises(TypeError):
            critic_input(r.obs.astype(np.float64), r.critic)
        with self.assertRaises(ValueError):
            critic_input(r.obs[:, :-1], r.critic)
        with self.assertRaises(ValueError):
            critic_input(r.obs, r.critic[:-1])


if __name__ == "__main__":
    unittest.main()
