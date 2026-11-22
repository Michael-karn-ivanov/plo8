"""Reproducible construction, state dict round trip, and config / encoder
version recorded in the state dict."""

import os
import tempfile
import unittest

import torch

from rl_env import ENCODER_VERSION, OBS_DIM, PRIV_DIM
from trainer import CriticNet, NetConfigError, PolicyNet, critic_input

try:
    from . import helpers
except ImportError:  # run as a top-level module
    import helpers  # type: ignore[no-redef]


def build(cls, seed, **kw):
    torch.manual_seed(seed)
    return cls(**kw)


def assert_same_state(test, a, b):
    sa, sb = a.state_dict(), b.state_dict()
    test.assertEqual(list(sa), list(sb))
    for k in sa:
        if k.endswith("_extra_state"):
            test.assertEqual(sa[k], sb[k])
        else:
            test.assertTrue(torch.equal(sa[k], sb[k]), k)


class StateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        recs = helpers.records()
        cls.obs = torch.from_numpy(recs.obs)
        cls.mask = torch.from_numpy(recs.mask)
        cls.x = torch.from_numpy(critic_input(recs.obs, recs.critic))

    def outputs(self, net):
        with torch.no_grad():
            if isinstance(net, PolicyNet):
                return net(self.obs, self.mask)
            return net(self.x)

    def test_same_seed_same_weights(self):
        for cls in (PolicyNet, CriticNet):
            a, b = build(cls, 11), build(cls, 11)
            assert_same_state(self, a, b)
            self.assertTrue(torch.equal(self.outputs(a), self.outputs(b)))
            c = build(cls, 12)
            self.assertFalse(torch.equal(a.body[0].weight, c.body[0].weight))
            self.assertFalse(torch.equal(a.head.weight, c.head.weight))

    def test_initialisation(self):
        net = build(PolicyNet, 13, hidden=64, layers=2)
        for layer in (net.body[0], net.body[2]):
            self.assertTrue(torch.all(layer.bias == 0))
            w = layer.weight
            # Orthogonal rows (64 <= in_features), scaled by sqrt(2).
            torch.testing.assert_close(w @ w.T, 2.0 * torch.eye(64), atol=1e-4, rtol=0)
        self.assertTrue(torch.all(net.head.bias == 0))
        h = net.head.weight
        torch.testing.assert_close(h @ h.T, 1e-4 * torch.eye(3), atol=1e-7, rtol=0)
        c = build(CriticNet, 13, hidden=64, layers=1)
        self.assertTrue(torch.all(c.head.bias == 0))
        torch.testing.assert_close(c.head.weight.norm(), torch.tensor(1.0), atol=1e-5, rtol=0)

    def test_round_trip(self):
        for cls in (PolicyNet, CriticNet):
            a, b = build(cls, 14), build(cls, 15)
            self.assertFalse(torch.equal(self.outputs(a), self.outputs(b)))
            b.load_state_dict(a.state_dict())
            self.assertTrue(torch.equal(self.outputs(a), self.outputs(b)))

            c = build(cls, 16)
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "net.pt")
                torch.save(a.state_dict(), path)
                state = torch.load(path, weights_only=True)
            c.load_state_dict(state)
            self.assertTrue(torch.equal(self.outputs(a), self.outputs(c)))

    def test_extra_state(self):
        p, c = PolicyNet(), CriticNet()
        self.assertEqual(
            p.state_dict()["_extra_state"],
            {
                "net": "policy",
                "encoder_version": ENCODER_VERSION,
                "in_dim": OBS_DIM,
                "hidden": 512,
                "layers": 3,
                "n_actions": 3,
            },
        )
        extra = c.state_dict()["_extra_state"]
        self.assertEqual(extra["net"], "critic")
        self.assertEqual(extra["encoder_version"], ENCODER_VERSION)
        self.assertEqual(extra["in_dim"], OBS_DIM + PRIV_DIM)

    def test_refuses_other_encoder_version(self):
        for cls in (PolicyNet, CriticNet):
            a, b = build(cls, 17), build(cls, 18)
            state = a.state_dict()
            state["_extra_state"] = dict(state["_extra_state"], encoder_version=ENCODER_VERSION + 1)
            before = self.outputs(b)
            with self.assertRaises(NetConfigError) as cm:
                b.load_state_dict(state)
            self.assertIn("encoder_version", str(cm.exception))
            # Refused before any weight was copied (the root's extra state is
            # checked before its children are loaded).
            self.assertTrue(torch.equal(before, self.outputs(b)))
            # A network built for another encoder version refuses the default one.
            other = cls(encoder_version=ENCODER_VERSION + 1)
            with self.assertRaises(NetConfigError):
                other.load_state_dict(a.state_dict())

    def test_refuses_other_config(self):
        policy = PolicyNet()
        with self.assertRaises((NetConfigError, RuntimeError)):
            CriticNet().load_state_dict(policy.state_dict())
        with self.assertRaises((NetConfigError, RuntimeError)):
            PolicyNet().load_state_dict(CriticNet().state_dict())
        with self.assertRaises((NetConfigError, RuntimeError)):
            PolicyNet(hidden=256).load_state_dict(policy.state_dict())
        with self.assertRaises((NetConfigError, RuntimeError)):
            PolicyNet(layers=2).load_state_dict(policy.state_dict())
        # A state dict without the recorded config is refused too (strict load).
        state = policy.state_dict()
        del state["_extra_state"]
        with self.assertRaises(RuntimeError):
            PolicyNet().load_state_dict(state)

    def test_train_eval_same_outputs(self):
        net = build(PolicyNet, 19)
        net.train()
        a = self.outputs(net)
        net.eval()
        self.assertTrue(torch.equal(a, self.outputs(net)))
        # Pure function: repeated calls agree.
        self.assertTrue(torch.equal(a, self.outputs(net)))


if __name__ == "__main__":
    unittest.main()
