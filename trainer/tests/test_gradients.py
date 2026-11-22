"""Gradients flow: one optimisation step changes every parameter tensor of each
network, with finite gradients; policy and critic share nothing."""

import unittest

import numpy as np
import torch
from torch.distributions import Categorical

from trainer import CriticNet, PolicyNet, critic_input

try:
    from . import helpers
except ImportError:  # run as a top-level module
    import helpers  # type: ignore[no-redef]


def snapshot(net):
    return {name: p.detach().clone() for name, p in net.named_parameters()}


class GradientsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        recs = helpers.records()
        idx = helpers.rows_with_masks(recs, 256, per_mask=40)
        cls.obs = np.ascontiguousarray(recs.obs[idx])
        cls.mask = np.ascontiguousarray(recs.mask[idx])
        cls.x = critic_input(cls.obs, np.ascontiguousarray(recs.critic[idx]))
        cls.reward = np.ascontiguousarray(recs.reward[idx])

    def assert_step_changes_all(self, net, loss_fn):
        before = snapshot(net)
        opt = torch.optim.SGD(net.parameters(), lr=0.1)
        opt.zero_grad()
        loss = loss_fn()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, p in net.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertTrue(torch.any(p.grad != 0), f"{name}: all-zero gradient")
        opt.step()
        for name, p in net.named_parameters():
            self.assertFalse(torch.equal(before[name], p.detach()), f"{name} unchanged")
        names = set(before)
        self.assertTrue(any(n.startswith("body.") for n in names))
        self.assertTrue(any(n.startswith("head.") for n in names))

    def test_policy_step(self):
        torch.manual_seed(7)
        net = PolicyNet()
        obs = torch.from_numpy(self.obs)
        mask = torch.from_numpy(self.mask)
        gen = torch.Generator().manual_seed(8)
        adv = torch.randn(len(self.obs), generator=gen)

        def loss_fn():
            dist = Categorical(logits=net(obs, mask))
            with torch.no_grad():
                a = torch.multinomial(dist.probs, 1, generator=gen).squeeze(1)
            self.assertTrue(torch.all(mask.gather(1, a[:, None])))
            return -(dist.log_prob(a) * adv).mean() - 0.01 * dist.entropy().mean()

        self.assert_step_changes_all(net, loss_fn)

    def test_critic_step(self):
        torch.manual_seed(9)
        net = CriticNet()
        x = torch.from_numpy(self.x)
        target = torch.from_numpy(self.reward)
        self.assert_step_changes_all(net, lambda: ((net(x) - target) ** 2).mean())

    def test_disjoint_parameters(self):
        torch.manual_seed(10)
        policy, critic = PolicyNet(), CriticNet()
        p_ptrs = {p.data_ptr() for p in policy.parameters()}
        c_ptrs = {p.data_ptr() for p in critic.parameters()}
        self.assertEqual(len(p_ptrs), len(list(policy.parameters())))
        self.assertFalse(p_ptrs & c_ptrs)

        before = snapshot(policy)
        opt = torch.optim.SGD(critic.parameters(), lr=0.1)
        loss = ((critic(torch.from_numpy(self.x)) - torch.from_numpy(self.reward)) ** 2).mean()
        loss.backward()
        opt.step()
        for name, p in policy.named_parameters():
            self.assertTrue(torch.equal(before[name], p.detach()), name)
            self.assertIsNone(p.grad, name)


if __name__ == "__main__":
    unittest.main()
