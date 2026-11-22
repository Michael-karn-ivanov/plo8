"""The acting policy: an rl_env `Controller` over a `PolicyNet`.

`NeuralPolicy` answers a whole `DecisionBatch` with one network call, reading
only `batch.obs` and `batch.mask` (a DecisionBatch never carries the critic's
privileged input). It samples from the masked distribution with its own seeded
`torch.Generator` - torch's global RNG is never used - and returns
`(actions, logp)`, the tuple form of `Controller.act`, so rl_env's Recorder
keeps the log-probability of every chosen action.

The policy holds a private copy of the network (eval mode, no gradients); it
changes only through `load_state_dict`, which validates the state dict before
copying anything.
"""

from __future__ import annotations

import copy
import numbers
from typing import Any, Mapping, Optional

import numpy as np
import torch

from rl_env import ENCODER_VERSION, N_ACTIONS, OBS_DIM, DecisionBatch

from .nets import NetConfigError, PolicyNet, check_state_dict, log_probs

__all__ = ["NeuralPolicy"]

_SEED_LIMIT = 2**64  # torch.Generator.manual_seed accepts [0, 2**64)


def _check_seed(seed: Any) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral):
        raise TypeError(f"seed must be an int, got {seed!r}")
    seed = int(seed)
    if not 0 <= seed < _SEED_LIMIT:
        raise ValueError(f"seed must be in [0, 2**64), got {seed}")
    return seed


def _check_threads(num_threads: Any) -> Optional[int]:
    if num_threads is None:
        return None
    if isinstance(num_threads, (bool, np.bool_)) or not isinstance(num_threads, numbers.Integral):
        raise TypeError(f"num_threads must be an int or None, got {num_threads!r}")
    num_threads = int(num_threads)
    if num_threads < 1:
        raise ValueError(f"num_threads must be >= 1, got {num_threads}")
    return num_threads


class NeuralPolicy:
    """rl_env Controller sampling from a private copy of a `PolicyNet`.

    `num_threads` (default 1: a rollout worker owns one core) is passed to
    `torch.set_num_threads` at construction; None leaves it alone. This is a
    PROCESS-WIDE torch setting: all policies of one process should use the same
    value (the last one constructed wins).
    """

    wants_views = False  # the batch builds no views for it
    wants_obs = True  # it reads DecisionBatch.obs

    def __init__(self, net: PolicyNet, *, seed: int, num_threads: Optional[int] = 1) -> None:
        if not isinstance(net, PolicyNet):
            raise TypeError(f"net must be a PolicyNet, got {type(net).__name__}")
        seed = _check_seed(seed)
        num_threads = _check_threads(num_threads)
        # deepcopy draws no random numbers; the learner's later steps on `net`
        # do not reach this copy.
        own = copy.deepcopy(net)
        own.eval()
        for p in own.parameters():
            p.requires_grad_(False)
            p.grad = None
        self._net = own
        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(seed)
        if num_threads is not None:
            torch.set_num_threads(num_threads)

    @classmethod
    def from_state_dict(
        cls, state_dict: Mapping[str, Any], *, seed: int, num_threads: Optional[int] = 1
    ) -> "NeuralPolicy":
        """A policy from weights only (e.g. a worker that receives a learner's
        `state_dict()`). The recorded config must be a policy for this rl_env
        encoder; the throw-away initialisation of the network does not move
        torch's global RNG."""
        if not isinstance(state_dict, Mapping):
            raise NetConfigError(f"a state dict must be a mapping, got {type(state_dict).__name__}")
        extra = state_dict.get("_extra_state")
        if not isinstance(extra, dict):
            raise NetConfigError("state dict has no recorded config (_extra_state)")
        wanted = {
            "net": "policy",
            "encoder_version": ENCODER_VERSION,
            "in_dim": OBS_DIM,
            "n_actions": N_ACTIONS,
        }
        bad = [k for k, v in wanted.items() if extra.get(k) != v]
        if bad:
            details = ", ".join(f"{k}: state {extra.get(k)!r} != {wanted[k]!r}" for k in bad)
            raise NetConfigError(f"not a policy state dict for this encoder ({details})")
        hidden, layers = extra.get("hidden"), extra.get("layers")
        for name, v in (("hidden", hidden), ("layers", layers)):
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise NetConfigError(f"recorded {name} must be a positive int, got {v!r}")
        seed = _check_seed(seed)
        num_threads = _check_threads(num_threads)
        with torch.random.fork_rng(devices=[]):
            net = PolicyNet(hidden=hidden, layers=layers)
        policy = cls(net, seed=seed, num_threads=num_threads)
        policy.load_state_dict(state_dict)
        return policy

    # --- state -------------------------------------------------------------------

    @property
    def net(self) -> PolicyNet:
        """The private network (eval mode, no gradients). Read-only use."""
        return self._net

    def seed(self, seed: int) -> None:
        """Re-seed the sampling generator (restarts its sequence)."""
        self._gen.manual_seed(_check_seed(seed))

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Replace all weights. Validated first (config, encoder version, keys,
        shapes, dtypes): a refused state dict raises NetConfigError and leaves the
        current weights untouched. Tensors are copied, never aliased."""
        check_state_dict(self._net, state_dict)
        self._net.load_state_dict(state_dict, strict=True)

    def state_dict(self) -> dict:
        """The private network's `state_dict()`."""
        return self._net.state_dict()

    # --- acting ------------------------------------------------------------------

    def act(self, batch: DecisionBatch) -> tuple[np.ndarray, np.ndarray]:
        """One sampled action per row (int64, rl_env action order) and its
        log-probability under the current weights (float32). Reads only
        `batch.obs` and `batch.mask`."""
        if batch.obs is None:
            raise ValueError("NeuralPolicy needs observation vectors")
        mask = batch.mask
        n = mask.shape[0]
        if n == 0:
            return np.empty(0, np.int64), np.empty(0, np.float32)
        with torch.inference_mode():
            logits = self._net(torch.from_numpy(batch.obs), torch.from_numpy(mask))
            lp = log_probs(logits)  # (n, N_ACTIONS)
            a = torch.multinomial(lp.exp(), 1, generator=self._gen)  # (n, 1)
            logp = lp.gather(1, a).squeeze(1)
        actions = a.squeeze(1).numpy()
        # Cannot happen (illegal probability is exactly 0); an error rather than a
        # silent illegal action.
        if not mask[np.arange(n), actions].all():
            raise RuntimeError("sampled an illegal action")
        return actions, logp.numpy()
