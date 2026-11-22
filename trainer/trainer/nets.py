"""The policy and critic networks.

- `PolicyNet(obs, mask)`: obs float32 (B, OBS_DIM), mask bool (B, N_ACTIONS)
  (True = legal, rl_env's order fold / check-or-call / bet-or-raise) -> masked
  logits (B, N_ACTIONS). It has no other input and refuses an input of any other
  width, so the critic's privileged input can never reach it.
- `CriticNet(x)`: x float32 (B, OBS_DIM + PRIV_DIM) from `critic_input` -> value
  (B,) in big bets.

Masking sets illegal logits to the finite `MASK_VALUE = -1e9` (never -inf) and
clamps legal logits to +-`LOGIT_LIMIT`, so softmax, log-probabilities and entropy
never produce NaN and illegal actions get exactly zero probability.

Both networks record their config and the rl_env `ENCODER_VERSION` in the state
dict (`_extra_state`) and refuse to load a state dict built for another one.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from rl_env import ENCODER_VERSION, N_ACTIONS, OBS_DIM

__all__ = [
    "MASK_VALUE",
    "LOGIT_LIMIT",
    "DEFAULT_HIDDEN",
    "DEFAULT_LAYERS",
    "NetConfigError",
    "mlp",
    "masked_logits",
    "PolicyNet",
]

MASK_VALUE = -1e9  # logit of an illegal action (finite: never -inf)
LOGIT_LIMIT = 1e4  # legal logits are clamped to [-LOGIT_LIMIT, LOGIT_LIMIT]
DEFAULT_HIDDEN = 512
DEFAULT_LAYERS = 3

_HIDDEN_GAIN = math.sqrt(2.0)
_POLICY_HEAD_GAIN = 0.01
_CRITIC_HEAD_GAIN = 1.0


class NetConfigError(ValueError):
    """A state dict was built for a network of another kind, config or encoder
    version."""


# --- building blocks -------------------------------------------------------------


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive int, got {value!r}")
    return value


def _linear(in_dim: int, out_dim: int, gain: float) -> nn.Linear:
    """A linear layer with orthogonal weights (`gain`) and zero bias. Built with
    `skip_init`, so torch's default init draws no random numbers: the only draws
    are the orthogonal init's, in construction order."""
    layer = torch.nn.utils.skip_init(nn.Linear, in_dim, out_dim)
    with torch.no_grad():
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.zeros_(layer.bias)
    return layer


def mlp(in_dim: int, hidden: int = DEFAULT_HIDDEN, layers: int = DEFAULT_LAYERS) -> nn.Sequential:
    """`layers` x (Linear(d, hidden), ReLU), d = `in_dim` then `hidden`; the body
    only (each network adds its own head). Hidden layers are initialised
    orthogonally with gain sqrt(2), bias 0, first to last."""
    _positive_int("in_dim", in_dim)
    _positive_int("hidden", hidden)
    _positive_int("layers", layers)
    mods: list[nn.Module] = []
    d = in_dim
    for _ in range(layers):
        mods.append(_linear(d, hidden, _HIDDEN_GAIN))
        mods.append(nn.ReLU())
        d = hidden
    return nn.Sequential(*mods)


def _check_input(x: Any, width: int, name: str) -> None:
    if not isinstance(x, Tensor):
        raise TypeError(
            f"{name} must be a torch.Tensor (use torch.from_numpy on rl_env's array), "
            f"got {type(x).__name__}"
        )
    if x.dtype != torch.float32:
        raise TypeError(f"{name} must be float32, got {x.dtype}")
    if x.ndim != 2 or x.shape[1] != width:
        raise ValueError(f"{name} must have shape (B, {width}), got {tuple(x.shape)}")


def _check_mask(mask: Any, shape: Any) -> None:
    if not isinstance(mask, Tensor):
        raise TypeError(
            f"mask must be a torch.Tensor (use torch.from_numpy on rl_env's array), "
            f"got {type(mask).__name__}"
        )
    if mask.dtype != torch.bool:
        # On a uint8 mask `~mask` is a bitwise not (1 -> 254): it would mask nothing.
        raise TypeError(f"mask must be torch.bool, got {mask.dtype}")
    if tuple(mask.shape) != tuple(shape):
        raise ValueError(f"mask must have shape {tuple(shape)}, got {tuple(mask.shape)}")


def masked_logits(raw: Tensor, mask: Tensor) -> Tensor:
    """Legal logits clamped to +-`LOGIT_LIMIT`, illegal logits exactly
    `MASK_VALUE`. Every legal logit exceeds every illegal one by at least
    1e9 - 1e4, so a float32 softmax gives illegal actions exactly 0."""
    _check_mask(mask, raw.shape)
    return raw.clamp(-LOGIT_LIMIT, LOGIT_LIMIT).masked_fill(~mask, MASK_VALUE)


# --- config in the state dict ----------------------------------------------------

_MISSING = object()


class _ConfigMixin:
    """`get_extra_state` / `set_extra_state`: the network's config (plain ints and
    strings, readable with `torch.load(weights_only=True)`) is stored in the state
    dict and must match on load."""

    def _config(self) -> dict:
        raise NotImplementedError

    def get_extra_state(self) -> dict:
        return dict(self._config())

    def set_extra_state(self, state: Any) -> None:
        own = self._config()
        if not isinstance(state, dict):
            raise NetConfigError(f"expected a config dict in _extra_state, got {type(state).__name__}")
        keys = sorted(set(own) | set(state), key=str)
        diff = [k for k in keys if own.get(k, _MISSING) != state.get(k, _MISSING)]
        if diff:
            details = ", ".join(
                f"{k}: state {state.get(k, '<missing>')!r} != net {own.get(k, '<missing>')!r}"
                for k in diff
            )
            raise NetConfigError(f"state dict does not match this network ({details})")


# --- networks --------------------------------------------------------------------


class PolicyNet(_ConfigMixin, nn.Module):
    """Masked action logits from the observation and the mask only."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        n_actions: int = N_ACTIONS,
        hidden: int = DEFAULT_HIDDEN,
        layers: int = DEFAULT_LAYERS,
        encoder_version: int = ENCODER_VERSION,
    ) -> None:
        super().__init__()
        self.obs_dim = _positive_int("obs_dim", obs_dim)
        self.n_actions = _positive_int("n_actions", n_actions)
        self.hidden = _positive_int("hidden", hidden)
        self.layers = _positive_int("layers", layers)
        self.encoder_version = int(encoder_version)
        # Body first (first to last layer), then the head: a fixed RNG order.
        self.body = mlp(obs_dim, hidden, layers)
        self.head = _linear(hidden, n_actions, _POLICY_HEAD_GAIN)

    def _config(self) -> dict:
        return {
            "net": "policy",
            "encoder_version": self.encoder_version,
            "in_dim": self.obs_dim,
            "hidden": self.hidden,
            "layers": self.layers,
            "n_actions": self.n_actions,
        }

    def forward(self, obs: Tensor, mask: Tensor) -> Tensor:
        """obs float32 (B, obs_dim), mask bool (B, n_actions) -> logits float32
        (B, n_actions), columns in rl_env's action order. No copy, reshape or
        dtype conversion of the inputs."""
        _check_input(obs, self.obs_dim, "obs")
        _check_mask(mask, (obs.shape[0], self.n_actions))
        return masked_logits(self.head(self.body(obs)), mask)
