"""The PPO learner: training records -> PPO updates of both networks.

- `make_batch(*records) -> PPOBatch` validates rl_env `TrainingRecords` (of one
  or several workers), concatenates them in numpy (always a copy) and builds the
  tensors the loss needs, including the critic input `critic_x` (rl_env's
  `expand_critic` through `critic_input`, once per update).
- `compute_advantages(critic, batch)`: return minus the pre-update critic value,
  computed once per update without gradient. Every decision belongs to a
  single-hand episode with one terminal reward, so the return is the reward
  (no discounting, no GAE, no bootstrapping).
- `ppo_loss(policy, critic, batch, adv, idx, cfg) -> (loss, LossParts)`: the
  clipped surrogate, the critic's MSE and the entropy bonus of one minibatch. A
  stateless function (the later magnet term extends it). Log-probabilities come
  from `log_probs(PolicyNet(obs, mask))`, exactly as `NeuralPolicy` acts, so an
  unchanged policy gives a ratio of 1. The policy sees only `obs` and `mask`;
  the privileged `critic_x` goes only to the critic.
- `Learner(config, seed=...)` owns both networks, one Adam optimiser over them
  and a private shuffle generator. `update(*records)` runs `epochs` passes of
  shuffled minibatches; `get_weights()` hands out an unaliased copy of the
  policy weights for the next collection; `save` / `load` write and restore
  everything needed to resume exactly.

Torch's global RNG is never used (network initialisation runs inside
`torch.random.fork_rng`), and every refused input raises before any weight,
optimiser state, generator or counter changes.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import math
import numbers
import os
import time
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import numpy as np
import torch
from torch import Tensor

from rl_env import ENCODER_VERSION, N_ACTIONS, OBS_DIM, TrainingRecords

from .nets import (
    DEFAULT_HIDDEN,
    DEFAULT_LAYERS,
    CriticNet,
    NetConfigError,
    PolicyNet,
    check_state_dict,
    critic_input,
    log_probs,
)

__all__ = [
    "PPOConfig",
    "PPOBatch",
    "make_batch",
    "compute_advantages",
    "explained_variance",
    "ppo_loss",
    "LossParts",
    "Learner",
    "UpdateStats",
    "CHECKPOINT_FORMAT",
]

CHECKPOINT_FORMAT = "trainer.learner/1"
_CHECKPOINT_KEYS = (
    "format",
    "encoder_version",
    "config",
    "seed",
    "n_updates",
    "policy",
    "critic",
    "optimizer",
    "shuffle_rng",
)
_CRITIC_CARDS_SHAPE = (9, 4)  # rl_env's critic_cards rows: 9 other seats x 4 cards
_NET_FIELDS = ("hidden", "layers")


# --- small validators --------------------------------------------------------------


def _is_int(value: Any) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))


def _check_seed(seed: Any) -> int:
    if not _is_int(seed):
        raise TypeError(f"seed must be an int, got {seed!r}")
    seed = int(seed)
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    return seed


def _check_threads(num_threads: Any) -> Optional[int]:
    if num_threads is None:
        return None
    if not _is_int(num_threads):
        raise TypeError(f"num_threads must be an int or None, got {num_threads!r}")
    num_threads = int(num_threads)
    if num_threads < 1:
        raise ValueError(f"num_threads must be >= 1, got {num_threads}")
    return num_threads


def _seed32(ss: np.random.SeedSequence) -> int:
    return int(ss.generate_state(1, np.uint32)[0])


@contextlib.contextmanager
def _torch_threads(num_threads: Optional[int]) -> Iterator[None]:
    """Run with `num_threads` torch threads and restore the previous count
    afterwards (also on error); None changes nothing. `torch.set_num_threads`
    is process-wide, so a NeuralPolicy (1 thread) and a learner in one process
    each run with their own count."""
    if num_threads is None:
        yield
        return
    previous = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


# --- configuration -----------------------------------------------------------------

_INT_FIELDS = ("epochs", "minibatch", "hidden", "layers")
_FLOAT_FIELDS = ("lr", "clip", "vf_coef", "ent_coef", "max_grad_norm", "adam_eps", "adv_eps")


@dataclass(frozen=True)
class PPOConfig:
    """PPO hyper-parameters and the size of both networks. Plain ints and floats
    only, so `dataclasses.asdict` goes into a `weights_only` checkpoint."""

    lr: float = 3e-4  # Adam
    clip: float = 0.2
    epochs: int = 3
    minibatch: int = 4096  # decisions per minibatch
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 1.0
    hidden: int = DEFAULT_HIDDEN  # both networks
    layers: int = DEFAULT_LAYERS  # both networks
    adam_eps: float = 1e-5  # CleanRL's PPO value
    adv_eps: float = 1e-8  # advantage normalisation

    def __post_init__(self) -> None:
        for name in _INT_FIELDS:
            v = getattr(self, name)
            if not _is_int(v):
                raise TypeError(f"PPOConfig.{name} must be an int, got {v!r}")
            object.__setattr__(self, name, int(v))
        for name in _FLOAT_FIELDS:
            v = getattr(self, name)
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real):
                raise TypeError(f"PPOConfig.{name} must be a float, got {v!r}")
            v = float(v)
            if not math.isfinite(v):
                raise ValueError(f"PPOConfig.{name} must be finite, got {v!r}")
            object.__setattr__(self, name, v)
        checks = (
            ("lr", self.lr > 0, "> 0"),
            ("clip", 0 < self.clip < 1, "in (0, 1)"),
            ("epochs", self.epochs >= 1, ">= 1"),
            ("minibatch", self.minibatch >= 1, ">= 1"),
            ("vf_coef", self.vf_coef >= 0, ">= 0"),
            ("ent_coef", self.ent_coef >= 0, ">= 0"),
            ("max_grad_norm", self.max_grad_norm > 0, "> 0"),
            ("hidden", self.hidden >= 1, ">= 1"),
            ("layers", self.layers >= 1, ">= 1"),
            ("adam_eps", self.adam_eps > 0, "> 0"),
            ("adv_eps", self.adv_eps > 0, "> 0"),
        )
        for name, ok, rule in checks:
            if not ok:
                raise ValueError(f"PPOConfig.{name} must be {rule}, got {getattr(self, name)!r}")


# --- the batch ---------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PPOBatch:
    """The tensors of one update (CPU, own memory)."""

    obs: Tensor  # float32 (N, OBS_DIM)
    mask: Tensor  # bool (N, N_ACTIONS)
    action: Tensor  # int64 (N,)
    logp: Tensor  # float32 (N,): recorded (old) log-probabilities
    critic_x: Tensor  # float32 (N, OBS_DIM + PRIV_DIM): PRIVILEGED, critic only
    reward: Tensor  # float32 (N,): the return of each decision, big bets

    def __len__(self) -> int:
        return int(self.action.shape[0])


def _check_array(i: int, name: str, x: Any, dtype: Any, shape: tuple) -> None:
    if not isinstance(x, np.ndarray):
        raise TypeError(f"records[{i}].{name} must be a numpy array, got {type(x).__name__}")
    if x.dtype != dtype:
        raise TypeError(f"records[{i}].{name} must be {np.dtype(dtype)}, got {x.dtype}")
    if x.shape != shape:
        raise ValueError(f"records[{i}].{name} must have shape {shape}, got {x.shape}")


def _check_record(i: int, r: TrainingRecords) -> int:
    """Shapes and dtypes of one record; returns its row count."""
    obs = r.obs
    if not isinstance(obs, np.ndarray):
        raise TypeError(f"records[{i}].obs must be a numpy array, got {type(obs).__name__}")
    if obs.ndim != 2 or obs.shape[1] != OBS_DIM:
        raise ValueError(
            f"records[{i}].obs must have shape (N, {OBS_DIM}), got {obs.shape} "
            f"(records of another encoder layout?)"
        )
    if obs.dtype != np.float32:
        raise TypeError(f"records[{i}].obs must be float32, got {obs.dtype}")
    n = obs.shape[0]
    _check_array(i, "mask", r.mask, np.bool_, (n, N_ACTIONS))
    _check_array(i, "action", r.action, np.int64, (n,))
    _check_array(i, "logp", r.logp, np.float32, (n,))
    _check_array(i, "reward", r.reward, np.float32, (n,))
    critic = r.critic
    if not isinstance(critic, np.ndarray):
        raise TypeError(f"records[{i}].critic must be a numpy array, got {type(critic).__name__}")
    if critic.dtype.kind not in "iu":
        raise TypeError(f"records[{i}].critic must be an integer array, got {critic.dtype}")
    if critic.shape != (n, *_CRITIC_CARDS_SHAPE):
        raise ValueError(
            f"records[{i}].critic must have shape {(n, *_CRITIC_CARDS_SHAPE)}, got {critic.shape}"
        )
    return n


def make_batch(*records: TrainingRecords) -> PPOBatch:
    """Validate the records, concatenate them (argument order, always a copy) and
    build the tensors of one update. Nothing is built before every check has
    passed, and the caller's records are never modified or aliased."""
    if not records:
        raise TypeError("make_batch needs at least one TrainingRecords")
    for i, r in enumerate(records):
        if not isinstance(r, TrainingRecords):
            raise TypeError(f"records[{i}] must be an rl_env.TrainingRecords, got {type(r).__name__}")
    total = sum(_check_record(i, r) for i, r in enumerate(records))
    if total == 0:
        raise ValueError("no decisions")
    no_logp = sum(int((~np.isfinite(r.logp)).sum()) for r in records)
    if no_logp:
        raise ValueError(
            f"{no_logp} records without log-probability, not from a NeuralPolicy "
            f"(e.g. rows of a RandomController or an arena bot)"
        )
    for i, r in enumerate(records):
        n = len(r.action)
        if n == 0:
            continue
        if ((r.action < 0) | (r.action >= N_ACTIONS)).any():
            raise ValueError(f"records[{i}] has actions outside 0..{N_ACTIONS - 1}")
        if not r.mask[np.arange(n), r.action].all():
            raise ValueError(f"records[{i}] has recorded actions not allowed by their mask")
        if not np.isfinite(r.reward).all():
            raise ValueError(f"records[{i}] has non-finite rewards")

    def cat(name: str) -> np.ndarray:  # np.concatenate always copies
        return np.concatenate([getattr(r, name) for r in records])

    obs, critic = cat("obs"), cat("critic")
    x = critic_input(obs, critic)  # PRIVILEGED: only for the critic
    return PPOBatch(
        obs=torch.from_numpy(obs),
        mask=torch.from_numpy(cat("mask")),
        action=torch.from_numpy(cat("action")),
        logp=torch.from_numpy(cat("logp")),
        critic_x=torch.from_numpy(x),
        reward=torch.from_numpy(cat("reward")),
    )


# --- advantages --------------------------------------------------------------------


def compute_advantages(
    critic: CriticNet, batch: PPOBatch, chunk: int = 8192
) -> tuple[Tensor, Tensor]:
    """(adv, values), both float32 (N,): `reward - critic(critic_x)` with the
    critic as it is now, without gradient, in chunks of `chunk` rows. Called once
    per update, before the first optimiser step."""
    if not _is_int(chunk) or chunk < 1:
        raise ValueError(f"chunk must be a positive int, got {chunk!r}")
    with torch.no_grad():
        values = torch.cat(
            [critic(batch.critic_x[s : s + chunk]) for s in range(0, len(batch), chunk)]
        )
    return batch.reward - values, values


def explained_variance(values: Tensor, reward: Tensor) -> float:
    """1 - Var(R - V) / Var(R), float64, population variances. NaN when every
    reward is the same - decided by exact equality, not Var(R) == 0: the float64
    variance of identical values can be a tiny positive number. `reward` has at
    least one row (make_batch refuses empty batches)."""
    if bool((reward == reward[0]).all()):  # every reward the same: undefined
        return math.nan
    r, v = reward.detach().double(), values.detach().double()
    return float(1.0 - (r - v).var(correction=0) / r.var(correction=0))


# --- the loss ----------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class LossParts:
    policy_loss: Tensor  # scalar, with graph
    value_loss: Tensor  # scalar, with graph
    entropy: Tensor  # scalar, with graph
    approx_kl: Tensor  # scalar, detached
    clip_frac: Tensor  # scalar, detached
    ratio: Tensor  # (len(idx),), detached


def ppo_loss(
    policy: PolicyNet,
    critic: CriticNet,
    batch: PPOBatch,
    adv: Tensor,
    idx: Tensor,
    cfg: PPOConfig,
) -> tuple[Tensor, LossParts]:
    """PPO loss of the minibatch `idx`: clipped surrogate + vf_coef * value MSE -
    ent_coef * entropy. Advantages are normalised within the minibatch
    (population std, so one row gives 0, not NaN). The policy receives only
    `obs` and `mask`; `critic_x` goes only to the critic."""
    lp = log_probs(policy(batch.obs[idx], batch.mask[idx]))  # (M, N_ACTIONS)
    new_logp = lp.gather(1, batch.action[idx].unsqueeze(1)).squeeze(1)
    log_ratio = new_logp - batch.logp[idx]
    ratio = log_ratio.exp()
    a = adv[idx]
    a = (a - a.mean()) / (a.std(correction=0) + cfg.adv_eps)
    pg = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
    v = 0.5 * (critic(batch.critic_x[idx]) - batch.reward[idx]).pow(2).mean()
    # Illegal actions: exp(lp) == 0 exactly and lp is finite, so 0 * lp == 0.
    ent = -(lp.exp() * lp).sum(-1).mean()
    loss = pg + cfg.vf_coef * v - cfg.ent_coef * ent
    with torch.no_grad():
        kl = ((ratio - 1) - log_ratio).mean()
        cf = ((ratio - 1).abs() > cfg.clip).float().mean()
    return loss, LossParts(pg, v, ent, kl, cf, ratio.detach())


# --- statistics --------------------------------------------------------------------


@dataclass(frozen=True)
class UpdateStats:
    policy_loss: float  # \
    value_loss: float  #  | means over the minibatches of the last epoch
    entropy: float  #  | (each minibatch weighs equally)
    approx_kl: float  #  | mean of (ratio - 1) - log ratio
    clip_frac: float  #  | share of decisions with |ratio - 1| > clip
    grad_norm: float  # /  global norm before clipping
    grad_norm_max: float  # max over all steps of the update, before clipping
    explained_variance: float  # 1 - Var(R - V_old) / Var(R), pre-update; nan if all R equal
    value_mean: float  # mean pre-update critic value, big bets
    reward_mean: float  # mean return, big bets
    adv_std: float  # std of the raw (unnormalised) advantages
    decisions: int  # rows of the update
    steps: int  # minibatch steps = epochs * ceil(decisions / minibatch)
    update: int  # n_updates after this update
    wall_time: float  # seconds


# --- the learner -------------------------------------------------------------------


class Learner:
    """Both networks, one Adam optimiser over them and a private shuffle
    generator; PPO updates from training records (see the module docstring)."""

    def __init__(
        self,
        config: Optional[PPOConfig] = None,
        *,
        seed: int,
        num_threads: Optional[int] = None,
    ) -> None:
        cfg = PPOConfig() if config is None else config
        if not isinstance(cfg, PPOConfig):
            raise TypeError(f"config must be a PPOConfig, got {type(cfg).__name__}")
        seed = _check_seed(seed)
        num_threads = _check_threads(num_threads)
        policy_ss, critic_ss, shuffle_ss = np.random.SeedSequence(seed).spawn(3)
        # The seed sets both networks' initial weights; torch's global RNG is
        # left where it was.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(_seed32(policy_ss))
            policy = PolicyNet(hidden=cfg.hidden, layers=cfg.layers)
            torch.manual_seed(_seed32(critic_ss))
            critic = CriticNet(hidden=cfg.hidden, layers=cfg.layers)
        policy.train()
        critic.train()
        self._cfg = cfg
        self._seed = seed
        self._num_threads = num_threads
        self._policy = policy
        self._critic = critic
        self._gen = torch.Generator(device="cpu")
        self._gen.manual_seed(_seed32(shuffle_ss))  # minibatch permutations only
        self._params = [*policy.parameters(), *critic.parameters()]
        self._opt = torch.optim.Adam(self._params, lr=cfg.lr, eps=cfg.adam_eps)
        self._n_updates = 0

    # --- read-only views ---------------------------------------------------------

    @property
    def config(self) -> PPOConfig:
        return self._cfg

    @property
    def policy(self) -> PolicyNet:
        """The live trainable policy network."""
        return self._policy

    @property
    def critic(self) -> CriticNet:
        """The live trainable critic network."""
        return self._critic

    @property
    def optimizer(self) -> torch.optim.Adam:
        return self._opt

    @property
    def n_updates(self) -> int:
        return self._n_updates

    @property
    def seed(self) -> int:
        return self._seed

    # --- training ----------------------------------------------------------------

    def update(self, *records: TrainingRecords) -> UpdateStats:
        """One PPO update from the records of one or several collections
        (concatenated in argument order). Refused records raise before anything
        changes; the records are only read."""
        batch = make_batch(*records)
        cfg, n = self._cfg, len(batch)
        with _torch_threads(self._num_threads):
            t0 = time.perf_counter()
            adv, values = compute_advantages(self._critic, batch)
            ev = explained_variance(values, batch.reward)  # nan if every reward is equal
            last: list[tuple[LossParts, float]] = []
            gmax, steps = 0.0, 0
            for epoch in range(cfg.epochs):
                perm = torch.randperm(n, generator=self._gen)
                for s in range(0, n, cfg.minibatch):
                    idx = perm[s : s + cfg.minibatch]  # the last one may be smaller
                    # Through the module global, so tests (and later stages) can wrap it.
                    loss, parts = ppo_loss(self._policy, self._critic, batch, adv, idx, cfg)
                    self._opt.zero_grad(set_to_none=True)
                    loss.backward()
                    gn = float(
                        torch.nn.utils.clip_grad_norm_(
                            self._params, cfg.max_grad_norm, error_if_nonfinite=True
                        )
                    )
                    self._opt.step()
                    steps += 1
                    gmax = max(gmax, gn)
                    if epoch == cfg.epochs - 1:
                        last.append((parts, gn))
            self._n_updates += 1
            wall = time.perf_counter() - t0

        def mean(get) -> float:
            return float(np.mean([get(p, g) for p, g in last]))

        return UpdateStats(
            policy_loss=mean(lambda p, g: float(p.policy_loss.detach())),
            value_loss=mean(lambda p, g: float(p.value_loss.detach())),
            entropy=mean(lambda p, g: float(p.entropy.detach())),
            approx_kl=mean(lambda p, g: float(p.approx_kl)),
            clip_frac=mean(lambda p, g: float(p.clip_frac)),
            grad_norm=mean(lambda p, g: g),
            grad_norm_max=float(gmax),
            explained_variance=ev,
            value_mean=float(values.double().mean()),
            reward_mean=float(batch.reward.double().mean()),
            adv_std=float(adv.double().std(correction=0)),
            decisions=n,
            steps=steps,
            update=self._n_updates,
            wall_time=float(wall),
        )

    def get_weights(self) -> dict[str, Any]:
        """The current policy weights (with config and encoder version) as an
        unaliased copy: the form NeuralPolicy and RolloutWorker.collect accept."""
        return {
            k: copy.deepcopy(v) if k.endswith("_extra_state") else v.detach().clone()
            for k, v in self._policy.state_dict().items()
        }

    # --- checkpoints -------------------------------------------------------------

    def save(self, path: str | os.PathLike) -> None:
        """Everything needed to resume, in one `weights_only`-readable file
        (written to `path + ".tmp"`, then renamed over `path`)."""
        path = os.fspath(path)
        tmp = path + ".tmp"
        state = {
            "format": CHECKPOINT_FORMAT,
            "encoder_version": ENCODER_VERSION,
            "config": dataclasses.asdict(self._cfg),
            "seed": self._seed,
            "n_updates": self._n_updates,
            "policy": self._policy.state_dict(),
            "critic": self._critic.state_dict(),
            "optimizer": self._opt.state_dict(),
            "shuffle_rng": self._gen.get_state(),
        }
        try:
            torch.save(state, tmp)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    @classmethod
    def load(
        cls,
        path: str | os.PathLike,
        *,
        config: Optional[PPOConfig] = None,
        num_threads: Optional[int] = None,
    ) -> "Learner":
        """A learner restored from `save`. `config` may change the training
        fields (lr, clip, ...) but not the network fields (`hidden`, `layers`:
        NetConfigError). Another encoder version raises NetConfigError, a
        malformed file ValueError."""
        if config is not None and not isinstance(config, PPOConfig):
            raise TypeError(f"config must be a PPOConfig, got {type(config).__name__}")
        num_threads = _check_threads(num_threads)
        with _torch_threads(num_threads):
            ckpt = torch.load(os.fspath(path), map_location="cpu", weights_only=True)
            if not isinstance(ckpt, dict):
                raise ValueError(f"not a learner checkpoint: {type(ckpt).__name__}")
            if ckpt.get("format") != CHECKPOINT_FORMAT:
                raise ValueError(
                    f"not a learner checkpoint: format {ckpt.get('format')!r} != {CHECKPOINT_FORMAT!r}"
                )
            missing = [k for k in _CHECKPOINT_KEYS if k not in ckpt]
            if missing:
                raise ValueError(f"learner checkpoint is missing {missing}")
            if ckpt["encoder_version"] != ENCODER_VERSION:
                raise NetConfigError(
                    f"checkpoint encoder_version {ckpt['encoder_version']!r} != "
                    f"rl_env.ENCODER_VERSION {ENCODER_VERSION!r}"
                )
            stored = ckpt["config"]
            names = [f.name for f in dataclasses.fields(PPOConfig)]
            if not isinstance(stored, dict) or set(stored) != set(names):
                raise ValueError(f"checkpoint config must have exactly the fields {names}")
            stored_cfg = PPOConfig(**stored)
            if config is None:
                cfg = stored_cfg
            else:
                diff = [k for k in _NET_FIELDS if getattr(config, k) != getattr(stored_cfg, k)]
                if diff:
                    details = ", ".join(
                        f"{k}: config {getattr(config, k)!r} != checkpoint {getattr(stored_cfg, k)!r}"
                        for k in diff
                    )
                    raise NetConfigError(f"network config differs from the checkpoint ({details})")
                cfg = config
            seed, n_updates = ckpt["seed"], ckpt["n_updates"]
            if not _is_int(n_updates) or n_updates < 0:
                raise ValueError(f"checkpoint n_updates must be an int >= 0, got {n_updates!r}")
            rng = ckpt["shuffle_rng"]
            if not isinstance(rng, Tensor) or rng.dtype != torch.uint8:
                raise ValueError("checkpoint shuffle_rng must be a uint8 tensor")
            if not isinstance(ckpt["optimizer"], dict):
                raise ValueError("checkpoint optimizer must be a state dict")

            learner = cls(cfg, seed=seed, num_threads=num_threads)
            check_state_dict(learner._policy, ckpt["policy"])  # nothing copied yet
            check_state_dict(learner._critic, ckpt["critic"])
            learner._policy.load_state_dict(ckpt["policy"])
            learner._critic.load_state_dict(ckpt["critic"])
            learner._opt.load_state_dict(ckpt["optimizer"])
            for group in learner._opt.param_groups:  # Adam's state carries the stored ones
                group["lr"] = cfg.lr
                group["eps"] = cfg.adam_eps
            learner._gen.set_state(rng)
            learner._n_updates = int(n_updates)
        return learner
