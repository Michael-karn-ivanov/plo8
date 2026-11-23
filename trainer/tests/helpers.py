"""Test data from rl_env: training records of real hands, every mask rl_env can
produce, hand-built decision batches, peaked policy networks, small weights,
rollout workers and learners, records built from NeuralPolicy actions, a view
counter and a table driver for controllers."""

from __future__ import annotations

import contextlib
import functools
from typing import Any, Callable, Hashable, Iterable, Mapping, Optional

import numpy as np
import torch

from rl_env import (
    DecisionBatch,
    HandSampler,
    RandomController,
    Recorder,
    Table,
    TableBatch,
    TrainingRecords,
)
from trainer import Learner, NeuralPolicy, PolicyNet, PPOConfig, RolloutWorker

# Every mask rl_env can produce, over (fold, check-or-call, bet-or-raise).
ALL_MASKS = np.array(
    [
        [False, True, False],  # check free, no raise (e.g. everyone else all-in)
        [False, True, True],  # check free, raise allowed
        [True, True, False],  # facing a bet, cannot raise (cap reached or short stack)
        [True, True, True],  # facing a bet, raise allowed
    ],
    dtype=bool,
)
COMMON_MASKS = ALL_MASKS[1:]
SINGLE_LEGAL = ALL_MASKS[0]


@functools.lru_cache(maxsize=None)
def _records(n_hands: int, seed: int, sizes: tuple) -> TrainingRecords:
    recorder = Recorder()
    batch = TableBatch(64, recorder=recorder)
    controllers = {"random": RandomController(seed)}
    sampler = HandSampler(seed, sizes)
    done = batch.run(controllers, sampler, n_hands, lambda spec: ["random"] * spec.n_seats)
    assert done == n_hands, (done, n_hands)
    return recorder.drain()


def records(n_hands: int = 400, seed: int = 0, sizes: Iterable[int] = range(2, 11)) -> TrainingRecords:
    """Records of `n_hands` random-play hands (every seat recorded) of 2- to
    10-player tables with training stacks. Cached: do not modify the arrays.
    Their logp is NaN (RandomController returns none)."""
    return _records(int(n_hands), int(seed), tuple(int(s) for s in sizes))


def mask_rows(mask: np.ndarray) -> set:
    return {tuple(bool(x) for x in row) for row in mask}


def all_mask_batch(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each real observation row paired with each mask of ALL_MASKS:
    obs (4 * n, OBS_DIM) float32, mask (4 * n, 3) bool, both C-contiguous."""
    n = obs.shape[0]
    o = np.ascontiguousarray(np.repeat(obs, len(ALL_MASKS), axis=0), dtype=np.float32)
    m = np.ascontiguousarray(np.tile(ALL_MASKS, (n, 1)))
    return o, m


def rows_with_masks(recs: TrainingRecords, n: int, per_mask: int) -> np.ndarray:
    """Indices of `n` rows of `recs` containing at least `per_mask` rows of each
    of the common masks."""
    picked: list[int] = []
    for m in COMMON_MASKS:
        idx = np.flatnonzero((recs.mask == m).all(axis=1))[:per_mask]
        assert len(idx) == per_mask, f"only {len(idx)} rows with mask {m}"
        picked.extend(int(i) for i in idx)
    chosen = set(picked)
    for i in range(len(recs)):
        if len(picked) >= n:
            break
        if i not in chosen:
            picked.append(i)
            chosen.add(i)
    assert len(picked) == n
    return np.array(sorted(picked), dtype=np.int64)


def decision_batch(obs: Optional[np.ndarray], mask: np.ndarray, key: Hashable = "nn") -> DecisionBatch:
    """A DecisionBatch from given rows (tables / seats -1, no legal actions, no
    views); 0-row batches included."""
    mask = np.ascontiguousarray(mask, dtype=bool)
    n = mask.shape[0]
    if obs is not None:
        obs = np.ascontiguousarray(obs, dtype=np.float32)
    return DecisionBatch(
        key=key,
        tables=np.full(n, -1, dtype=np.int32),
        seats=np.full(n, -1, dtype=np.int32),
        obs=obs,
        mask=mask,
        legal=[None] * n,
        views=None,
    )


def small_policy_net(seed: int, hidden: int = 64, layers: int = 2) -> PolicyNet:
    """A small PolicyNet built from `seed` without moving torch's global RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return PolicyNet(hidden=hidden, layers=layers)


def peaked_policy_net(seed: int, hidden: int = 64, layers: int = 2) -> PolicyNet:
    """A small net whose head weights are scaled (x300), so that action
    probabilities on real observations are clearly uneven."""
    net = small_policy_net(seed, hidden, layers)
    with torch.no_grad():
        net.head.weight.mul_(300.0)
    return net


def small_weights(seed: int, peaked: bool = False) -> dict:
    """`state_dict()` of a `PolicyNet(hidden=64, layers=2)` built from `seed`
    (head scaled as in `peaked_policy_net` with `peaked=True`). Global torch RNG
    untouched."""
    net = peaked_policy_net(seed) if peaked else small_policy_net(seed)
    return net.state_dict()


def small_worker(seed: int, **kw: Any) -> RolloutWorker:
    """A RolloutWorker with small networks (64 x 2) and 32 tables (overridable)."""
    kw.setdefault("n_tables", 32)
    kw.setdefault("hidden", 64)
    kw.setdefault("layers", 2)
    return RolloutWorker(seed=seed, **kw)


def small_config(**kw: Any) -> PPOConfig:
    """A PPOConfig with small networks (64 x 2) and minibatches of 256
    (overridable)."""
    kw.setdefault("hidden", 64)
    kw.setdefault("layers", 2)
    kw.setdefault("minibatch", 256)
    return PPOConfig(**kw)


def small_learner(seed: int, num_threads: Optional[int] = None, **kw: Any) -> Learner:
    """A Learner with `small_config(**kw)`."""
    return Learner(small_config(**kw), seed=seed, num_threads=num_threads)


def acted_records(
    weights: Mapping[str, Any],
    obs: np.ndarray,
    mask: np.ndarray,
    critic: np.ndarray,
    reward_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
    seed: int,
) -> TrainingRecords:
    """TrainingRecords of given rows acted on by a `NeuralPolicy` holding
    `weights` (its sampled actions and logp); `reward = reward_fn(actions,
    rows)` as float32. Table / seat / controller 0, hand_id = row index. The
    policy leaves torch's thread count alone."""
    obs = np.ascontiguousarray(obs, dtype=np.float32)
    mask = np.ascontiguousarray(mask, dtype=bool)
    n = obs.shape[0]
    policy = NeuralPolicy.from_state_dict(weights, seed=seed, num_threads=None)
    actions, logp = policy.act(decision_batch(obs, mask))
    rows = np.arange(n, dtype=np.int64)
    reward = np.asarray(reward_fn(actions, rows), dtype=np.float32).reshape(n)
    return TrainingRecords(
        table=np.zeros(n, dtype=np.int32),
        hand_id=rows.copy(),
        seat=np.zeros(n, dtype=np.int32),
        controller=np.zeros(n, dtype=np.int32),
        obs=obs.copy(),
        mask=mask.copy(),
        action=np.asarray(actions, dtype=np.int64).copy(),
        logp=np.asarray(logp, dtype=np.float32).copy(),
        critic=np.ascontiguousarray(critic, dtype=np.int8).copy(),
        reward=reward,
    )


def replace_records(recs: TrainingRecords, **fields: Any) -> TrainingRecords:
    """A TrainingRecords with some fields replaced (others shared)."""
    kw = {name: getattr(recs, name) for name in recs.__dataclass_fields__}
    kw.update(fields)
    return TrainingRecords(**kw)


def net_from_weights(weights: Mapping[str, Any]) -> PolicyNet:
    """A small PolicyNet holding `weights` (global torch RNG untouched)."""
    extra = weights["_extra_state"]
    with torch.random.fork_rng(devices=[]):
        net = PolicyNet(hidden=extra["hidden"], layers=extra["layers"])
    net.load_state_dict(weights)
    net.eval()
    return net


def recorded_logp_under(net: PolicyNet, recs: TrainingRecords) -> np.ndarray:
    """log pi(recorded action | recorded obs, mask) under `net`, float32 (N,)."""
    with torch.no_grad():
        lp = torch.log_softmax(
            net(torch.from_numpy(recs.obs), torch.from_numpy(recs.mask)), dim=-1
        )
        out = lp.gather(1, torch.from_numpy(recs.action).unsqueeze(1)).squeeze(1)
    return out.numpy()


class ViewCounter:
    def __init__(self) -> None:
        self.n = 0


@contextlib.contextmanager
def count_views():
    """Patch rl_env's `Table.view` (the batch builds views only through it) with
    a counting wrapper; yields the counter."""
    counter = ViewCounter()
    original = Table.view

    def view(self):
        counter.n += 1
        return original(self)

    Table.view = view  # type: ignore[method-assign]
    try:
        yield counter
    finally:
        Table.view = original  # type: ignore[method-assign]


class _Logged:
    """Forwards to a controller and logs `(key, actions, logp)` of every call."""

    def __init__(self, key: Hashable, inner: Any, log: list) -> None:
        self.key = key
        self.inner = inner
        self.log = log
        self.wants_views = bool(getattr(inner, "wants_views", False))
        self.wants_obs = bool(getattr(inner, "wants_obs", True))

    def act(self, batch: DecisionBatch):
        out = self.inner.act(batch)
        if isinstance(out, tuple):
            actions, logp = out
            self.log.append((self.key, np.array(actions, copy=True), np.array(logp, copy=True)))
        else:
            self.log.append((self.key, np.array(out, copy=True), None))
        return out


def play(
    controllers: Mapping[Hashable, Any],
    seat_keys: Callable[[Any, np.random.Generator], list],
    n_hands: int,
    seed: int,
    n_tables: int = 64,
    sizes: Iterable[int] = range(2, 11),
    recorder: Optional[Recorder] = None,
) -> tuple[list, list]:
    """Keep every table of a TableBatch busy with `HandSampler(seed, sizes)` hands
    (controller keys from `seat_keys(spec, rng)`) until `n_hands` hands finished,
    via `TableBatch.step`. Returns (finished hands, [(key, actions, logp)] per
    controller call)."""
    log: list = []
    wrapped = {k: _Logged(k, c, log) for k, c in controllers.items()}
    tb = TableBatch(n_tables, recorder=recorder)
    sampler = HandSampler(seed, sizes)
    rng = np.random.default_rng(seed)
    started = 0
    finished: list = []
    for t in range(n_tables):
        if started >= n_hands:
            break
        spec = sampler.sample()
        tb.start_hand(t, spec, seat_keys(spec, rng))
        started += 1
    while len(finished) < n_hands:
        done = tb.step(wrapped)
        if not done and not tb.any_in_hand():
            break
        for fh in done:
            finished.append(fh)
            if started < n_hands and fh.table in tb.idle_tables():
                spec = sampler.sample()
                tb.start_hand(fh.table, spec, seat_keys(spec, rng))
                started += 1
    assert len(finished) == n_hands, (len(finished), n_hands)
    return finished, log
