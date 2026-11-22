"""Test data from rl_env: training records of real hands, every mask rl_env can
produce, and a sampling controller over a PolicyNet."""

from __future__ import annotations

import functools
from typing import Iterable

import numpy as np
import torch

from rl_env import (
    N_ACTIONS,
    HandSampler,
    RandomController,
    Recorder,
    TableBatch,
    TrainingRecords,
)

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
    10-player tables with training stacks. Cached: do not modify the arrays."""
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


class SamplingController:
    """Test-only rl_env Controller: samples from the masked PolicyNet (its own
    seeded torch.Generator) and checks every action against its row's mask
    before returning it."""

    wants_views = False

    def __init__(self, net, seed: int) -> None:
        self.net = net
        self.gen = torch.Generator().manual_seed(seed)
        self.decisions = 0
        self.counts = np.zeros(N_ACTIONS, dtype=np.int64)

    def act(self, batch):
        with torch.no_grad():
            logits = self.net(torch.from_numpy(batch.obs), torch.from_numpy(batch.mask))
            probs = torch.distributions.Categorical(logits=logits).probs
            actions = torch.multinomial(probs, 1, generator=self.gen).squeeze(1).numpy()
        actions = actions.astype(np.int64)
        for r, a in enumerate(actions):
            assert batch.mask[r, a], f"illegal action {a} for mask {batch.mask[r]}"
        self.counts += np.bincount(actions, minlength=N_ACTIONS)
        self.decisions += len(actions)
        return actions
