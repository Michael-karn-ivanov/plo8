"""Fuzz test: random hands through a TableBatch at random table sizes (2-10)
with `StackConfig()` stacks and random controllers.

100,000 hands by default (the CI count); set RL_ENV_FUZZ_HANDS to run fewer
locally, e.g. `RL_ENV_FUZZ_HANDS=2000 buck2 test //rl_env:rl_env_fuzz_test`.

For every hand: no exception, the seats' nets sum to exactly zero in chips;
for every decision: the observation is finite and in [-1, 2] and the mask
allows check / call. A failure prints the chunk seed, so it can be replayed
with `play_chunk(seed, hands)`.
"""

import os
import sys
import unittest

import numpy as np

from rl_env import (
    CHECK_CALL,
    OBS_DIM,
    HandSampler,
    RandomController,
    Recorder,
    StackConfig,
    TableBatch,
)

DEFAULT_HANDS = 100_000
HANDS_PER_CHUNK = 5_000
TABLES = 64


def fuzz_hands() -> int:
    value = os.environ.get("RL_ENV_FUZZ_HANDS")
    if not value:
        return DEFAULT_HANDS
    hands = int(value)
    if hands < 1:
        raise ValueError(f"RL_ENV_FUZZ_HANDS must be a positive int, got {value!r}")
    return hands


def play_chunk(seed: int, hands: int) -> dict:
    """Play `hands` random hands through a batch built from `seed`. Raises
    AssertionError on any violated invariant."""
    recorder = Recorder() if seed % 4 == 0 else None  # some chunks also record
    batch = TableBatch(TABLES, recorder=recorder)
    sampler = HandSampler(seed, cfg=StackConfig())
    ctrls = {k: RandomController(seed * 10 + k) for k in range(3)}

    def seat_controllers(spec):
        return [(spec.deck_seed + s) % 3 for s in range(spec.n_seats)]

    stats = {"hands": 0, "decisions": 0}
    started = 0
    for t in batch.idle_tables():
        if started >= hands:
            break
        spec = sampler.sample()
        batch.start_hand(t, spec, seat_controllers(spec))
        started += 1
    while stats["hands"] < hands:
        groups = batch.pending(ctrls)
        for g in groups:
            obs = g.obs
            assert obs.shape == (len(g), OBS_DIM) and obs.dtype == np.float32
            assert np.isfinite(obs).all(), f"seed {seed}: non-finite observation"
            assert obs.min() >= -1.0 and obs.max() <= 2.0, f"seed {seed}: observation out of range"
            assert g.mask[:, CHECK_CALL].all(), f"seed {seed}: check / call masked"
            batch.submit(g, ctrls[g.key].act(g))
            stats["decisions"] += len(g)
        finished = batch.collect_finished()
        assert groups or finished, f"seed {seed}: no progress"
        for fh in finished:
            stats["hands"] += 1
            chips = [int(round(x * 100)) for x in fh.net_bb.tolist()]
            assert sum(chips) == 0, f"seed {seed}: nets {chips} do not sum to 0"
            assert all(s >= 0 for s in fh.result.stacks.values()), f"seed {seed}: negative stack"
            if started < hands:
                spec = sampler.sample()
                batch.start_hand(fh.table, spec, seat_controllers(spec))
                started += 1
        if recorder is not None and recorder.n_ready > 20_000:
            rec = recorder.drain()
            assert np.isfinite(rec.reward).all()
    assert stats["hands"] == hands, f"seed {seed}: {stats['hands']} of {hands} hands finished"
    assert not batch.any_in_hand(), f"seed {seed}: hands still running"
    if recorder is not None:
        assert recorder.n_pending == 0, f"seed {seed}: pending records after the last hand"
    return stats


class FuzzBatchTest(unittest.TestCase):
    def test_random_hands(self):
        total = fuzz_hands()
        played = 0
        decisions = 0
        seed = 0
        while played < total:
            hands = min(HANDS_PER_CHUNK, total - played)
            try:
                stats = play_chunk(seed, hands)
            except BaseException:
                print(f"rl_env fuzz: failing chunk seed {seed} ({hands} hands)", file=sys.stderr)
                raise
            played += stats["hands"]
            decisions += stats["decisions"]
            seed += 1
        self.assertEqual(played, total)
        self.assertGreater(decisions, played)


if __name__ == "__main__":
    unittest.main()
