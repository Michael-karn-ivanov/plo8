"""Determinism: the same seeds give identical decisions, actions, records and
results; the same deck seed deals the same button-relative cards whatever the
stacks."""

import unittest

import numpy as np

from rl_env import HandSampler, HandSpec, RandomController, Recorder, Table, TableBatch

try:
    from . import helpers
except ImportError:  # run as a top-level module
    import helpers  # type: ignore[no-redef]


def alternating(spec):
    return ["a" if s % 2 == 0 else "b" for s in range(spec.n_seats)]


def run(n_tables=12, hands=300):
    recorder = Recorder()
    batch = TableBatch(n_tables, recorder=recorder)
    sampler = HandSampler(21)
    ctrls = {"a": RandomController(1, with_logp=True), "b": RandomController(2)}
    log = []
    results = []
    started = 0
    for t in batch.idle_tables():
        spec = sampler.sample()
        batch.start_hand(t, spec, alternating(spec))
        started += 1
    while batch.any_in_hand() or started < hands:
        for g in batch.pending(ctrls):
            out = ctrls[g.key].act(g)
            actions, logp = out if isinstance(out, tuple) else (out, None)
            log.append(
                (
                    g.key,
                    g.tables.tolist(),
                    g.seats.tolist(),
                    g.obs.tobytes(),
                    g.mask.tobytes(),
                    np.asarray(actions).tolist(),
                )
            )
            batch.submit(g, actions, logp)
        for fh in batch.collect_finished():
            results.append(
                (fh.table, fh.hand_id, fh.spec, fh.controllers, fh.net_bb.tolist(), fh.result.history)
            )
            if started < hands:
                spec = sampler.sample()
                batch.start_hand(fh.table, spec, alternating(spec))
                started += 1
    return log, results, recorder.drain()


class DeterminismTest(unittest.TestCase):
    def test_identical_runs(self):
        log1, res1, rec1 = run()
        log2, res2, rec2 = run()
        self.assertGreater(len(log1), 100)
        self.assertEqual(len(res1), 300)
        self.assertEqual(log1, log2)
        self.assertEqual(res1, res2)
        for name in (
            "table",
            "hand_id",
            "seat",
            "controller",
            "obs",
            "mask",
            "action",
            "logp",
            "critic",
            "reward",
        ):
            a, b = getattr(rec1, name), getattr(rec2, name)
            self.assertEqual(a.dtype, b.dtype, name)
            self.assertEqual(a.tobytes(), b.tobytes(), name)

    def test_same_deck_seed_same_button_relative_cards(self):
        sampler = HandSampler(22)
        for _ in range(100):
            spec = sampler.sample()
            n = spec.n_seats
            other_stacks = tuple(5.0 + 3 * s for s in range(n))
            for button in range(n):
                other = HandSpec(n, other_stacks, button, spec.deck_seed)
                t1, t2 = Table(), Table()
                t1.start_hand(n, spec.stacks_bb, spec.button, spec.deck_seed)
                t2.start_hand(n, other.stacks_bb, other.button, other.deck_seed)
                c1, c2 = t1.privileged_hole_cards(), t2.privileged_hole_cards()
                for i in range(n):
                    self.assertEqual(
                        c1[helpers.seat_of(n, spec.button, i)], c2[helpers.seat_of(n, button, i)]
                    )


if __name__ == "__main__":
    unittest.main()
