"""Training records: every drained record belongs to exactly one finished hand
and carries that seat's reward; unfinished hands are not drained; key filter;
logp; pending rows released."""

import unittest

import numpy as np

from engine import PlayerActed
from rl_env import (
    OBS_DIM,
    HandSampler,
    RandomController,
    Recorder,
    TableBatch,
    critic_cards,
)


def alternating(spec):
    return ["a" if s % 2 == 0 else "b" for s in range(spec.n_seats)]


def controllers():
    # "a" returns log-probabilities, "b" actions only.
    return {"a": RandomController(1, with_logp=True), "b": RandomController(2)}


def decisions_by_seat(result):
    out = {}
    for ev in result.history:
        if isinstance(ev, PlayerActed):
            out[ev.seat] = out.get(ev.seat, 0) + 1
    return out


class RecorderTest(unittest.TestCase):
    def test_records_match_finished_hands(self):
        recorder = Recorder()
        batch = TableBatch(16, recorder=recorder)
        finished = {}

        def keep(fh):
            self.assertNotIn(fh.hand_id, finished)
            finished[fh.hand_id] = fh

        done = batch.run(controllers(), HandSampler(3), 400, alternating, keep)
        self.assertEqual(done, 400)
        self.assertEqual(recorder.n_pending, 0)
        self.assertEqual(recorder.n_ready, batch.n_decisions)
        rec = recorder.drain()
        n = len(rec)
        self.assertEqual(n, batch.n_decisions)
        self.assertEqual(recorder.n_ready, 0)
        self.assertEqual(rec.obs.shape, (n, OBS_DIM))
        self.assertEqual(rec.obs.dtype, np.float32)
        self.assertEqual(rec.mask.shape, (n, 3))
        self.assertEqual(rec.critic.shape, (n, 9, 4))
        self.assertEqual(rec.critic.dtype, np.int8)
        counts = {}
        for i in range(n):
            hid = int(rec.hand_id[i])
            self.assertIn(hid, finished)
            fh = finished[hid]
            seat = int(rec.seat[i])
            self.assertEqual(int(rec.table[i]), fh.table)
            self.assertEqual(rec.reward[i], fh.rewards[seat])
            self.assertTrue(rec.mask[i, int(rec.action[i])])
            key = fh.controllers[seat]
            self.assertEqual(int(rec.controller[i]), recorder.codes[key])
            if key == "a":
                allowed = int(rec.mask[i].sum())
                self.assertAlmostEqual(float(rec.logp[i]), -np.log(allowed), places=5)
            else:
                self.assertTrue(np.isnan(rec.logp[i]))
            counts[(hid, seat)] = counts.get((hid, seat), 0) + 1
        # Exactly the decisions of each hand, per seat.
        for hid, fh in finished.items():
            for seat, k in decisions_by_seat(fh.result).items():
                self.assertEqual(counts.get((hid, seat), 0), k, (hid, seat))
        self.assertEqual(sum(counts.values()), n)

    def test_unfinished_hands_are_not_drained(self):
        recorder = Recorder()
        batch = TableBatch(8, recorder=recorder)
        sampler = HandSampler(4)
        ctrls = controllers()
        for t in range(8):
            spec = sampler.sample()
            batch.start_hand(t, spec, alternating(spec))
        finished = {}
        drained = []
        for _ in range(2):
            for fh in batch.step(ctrls):
                finished[fh.hand_id] = fh
            rec = recorder.drain()
            drained.append(rec)
            for hid in rec.hand_id.tolist():
                self.assertIn(hid, finished)
        self.assertTrue(batch.any_in_hand())
        self.assertGreater(recorder.n_pending, 0)
        running = {batch.hand_id(t) for t in range(8) if t not in batch.idle_tables()}
        for rec in drained:
            self.assertFalse(running & set(rec.hand_id.tolist()))
        # Finish everything: pending rows are released.
        while batch.any_in_hand():
            for fh in batch.step(ctrls):
                finished[fh.hand_id] = fh
        self.assertEqual(recorder.n_pending, 0)
        rest = recorder.drain()
        total = sum(len(r) for r in drained) + len(rest)
        self.assertEqual(total, batch.n_decisions)
        self.assertGreater(len(rest), 0)
        self.assertTrue(running <= set(rest.hand_id.tolist()))

    def test_key_filter(self):
        recorder = Recorder(keys=["a"])
        batch = TableBatch(8, recorder=recorder)
        finished = []
        batch.run(controllers(), HandSampler(5), 100, alternating, finished.append)
        rec = recorder.drain()
        self.assertGreater(len(rec), 0)
        self.assertEqual(set(rec.controller.tolist()), {recorder.codes["a"]})
        self.assertNotIn("b", recorder.codes)
        want = 0
        for fh in finished:
            for seat, k in decisions_by_seat(fh.result).items():
                if fh.controllers[seat] == "a":
                    want += k
        self.assertEqual(len(rec), want)
        self.assertFalse(np.isnan(rec.logp).any())

    def test_critic_taken_before_the_action(self):
        recorder = Recorder()
        batch = TableBatch(4, recorder=recorder)
        sampler = HandSampler(6)
        for t in range(4):
            spec = sampler.sample()
            batch.start_hand(t, spec, alternating(spec))
        want = []
        for g in batch.pending():
            for r, t in enumerate(g.tables.tolist()):
                want.append(critic_cards(batch.tables[t], int(g.seats[r])))
            batch.submit(g, np.full(len(g), 1))
        while batch.any_in_hand():
            batch.step(controllers())
        rec = recorder.drain()
        self.assertGreaterEqual(len(rec), len(want))
        for i, w in enumerate(want):
            np.testing.assert_array_equal(rec.critic[i], w)

    def test_empty_drain(self):
        rec = Recorder().drain()
        self.assertEqual(len(rec), 0)
        self.assertEqual(rec.obs.shape, (0, OBS_DIM))
        self.assertEqual(rec.critic.shape, (0, 9, 4))


if __name__ == "__main__":
    unittest.main()
