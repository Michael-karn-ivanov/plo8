"""Fuzz test: random step-mode hands at random table sizes and stacks.

100,000 hands by default (the CI count); set ENGINE_FUZZ_HANDS to run fewer
(or more) locally, e.g. `ENGINE_FUZZ_HANDS=2000 buck2 test //engine:engine_fuzz_test`.

For every hand: no exception, the players' net results sum to exactly zero in
chips (there is no rake), every final stack is >= 0, and the chips at the table
before the hand equal the chips after it. A failure names the table seed, so it
can be replayed with `play_table(seed, hands)`.
"""

import os
import random
import sys
import unittest

from engine import Deck, Environment, Player, PlayerStatus

DEFAULT_HANDS = 100_000
HANDS_PER_TABLE = 200


def fuzz_hands() -> int:
    value = os.environ.get("ENGINE_FUZZ_HANDS")
    if not value:
        return DEFAULT_HANDS
    hands = int(value)
    if hands < 1:
        raise ValueError(f"ENGINE_FUZZ_HANDS must be a positive int, got {value!r}")
    return hands


def play_table(seed: int, hands: int) -> dict:
    """Play `hands` random step-mode hands at a table built from `seed`.
    Raises AssertionError on any violated invariant."""
    rng = random.Random(seed)
    n = rng.randint(2, 10)
    players = [Player(f"p{i}", round(rng.uniform(0.5, 200.0), 2)) for i in range(n)]
    env = Environment(players, button=rng.randrange(n), deck=Deck(random.Random(seed)))
    cpb = env.chips_per_bb
    views = rng.random() < 0.2  # some tables also request a view at every decision
    stats = {"hands": 0, "decisions": 0, "showdowns": 0}
    for _ in range(hands):
        # Between hands: top up / rebuy (busted players come back this way).
        for p in env.players:
            if p.chips == 0 or rng.random() < 0.05:
                p.rebuy(round(rng.uniform(0.5, 200.0), 2))
        assert env.can_start(), f"seed {seed}: table cannot start"
        before = sum(p.chips for p in env.players)
        d = env.start_hand()
        while d is not None:
            assert d.legal.actions, f"seed {seed}: decision without legal actions"
            if views:
                view = env.view()
                assert view.seat == d.seat and view.hole_cards == d.player.hole_cards
            stats["decisions"] += 1
            d = env.step(rng.choice(d.legal.actions))
        result = env.last_result
        assert not env.hand_in_progress, f"seed {seed}: hand still in progress"
        net = sum(round(v * cpb) for v in result.net.values())
        assert net == 0, f"seed {seed}, hand {result.hand_number}: net sums to {net} chips"
        assert all(p.chips >= 0 for p in env.players), f"seed {seed}: negative stack"
        assert all(s >= 0 for s in result.stacks.values()), f"seed {seed}: negative stack"
        after = sum(p.chips for p in env.players)
        assert before == after, f"seed {seed}, hand {result.hand_number}: {before} != {after} chips"
        assert all(
            p.status in (PlayerStatus.ACTIVE, PlayerStatus.SITTING_OUT) for p in env.players
        )
        stats["hands"] += 1
        stats["showdowns"] += result.showdown
    return stats


class FuzzStepTest(unittest.TestCase):
    def test_random_step_mode_hands(self):
        total = fuzz_hands()
        master = random.Random(20240601)
        done = decisions = showdowns = 0
        while done < total:
            seed = master.randrange(1 << 30)
            hands = min(HANDS_PER_TABLE, total - done)
            try:
                stats = play_table(seed, hands)
            except Exception as e:  # noqa: BLE001 - report the seed for replay
                print(f"fuzz failure: play_table({seed}, {hands})", file=sys.stderr)
                raise AssertionError(f"fuzz failure, replay with play_table({seed}, {hands})") from e
            done += stats["hands"]
            decisions += stats["decisions"]
            showdowns += stats["showdowns"]
        self.assertEqual(done, total)
        self.assertGreater(decisions, 0)
        self.assertGreater(showdowns, 0)


if __name__ == "__main__":
    unittest.main()
