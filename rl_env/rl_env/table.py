"""Table: one engine `Environment` in step mode, one independent hand at a time.

Every hand is independent: stacks, table size and button are given anew and a
fresh `Environment` is built for it, so there is no rebuy, sit-out, join, leave
or button movement to manage. Players have no agent, so the engine builds no
views and queues no events for them.

The deck is `Deck(random.Random(deck_seed))` (the environment's default for
`seed=deck_seed`), shuffled once at `start_hand()`. By the engine's deal-order
contract (hole cards round-robin from the left of the button, then flop / turn
/ river, no burns) the same seed deals the same cards to the same
button-relative positions, whatever the stacks.

Amounts stay in BB at the engine boundary; rewards are in big bets (1 big bet
= 2 BB). Zero-sum is checked in chips.

Hidden information: `privileged_hole_cards()` is the only place in rl_env that
reads the engine's `env.privileged_hole_cards()`. Its two callers are the
critic input (`critic.critic_cards`, during a hand) and `finished_hole_cards()`
(after a hand, for evaluation tools). Nothing a controller receives is built
from either.
"""

from __future__ import annotations

import numbers
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from entities import Deck, Hand
from engine import ActionType, Decision, Environment, HandResult, LegalActions, Player, TableView

from .actions import action_mask, mask_array, to_engine
from .encoder import OBS_DIM, collect, encode
from .tracker import HandTracker

__all__ = ["Table", "NoHandError", "MIN_SEATS", "MAX_SEATS"]

MIN_SEATS = 2
MAX_SEATS = 10


class NoHandError(RuntimeError):
    """A decision read / action while no decision is pending at the table (or a
    finished-hand read while no hand is over)."""


def _is_number(x: Any) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, (bool, np.bool_))


class Table:
    """One engine environment in step mode (see the module docstring)."""

    def __init__(self, chips_per_bb: int = 100) -> None:
        if (
            isinstance(chips_per_bb, bool)
            or not isinstance(chips_per_bb, int)
            or chips_per_bb < 2
            or chips_per_bb % 2
        ):
            raise ValueError(f"chips_per_bb must be an even int >= 2, got {chips_per_bb!r}")
        self.chips_per_bb = chips_per_bb
        self.env: Optional[Environment] = None
        self.n_seats = 0
        self.button: Optional[int] = None
        self.result: Optional[HandResult] = None
        self.net_bb: Optional[np.ndarray] = None
        self.rewards: Optional[np.ndarray] = None
        self._tracker: Optional[HandTracker] = None
        self._seen = 0

    # --- starting a hand ----------------------------------------------------------

    def start_hand(
        self,
        n_seats: int,
        stacks_bb: Sequence[float],
        button: int,
        deck_seed: int,
        *,
        deck: Optional[Deck] = None,
    ) -> bool:
        """Start a new hand (any running hand is abandoned). Returns whether a
        decision is pending (False when the hand needed none: it is already over).

        The hand's tracker starts empty and is fed only when an observation (or
        `tracker`) is read, so a hand whose observations are never read never
        processes an event. Starting a hand ends the readability of the previous
        hand's `finished_hole_cards()`.

        `deck` (tests and replays only): any `entities.Deck`, e.g. a stacked deck,
        used instead of the seeded one."""
        if isinstance(n_seats, (bool, np.bool_)) or not isinstance(n_seats, numbers.Integral):
            raise ValueError(f"n_seats must be an int, got {n_seats!r}")
        n = int(n_seats)
        if not MIN_SEATS <= n <= MAX_SEATS:
            raise ValueError(f"n_seats must be {MIN_SEATS}-{MAX_SEATS}, got {n}")
        stacks = list(stacks_bb)
        if len(stacks) != n:
            raise ValueError(f"need {n} stacks, got {len(stacks)}")
        cpb = self.chips_per_bb
        clean: list[float] = []
        for i, s in enumerate(stacks):
            if not _is_number(s):
                raise ValueError(f"stack of seat {i} must be a number, got {s!r}")
            s = float(s)
            chips = s * cpb
            if not np.isfinite(chips) or s <= 0:
                raise ValueError(f"stack of seat {i} must be positive, got {s!r}")
            whole = round(chips)
            if whole < 1 or abs(chips - whole) > 1e-6 * max(1.0, abs(chips)):
                raise ValueError(
                    f"stack of seat {i} ({s!r} BB) is not a whole number of chips "
                    f"at {cpb} chips/BB"
                )
            clean.append(s)
        if isinstance(button, (bool, np.bool_)) or not isinstance(button, numbers.Integral):
            raise ValueError(f"button must be an int, got {button!r}")
        button = int(button)
        if not 0 <= button < n:
            raise ValueError(f"button must be a seat 0..{n - 1}, got {button}")
        if isinstance(deck_seed, (bool, np.bool_)) or not isinstance(deck_seed, numbers.Integral):
            raise ValueError(f"deck_seed must be an int, got {deck_seed!r}")
        deck_seed = int(deck_seed)
        if deck is not None and not isinstance(deck, Deck):
            raise ValueError(f"deck must be an entities.Deck, got {type(deck).__name__}")

        players = [Player(f"s{i}", clean[i]) for i in range(n)]
        env = Environment(
            players=players,
            button=button,
            chips_per_bb=cpb,
            seed=deck_seed,
            action_timeout=None,
            deck=deck,
        )
        self.env = env
        self.n_seats = n
        self.button = button
        self.result = None
        self.net_bb = None
        self.rewards = None
        self._tracker = HandTracker()
        self._seen = 0
        decision = env.start_hand()
        if decision is None:
            self._finish()
            return False
        return True

    # --- during a hand --------------------------------------------------------------

    @property
    def decision(self) -> Optional[Decision]:
        env = self.env
        return None if env is None else env.decision

    @property
    def decision_pending(self) -> bool:
        env = self.env
        return env is not None and env.decision is not None

    def _decision(self) -> Decision:
        env = self.env
        decision = None if env is None else env.decision
        if decision is None:
            raise NoHandError("no decision is pending at this table")
        return decision

    @property
    def seat_to_act(self) -> int:
        return self._decision().seat

    @property
    def legal(self) -> LegalActions:
        return self._decision().legal

    @property
    def mask(self) -> tuple[bool, bool, bool]:
        return action_mask(self._decision().legal)

    def mask_array(self) -> np.ndarray:
        return mask_array(self._decision().legal)

    def view(self) -> TableView:
        """The engine's view for the acting seat (built on request, cached by the
        engine for the decision)."""
        self._decision()
        return self.env.view()  # type: ignore[union-attr]

    def hole_cards(self) -> Hand:
        """The acting seat's own hole cards (no view is built)."""
        self._decision()
        return self.env.hole_cards()  # type: ignore[union-attr]

    @property
    def history(self):
        """The engine's public event history of the current hand (read-only)."""
        if self.env is None:
            return ()
        return self.env.history

    def privileged_hole_cards(self) -> Mapping[int, Hand]:
        """PRIVILEGED - hidden information: seat -> hole cards of every participant
        of the current hand (or of the last one once it is over).

        The only place in rl_env that reads the engine's privileged accessor. Its
        callers are the critic input (`critic_cards`) and `finished_hole_cards()`;
        it must never reach a controller."""
        if self.env is None:
            return {}
        return self.env.privileged_hole_cards()

    @property
    def tracker(self) -> HandTracker:
        """The hand's public-state tracker, brought up to date."""
        self._sync()
        return self._tracker  # type: ignore[return-value]

    @property
    def events_fed(self) -> int:
        """How many of the hand's events the tracker has processed so far (0 while
        no observation of the hand has been read)."""
        return self._seen

    def _sync(self) -> None:
        """Feed the tracker the events added since the previous read (cursor into
        the engine's history; no copy of the whole history)."""
        env = self.env
        tracker = self._tracker
        if env is None or tracker is None:
            raise NoHandError("no hand at this table")
        history = env.history
        end = len(history)
        seen = self._seen
        if end > seen:
            feed = tracker.feed
            for i in range(seen, end):
                feed(history[i])
            self._seen = end

    def observation(self, out: Optional[np.ndarray] = None) -> np.ndarray:
        """The acting seat's observation (float32, `OBS_DIM`), without a view."""
        decision = self._decision()
        self._sync()
        return encode(
            self._tracker,  # type: ignore[arg-type]
            decision.seat,
            self.env.hole_cards(),  # type: ignore[union-attr]
            decision.legal,
            out,
        )

    def collect_observation(self, cols: list, vals: list) -> None:
        """Append the acting seat's non-zero (column, value) pairs (batch path)."""
        decision = self._decision()
        self._sync()
        collect(
            self._tracker,  # type: ignore[arg-type]
            decision.seat,
            self.env.hole_cards(),  # type: ignore[union-attr]
            decision.legal,
            cols,
            vals,
        )

    def act(self, action: int) -> None:
        """Submit a masked action. Raises IllegalMaskedActionError for an action
        outside the mask (nothing changes)."""
        decision = self._decision()
        self._step(to_engine(action, decision.legal))

    def act_engine(self, action_type: ActionType) -> None:
        """Submit a raw engine action (e.g. a fold when checking is free). Only for
        replays of recorded hands; not part of the policy path."""
        self._decision()
        self._step(action_type)

    def _step(self, action_type: ActionType) -> None:
        if self.env.step(action_type) is None:  # type: ignore[union-attr]
            self._finish()

    # --- end of a hand ----------------------------------------------------------------

    @property
    def hand_over(self) -> bool:
        return self.result is not None

    def finished_hole_cards(self) -> Mapping[int, Hand]:
        """HIDDEN INFORMATION - evaluation tools only (e.g. arena's hand histories);
        must never reach a controller.

        seat -> `Hand` of every seat of the finished hand (all seats of an rl_env
        table are dealt in, folded ones included), as the engine's read-only
        mapping. Raises NoHandError unless the hand is over, so it reveals nothing
        while a decision is pending. Readable until this table's next
        `start_hand` (which builds a new environment)."""
        if not self.hand_over:
            raise NoHandError("no finished hand at this table")
        return self.privileged_hole_cards()

    def _finish(self) -> None:
        env = self.env
        assert env is not None
        result = env.last_result
        assert result is not None
        net = np.array([result.net[seat] for seat in range(self.n_seats)], dtype=np.float64)
        cpb = self.chips_per_bb
        chips = sum(int(round(x * cpb)) for x in net.tolist())
        if chips != 0:
            raise AssertionError(f"nets do not sum to 0 in chips: {chips} ({net.tolist()})")
        self.result = result
        self.net_bb = net
        self.rewards = (net / 2.0).astype(np.float32)


# Re-exported for callers that build observation buffers.
Table.OBS_DIM = OBS_DIM  # type: ignore[attr-defined]
