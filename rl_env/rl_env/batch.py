"""Many tables at once: decisions grouped by controller, actions submitted in
batches, finished hands reported with each seat's result.

Each seat of a hand is assigned a controller key by the caller when the hand
starts. `pending()` groups the decisions of all tables (at most one per table)
by the acting seat's controller key; a `Controller` answers a whole
`DecisionBatch` at once. Grouping and iteration orders are by table index
(never over sets or `id()`), so a run is deterministic.

`TableBatch(observations=False)` builds no observation vectors (and feeds no
tracker): for batches whose controllers all declare `wants_obs = False`
(evaluation matches with rule-based bots, the bench's `--no-obs`). Everything
else - masks, legal actions, grouping, order, views, actions and results - is
the same as with observations.

Nothing a controller receives (`DecisionBatch`) or `FinishedHand` carries any
hidden information, nor any reference to an engine object that could reach it.
The full deal of a finished hand is read separately with
`finished_hole_cards(table)`, for evaluation tools only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Hashable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

import numpy as np

from engine import HandResult, LegalActions, TableView
from entities import Hand

from .actions import IllegalMaskedActionError, action_mask, check_action, mask_array
from .encoder import OBS_DIM, encode_view
from .recorder import Recorder
from .stacks import HandSampler, HandSpec
from .table import NoHandError, Table

__all__ = ["Controller", "DecisionBatch", "FinishedHand", "TableBatch"]


class _BatchToken:
    """Opaque identity of a TableBatch inside its decision tokens."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "<TableBatch token>"


@dataclass(eq=False)
class DecisionBatch:
    """The pending decisions of one controller: one row per table. Holds only
    what the acting seats may see (no critic input, no other seat's cards, no
    engine object)."""

    key: Hashable
    tables: np.ndarray  # int32 (B,)
    seats: np.ndarray  # int32 (B,)
    obs: Optional[np.ndarray]  # float32 (B, OBS_DIM); None in a batch without observations
    mask: np.ndarray  # bool (B, 3)
    legal: list  # list[LegalActions]
    views: Optional[list] = None  # list[TableView], only for controllers that want them
    # Opaque per-row tokens of the decisions the rows answer (to reject a stale
    # batch); None for `single()` batches, which belong to no TableBatch.
    tokens: Optional[list] = field(default=None, repr=False)

    def __len__(self) -> int:
        return int(self.tables.shape[0])

    @classmethod
    def single(
        cls, view: TableView, legal: LegalActions, key: Hashable = None
    ) -> "DecisionBatch":
        """A one-row batch from one engine view and its legal actions (an engine
        `on_turn`, or a decoded server `turn` message). Table -1, seat `view.seat`.
        Always builds `obs`."""
        return cls(
            key=key,
            tables=np.array([-1], dtype=np.int32),
            seats=np.array([view.seat], dtype=np.int32),
            obs=encode_view(view, legal)[None, :],
            mask=mask_array(legal)[None, :],
            legal=[legal],
            views=[view],
        )


@runtime_checkable
class Controller(Protocol):
    """Decides for the seats assigned to it.

    `wants_views` (class or instance attribute): build `DecisionBatch.views`.
    `wants_obs` (optional; True when absent): reads `DecisionBatch.obs`. Only a
    controller declaring `wants_obs = False` may sit in a batch without
    observations.
    `act` returns one action per row (int array), or `(actions, logp)` with the
    log-probability of each chosen action (float32), which the Recorder keeps.
    """

    wants_views: bool

    def act(self, batch: DecisionBatch) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]: ...


@dataclass(frozen=True, eq=False)
class FinishedHand:
    """Public results of a finished hand (`result` shows hole cards only for the
    hands shown at showdown)."""

    table: int
    hand_id: int
    spec: HandSpec
    controllers: tuple  # controller key per seat
    net_bb: np.ndarray  # float64 (n_seats,)
    rewards: np.ndarray  # float32 (n_seats,), big bets
    result: HandResult


def _wants_views(controller: Any) -> bool:
    return bool(getattr(controller, "wants_views", False))


def _wants_obs(controller: Any) -> bool:
    return bool(getattr(controller, "wants_obs", True))


def _scalar(x: Any) -> Any:
    item = getattr(x, "item", None)
    return item() if callable(item) else x


class TableBatch:
    """`n_tables` tables played side by side (see the module docstring)."""

    def __init__(
        self,
        n_tables: int,
        *,
        chips_per_bb: int = 100,
        recorder: Optional[Recorder] = None,
        observations: bool = True,
    ) -> None:
        if isinstance(n_tables, bool) or not isinstance(n_tables, int) or n_tables < 1:
            raise ValueError(f"n_tables must be a positive int, got {n_tables!r}")
        observations = bool(observations)
        if recorder is not None and not observations:
            raise ValueError("a Recorder needs observations: TableBatch(observations=True)")
        self.tables = [Table(chips_per_bb) for _ in range(n_tables)]
        self.recorder = recorder
        # False: no observation is built (DecisionBatch.obs is None); only
        # controllers with `wants_obs = False` may act in such a batch.
        self.observations = observations
        self._hand_id: list[int] = [-1] * n_tables
        self._spec: list[Optional[HandSpec]] = [None] * n_tables
        self._controllers: list[tuple] = [()] * n_tables
        self._running: list[bool] = [False] * n_tables
        # Actions applied in the table's current hand (part of the decision token).
        self._acted: list[int] = [0] * n_tables
        self._token = _BatchToken()
        self._finished: list[FinishedHand] = []
        self._next_hand_id = 0
        self.n_decisions = 0  # decisions submitted so far
        self.n_hands = 0  # hands finished so far

    def __len__(self) -> int:
        return len(self.tables)

    # --- hands ---------------------------------------------------------------------

    def start_hand(self, table: int, spec: HandSpec, controllers: Sequence[Hashable]) -> int:
        """Start a hand on an idle table with one controller key per seat. Returns
        the hand's batch-unique, increasing id. A hand that needs no decision is
        finished at once (reported by the next `collect_finished()`). Ends the
        readability of the table's previous `finished_hole_cards`."""
        if self._running[table]:
            raise RuntimeError(f"table {table} still has a hand running")
        keys = tuple(controllers)
        if len(keys) != spec.n_seats:
            raise ValueError(f"need {spec.n_seats} controller keys, got {len(keys)}")
        tbl = self.tables[table]
        pending = tbl.start_hand(spec.n_seats, spec.stacks_bb, spec.button, spec.deck_seed)
        hand_id = self._next_hand_id
        self._next_hand_id += 1
        self._hand_id[table] = hand_id
        self._spec[table] = spec
        self._controllers[table] = keys
        self._running[table] = True
        self._acted[table] = 0
        if not pending:
            self._finish(table)
        return hand_id

    def hand_id(self, table: int) -> int:
        """Id of the table's current (or last) hand, -1 before the first."""
        return self._hand_id[table]

    def controllers_of(self, table: int) -> tuple:
        """Controller key per seat of the table's current (or last) hand."""
        return self._controllers[table]

    def idle_tables(self) -> list[int]:
        """Tables without a running hand (waiting for the caller's next hand)."""
        return [t for t, running in enumerate(self._running) if not running]

    def any_in_hand(self) -> bool:
        return any(self._running)

    def _finish(self, table: int) -> None:
        tbl = self.tables[table]
        self._running[table] = False
        hand_id = self._hand_id[table]
        fh = FinishedHand(
            table=table,
            hand_id=hand_id,
            spec=self._spec[table],  # type: ignore[arg-type]
            controllers=self._controllers[table],
            net_bb=tbl.net_bb,  # type: ignore[arg-type]
            rewards=tbl.rewards,  # type: ignore[arg-type]
            result=tbl.result,  # type: ignore[arg-type]
        )
        self._finished.append(fh)
        self.n_hands += 1
        if self.recorder is not None:
            self.recorder.finish(hand_id, tbl.rewards)  # type: ignore[arg-type]

    def collect_finished(self) -> list[FinishedHand]:
        """Hands finished since the last call, in finishing order."""
        out = self._finished
        self._finished = []
        return out

    def finished_hole_cards(self, table: int) -> Mapping[int, Hand]:
        """HIDDEN INFORMATION - evaluation tools only (arena's hand histories,
        deal-order checks); must never reach a controller.

        seat -> `Hand` of every seat of the table's last hand once it is over
        (whether or not `collect_finished()` has returned it yet), until the
        caller starts that table's next hand: read it for a `FinishedHand` before
        restarting `fh.table`. Raises RuntimeError while the table has a hand
        running and NoHandError if it never had one."""
        if self._running[table]:
            raise RuntimeError(f"table {table} still has a hand running")
        if self._hand_id[table] < 0:
            raise NoHandError(f"table {table} has not had a hand yet")
        return self.tables[table].finished_hole_cards()

    # --- decisions -----------------------------------------------------------------

    def _decision_token(self, table: int) -> tuple:
        return (self._token, self._hand_id[table], self._acted[table])

    def pending(
        self, controllers: Optional[Mapping[Hashable, Any]] = None
    ) -> list[DecisionBatch]:
        """Every table with a decision, grouped by the acting seat's controller key.
        Groups are ordered by their smallest table index, rows by table index.
        Views are built only for groups whose controller (looked up in
        `controllers`) has `wants_views`.

        In a batch without observations `obs` is None; if `controllers` is given
        and a pending group's controller has `wants_obs` true (or absent), it
        raises ValueError before anything is built."""
        groups: dict[Hashable, list[int]] = {}
        tables = self.tables
        running = self._running
        for t, tbl in enumerate(tables):
            if running[t]:
                decision = tbl.decision
                if decision is not None:
                    key = self._controllers[t][decision.seat]
                    group = groups.get(key)
                    if group is None:
                        groups[key] = [t]
                    else:
                        group.append(t)
        build_obs = self.observations
        if not build_obs and controllers is not None:
            for key in groups:
                controller = controllers.get(key)
                if controller is not None and _wants_obs(controller):
                    raise ValueError(
                        f"controller {key!r} wants observations (wants_obs is true or "
                        f"absent) but this TableBatch builds none (observations=False)"
                    )
        out = []
        for key, ts in groups.items():
            b = len(ts)
            seats = []
            legal = []
            masks = []
            tokens = []
            obs = None
            if build_obs:
                obs = np.zeros((b, OBS_DIM), dtype=np.float32)
                rows: list[int] = []
                cols: list[int] = []
                vals: list[float] = []
            for r, t in enumerate(ts):
                tbl = tables[t]
                decision = tbl.decision
                tokens.append(self._decision_token(t))
                seats.append(decision.seat)  # type: ignore[union-attr]
                lg = decision.legal  # type: ignore[union-attr]
                legal.append(lg)
                masks.append(action_mask(lg))
                if build_obs:
                    before = len(cols)
                    tbl.collect_observation(cols, vals)
                    rows.extend([r] * (len(cols) - before))
            if build_obs and cols:
                obs[rows, cols] = vals  # type: ignore[index]
            views = None
            if controllers is not None and _wants_views(controllers.get(key)):
                views = [tables[t].view() for t in ts]
            out.append(
                DecisionBatch(
                    key=key,
                    tables=np.array(ts, dtype=np.int32),
                    seats=np.array(seats, dtype=np.int32),
                    obs=obs,
                    mask=np.array(masks, dtype=bool).reshape(b, 3),
                    legal=legal,
                    views=views,
                    tokens=tokens,
                )
            )
        return out

    def submit(self, batch: DecisionBatch, actions: Any, logp: Any = None) -> None:
        """Apply one action per row. Every action is validated against its mask
        (and the batch against the tables) first: if any is invalid, it raises and
        no table is touched. Then each table acts; decisions are recorded (with
        `logp` if given) and finished hands move to the finished list."""
        b = len(batch)
        acts = np.asarray(actions)
        if acts.shape != (b,):
            raise IllegalMaskedActionError(f"need {b} actions, got shape {acts.shape}")
        lp = None
        if logp is not None:
            lp = np.asarray(logp, dtype=np.float32)
            if lp.shape != (b,):
                raise ValueError(f"need {b} log-probabilities, got shape {lp.shape}")
        checked = [check_action(_scalar(acts[r]), batch.mask[r]) for r in range(b)]
        tokens = batch.tokens
        for r in range(b):
            t = int(batch.tables[r])
            if not 0 <= t < len(self.tables):
                raise ValueError(f"row {r}: no table {t} in this batch")
            tbl = self.tables[t]
            decision = tbl.decision
            if (
                not self._running[t]
                or decision is None
                or decision.seat != int(batch.seats[r])
                or (tokens is not None and tokens[r] != self._decision_token(t))
            ):
                raise ValueError(f"row {r}: table {t} has no such pending decision (stale batch)")
        recorder = self.recorder
        record = recorder is not None and recorder.wants(batch.key)
        if record and batch.obs is None:
            raise ValueError("cannot record a batch without observations")
        for r in range(b):
            t = int(batch.tables[r])
            tbl = self.tables[t]
            a = checked[r]
            if record:
                recorder.record(  # type: ignore[union-attr]
                    tbl,
                    t,
                    self._hand_id[t],
                    int(batch.seats[r]),
                    batch.key,
                    batch.obs[r],  # type: ignore[index]
                    batch.mask[r],
                    a,
                    None if lp is None else float(lp[r]),
                )
            tbl.act(a)
            self._acted[t] += 1
            self.n_decisions += 1
            if tbl.hand_over:
                self._finish(t)

    def step(self, controllers: Mapping[Hashable, Any]) -> list[FinishedHand]:
        """`pending`, each controller's `act` in group order, `submit`, then
        `collect_finished()`. In a batch without observations a controller that
        wants them makes it raise ValueError before any table changes."""
        for batch in self.pending(controllers):
            out = controllers[batch.key].act(batch)
            if isinstance(out, tuple):
                actions, logp = out
            else:
                actions, logp = out, None
            self.submit(batch, actions, logp)
        return self.collect_finished()

    def run(
        self,
        controllers: Mapping[Hashable, Any],
        sampler: HandSampler,
        n_hands: int,
        seat_controllers: Callable[[HandSpec], Sequence[Hashable]],
        on_finished: Optional[Callable[[FinishedHand], None]] = None,
    ) -> int:
        """Keep every table busy with `sampler.sample()` hands (controller keys from
        `seat_controllers(spec)`) until `n_hands` hands started here have finished.
        Calls `on_finished` for every finished hand (before its table is
        restarted); returns how many finished."""
        started = 0
        done = 0
        for t in self.idle_tables():
            if started >= n_hands:
                break
            spec = sampler.sample()
            self.start_hand(t, spec, seat_controllers(spec))
            started += 1
        while done < n_hands:
            finished = self.step(controllers)
            if not finished and not self.any_in_hand():
                break
            for fh in finished:
                done += 1
                if on_finished is not None:
                    on_finished(fh)
                if started < n_hands and not self._running[fh.table]:
                    spec = sampler.sample()
                    self.start_hand(fh.table, spec, seat_controllers(spec))
                    started += 1
        return done
