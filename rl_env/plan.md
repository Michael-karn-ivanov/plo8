# rl_env — implementation plan

Python library (`rl_env`): the reinforcement-learning environment for fixed-limit
Omaha Hi-Lo 8-or-better. It drives `//engine` in step mode and gives the learning
side what it needs: tables, decisions, observation vectors, action masks, results and
training records. Projects above it (arena, trainer, bot_client) never touch the
engine directly.

Guiding rules:
- **No rules here.** Betting, pots, showdown and hand evaluation stay in the engine.
  rl_env only translates: the engine's `LegalActions` into a 3-action mask, a masked
  action into an engine `ActionType`, the engine's public events into an observation,
  the engine's `HandResult` into rewards. Hand-strength hints call the engine's
  evaluator.
- **One observation function, one input.** The observation is a function of the
  acting player's public information only: the public events of the hand so far, the
  player's own hole cards and its `LegalActions`. All of it is in the engine's
  `TableView` (`history`, `hole_cards`) plus the legal actions. The same code computes
  it at a real table (from a view received over the cli_server protocol) and in
  training (from the same events, read without building a view). See Observation.
- **Hidden information leaves through two separate doors, never through a
  controller.** During a hand only the critic input (`critic.py`, a separate function
  and a separate array that the observation code never imports). After a hand only
  `finished_hole_cards` (every seat's cards of a finished hand, for evaluation tools
  such as arena's hand histories), readable until the table's next hand starts. Both
  go through the one privileged read of `table.py` (`Table.privileged_hole_cards`);
  nothing in `pending` / `step` / `submit` / `DecisionBatch` / `FinishedHand` touches
  it, except the `Recorder` taking the critic input.

## Alignment with `uber-plan.md` (Phase 1B)

The uber-plan's 1B code is a sketch written before step mode existed; this plan follows
the engine as built (`Environment.start_hand()` / `step()` / `decision` / `view()`,
not `begin_hand()` / `submit()` / `turn.view()`) and the repo's `bench/` convention
(as `//engine:bench`). Names the later phases (arena, trainer, bot_client) should use:

| uber-plan sketch | here |
|---|---|
| `rl_env/adapter.py`: `Act`, `legal_mask`, `to_engine`, `EngineTable` | `actions.py`: `FOLD` / `CHECK_CALL` / `BET_RAISE`, `action_mask` / `mask_array`, `to_engine`; `table.py`: `Table` |
| `rl_env/encoding.py`: `encode_obs(view, legal)`, `OBS_DIM`, `ENCODER_VERSION` | `encoder.py`: `encode_view(view, legal)`, `OBS_DIM`, `ENCODER_VERSION` (training uses `Table.observation()`, byte-identical) |
| `encode_priv(table, seat)`, `PRIV_DIM`, `expand_cards` on the learner | `critic.py`: `critic_cards(table, seat)` (9×4 card indices), `PRIV_DIM`, `expand_critic` (numpy reference for the learner's expansion) |
| `rl_env/tables.py`: `TableBatch.start_hand / pending / view / apply / finished_hands / other_cards`, `idle()`, `any_in_hand()` | `batch.py`: `TableBatch(observations=...)`, `start_hand / pending / submit / step / collect_finished / idle_tables / any_in_hand / finished_hole_cards`; critic input and rewards attached by `Recorder` |
| `rl_env/controller.py`: `act(obs, mask, views, legals) -> actions` (Phase 3 returns `(actions, logp)`) | `batch.py`: `Controller.act(DecisionBatch) -> actions` or `(actions, logp)`; `wants_views`, `wants_obs` |
| trainer's `open_steps` / `StepBuffer` matching decisions to results | `recorder.py`: `Recorder` / `TrainingRecords` (incl. `logp`) |
| `rl_env/tools/bench.py` | `bench/bench.py` (`//rl_env:bench`) |

Deliberate differences from the sketch:
- **No view per decision in training.** The sketch's rollout builds a `TableView` for
  every decision and encodes it; here the observation comes from an incremental
  tracker over the engine's events (views only for controllers that ask), checked
  byte-for-byte against `encode_view`. Views were a third of the engine's baseline time.
- **Wider observation** (554 instead of 478): pot odds, cap/raises-left flags, own
  action history, button flag, street of each recent action, finer low hints.
- **Critic input** uses `NO_CARD = 52` (a valid embedding row) instead of −1.
- **Baseline bots stay in arena.** rl_env ships only `RandomController` (needed by its
  own bench, fuzz and tests, which cannot depend on arena); arena's `RandomBot` may
  wrap it. The rule-based controller that proves the view path lives in the tests.

## Layout

```
rl_env/
  spec.md
  plan.md
  BUCK
  rl_env/
    __init__.py       # public exports
    actions.py        # FOLD / CHECK_CALL / BET_RAISE, mask, action -> engine ActionType
    table.py          # Table: one engine Environment in step mode, one hand at a time
    stacks.py         # StackConfig, draw_stacks, HandSpec, HandSampler
    tracker.py        # HandTracker: public state of a hand folded from engine events
    hints.py          # hand-strength hints (engine evaluator), cached per street
    encoder.py        # OBS_DIM, ENCODER_VERSION, OBS_LAYOUT, encode(), encode_view()
    critic.py         # critic input: other seats' hole cards, relative slots
    batch.py          # TableBatch, DecisionBatch, FinishedHand, Controller protocol
    controllers.py    # RandomController (bench, fuzz, tests; arena owns the baseline bots)
    recorder.py       # Recorder, TrainingRecords
  bench/
    bench.py          # rollout benchmark (buck2 binary)
  tests/
    helpers.py        # StackedDeck (entities put_on_top, engine deal-order contract), random play,
                      # CheckCallController (tiny view-reading controller for the view path)
    test_actions.py
    test_table.py
    test_stacks.py
    test_tracker.py
    test_encoder.py
    test_leakage.py
    test_rotation.py
    test_hints.py
    test_server_path.py
    test_recorded_hands.py
    test_critic.py
    test_batch.py
    test_no_obs.py
    test_finished_cards.py
    test_recorder.py
    test_determinism.py
  fuzz/
    test_fuzz_batch.py   # 100,000 random hands through the batch (own test target)
```

## Build (buck2)

`rl_env/BUCK`:
- `python_library(name = "rl_env", srcs = glob(["rl_env/**/*.py"]), base_module = "",
  deps = ["//engine:engine", "//entities:entities"], visibility = ["PUBLIC"])`.
  numpy is not a buck2 target: the repo's Python toolchain (`toolchains/BUCK`) runs the
  repo venv interpreter (`.venv`, `requirements-rl.txt`, `numpy==2.5.5`), so
  `import numpy` works in every target. No other third-party import is allowed in
  `rl_env/` (checked by a test that scans the imports of the package).
- `python_test(name = "rl_env_test", srcs = glob(["tests/*.py"]), base_module =
  "rl_env_tests", deps = [":rl_env", "//cli_server:protocol", "//cli_server:e2e"])`.
  Only the tests depend on cli_server (protocol round trip and recorded hands); the
  library does not. `unittest`, like the other projects.
- `python_test(name = "rl_env_fuzz_test", srcs = glob(["fuzz/*.py"]), base_module =
  "rl_env_fuzz", deps = [":rl_env"])` — kept apart because it runs for minutes;
  `RL_ENV_FUZZ_HANDS` lowers the hand count for local runs (default and CI: 100,000).
- `python_binary(name = "bench", main = "bench/bench.py", deps = [":rl_env"])` — see
  Speed (the engine-only number comes from `engine.benchmark.run_benchmark`, part of
  `//engine:engine`).

Run: `buck2 test //rl_env/...`, `buck2 run //rl_env:bench -- [options]`.

## Units

- The engine reports amounts in big blinds (floats over integer chips). rl_env keeps
  BB at the engine boundary and converts to **big bets** (1 big bet = 2 BB) only in
  the observation and in rewards.
- A `Table` uses `chips_per_bb = 100` by default (stacks have 0.01 BB resolution);
  replays of recorded hands set it to the recording's big blind, as cli_server does.
- Zero-sum checks are done in chips (`round(net_bb * chips_per_bb)`), never on floats.

## 1. Actions (`actions.py`)

```python
FOLD, CHECK_CALL, BET_RAISE = 0, 1, 2
N_ACTIONS = 3
```
- `action_mask(legal: LegalActions) -> tuple[bool, bool, bool]`:
  - `FOLD`: `not legal.can_check` (only when facing a bet; folding when checking is
    free is masked, although the engine accepts it);
  - `CHECK_CALL`: always `True` (the engine always offers check or call to a player
    who has a decision: `can_check` or `call_amount > 0`; asserted, a violation is an
    engine bug and raises);
  - `BET_RAISE`: `legal.can_raise` (the engine already excludes cap reached, cannot
    put in more than a call, and nobody else able to act).
- `to_engine(action: int, legal) -> ActionType`: `FOLD → FOLD`, `CHECK_CALL → CHECK`
  if `legal.can_check` else `CALL`, `BET_RAISE → RAISE`. An action outside the mask
  (or not 0..2) raises `IllegalMaskedActionError(ValueError)` before anything is sent
  to the engine, so nothing changes.
- Engine `ActionType` → action kind for the observation's action history:
  `FOLD → FOLD`, `CHECK`/`CALL → CHECK_CALL`, `RAISE → BET_RAISE`.

The mask is also returned as a numpy `bool` row in batches (`mask[:, a]`);
`mask_array(legal) -> np.ndarray[bool, (3,)]` gives the same row for a single
decision (bot_client, arena's `BotAgent`).

## 2. One table (`table.py`)

`Table(chips_per_bb: int = 100)` — one engine `Environment` in step mode, rebuilt for
each hand. Every hand is independent: stacks, size and button are given anew, so
there is no rebuy, sit-out, join, leave or button movement to manage.

### Starting a hand
`start_hand(n_seats, stacks_bb, button, deck_seed, *, deck=None) -> bool`
- Validates `2 <= n_seats <= 10`, `len(stacks_bb) == n_seats`, every stack > 0 and a
  whole number of chips, `0 <= button < n_seats`, `deck_seed` an int.
- Builds `Environment(players=[Player(f"s{i}", stacks_bb[i]) for i in range(n)],
  button=button, chips_per_bb=chips_per_bb, seed=deck_seed, action_timeout=None)`.
  Players have no agent, so the engine builds no views and queues no events for them.
  The environment's deck is `Deck(random.Random(deck_seed))`, shuffled once at
  `start_hand()`; by the engine's deal-order contract (hole cards round-robin from the
  left of the button, then flop / turn / river, no burns) the same seed deals the same
  cards to the same button-relative positions, whatever the stacks.
- `deck` (tests and replays only): any `entities.Deck`, e.g. a stacked deck, passed to
  `Environment(deck=...)` instead of the seeded one.
- Calls `env.start_hand()`. Returns whether a decision is pending (`False` when the
  hand needed no decision, e.g. everyone all-in on the blinds: it is already over).
- Creates the hand's (empty) `HandTracker` (see Observation) and resets the history
  cursor. The tracker is fed only when an observation (or `tracker`) is read, so a
  table whose observations are never read never processes an event.
- Building a fresh `Environment` per hand costs a few tens of microseconds; at 512
  tables and several decisions per hand it is expected to be a small share. The bench
  profile shows it (`Table.start_hand`); if it exceeds ~10% of the time, reuse one
  environment per table and size (top-up via `rebuy()` as `//engine:bench` does, plus
  an engine-supported way to set stacks and button per hand, requested in
  `engine/spec.md` then) — not before.

### During a hand
- `decision_pending: bool`, `seat_to_act: int`, `legal: LegalActions` (from the
  engine's `Decision`), `mask` (see Actions).
- `view() -> TableView` — the engine's view for the acting seat (`env.view()`, cached
  by the engine per decision). Only built when asked.
- `observation(out=None) -> np.ndarray` — see Observation; does not build a view.
- `act(action: int)` — `to_engine` (raises on a masked action, nothing changes), then
  `env.step(type)`. Engine errors are not expected after the mask check; if one
  happens it propagates (it is a bug in rl_env or the engine).
- `act_engine(action_type)` — submits a raw engine action (e.g. a fold when checking
  is free). Used only by the recorded-hand replay test; not part of the policy path.

### End of a hand
- `hand_over: bool`; `result: HandResult` (`env.last_result`).
- `net_bb: np.ndarray[float64]` — per seat `result.net[seat]`.
- `rewards: np.ndarray[float32]` — `net_bb / 2` (big bets), no rake.
- Asserted when the hand ends: `sum(round(net_bb * chips_per_bb)) == 0`.
- `finished_hole_cards() -> Mapping[int, Hand]` — **evaluation tools only**: seat →
  `Hand` for every seat of the finished hand (all seats of an rl_env table are dealt
  in, folded ones included). Raises `NoHandError` unless `hand_over` (so it cannot
  reveal anything while a decision is pending). It returns the engine's read-only
  `privileged_hole_cards()` mapping, which the engine keeps for the last hand until
  the next one starts; since `start_hand` builds a new `Environment`, the cards stay
  readable exactly until this table's next `start_hand`. Its docstring states that it
  is hidden information and must never reach a controller.

### Privileged read
`Table.privileged_hole_cards()` is the only place in rl_env that calls the engine's
`env.privileged_hole_cards()`. It has two callers: `critic_cards` (during a hand, for
the critic) and `finished_hole_cards` (after a hand). A test scans the package and
checks that no other module mentions `privileged_hole_cards`, and that
`encoder.py`, `tracker.py`, `hints.py` and `controllers.py` call neither.

### Critic input (`critic.py`)
`critic_cards(table, seat) -> np.ndarray[int8, (9, 4)]` — row `k-1` holds the hole
cards of relative slot `k = 1..9` (same slot order as the observation, see
Observation), as `Card.index`, sorted ascending; `NO_CARD = 52` marks a slot beyond
the table size (the marker is a valid embedding index for a 53-row card embedding).
All seats of an rl_env table are dealt in, so there is no "not dealt in" case in
training; the marker covers it anyway. Hole cards are read through
`Table.privileged_hole_cards()` (engine step-mode accessor, see Required engine
capabilities). `encoder.py` and `tracker.py` never import `critic.py`; a test checks
it.

`PRIV_DIM = 9 * 52 = 468` and `expand_critic(cards: (B, 9, 4) int8) -> (B, PRIV_DIM)
float32`: per slot a 52-wide multi-hot of its hole cards (`NO_CARD` contributes
nothing). It is the reference for the critic's input (`OBS_DIM + PRIV_DIM` in the
uber-plan's `CriticNet`); the learner may do the same expansion in torch, checked
against this function in trainer's tests.

## 3. Stacks used for training (`stacks.py`)

```python
@dataclass(frozen=True)
class StackConfig:
    deep_share: float = 0.8
    deep_bb: float = 100.0
    short_min_bb: float = 5.0
    short_max_bb: float = 150.0

def draw_stacks(rng: np.random.Generator, n: int, cfg=StackConfig()) -> tuple[float, ...]
```
- With probability `deep_share` all `n` seats get `deep_bb`; otherwise each seat gets
  an independent uniform draw in `[short_min_bb, short_max_bb]`, rounded to 0.01 BB
  (whole chips at 100 chips/BB).
- `HandSpec(n_seats, stacks_bb, button, deck_seed)` (frozen) — the four inputs of a
  hand.
- `HandSampler(seed, sizes=range(2, 11), cfg=StackConfig())` — one
  `np.random.Generator(np.random.PCG64(seed))`; `sample(n_seats=None) -> HandSpec`
  draws the size (if not given), the stacks, a uniform button and a 63-bit deck seed,
  always in that order, so a run is reproducible from the sampler's seed.

## 4. Observation

### How it is computed
Two layers:
1. `HandTracker` (`tracker.py`) folds the public events of one hand into a compact
   public state. It accepts events one by one (`feed(event)`), so it can be kept up
   to date incrementally. It reads only these event types: `HandStarted` (seats,
   button, blinds, starting stacks of the participants), `BlindPosted`,
   `PlayerActed`, `BoardDealt`. `HandShown`, `BetReturned` and `PotAwarded` only
   happen after the last decision and are ignored. It holds, per participant index
   (participants in seat order, from `HandStarted`): stack, chips in the pot in
   total and on this street, folded, all-in, number of bets/raises per street, acted
   on this street (a voluntary action; blinds do not count); and for the hand: the
   street, board cards per street, pot, current bet, bets made this street (the big
   blind counts preflop; every `RAISE` event counts), the last aggressor on this
   street and the last aggressor before this street, and the last 8 actions
   (`deque(maxlen=8)` of participant index, action kind, street). All amounts are
   taken from the events (`amount_bb`, `to_bb`, `stack_bb`), never recomputed by
   rules.
2. `encode(tracker, me_seat, hole_cards, legal, out_row)` (`encoder.py`) writes the
   vector from the tracker, the acting player's own hole cards and `legal`.

Two entry points produce the same bytes:
- `encode_view(view: TableView, legal: LegalActions) -> np.ndarray` — builds a fresh
  tracker from `view.history`, takes `view.seat` and `view.hole_cards`. This is what a
  bot at a real table (plo8/bot_client, via the cli_server protocol) calls. It uses
  nothing else from the view.
- `Table.observation()` — the training path. The table's tracker is fed only the
  events added since the previous decision (`env` public event history of the
  current hand, read without copying, from a cursor), the acting seat's own hole
  cards come from `env.hole_cards()`, and `legal` from the decision. No
  view is built and the hand history is never re-read.

Equality of the two paths is a test at every decision (see Tests), which also makes
the training path inherit the leakage guarantee of the view (only the acting
player's cards; other hole cards appear in events only through `HandShown`, which
cannot precede a decision and which the tracker ignores anyway).

### Relative slots
Slots are over the hand's participants (the `HandStarted` list, seat order). Slot 0
is the acting player, slot `k` the `k`-th participant clockwise after it. Seats not
dealt in (sitting-out seats at a server table) are not slots, so a table with a
sitting-out seat looks like the smaller table it is. Slots `k >= n` are all zeros.
Positions are relative to the button. No seat number, name or hand number enters the
vector, so the same situation from different physical seats encodes identically.

### Scaling
Every amount is in big bets and written as `min(x / S, 2.0)` with a fixed scale `S`
per kind: pot and chips in the pot `S = 25` (2.0 = 50 big bets = 100 BB), stacks
`S = 50` (150 BB = 1.5), street bets and amount to call `S = 2`. Counts are
`min(c, cap) / cap`. One-hots are 0/1. All values are therefore finite and in
`[0, 2]` (inside the spec's `[-1, 2]`). Values are computed in Python floats and
stored as float32.

### Layout (`ENCODER_VERSION = 1`, `OBS_DIM = 554`)
`OBS_LAYOUT` in `encoder.py` names every block with its offset and size; a test
checks that the blocks are contiguous, non-overlapping and end at `OBS_DIM`.

| Offset | Size | Block | Encoding |
|---|---|---|---|
| 0 | 52 | hole cards | multi-hot over `Card.index` (4 ones); order-free, as in Omaha |
| 52 | 52 | flop | multi-hot (3 ones from the flop on) |
| 104 | 52 | turn | one-hot |
| 156 | 52 | river | one-hot |
| 208 | 4 | street | one-hot preflop / flop / turn / river |
| 212 | 9 | players dealt in | one-hot 2..10 |
| 221 | 10 | position | one-hot of the acting player's offset from the button among participants: 0 = button, 1 = first after the button (SB; heads-up the BB), ... |
| 231 | 17 | betting state | see below |
| 248 | 6 | own action history | see below |
| 254 | 135 | other seats | 9 slots × 15, slot `k` at `254 + 15·(k−1)` |
| 389 | 144 | recent actions | 8 entries × 18, most recent first |
| 533 | 21 | hand-strength hints | see below |

Betting state (231):
| Offset | Item |
|---|---|
| 231 | pot (all chips put in so far), `S = 25` |
| 232 | amount to call (`legal.call_amount`), `S = 2` |
| 233 | pot odds `to_call / (pot + to_call)` (0 when checking is free) |
| 234–238 | bets made this street, one-hot 0, 1, 2, 3, 4-or-more (the cap is lifted heads-up) |
| 239 | own chips in the pot, total, `S = 25` |
| 240 | own chips in the pot, this street, `S = 2` |
| 241 | players still in the hand (not folded) / 10 |
| 242 | players still able to act (not folded, not all-in) / 10 |
| 243 | own stack, `S = 50` |
| 244 | `legal.can_raise` |
| 245 | `legal.can_check` |
| 246 | cap lifted (`legal.raises_left is None`) |
| 247 | `legal.raises_left / 4` (0 when lifted) |

Own action history (248): bets/raises made by the acting player on preflop, flop,
turn, river (`min(c, 4) / 4`, 4 values); acting player is the last aggressor on this
street; acting player was the last aggressor before this street.

Per other seat (slot `k`, 15 values, offsets relative to the slot start):
| +0 | occupied (a participant) |
|---|---|
| +1 | still in the hand (not folded) |
| +2 | all-in |
| +3 | chips in the pot, total, `S = 25` |
| +4 | chips in the pot, this street, `S = 2` |
| +5..+8 | bets/raises made on preflop / flop / turn / river, `min(c, 4) / 4` |
| +9 | last aggressor on this street |
| +10 | last aggressor before this street |
| +11 | acts after the acting player on this street (later in this street's action order: from the left of the big blind preflop, from the left of the button after; only if still in the hand and not all-in) |
| +12 | has acted on this street (voluntary action) |
| +13 | stack, `S = 50` |
| +14 | is the button |

Recent actions (389): the last 8 `PlayerActed` events of the hand (blinds excluded,
forced folds included), entry `j` (0 = most recent) at `389 + 18·j`: +0 present, +1..+10
relative slot one-hot, +11..+13 action kind (fold / check-or-call / bet-or-raise),
+14..+17 street one-hot. Missing entries are zeros.

Hand-strength hints (533):
| +0..+8 | category of the best high hand, one-hot over the evaluator's 9 categories (high card … straight flush); flop on |
|---|---|
| +9 | has a qualifying low; flop on |
| +10..+13 | highest card of the best low, one-hot 5 / 6 / 7 / 8; flop on |
| +14 | low strength: `1 − i / 55`, `i` = rank of the best low among the 56 possible 8-or-better lows (best first; the wheel is 1.0); flop on |
| +15..+19 | distinct low ranks (A..8, ace counted once as low) in the hole cards, one-hot 0..4; every street |
| +20 | hole cards contain an ace and a deuce; every street |

`hints.py` gets high and low from `engine.evaluator.best_omaha(hand, board)` (exactly
two hole cards and three board cards — the engine's rule), the category from the
high rank's first element, and the low index from a 56-entry table built at import
by sorting all 5-of-8 low tuples the same way the evaluator compares them. Preflop
the evaluator is not called and the flop-on entries are zero. The hints are cached
in the tracker per `(seat, street)`: they depend only on the hole cards and the
board, so a player's second decision on the same street reuses them.

## 5. Many tables at once (`batch.py`)

### Controllers
```python
class Controller(Protocol):
    wants_views: bool                     # class or instance attribute
    # optional: wants_obs: bool           # default True when absent
    def act(self, batch: DecisionBatch) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """One action per row (int); optionally also the log-probability of each
        chosen action (float32), which the Recorder keeps for PPO."""
```
- `wants_obs` (optional attribute, read with `getattr(c, "wants_obs", True)`): a
  controller that reads no observation vectors declares `False`. It does not change
  what a batch with observations builds; it only allows the controller to sit in a
  batch without observations (below). Absent means `True`, so a neural policy that
  forgets the attribute can never be handed a batch without vectors.
- `DecisionBatch` (one per controller per step): `key` (the controller's key),
  `tables: np.ndarray[int32]`, `seats: np.ndarray[int32]`, `obs: np.ndarray[float32,
  (B, OBS_DIM)] | None` (`None` in a batch without observations), `mask:
  np.ndarray[bool, (B, 3)]`, `legal: list[LegalActions]`, `views: list[TableView] |
  None` (only built when `wants_views`). It never carries anything hidden: no
  critic input, no other seat's hole cards.
- `DecisionBatch.single(view, legal, key=None) -> DecisionBatch` — a one-row batch
  from one engine view and legal actions (`encode_view`, `mask_array`, `views=[view]`,
  table and seat `-1` / `view.seat`). This is how a `Controller` answers a single
  engine `on_turn` (arena's `BotAgent`) or a decoded server `turn` message
  (bot_client), so neither re-assembles the batch. It always builds `obs`.
- `controllers.py`: `RandomController(seed)` — uniform over the masked actions with
  its own `np.random.Generator` (used by bench, fuzz and tests); `wants_views =
  False`, `wants_obs = False`. Baseline bots (RandomBot, CallingStation, Maniac,
  RuleBot) belong to arena. `tests/helpers.py` has `CheckCallController`, a tiny
  controller that reads `views` and `legal` (`wants_views = True`, `wants_obs =
  False`), to prove the view path of the interface.

### TableBatch
`TableBatch(n_tables, *, chips_per_bb=100, recorder: Recorder | None = None,
observations: bool = True)` holds `n_tables` `Table`s and, per table, the current
hand's id, its spec, per-seat controller keys and whether a hand is running.
- `start_hand(table, spec: HandSpec, controllers: Sequence[Hashable]) -> int` — one
  controller key per seat, chosen by the caller. Assigns the hand a batch-unique,
  increasing `hand_id`. A hand that needs no decision is finished at once and shows
  up in the next `collect_finished()`. Starting a hand on a table ends the
  readability of that table's previous `finished_hole_cards`.
- `pending(controllers=None) -> list[DecisionBatch]` — every table with a decision
  (at most one per table), grouped by the acting seat's controller key. Group order:
  by the smallest table index in the group; rows inside a group: by table index.
  Observations are written straight into the group's preallocated `obs` array; views
  are built only for groups whose controller has `wants_views` (the controller objects
  are passed to `pending(controllers)` so the batch knows).
- `submit(batch: DecisionBatch, actions, logp=None)` — validates every action against
  its mask first (and the batch against the tables: a stale batch raises); if any is
  invalid it raises and no table is touched. Then `act` on each table, recording the
  decisions with their `logp` if given (see Recorder) and moving finished hands to the
  finished list.
- `step(controllers: Mapping[Hashable, Controller]) -> list[FinishedHand]` —
  convenience: `pending`, each controller's `act` in group order, `submit`, then
  `collect_finished()`.
- `collect_finished() -> list[FinishedHand]` — hands finished since the last call, in
  finishing order: `FinishedHand(table, hand_id, spec, controllers, net_bb, rewards,
  result)`. The table is then idle until the caller starts its next hand
  (`idle_tables()` lists them; `any_in_hand()` tells whether any table still has a
  hand running, so a rollout worker can stop dealing and finish the open hands).
  `result.history` gives arena its sample hand histories; `controllers` tells which
  controller sat where (duplicate evaluation rotates them over the same `spec`).
  `FinishedHand` holds only public results (the engine's `HandResult` shows hole
  cards only for hands shown at showdown); the full deal is read separately, below.
- `run(controllers, sampler, n_hands, seat_controllers, on_finished=None)` — helper
  loop used by the bench and the fuzz test: keeps every table busy with
  `sampler.sample()` and the controller keys from `seat_controllers(spec)`, until
  `n_hands` hands finished.

### Batches without observations
`TableBatch(..., observations=False)` is for batches whose controllers read no
vectors (arena's evaluation matches with rule-based bots, the bench's `--no-obs`):
- `pending` builds no `obs` array and calls no encoder: `DecisionBatch.obs is None`.
  Since the table's tracker is only fed when an observation is read, no event is
  processed and no hint is evaluated either. Masks, legal actions, grouping, row
  order and views (for `wants_views` controllers) are built exactly as with
  observations.
- `pending(controllers)` / `step(controllers)` raise `ValueError` (before anything is
  built or submitted) if the batch has no observations and a controller of a pending
  group has `wants_obs` true (or absent), so a controller that needs vectors fails
  loudly instead of receiving `None`. `pending()` without controllers skips this
  check (the caller then knows what it asked for).
- A `Recorder` needs observations: `TableBatch(recorder=..., observations=False)`
  raises `ValueError`.
- Nothing else changes: the engine sees the same actions, so actions, results,
  `hand_id`s, finishing order and `FinishedHand`s are identical to a batch with
  observations for the same seeds and controller decisions (rl_env draws no random
  numbers in `pending`, see Determinism). This is a test.

### Hole cards of finished hands (evaluation tools only)
`TableBatch.finished_hole_cards(table) -> Mapping[int, Hand]` — seat → `Hand` of
every seat of the table's last hand, once it is over (whether or not
`collect_finished()` has returned it yet), until the caller starts that table's next
hand. It is `Table.finished_hole_cards()` of that table; it raises `RuntimeError` if
the table has a hand running and `NoHandError` if it never had one.
- For arena's hand histories and deal-order checks: read it for a `FinishedHand`
  before restarting `fh.table`.
- Kept apart from everything controllers receive: `pending`, `submit`, `step`, `run`
  and `DecisionBatch` never call it and `FinishedHand` does not contain it. The
  docstring marks it as hidden information, like the critic input.

### Training records (`recorder.py`)
`Recorder(keys=None)` — records the decisions of the given controller keys (all when
`None`). For each submitted decision it appends to column buffers: `table`, `hand_id`,
`seat`, `controller` (small int code; `Recorder.codes` maps keys to codes), `obs`,
`mask`, `action`, `logp` (from the controller's `(actions, logp)` return; NaN when it
returned actions only), `critic` (`critic_cards(table, seat)`, taken before the action
is applied). Rows of a running hand stay pending; when the hand finishes, every pending
row of that `hand_id` gets `reward = rewards[seat]` and moves to the ready buffer.
`n_ready` / `n_pending` give the row counts (the worker's "enough decisions" check;
pending rows are always released when their hand ends, so nothing accumulates).
`drain() -> TrainingRecords` returns numpy arrays (one row per decision, decision
order) for finished hands only, so a record is never matched to the wrong or an
unfinished hand: `hand_id` is unique in the batch and the reward is taken from that
hand's `FinishedHand`. A recorder only works with a batch that builds observations
(see above).

### Determinism
rl_env draws random numbers only in `HandSampler` and `RandomController`, each from its
own seeded `np.random.Generator`; the engine draws only from the deck seed. Grouping
and iteration orders are by table index (never over sets or `id()`), so the same
seeds and the same controller decisions give the same decisions, actions, records
and results. Whether observations are built does not touch any random source or any
order, so it does not change a run either.

## 6. Speed

Target: ≥ 5,000 decisions/s on one core, observation and mask built at every
decision, 512 tables in one batch, `RandomController`, at 2, 6 and 9 players.

### Benchmark (`bench/bench.py`, `buck2 run //rl_env:bench`)
Options: `--tables 512`, `--players 2,6,9`, `--hands 20000` per size, `--seed 0`,
`--stacks deep|mixed` (default `mixed`, i.e. `StackConfig()`), `--no-obs`
(`TableBatch(observations=False)`: mask only, to isolate the observation cost and to
show what an arena match pays), `--record` (with a `Recorder`, to see its cost;
incompatible with `--no-obs`), `--profile` (cProfile, top functions). For each size
it reports hands, decisions, wall time, decisions/s, hands/s, and next to it the
engine-only number for the same size, obtained by calling the engine benchmark's
workload function (`engine.benchmark.run_benchmark`, see Required engine
capabilities), plus the share of time spent in rl_env (`1 − engine_time /
total_time`). Timing with `time.perf_counter()`, first 1,000 hands excluded.

### Implementation choices for speed
- Incremental tracker: each event is processed once per hand (cursor into the
  engine's event list), no history copy per decision; nothing is processed when no
  observation is read.
- Sparse writes: the vector is ~95% zeros. Group arrays are allocated with
  `np.zeros`; the encoder collects `(column, value)` pairs in Python lists for all rows
  of a group and writes them with one fancy-indexed assignment per group.
- Static parts cached per hand and seat: hole-card multi-hot indices, hole low-rank
  hints, dealt-in one-hot, position one-hot.
- Hints cached per `(seat, street)`; evaluator called at most once per player per
  street.
- No views unless a controller asks; agents-less players so the engine queues no
  events and builds no `on_hole_cards` views.
- No observations at all in batches that declare they need none.

### Results (development Mac)
| Version | 2p dec/s | 6p dec/s | 9p dec/s | engine-only 6p dec/s | rl_env share (6p) |
|---|---|---|---|---|---|
| First implementation | 31,837 | 38,863 | 35,104 | 141,206 | 72.5% |

First implementation: `buck2 run //rl_env:bench` with the defaults (512 tables,
20,000 hands per size, seed 0, `mixed` stacks, observation built at every
decision), 2026-11-18. Engine-only at 2p / 9p: 84,672 / 152,215 dec/s; rl_env share
62.4% / 76.9%. The target (≥ 5,000 dec/s) is met at every size, so none of the
steps in "If the target is not met" is needed yet.

A row is added after each speed-up, from the bench output. A `--no-obs` row (batch
without observations) is added once that mode is in place, as the reference for
arena's match speed.

### If the target is not met
Record the gap and the profile here, then in this order: (1) precompute per-hand
index tables (slot of each seat, street order) so the per-decision encoder is a
straight loop over slots; (2) encode all rows of a group from tracker arrays held as
flat Python lists instead of per-player objects; (3) if the engine dominates the
profile, request the speed-up in `engine/spec.md` (evaluator or betting loop), since
rules and evaluation belong there; (4) a native encoder (`cxx_python_extension`)
behind the same `encode` signature, checked byte-for-byte against the Python one.

## Required engine capabilities

Used as they are: `Environment(players, button, chips_per_bb, seed, deck,
action_timeout=None)`, `start_hand()` / `step()` / `decision` / `view()` /
`last_result`, `Player(name, stack)`, `Decision`, `LegalActions`, `ActionType`,
`TableView`, `HandResult`, the event classes, `engine.evaluator.best_omaha`, and the
deal-order contract.

Requested in `engine/spec.md` (step mode / speed; names from `engine/plan.md`):
- **Public events of the running hand without a view**: `env.history`, a read-only
  `HandHistory` sequence over the current hand's event list (no copy), so the tracker
  can be fed incrementally (`env.history[seen:]`).
- **Hole cards without a view**: `env.hole_cards()` — the acting player's own hole
  cards for the observation; `env.privileged_hole_cards()` — seat → hand of every
  participant, for the critic input and (after the hand) `finished_hole_cards`,
  clearly separate from anything an agent receives. It keeps answering for the last
  hand once it is over, until the next hand starts (`engine/plan.md`, view-free
  reads), which is all `finished_hole_cards` needs; no further engine change.
- **Exact stacks at any chip unit for initial players**: a stack given in BB that is
  a whole number of the table's chips is seated exactly (today `Player(stack)` first
  rounds to 1/100 BB, which can lose chips of a recorded hand whose big blind is not
  a divisor of 100).
- **Benchmark workload as a function**: `engine.benchmark.run_benchmark(...)`
  (returns a `BenchResult`), so this bench can print the engine-only number from the
  engine's own code.

Until these land, the table reads `env.dealer.history`, `Player.hole_cards` of
`env.players`, and seats replay players through `Environment.join` in cli_server's
e2e seating order; each of these is isolated in one function of `table.py`.

## Required cli_server capabilities

Tests use `//cli_server:protocol` (`encode_table_view` / `decode_table_view`,
`encode_legal` / `decode_legal`, plus `json` text round trip) and `//cli_server:e2e`
(`load_case`, the case seating and `StackedDeck` script builder, the files in
`e2e_test_sets`). `:e2e` must be visible to `//rl_env/...` (requested in
`cli_server/spec.md`).

## Tests

- `test_actions`: mask for every `LegalActions` shape (check free → fold masked;
  facing a bet → fold allowed; raise allowed iff `can_raise`, including cap reached,
  short stack that can only call, others all-in); `to_engine` mapping; masked or
  out-of-range actions raise; the scan of `rl_env/` imports allows only the standard
  library, numpy, `engine` and `entities`.
- `test_table`: start / act / end of a hand at 2..10 seats; `start_hand` argument
  validation; a no-decision hand (all-in on the blinds) is over at once; every
  masked-allowed action is accepted by the engine and a masked one raises with the
  engine state unchanged (same decision, history length, stacks); rewards = net / 2
  and nets sum to 0 in chips; the same deck seed with different stacks deals the same
  cards relative to the button; a hand whose observation is never read leaves the
  tracker unfed.
- `test_stacks`: share of deep hands ≈ `deep_share` over many draws; short stacks in
  range, 0.01 BB resolution; same seed → same specs; custom `StackConfig`.
- `test_tracker`: at every decision of many random hands, the tracker's pot, stacks,
  street bets, total bets, folded / all-in flags and street equal the engine view's
  (`view.pot`, `SeatView`s, `view.street`); incremental feeding equals feeding the
  whole history at once.
- `test_encoder`: `OBS_LAYOUT` contiguous and summing to `OBS_DIM`; hand-written
  situations give the expected values at the documented offsets (one per block);
  shape, dtype and range (`finite`, `[-1, 2]`) over many random hands at all sizes
  and stacks including short stacks, all-ins and side pots; **training path equals
  `encode_view(env.view(), legal)` byte for byte at every decision**.
- `test_leakage` (most important): play a random hand to a random decision; replay it
  with a stacked deck that keeps the acting player's hole cards and the board but
  gives every other player different cards (drawn from the unused cards), with the
  same actions; the observation at every decision of the acting seat is
  byte-identical, on both paths. Thousands of situations, all table sizes. Also:
  `encoder.py` / `tracker.py` do not import `critic.py`; only `table.py` mentions
  `privileged_hole_cards`, and `encoder.py`, `tracker.py`, `hints.py`,
  `controllers.py` call neither it nor `finished_hole_cards`.
- `test_rotation`: same deck seed, same button-relative stacks and actions, button at
  every physical seat → identical observation sequences.
- `test_hints`: known hands — each high category, wheel straight, no flush with one
  suited hole card and four on board (exactly-two rule), qualifying low, no low,
  counterfeited low, a low that only completes on the river, A-2 vs A-3 lows, the
  hole low-rank count with paired low cards, zeros preflop; cache reuse within a
  street and refresh on the next street.
- `test_server_path`: for every decision of a few thousand random hands, the
  observation from `(view, legal)` equals the one from
  `decode(json.loads(json.dumps(encode(...))))` of the same view and legal actions.
- `test_recorded_hands`: each of the 26 files in `cli_server/e2e_test_sets` replayed
  through a `Table` (`chips_per_bb` = the case's big blind, case seating with the
  button on seat 0, stacked deck from the case, recorded actions mapped to masked
  actions, `act_engine` only where the mask does not express the recorded action)
  gives the recorded final stacks and pot splits (per pot, high / low winners and
  amounts in chips).
- `test_critic`: slot order matches the observation's, ascending card indices,
  `NO_CARD` for missing slots, never contains the acting player's cards;
  `expand_critic` has 4 ones per dealt slot, zeros for `NO_CARD`, width `PRIV_DIM`.
- `test_batch`: grouping by controller, at most one decision per table, group and
  row order; views built only for `wants_views` controllers (counted); `submit` with
  one masked action raises and changes no table; finished hands reported once with
  correct rewards; idle tables are not advanced; `any_in_hand` false exactly when
  every table is idle; `DecisionBatch.single(view, legal)` equals the batch's row
  (obs, mask) for the same decision.
- `test_no_obs` (batches without observations): for random batch runs at all sizes
  with `RandomController` and `CheckCallController` (views on), two batches with the
  same sampler and controller seeds, one with and one without observations, give
  identical pending tables / seats / masks / legal actions per step, identical
  actions, `hand_id`s, finishing order, nets and `HandResult` histories; without
  observations `obs is None` and the encoder is never called (counted by patching
  `Table.collect_observation`); views are still built for `wants_views` controllers
  and only for them; a controller with `wants_obs` true or absent makes `step` /
  `pending(controllers)` raise before any table changes; `TableBatch(recorder=...,
  observations=False)` raises.
- `test_finished_cards` (hole cards of finished hands): for random batch runs and for
  stacked-deck hands (cards known from the script and the deal-order contract),
  `finished_hole_cards(t)` of every finished hand gives every seat's dealt hand —
  equal to the stacked script, to `privileged_hole_cards()` read at the hand's last
  decision, and to `result`'s shown hands for every seat shown at showdown; it stays
  readable after `collect_finished()` and on other tables' restarts, changes when
  that table's next hand starts, and raises while the table has a hand running and
  before its first hand. Controllers never receive it: during `step` /
  `run` (no recorder) `Table.privileged_hole_cards` is never called (counted), and
  every `DecisionBatch` field and every view contains no card of another seat (views
  only the acting seat's `hole_cards`, `HandShown` cannot precede a decision).
- `test_recorder`: for random batch runs, every drained record belongs to exactly one
  finished hand, its reward equals that hand's `rewards[seat]`; records of unfinished
  hands are not drained; filtering by controller keys; `logp` stored when the
  controller returns it and NaN otherwise; `n_pending` returns to 0 once all hands end.
- `test_determinism`: two runs with the same sampler, controller seeds and table count
  give identical decisions, actions, records and results; the same deck seed deals
  the same button-relative cards for different stacks.
- `fuzz/test_fuzz_batch` (`rl_env_fuzz_test`): 100,000 random hands through a
  `TableBatch` (sizes 2–10, `StackConfig()` stacks, random controllers), no
  exceptions, nets sum to 0 in chips in every hand, every observation in range; a
  failing seed is printed.
- Regression: `buck2 test //...` — entities, engine, cli_server (including the
  end-to-end cases) and cli_client tests unchanged and green.
