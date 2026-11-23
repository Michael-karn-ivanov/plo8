# trainer — implementation plan

Python library (`trainer`): the neural networks and, in later stages, the training loop
for a fixed-limit Omaha Hi-Lo 8-or-better bot, on top of `//rl_env` and `//arena`. The
current spec covers the first stages: the policy and critic networks, the acting policy
(`NeuralPolicy`, an rl_env `Controller` over a `PolicyNet`) and the rollout loop
(`RolloutWorker`, which plays the learner against arena's bots in one `TableBatch` and
returns the learner's training records). The PPO learner, driver, Ray, league, AWS and
ONNX export are not part of this plan; nothing here is built for them beyond keeping the
pieces easy to reuse (constructor parameters, `state_dict`, the recorded encoder
version, the `(actions, logp)` return that rl_env's `Recorder` stores, a controller key
per seat for later opponent kinds, `collect` returning `(records, stats)` for a later
`@ray.method(num_returns=2)`).

Guiding rules:
- **Consume rl_env's arrays, never re-create them.** The networks take exactly what
  rl_env produces: `obs` float32 `(B, OBS_DIM)`, `mask` bool `(B, N_ACTIONS)`
  (True = legal, order fold / check-or-call / bet-or-raise), and for the critic
  `obs` concatenated with `rl_env.expand_critic(critic_cards)`. trainer has no
  encoder, no mask logic and no torch version of `expand_critic`.
- **The policy cannot receive hidden information.** `PolicyNet.forward(obs, mask)` has
  no other input, and checks that `obs` is exactly `OBS_DIM` wide, so a critic-sized
  input (`OBS_DIM + PRIV_DIM`) is rejected instead of silently used. `NeuralPolicy`
  reads only `DecisionBatch.obs` and `.mask` (a `DecisionBatch` never carries critic
  input). Only `CriticNet` and the `critic_input` helper touch the privileged part.
- **Masking cannot produce NaN.** Illegal logits are set to the finite `MASK_VALUE =
  -1e9`, never `-inf`; legal logits are bounded far above it (see Masking).
- **Acting samples, from its own generator.** `NeuralPolicy` samples from the masked
  distribution with a private `torch.Generator`; it never touches torch's global RNG,
  so a learner in the same process stays reproducible.
- **Records only from complete hands.** The rollout loop never cuts a hand: it stops
  dealing, plays every running hand to its end, then drains rl_env's `Recorder`, which
  only releases rows of finished hands. Nothing is carried from one collection to the
  next.
- **The rollout loop drives, rl_env plays.** `RolloutWorker` only chooses hand specs,
  lineups and controller keys and calls `TableBatch.start_hand` / `step`; batching per
  controller, mask checks, recording and reward attachment are rl_env's.
- **No torch outside trainer.** rl_env and arena stay numpy-only; a trainer test
  enforces it.

## Alignment with `uber-plan.md` (Phase 3a, 3b, 3c)

| uber-plan sketch | here |
|---|---|
| `trainer/nets.py`: `mlp(inp, hidden=512, layers=3)` | `nets.py`: `mlp(in_dim, hidden, layers)` (same shape) |
| `PolicyNet(obs_dim=OBS_DIM, n_actions=3)`, `forward(obs, mask)` with `masked_fill(~mask, -1e9)` | `PolicyNet(obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=512, layers=3)`; masking in `masked_logits` (adds a bound on legal logits and dtype/shape checks) |
| `CriticNet(in_dim=OBS_DIM + PRIV_DIM)`, `forward(obs_priv) -> (B,)` | same, plus `hidden`, `layers` |
| learner's `torch.cat([d.obs, expand_critic(d.critic)])` with a torch twin of `expand_critic` | `critic_input(obs, critic_cards)` in numpy, calling rl_env's `expand_critic` (the spec forbids re-implementing the expansion) |
| default `nn.Linear` init | orthogonal init as in CleanRL's PPO (see Initialisation) |
| — | encoder version and net config recorded in the `state_dict` (extra state) |
| `trainer/policy.py`: `NeuralPolicy()` builds its own `PolicyNet().eval()`, `torch.set_num_threads(1)` | `NeuralPolicy(net, *, seed, num_threads=1)` holds a private copy of the given net (eval, no grad); `NeuralPolicy.from_state_dict(sd, seed=...)` for workers that only receive weights |
| `load(weights)` = `net.load_state_dict(weights)` | `load_state_dict(sd)`: validated before anything is copied (config, encoder version, keys, shapes), so a refused state dict leaves the old weights intact |
| `@torch.no_grad()`, `Categorical(logits).sample()` from the global RNG | `torch.inference_mode()`, `torch.multinomial` with the policy's own seeded `torch.Generator`; log-probabilities via `log_softmax` (`nets.log_probs`, also for the later PPO loss) |
| `wants_views = False` | `wants_views = False`, `wants_obs = True` (explicit) |
| `trainer/lineup.py` "who sits where" | `lineup.py`: `draw_lineup(rng, n_seats, opponents, max_learner_seats=3)` (pure, controller keys per seat); size, stacks, button, deck seed from rl_env's `HandSampler` |
| `RolloutWorker.collect(learner_weights, league_view, n_decisions)` | `RolloutWorker.collect(weights, n_decisions, *, on_finished=None) -> (TrainingRecords, RolloutStats)`; no `league_view` yet (opponents are a constructor parameter: arena bot names) |
| `while rec.n_ready + rec.n_pending < n or any_in_hand()` loop, `self.stats.add(hand)` | same loop shape (plus the no-decision-hand case, see Collection); statistics accumulated per `FinishedHand` in integer chips |
| `Recorder(keys={"learner"})` | `Recorder(keys=[LEARNER_KEY])`, `LEARNER_KEY = "learner"`; bot keys = arena bot names |

## Layout

```
trainer/
  spec.md
  plan.md
  BUCK
  trainer/
    __init__.py      # public exports
    nets.py          # mlp, PolicyNet, CriticNet, masked_logits, log_probs, critic_input,
                     # check_state_dict, constants
    policy.py        # NeuralPolicy (rl_env Controller over a PolicyNet)
    lineup.py        # LEARNER_KEY, DEFAULT_OPPONENTS, draw_lineup
    rollout.py       # RolloutWorker, RolloutStats, ResultStats
  tests/
    helpers.py       # rl_env data: records from TableBatch runs, all mask shapes, peaked nets, view counter
    test_shapes.py
    test_masking.py
    test_sampling.py
    test_gradients.py
    test_state.py
    test_policy.py
    test_lineup.py
    test_rollout.py
    test_imports.py
```

## Build (buck2)

`trainer/BUCK`:
- `python_library(name = "trainer", srcs = glob(["trainer/**/*.py"]), base_module = "",
  deps = ["//rl_env:rl_env", "//arena:arena"], visibility = ["PUBLIC"])`. torch and
  numpy are not buck2 targets: the repo Python toolchain (`toolchains/BUCK`) runs the
  venv interpreter (`.venv`, `requirements-rl.txt`: `torch==2.15.0`, `numpy==2.5.5`),
  so `import torch` works in every target, as numpy does in rl_env. `//arena` is a
  library dependency now (the baseline bots of the rollout loop); the evaluator is used
  in a later stage.
- `python_test(name = "trainer_test", srcs = glob(["tests/*.py"]), base_module =
  "trainer_tests", deps = [":trainer", "//rl_env:rl_env", "//arena:arena"])`.
  `unittest`, like the other projects. All tests run on CPU in seconds (the table runs
  of `test_policy` and `test_rollout` use small networks and few tables, see Tests).

Run: `buck2 test //trainer/...`.

## 1. Networks (`nets.py`)

### Constants and imports
```python
from rl_env import OBS_DIM, ENCODER_VERSION, N_ACTIONS, PRIV_DIM, expand_critic
MASK_VALUE = -1e9        # logit of an illegal action (finite: never -inf)
LOGIT_LIMIT = 1e4        # legal logits are clamped to [-LOGIT_LIMIT, LOGIT_LIMIT]
DEFAULT_HIDDEN = 512
DEFAULT_LAYERS = 3
```
`nets.py` imports only torch, numpy and `rl_env`'s public constants and
`expand_critic`; it never imports `rl_env.encoder` internals or the batch.

### `mlp(in_dim, hidden=512, layers=3) -> nn.Sequential`
`layers` × (`nn.Linear(d, hidden)`, `nn.ReLU()`), `d` = `in_dim` then `hidden`.
`layers >= 1` and `hidden >= 1` (else `ValueError`). Returns the body only; each
network adds its own linear head.

### Initialisation
Applied in each network's constructor, after building all layers, so it uses the torch
global RNG in a fixed order (body layers first-to-last, then the head):
- hidden layers: `nn.init.orthogonal_(weight, gain=sqrt(2))`, bias 0;
- policy head: orthogonal, gain 0.01, bias 0 — the initial policy is close to uniform
  over the legal actions (entropy near `ln(#legal)`), which PPO wants at the start;
- critic head: orthogonal, gain 1.0, bias 0.

Same `torch.manual_seed(s)` before construction → bit-identical weights (orthogonal
init draws from the CPU generator deterministically). Nothing else in construction
draws random numbers.

### Masking: `masked_logits(raw, mask) -> Tensor`
```python
def masked_logits(raw: Tensor, mask: Tensor) -> Tensor:
    _check_mask(mask, raw.shape)                      # bool, same (B, N_ACTIONS) shape
    return raw.clamp(-LOGIT_LIMIT, LOGIT_LIMIT).masked_fill(~mask, MASK_VALUE)
```
- `mask` must be `torch.bool` (`TypeError` otherwise): on a `uint8` mask `~mask` is a
  bitwise not (1 → 254), which would silently mask nothing. `mask.shape == raw.shape`
  (`ValueError` otherwise).
- Illegal logits are exactly `MASK_VALUE`. Legal logits are within ±1e4 (±inf
  included, clamped), so every legal logit exceeds every illegal one by at least
  `1e9 − 1e4`; in float32 a softmax gives illegal actions exactly 0 probability, also
  when only one action is legal or when the raw network output is extreme. The clamp
  is never active in normal training (initial logits are ~0.01 scale); it only
  guarantees the invariant. Its gradient is 0 beyond the limit.
- With these logits, `torch.distributions.Categorical(logits=...)` gives finite
  `log_prob` (an illegal action: ≈ −1e9, never sampled) and finite `entropy` (torch
  multiplies probability 0 by the finite normalised logit, giving 0), and no NaN in
  their gradients.
- A row with no legal action cannot come from rl_env (check-or-call is always legal);
  such a row would give three equal logits, i.e. a uniform distribution — finite, not
  an error. Not checked per call (cost); the tests cover every mask rl_env produces.
- NaN inputs are not sanitised (rl_env guarantees finite observations in [−1, 2]; a
  NaN is a bug upstream and should stay visible).

### `log_probs(logits) -> Tensor`
`torch.log_softmax(logits, dim=-1)` over masked logits, `(B, N_ACTIONS)` float32. The
one place where action log-probabilities are computed from logits: `NeuralPolicy` uses
it for the `logp` it returns, and the later PPO loss will use it for the new
log-probabilities, so the probability ratio of an unchanged network is 1 up to float
rounding. (It equals `Categorical(logits=...).log_prob` to ~1 ulp; the tests use
either.)

### `PolicyNet`
```python
class PolicyNet(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=512, layers=3,
                 encoder_version=ENCODER_VERSION): ...
    def forward(self, obs: Tensor, mask: Tensor) -> Tensor:   # (B, n_actions) float32
```
- `body = mlp(obs_dim, hidden, layers)`, `head = nn.Linear(hidden, n_actions)`.
- `forward`: checks `obs` (float32, 2-D, `obs.shape[1] == obs_dim`) and `mask` (bool,
  `(obs.shape[0], n_actions)`), raising `ValueError` / `TypeError` with the expected
  and the actual shape/dtype; then `masked_logits(self.head(self.body(obs)), mask)`.
  No reshape, no dtype conversion, no copy of the inputs: tensors made with
  `torch.from_numpy` from rl_env's arrays (`DecisionBatch.obs` / `.mask`,
  `TrainingRecords.obs` / `.mask`) are used as they are.
- Output columns are in rl_env's action order (`FOLD`, `CHECK_CALL`, `BET_RAISE`),
  since the head has `N_ACTIONS` outputs and the mask is applied column by column.
- Works for any `B >= 1` (and `B = 0`, returning `(0, n_actions)`), on CPU. Pure
  function of the inputs and weights: no state is kept between calls, the module has
  no dropout or normalisation layers, so `train()` / `eval()` do not change outputs.
- Attributes: `obs_dim`, `n_actions`, `hidden`, `layers`, `encoder_version`.

### `CriticNet`
```python
class CriticNet(nn.Module):
    def __init__(self, in_dim=OBS_DIM + PRIV_DIM, hidden=512, layers=3,
                 encoder_version=ENCODER_VERSION): ...
    def forward(self, x: Tensor) -> Tensor:                   # (B,) float32
```
- `body = mlp(in_dim, hidden, layers)`, `head = nn.Linear(hidden, 1)`; `forward`
  checks `x` (float32, 2-D, `x.shape[1] == in_dim`) and returns
  `self.head(self.body(x)).squeeze(-1)` — shape `(B,)` also for `B = 1` (`squeeze(-1)`,
  never a bare `squeeze()`).
- The output is in big bets, the unit of rl_env's rewards (`TrainingRecords.reward`);
  no output scaling.
- Separate module, separate parameters: a `PolicyNet` and a `CriticNet` share no
  `nn.Parameter` (checked by a test via `data_ptr()` of all parameters).

### Critic input: `critic_input(obs, critic_cards) -> np.ndarray`
```python
def critic_input(obs: np.ndarray, critic_cards: np.ndarray) -> np.ndarray:
    # obs float32 (B, OBS_DIM); critic_cards int8 (B, 9, 4) (rl_env.critic_cards rows,
    # TrainingRecords.critic) -> float32 (B, OBS_DIM + PRIV_DIM), C-contiguous
    return np.concatenate([obs, expand_critic(critic_cards)], axis=1)
```
Validates `obs.shape == (B, OBS_DIM)`, dtype float32, and `len(critic_cards) == B`
(`expand_critic` validates the rest). It only concatenates; the expansion is rl_env's.
Its result goes to `CriticNet` via `torch.from_numpy`. This is the only function in
trainer that handles the privileged input, and its docstring says so; `PolicyNet` and
`NeuralPolicy` never receive it.

### Encoder version and config in the state dict
Both networks implement `get_extra_state()` / `set_extra_state(state)` (torch's hook,
stored in `state_dict()` under `_extra_state`):
- `get_extra_state()` → `{"net": "policy" | "critic", "encoder_version": ...,
  "in_dim": ..., "hidden": ..., "layers": ..., "n_actions": ...}` (plain ints and
  strings, so `torch.load(weights_only=True)` reads it).
- `set_extra_state(state)` raises `NetConfigError(ValueError)` if any entry differs
  from the receiving network's own, naming the differing keys. So a state dict built
  for another encoder version is refused at `load_state_dict`, even when the layer
  shapes happen to match (shape mismatches are already refused by torch).
- `encoder_version` defaults to `rl_env.ENCODER_VERSION` and `obs_dim` / `in_dim` to
  `OBS_DIM` / `OBS_DIM + PRIV_DIM`, read at construction, so an encoder change changes
  the networks with it. Later checkpoints and ONNX exports read `net.encoder_version`.

`check_state_dict(net, state_dict)` — validates a state dict against a network
**without copying anything**: `_extra_state` present and equal to
`net.get_extra_state()` (differences named), the same set of parameter keys (missing
and unexpected keys named), and for each key the same shape and dtype. Raises
`NetConfigError` on any mismatch. Torch's own `load_state_dict` may have copied some
tensors before it detects a mismatch; callers that must stay intact on refusal
(`NeuralPolicy.load_state_dict`) call `check_state_dict` first.

Round trip: `b.load_state_dict(a.state_dict())` for two networks of the same config
gives `torch.equal` outputs for the same inputs.

## 2. Acting policy (`policy.py`)

```python
class NeuralPolicy:                                   # rl_env Controller
    wants_views = False                               # the batch builds no views for it
    wants_obs = True                                  # refused by batches without observations

    def __init__(self, net: PolicyNet, *, seed: int, num_threads: int | None = 1): ...
    @classmethod
    def from_state_dict(cls, state_dict, *, seed: int, num_threads: int | None = 1) -> "NeuralPolicy": ...

    net: PolicyNet                                    # read-only property (the private copy)
    def seed(self, seed: int) -> None: ...
    def load_state_dict(self, state_dict) -> None: ...
    def state_dict(self) -> dict: ...                 # the private net's state_dict()
    def act(self, batch: DecisionBatch) -> tuple[np.ndarray, np.ndarray]: ...
```

### Construction
- `NeuralPolicy(net, seed=...)` takes a `PolicyNet` and keeps a **private copy**
  (`copy.deepcopy(net)`, which draws no random numbers): later training steps on the
  learner's net do not change the acting policy, and the policy changes only through
  `load_state_dict`. The copy is put in `eval()` mode and all its parameters get
  `requires_grad_(False)`. A non-`PolicyNet` argument raises `TypeError`.
- `from_state_dict(sd, seed=...)` — for workers that receive only weights (later
  Ray): reads `sd["_extra_state"]`, requires `net == "policy"`, `encoder_version ==
  rl_env.ENCODER_VERSION`, `in_dim == OBS_DIM`, `n_actions == N_ACTIONS`
  (`NetConfigError` otherwise), builds `PolicyNet(hidden=..., layers=...)` inside
  `torch.random.fork_rng(devices=[])` (its throw-away initialisation does not move
  torch's global RNG), then `load_state_dict(sd)`.
- `num_threads`: `torch.set_num_threads(num_threads)` at construction (default 1: a
  rollout worker owns one core); `None` leaves the setting alone. This is a
  process-wide torch setting — the docstring says so; in one process all policies
  should use the same value (the last constructed wins). `num_threads < 1` raises
  `ValueError`.
- `seed` is required (an `int >= 0`, else `ValueError` / `TypeError`); see Sampling.

### `act(batch)`
```python
def act(self, batch):
    if batch.obs is None:
        raise ValueError("NeuralPolicy needs observation vectors")
    n = batch.mask.shape[0]
    if n == 0:
        return np.empty(0, np.int64), np.empty(0, np.float32)
    with torch.inference_mode():
        logits = self._net(torch.from_numpy(batch.obs), torch.from_numpy(batch.mask))
        lp = log_probs(logits)                                    # (n, N_ACTIONS)
        a = torch.multinomial(lp.exp(), 1, generator=self._gen)   # (n, 1)
        logp = lp.gather(1, a).squeeze(1)
    actions = a.squeeze(1).numpy()                                # int64, rl_env action order
    if not batch.mask[np.arange(n), actions].all():               # cannot happen: illegal p == 0
        raise RuntimeError("sampled an illegal action")
    return actions, logp.numpy()                                  # float32
```
- One network call per decision batch, whatever its size; the inputs are
  `torch.from_numpy` views of `DecisionBatch.obs` / `.mask` (no copy). It reads no
  other field of the batch: not `views` (never built, `wants_views = False`), not
  `legal`, not `tables` / `seats`.
- Sampling, not argmax: `torch.multinomial` over the probabilities `exp(log_probs)`.
  Illegal actions have probability exactly 0 in float32 (see Masking), and
  `multinomial` never draws a zero-probability category; the explicit mask check
  after sampling turns any violation into an error instead of a silent illegal action
  (rl_env's `submit` would also refuse it).
- Return value: `(actions, logp)` — the tuple form of rl_env's `Controller.act`, so
  `TableBatch.step` passes `logp` to `submit` and the `Recorder` stores it per
  decision. `actions` int64 `(n,)`, `logp` float32 `(n,)`, the log-probability of each
  chosen action under the current weights.
- Gradient-free and in evaluation mode: `torch.inference_mode()`, parameters without
  `requires_grad`, net in `eval()`. The returned arrays share memory with tensors
  created in this call only, so nothing is aliased across calls.
- Batch sizes: any `n >= 1`, including `DecisionBatch.single(view, legal)` (one row,
  `obs` always built); `n = 0` returns two empty arrays without calling the network
  or the generator (rl_env never sends an empty batch, but callers composing batches
  may).
- `batch.obs is None` (a batch without observations) raises `ValueError`. rl_env
  refuses earlier anyway, because `wants_obs = True`.

### Sampling generator
- `self._gen = torch.Generator(device="cpu")`, `manual_seed(seed)` at construction;
  `seed(s)` re-seeds it. Only `act` draws from it, one `multinomial` call per
  non-empty batch. torch's global generator is never used by the policy (construction
  via `__init__` draws nothing; `from_state_dict` forks the global RNG around its
  initialisation).
- Reproducible: the same seed, the same weights and the same sequence of decision
  batches give the same actions and log-probabilities (CPU, same thread count; the
  network is deterministic on CPU for equal inputs).

### Weights
- `load_state_dict(sd)`: `check_state_dict(self._net, sd)` first — a state dict of
  another config (hidden, layers, critic, ...) or another `ENCODER_VERSION`, or with
  missing / unexpected keys or other shapes, raises `NetConfigError` and leaves the
  current weights untouched. Then `self._net.load_state_dict(sd, strict=True)`, which
  copies every tensor into the private parameters (`copy_`), replacing all weights;
  no tensor of `sd` is aliased, so a learner that keeps training after handing over
  its `state_dict()` does not change the policy.
- Accepted sources: `PolicyNet.state_dict()`, `NeuralPolicy.state_dict()`, and a
  state dict through `torch.save` / `torch.load(weights_only=True)`.
- No other state: besides the weights and the generator, the policy keeps nothing
  between batches (no caches, no per-table memory).

## 3. Rollout loop (`lineup.py`, `rollout.py`)

### Controller keys and lineups (`lineup.py`)
```python
LEARNER_KEY = "learner"
DEFAULT_OPPONENTS = ("random", "calling", "maniac", "rule")    # = arena.BOT_NAMES
MAX_LEARNER_SEATS = 3

def draw_lineup(rng: np.random.Generator, n_seats: int, opponents: Sequence[str],
                max_learner_seats: int = MAX_LEARNER_SEATS) -> tuple[str, ...]:
    k = int(rng.integers(1, min(max_learner_seats, n_seats - 1) + 1))  # 1..3, never every seat
    learner = rng.choice(n_seats, size=k, replace=False)               # random positions
    keys = [opponents[int(i)] for i in rng.integers(len(opponents), size=n_seats)]
    for s in learner:
        keys[int(s)] = LEARNER_KEY
    return tuple(keys)
```
- One controller key per seat: `LEARNER_KEY` or an arena bot name. Later opponent kinds
  (frozen snapshots, league entries) only add other keys (e.g. `("snapshot", id)`) and
  their controllers; the batch and the recorder already work per key.
- Draw order is fixed (count, positions, one bot draw for every seat, learner seats
  overwritten), so the lineup is a function of the generator state and `n_seats`. The
  bot draws for learner seats are thrown away rather than drawn conditionally, which
  keeps the number of draws per hand independent of where the learner sits.
- Heads-up `k` is exactly 1 (`min(3, 1)`); at 3 seats it is 1 or 2; from 4 seats 1..3,
  uniform. Each opponent seat gets a bot uniformly from `opponents`, independently.
- Pure function: no table, no engine; `test_lineup` checks the distributions directly.

### `RolloutWorker`
```python
class RolloutWorker:
    def __init__(self, *, seed: int, n_tables: int = 256,
                 opponents: Sequence[str] = DEFAULT_OPPONENTS,
                 sizes: Iterable[int] = range(2, 11),
                 max_learner_seats: int = MAX_LEARNER_SEATS,
                 hidden: int = DEFAULT_HIDDEN, layers: int = DEFAULT_LAYERS,
                 num_threads: int | None = 1): ...

    def collect(self, weights: Mapping[str, Any], n_decisions: int, *,
                on_finished: Callable[[FinishedHand], None] | None = None
                ) -> tuple[TrainingRecords, RolloutStats]: ...

    policy: NeuralPolicy        # read-only properties, for tests and later stages
    batch: TableBatch
    recorder: Recorder
    opponents: tuple[str, ...]
```

#### Construction
- Validation (`ValueError` / `TypeError`): `seed` an int `>= 0`; `n_tables >= 1`;
  `opponents` non-empty, no duplicates, every name in `arena.BOT_NAMES` (unknown names
  are refused by `make_bot` too, with the list of known ones), none equal to
  `LEARNER_KEY`; `sizes` non-empty, each in 2..10 (`HandSampler` checks it);
  `max_learner_seats >= 1`.
- Seeds: `ss = np.random.SeedSequence(seed)`; `hand_ss, lineup_ss, policy_ss, bot_ss =
  ss.spawn(4)`, turned into ints with `int(x.generate_state(1, np.uint64)[0])` (policy:
  `np.uint32`, a torch seed):
  - `self._sampler = HandSampler(hand_seed, sizes)` (rl_env's default `StackConfig()`:
    80% all 100 BB, otherwise per-seat 5–150 BB). It draws size, stacks, button, deck
    seed — the size uniformly over `sizes` (default 2..10).
  - `self._lineup_rng = np.random.Generator(np.random.PCG64(lineup_seed))`, only for
    `draw_lineup`.
  - bots: `self._bots = {name: make_bot(name, seed=bot_seeds[i]) for i, name in
    enumerate(opponents)}` with `bot_seeds = bot_ss.spawn(len(opponents))`; created
    once per worker, each sits at any number of tables (they are stateless batched
    controllers; RandomBot's generator is consumed in batch order, deterministic).
  - learner: `PolicyNet(hidden=hidden, layers=layers)` built inside
    `torch.random.fork_rng(devices=[])` (its initial weights are irrelevant — every
    collection loads weights first — and the global torch RNG is not moved), then
    `NeuralPolicy(net, seed=policy_seed, num_threads=num_threads)`.
- `self._controllers = {LEARNER_KEY: self.policy, **self._bots}`.
- `self._recorder = Recorder(keys=[LEARNER_KEY])` (learner code 0; bot decisions are
  not recorded) and `self._batch = TableBatch(n_tables, recorder=self._recorder)`
  (observations on: the learner needs them; RuleBot groups also get views, the other
  bots none).
- The worker persists across collections: batch, recorder, sampler, lineup generator,
  bot generators and the policy's sampling generator all continue, so a sequence of
  collections is reproducible from the seed; `hand_id`s keep increasing.

#### `collect(weights, n_decisions)`
```python
def collect(self, weights, n_decisions, *, on_finished=None):
    n = _check_positive_int(n_decisions)                 # before anything else
    self._policy.load_state_dict(weights)                # NetConfigError: nothing played
    batch, rec = self._batch, self._recorder
    assert not batch.any_in_hand() and rec.n_pending == 0 and rec.n_ready == 0
    acc = _StatsAccumulator(self.opponents, self._sampler.sizes)
    t0, all0 = time.perf_counter(), batch.n_decisions

    def recorded():                                       # finished or still running
        return rec.n_ready + rec.n_pending

    def account(hands):
        for fh in hands:
            acc.add(fh)
            if on_finished is not None:
                on_finished(fh)                           # before fh.table restarts

    while True:
        if recorded() < n:
            for t in batch.idle_tables():
                spec = self._sampler.sample()
                keys = draw_lineup(self._lineup_rng, spec.n_seats, self.opponents,
                                   self.max_learner_seats)
                acc.start(batch.start_hand(t, spec, keys), spec, keys)
        if not batch.any_in_hand():
            account(batch.collect_finished())             # hands that needed no decision
            if recorded() >= n:
                break
            continue
        account(batch.step(self._controllers))

    records = rec.drain()
    assert rec.n_pending == 0 and rec.n_ready == 0 and len(records) >= n
    stats = acc.finish(records, wall=time.perf_counter() - t0,
                       all_decisions=batch.n_decisions - all0)
    return records, stats
```
- Weights first: `NeuralPolicy.load_state_dict` validates (config, encoder version,
  keys, shapes) before copying, so mismatched weights raise `NetConfigError` with no
  hand started, no generator advanced and the previous weights intact. Every decision
  of the collection is made with exactly the loaded weights (nothing else changes the
  policy during `collect`).
- Dealing: new hands are started only on idle tables and only while
  `recorded() < n_decisions`. At the first refill every table gets a hand; afterwards a
  table is refilled right after its hand finished (at the top of the next loop
  iteration, after its `FinishedHand` has been accounted and passed to
  `on_finished`). Once `n_decisions` learner decisions exist (ready or pending) no hand
  is started; the loop steps until no hand is running. So the records exceed
  `n_decisions` by at most the learner decisions of hands already running at that
  moment.
- A hand that needs no decision (e.g. both blinds all-in) is finished inside
  `start_hand` and reported by the next `collect_finished()` (also through `step`); the
  `not any_in_hand()` branch collects such hands when no table is running, so none is
  lost or carried into the next collection.
- One batched call per controller per step: `TableBatch.step` groups all pending
  decisions by controller key, so the learner's decisions at all tables (whichever of
  its seats is acting) are one `NeuralPolicy.act` call, and each bot one call.
- Nothing left behind: at return every table is idle, `collect_finished()` is empty and
  the recorder has no pending and no ready rows, so consecutive collections are
  independent (only the generators carry over, by design).
- `drain()` returns rows in decision order, finished hands only, with `reward` = the
  seat's net in big bets from that hand's `FinishedHand` (rl_env's Recorder).
- Termination: each loop iteration either steps a running hand (every hand ends) or
  starts hands; while `recorded() < n` hands keep being dealt and every lineup has a
  learner seat, so learner decisions keep arriving.

#### Statistics (`RolloutStats`)
```python
@dataclass(frozen=True)
class ResultStats:              # one group of hands
    hands: int                  # hands in the group
    learner_seats: int          # learner seats over those hands
    decisions: int              # learner decisions in those hands
    total_chips: int            # learner's net over all its seats, chips (100 per BB)
    total_bb: float             # big bets: total_chips / 200
    bb_per_100: float           # 100 * total_bb / hands (nan when hands == 0)

@dataclass(frozen=True)
class RolloutStats:
    overall: ResultStats
    by_size: dict[int, ResultStats]        # every size of `sizes`, ascending
    by_opponent: dict[str, ResultStats]    # every opponent name, in `opponents` order
    all_decisions: int                     # every seat's decisions, bots included
    wall_time: float                       # seconds, time.perf_counter
    decisions_per_second: float            # learner decisions / wall_time
    all_decisions_per_second: float
    # convenience properties: hands, decisions, total_bb, bb_per_100 (= overall's)
```
- `_StatsAccumulator.start(hand_id, spec, keys)` remembers per hand: table size, learner
  seats, the set of bot names at the table. `add(fh)`: learner chips of the hand =
  `Σ round(fh.net_bb[s] * 100)` over its learner seats (integer chips, as arena does,
  so sums are exact and order-free); the hand is added to `overall`, to
  `by_size[n_seats]` and to `by_opponent[name]` for every distinct bot name present
  (a hand with two Maniacs counts once for `maniac`).
- Learner decisions per hand come from the drained records (`np.unique(records.hand_id,
  return_counts=True)`) in `finish`, so `overall.decisions == len(records)`.
- Hand counts include hands in which no learner seat had a decision (a walk in the big
  blind, a blind all-in): they are played and their learner result counts, but they
  contribute no record. So `overall.hands >=` the number of distinct `hand_id`s in the
  records, and `overall.total_chips` = the records' per-(hand, seat) results plus those
  seats' results.
- "Mean result per hand" is per hand, summing all learner seats of that hand
  (`bb_per_100 = 100 · total_bb / hands`); `learner_seats` is kept so a per-seat rate
  can be derived. Big bets = 2 BB = 200 chips, as everywhere else.
- These are logging figures, not an evaluation (no duplicate, no CI); arena's
  `play_match` remains the evaluator.

### Public exports (`trainer/__init__.py`)
`PolicyNet`, `CriticNet`, `NeuralPolicy`, `RolloutWorker`, `RolloutStats`,
`ResultStats`, `draw_lineup`, `LEARNER_KEY`, `DEFAULT_OPPONENTS`, `mlp`,
`masked_logits`, `log_probs`, `critic_input`, `check_state_dict`, `MASK_VALUE`,
`LOGIT_LIMIT`, `NetConfigError`.

## 4. Tests

### Data (`tests/helpers.py`)
- `records(n_hands, seed, sizes=range(2, 11)) -> TrainingRecords` — a `TableBatch`
  (64 tables, `Recorder()` recording every seat) driven by `rl_env.RandomController`
  with a `HandSampler(seed, sizes)` (training stacks: deep, short, all-ins, side pots)
  until `n_hands` hands finished, then `drain()`. Real observations, masks and critic
  inputs of 2- to 10-player hands; enough hands for several thousand rows.
- `ALL_MASKS` — every mask rl_env can produce, as bool rows over (fold, check-or-call,
  bet-or-raise): `(F,T,F)` (check free, no raise — e.g. everyone else all-in),
  `(F,T,T)` (check free, raise), `(T,T,F)` (facing a bet, cannot raise: cap reached
  or short stack), `(T,T,T)`. A test asserts that every mask found in `records(...)`
  is one of `ALL_MASKS` and that the three common ones `(F,T,T)`, `(T,T,F)`,
  `(T,T,T)` occur; the rarer `(F,T,F)` is always covered by the synthetic
  `ALL_MASKS` batches (on real observations) whether or not the run hits it.
- `decision_batch(obs, mask, key="nn") -> DecisionBatch` — builds a batch from given
  rows (`tables` / `seats` = `-1`, `legal` = empty list entries, `views = None`),
  including 0-row batches.
- `peaked_policy_net(seed, hidden=64, layers=2) -> PolicyNet` — a small net whose head
  weights are scaled (×300) so that, on real observations, action probabilities are
  clearly uneven (rows with probabilities between ~0.05 and ~0.9), for the frequency
  test; small nets keep table runs fast.
- `count_views()` — context manager patching `rl_env`'s `Table.view` with a counting
  wrapper (the batch builds views only through it), yielding the counter.
- `play(controllers, seat_keys, n_hands, seed, n_tables=64, sizes=range(2, 11),
  recorder=None)` — keeps every table of a `TableBatch` busy with `HandSampler(seed)`
  specs and `seat_keys(spec, rng)` controller keys until `n_hands` hands finished,
  via `TableBatch.step`; returns the finished hands and the list of `(key, actions,
  logp)` per step (for determinism comparisons).
- `small_weights(seed) -> dict` — `PolicyNet(hidden=64, layers=2)` built under
  `torch.manual_seed(seed)`, its `state_dict()`; with `peaked=True` the head scaled as
  in `peaked_policy_net` (so two weight sets give clearly different actions).
- `small_worker(seed, **kw) -> RolloutWorker` — `RolloutWorker(seed=seed, n_tables=32,
  hidden=64, layers=2, **kw)`; collections of ~2,000 decisions take a second or two
  (RuleBot's evaluator calls dominate).

### `test_shapes`
- From `records(...)`: batch sizes 1, 7 and 512 (rows of mixed table sizes); policy
  returns `(B, N_ACTIONS)` float32, critic `(B,)` float32 on `critic_input(obs,
  critic)`; also directly on `DecisionBatch.obs` / `.mask` from `TableBatch.pending()`.
- No copy: `torch.from_numpy(arr).data_ptr() == arr.ctypes.data` for `obs`, `mask`
  and the critic input, and the networks accept those tensors without error.
- Defaults follow rl_env: `PolicyNet().obs_dim == OBS_DIM`, `n_actions == N_ACTIONS`,
  `CriticNet().in_dim == OBS_DIM + PRIV_DIM`, both `encoder_version ==
  ENCODER_VERSION`; non-default `hidden` / `layers` give the expected number and size
  of `nn.Linear` layers.
- Wrong inputs raise: policy given a critic-sized input (`OBS_DIM + PRIV_DIM`), a
  float64 obs, a `uint8` or int mask, a mask of the wrong shape; critic given an
  `OBS_DIM`-wide input.
- `critic_input` equals `np.concatenate([obs, rl_env.expand_critic(cards)], 1)` and
  is float32 `(B, OBS_DIM + PRIV_DIM)`.

### `test_masking`
For each mask in `ALL_MASKS` (applied to real observations), for every real mask of
`records(...)`, and for extreme inputs (observations scaled by ±1e3, random normal
×1e3; a network whose head weights are multiplied by 1e6; `masked_logits` called
directly with raw logits ±1e30, ±inf and 0):
- illegal actions: logit exactly `MASK_VALUE`, softmax probability `== 0` (≤ 1e-30
  for safety in the assertion);
- legal probabilities sum to 1 (`atol 1e-6`);
- `Categorical(logits=...)`: `log_prob` of every action and `entropy` are finite;
  with one legal action (`(F,T,F)`), probability 1, log-probability 0, entropy 0;
- `log_probs(logits)` is finite and matches `Categorical(logits=...).log_prob` for
  every action (`atol 1e-6` on legal actions);
- backward of `entropy().sum() + log_prob(legal_actions).sum()` gives finite gradients.

### `test_sampling`
- 100,000 samples from `Categorical` over masked logits of real decisions (seeded):
  no illegal action ever; every legal action appears (initial policy ≈ uniform).
- The same with `torch.multinomial(log_probs(logits).exp(), 1)` (the sampler
  `NeuralPolicy` uses), also on the extreme-input logits of `test_masking`.

### `test_gradients`
- Policy: on 256 real rows (with `(F,T,T)`, `(T,T,F)` and `(T,T,T)` masks present,
  so every head row gets gradient), loss = `−(log_prob(a) · adv).mean() − 0.01 ·
  entropy.mean()` with sampled `a` and random `adv`; one `torch.optim.SGD(lr=0.1)`
  step. Every parameter tensor has a finite, not-all-zero gradient and changed value
  (`not torch.equal(before, after)`), body and head.
- Critic: MSE against the rows' rewards on `critic_input(...)`, one step, same checks.
- Policy and critic parameters are disjoint (`data_ptr()` sets do not intersect), and
  a critic step leaves the policy unchanged.

### `test_state`
- `torch.manual_seed(s)`: two `PolicyNet`s (two `CriticNet`s) have identical
  `state_dict`s; a different seed gives different weights.
- `state_dict` / `load_state_dict` into a freshly constructed (differently seeded)
  network → identical outputs on real batches (`torch.equal`); also through
  `torch.save` / `torch.load(weights_only=True)` in a temporary file.
- `_extra_state` holds the encoder version and config; loading a state dict whose
  `encoder_version` was altered raises `NetConfigError`; loading a policy state dict
  into a critic, or one with other `hidden` / `layers`, raises.
- `check_state_dict`: passes for a matching state dict; raises `NetConfigError`
  naming the problem for an altered `encoder_version`, other `hidden`, a missing key,
  an extra key and a wrongly shaped tensor; never modifies the network (parameters
  `torch.equal` before and after).

### `test_policy`
Unit level (fixed batches of real rows from `records(...)`, `ALL_MASKS` batches):
- Protocol: `wants_views is False`, `wants_obs is True`; `act` returns `(actions,
  logp)` with `actions` int64 `(B,)`, values in `{0, 1, 2}`, every one allowed by its
  row's mask, `logp` float32 `(B,)`, finite and ≤ 0; for `B` = 1, 7, 512, a
  `DecisionBatch.single(view, legal)` built from a real `TableBatch` view, and `B = 0`
  (two empty arrays, generator state unchanged: the next draw equals that of a fresh
  policy with the same seed).
- `logp` equals `log_probs(net(obs, mask))` at the returned actions (`torch.equal`:
  same computation) and `Categorical(logits).log_prob` (`atol 1e-6`).
- Only obs and mask are read: a batch whose `legal`, `views`, `tables` and `seats`
  are replaced by objects that raise on any access (a small `Poison` class) gives the
  same result; a batch with `obs = None` raises `ValueError`; a critic-sized `obs`
  raises (via `PolicyNet`).
- No gradients, eval mode: `policy.net.training is False`, no parameter
  `requires_grad`; calling `act` inside `torch.enable_grad()` still records no graph.
- Private copy: training steps on the source net after `NeuralPolicy(net, ...)`
  leave the policy's outputs unchanged.
- Threads: after `NeuralPolicy(net, seed=0)` `torch.get_num_threads() == 1`; with
  `num_threads=2` it is 2; with `None` unchanged (the test restores the original
  value in `tearDown`).
- Generator independence: `torch.manual_seed(123)`, then constructing a policy (both
  ways) and calling `act` many times leaves `torch.random.get_rng_state()` unchanged;
  changing the global seed between two runs does not change the policy's actions.
- Reproducibility: two policies with the same seed and weights give identical
  actions and logp over a sequence of batches; `seed(s)` restarts the sequence;
  another seed gives different actions over a 512-row batch.

Sampling frequencies:
- A fixed batch of 16 real rows (masks `(F,T,T)`, `(T,T,F)`, `(T,T,T)` and `(F,T,F)`
  present) and `peaked_policy_net`: 5,000 `act` calls (80,000 draws). Per row and
  action, the frequency is within 5 standard deviations
  (`5·sqrt(p(1−p)/5000)`) of `p = softmax(logits)`; a chi-square over all rows
  stays below the 1e-6 quantile; illegal actions never appear. Seeded, so the test is
  deterministic.

Weights:
- Two nets A, B (different seeds); `NeuralPolicy(A)`, `load_state_dict(B.state_dict())`
  → on a fixed batch, the returned logp equal B's `log_probs` at the returned
  actions; the same after loading another `NeuralPolicy`'s `state_dict()` and after a
  `torch.save` / `torch.load(weights_only=True)` round trip; `from_state_dict(B
  .state_dict(), seed=s)` behaves like `NeuralPolicy(B, seed=s)` (identical actions
  and logp).
- Loading replaces completely: every parameter equals B's afterwards; continuing to
  train B afterwards does not change the policy.
- Refused: a state dict with another `hidden` / `layers`, a `CriticNet` state dict,
  an altered `encoder_version`, a missing key → `NetConfigError`, and the policy's
  parameters and outputs are unchanged; `from_state_dict` with another
  `encoder_version` or a critic state dict raises.

At the table (small nets, `hidden=64, layers=2`, so a few hundred hands take seconds):
- **All seats**: `play({"nn": NeuralPolicy(...)}, every seat "nn", 300 hands, sizes
  2–10, 64 tables, Recorder())` runs without errors; inside `count_views()` the
  counter stays 0; every recorded action is allowed by its recorded mask (rl_env's
  `submit` also refuses illegal actions, so completing proves it too); every hand's
  net results sum to 0.
- **Mixed**: per hand, each seat is `"nn"` or `"rnd"` (`rl_env.RandomController`)
  at random, at least one of each; `Recorder(keys={"nn"})`; same checks; views still
  0 (both controllers want none).
- **Recorded logp**: for the drained records of both runs, `log_probs(policy.net(obs,
  mask))` at the recorded actions equals the recorded `logp` (`atol 1e-5`: the
  recomputation runs as one large batch, the recording in per-step groups of other
  sizes, so matrix products may round differently in the last bits); no NaN in the
  NeuralPolicy rows.
- **Determinism**: two runs with the same sampler seed, policy seed and weights give
  identical per-step actions and logp, the same `hand_id`s and nets; another policy
  seed gives different actions.

### `test_lineup`
`draw_lineup` with a seeded generator, 20,000 lineups at every size 2..10 (and at
sizes drawn as the worker draws them):
- length `n_seats`; learner count in `1..min(3, n_seats − 1)` — exactly 1 heads-up,
  never every seat; each allowed count occurs with frequency ≈ uniform (5-sigma
  bounds); at 6 seats every seat index holds the learner with frequency ≈ the mean
  learner count / 6.
- every non-learner key is in `opponents`; with the default set all four names occur
  with frequency ≈ 1/4 per seat; with `opponents=("calling", "maniac")` only those two.
- same generator state → same lineup; the number of draws does not depend on the
  outcome (two generators stay in lockstep after lineups of equal sizes).

### `test_rollout`
All with `small_worker` and `small_weights`; `n_decisions` of 1,000–3,000.

Construction:
- invalid arguments raise (`n_tables=0`, empty / duplicate / unknown opponents, an
  opponent named `"learner"`, a size 1 or 11, `max_learner_seats=0`);
- `torch.random.get_rng_state()` is unchanged by constructing a worker and by a
  `collect`.

Lineups (through `collect` with an `on_finished` that keeps every `FinishedHand`,
~600 hands):
- table sizes (`fh.spec.n_seats`) cover 2..10; per hand the learner holds 1..3 seats
  (`fh.controllers.count(LEARNER_KEY)`), never every seat, exactly one heads-up;
  every other key is a configured opponent name; all four default bots appear; with
  `opponents=("calling", "maniac")` and `sizes=(2, 6)` only those names and sizes.

Collection:
- `len(records) >= n_decisions`; `records.controller` all equal the learner's code
  (`worker.recorder.codes[LEARNER_KEY]`); every `(hand_id, seat)` of the records is a
  learner seat of a hand passed to `on_finished` in this collection (so all from
  finished hands, none from bots);
- every record's `reward` equals `fh.rewards[seat]` of its hand (exactly: the same
  float32), i.e. `net_bb[seat] / 2`;
- every recorded action is allowed by its recorded mask;
- recorded `logp` equals `log_probs(PolicyNet loaded with the weights)(obs, mask)` at
  the recorded actions (`atol 1e-5`, as in `test_policy`), no NaN;
- stop rule: `worker.batch.start_hand` wrapped by the test (instance attribute
  delegating to the original) asserts `recorder.n_ready + recorder.n_pending <
  n_decisions` at every call during `collect`; with `n_decisions=1` on 8 tables the
  records come only from hands started while no learner decision existed;
- nothing pending afterwards: `not worker.batch.any_in_hand()`,
  `worker.batch.collect_finished() == []`, `worker.recorder.n_pending == 0` and
  `n_ready == 0`; a second `collect` works and its records' `hand_id`s are all larger
  than the first's.
- one learner call per step: a wrapper around `worker.policy.act` counts calls and
  rows; per `TableBatch.step` (wrapped too) at most one learner call, and the rows of
  that call equal the number of pending learner decisions of that step.

Weights:
- `collect(B_weights, ...)` after `collect(A_weights, ...)`: the second collection's
  recorded logp match B's network (recomputed, `atol 1e-5`) and not A's (peaked
  weights make the difference large); `worker.policy.state_dict()` equals B's.
- Mismatched weights (other `hidden`, a critic state dict, an altered
  `encoder_version`, a missing key) raise `NetConfigError` before any hand:
  `worker.batch.n_hands` and `n_decisions` unchanged, no table running, recorder
  empty, the policy's parameters unchanged; a following valid `collect` gives exactly
  the records a fresh worker with the same seed gives (no generator was advanced).
- `n_decisions` of 0, negative or not an int raises `ValueError` / `TypeError`
  before the weights are loaded.

Statistics:
- `stats.overall.hands == len(finished hands)`, `stats.overall.decisions ==
  len(records)`, `Σ by_size hands == overall.hands`, per-size and per-opponent
  `decisions` equal the records of those hands;
- `overall.total_chips == Σ_hands Σ_learner seats round(fh.net_bb[s] * 100)`; from
  the records: `Σ over distinct (hand_id, seat) of round(reward · 200)` plus the
  results of learner seats without any record (found from the kept `FinishedHand`s)
  equals `total_chips` — each learner seat counted once, not once per decision;
  `total_bb == total_chips / 200`, `bb_per_100 == 100 · total_bb / hands`;
- `by_opponent[name].hands` = number of hands with at least one seat of that bot; a
  hand with two copies of one bot counts once;
- `wall_time > 0`, `decisions_per_second == decisions / wall_time`, `all_decisions >=
  decisions`.

Determinism:
- two workers with the same seed, same weights and the same `n_decisions` over two
  consecutive collections: every `TrainingRecords` field equal (`np.array_equal`,
  logp bit-identical), stats equal after `dataclasses.replace` of `wall_time` and the
  two rates; the `FinishedHand` specs and controller keys equal;
- another worker seed gives different hand specs and records.

### `test_imports`
- Static: for every module of the `rl_env` and `arena` packages (found with
  `pkgutil.walk_packages` over `rl_env.__path__` / `arena.__path__`, source via
  `importlib.util.find_spec(...).origin`), parse with `ast` and assert no `import` /
  `from` of `torch` or `trainer` (top-level or nested).
- Runtime: a fresh interpreter (`subprocess.run([sys.executable, "-c", ...])` with
  `PYTHONPATH` = the test's `sys.path`) imports `rl_env`, `arena` and all their
  submodules and reports that neither `torch` nor `trainer` is in `sys.modules`.
- trainer's own modules import only the standard library, numpy, torch, `rl_env` and
  `arena` (only `lineup.py` / `rollout.py` import `arena`).

Regression: `buck2 test //...` — entities, engine, rl_env, arena, cli_server and
cli_client tests unchanged and green.

## Required capabilities from other projects

Used as they are, no change requested:
- `//rl_env`: `OBS_DIM`, `ENCODER_VERSION`, `N_ACTIONS`, `FOLD` / `CHECK_CALL` /
  `BET_RAISE`, `PRIV_DIM`, `expand_critic`; the `Controller` protocol with
  `wants_views`, the optional `wants_obs` and the `(actions, logp)` return of `act`;
  `DecisionBatch` (fields `obs`, `mask`, `legal`, `views`, `tables`, `seats`, `key`;
  constructible directly, including with 0 rows) and `DecisionBatch.single`;
  `TableBatch` (`start_hand` returning the `hand_id` and finishing no-decision hands
  at once, `step` grouping by controller key and passing a controller's `logp` to
  `submit`, `collect_finished`, `idle_tables`, `any_in_hand`, counters `n_hands` /
  `n_decisions`, views built only for `wants_views` controllers through `Table.view`),
  `FinishedHand` (`hand_id`, `spec`, `controllers`, `net_bb`, `rewards`), `Recorder`
  (`keys`, `codes`, `n_ready`, `n_pending`, `drain`) / `TrainingRecords` (`hand_id`,
  `seat`, `controller`, `obs` float32, `mask` bool, `action`, `logp` (NaN for
  controllers that return none), `critic` int8 (N, 9, 4), `reward` float32 big bets),
  `HandSampler` (`sizes`, default `StackConfig()`), `RandomController`.
- `//arena`: `make_bot(name, seed)` and `BOT_NAMES`; the bots are batched `Controller`s
  with `wants_obs = False` (they sit in a batch with observations without using them),
  RuleBot with `wants_views = True`; RandomBot seeded at creation. Also the package
  scanned by `test_imports`.
- Repo venv: `torch==2.15.0` is already pinned in `requirements-rl.txt`; buck2 already
  runs the venv interpreter.
