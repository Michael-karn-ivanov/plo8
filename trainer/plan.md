# trainer — implementation plan

Python library (`trainer`): the neural networks and, in later stages, the training loop
for a fixed-limit Omaha Hi-Lo 8-or-better bot, on top of `//rl_env` (and later
`//arena`). The current spec covers the first stage only: the policy and critic
networks. The acting policy (`NeuralPolicy`), rollout loop, PPO learner, driver, Ray,
league, AWS and ONNX export are not part of this plan; nothing here is built for them
beyond keeping the networks easy to reuse (constructor parameters, `state_dict`, the
recorded encoder version).

Guiding rules:
- **Consume rl_env's arrays, never re-create them.** The networks take exactly what
  rl_env produces: `obs` float32 `(B, OBS_DIM)`, `mask` bool `(B, N_ACTIONS)`
  (True = legal, order fold / check-or-call / bet-or-raise), and for the critic
  `obs` concatenated with `rl_env.expand_critic(critic_cards)`. trainer has no
  encoder, no mask logic and no torch version of `expand_critic`.
- **The policy cannot receive hidden information.** `PolicyNet.forward(obs, mask)` has
  no other input, and checks that `obs` is exactly `OBS_DIM` wide, so a critic-sized
  input (`OBS_DIM + PRIV_DIM`) is rejected instead of silently used. Only
  `CriticNet` and the `critic_input` helper touch the privileged part.
- **Masking cannot produce NaN.** Illegal logits are set to the finite `MASK_VALUE =
  -1e9`, never `-inf`; legal logits are bounded far above it (see Masking).
- **No torch outside trainer.** rl_env and arena stay numpy-only; a trainer test
  enforces it.

## Alignment with `uber-plan.md` (Phase 3a)

| uber-plan sketch | here |
|---|---|
| `trainer/nets.py`: `mlp(inp, hidden=512, layers=3)` | `nets.py`: `mlp(in_dim, hidden, layers)` (same shape) |
| `PolicyNet(obs_dim=OBS_DIM, n_actions=3)`, `forward(obs, mask)` with `masked_fill(~mask, -1e9)` | `PolicyNet(obs_dim=OBS_DIM, n_actions=N_ACTIONS, hidden=512, layers=3)`; masking in `masked_logits` (adds a bound on legal logits and dtype/shape checks) |
| `CriticNet(in_dim=OBS_DIM + PRIV_DIM)`, `forward(obs_priv) -> (B,)` | same, plus `hidden`, `layers` |
| learner's `torch.cat([d.obs, expand_critic(d.critic)])` with a torch twin of `expand_critic` | `critic_input(obs, critic_cards)` in numpy, calling rl_env's `expand_critic` (the spec forbids re-implementing the expansion) |
| default `nn.Linear` init | orthogonal init as in CleanRL's PPO (see Initialisation) |
| — | encoder version and net config recorded in the `state_dict` (extra state) |

## Layout

```
trainer/
  spec.md
  plan.md
  BUCK
  trainer/
    __init__.py      # public exports
    nets.py          # mlp, PolicyNet, CriticNet, masked_logits, critic_input, constants
  tests/
    helpers.py       # rl_env data: records from TableBatch runs, all mask shapes, SamplingController
    test_shapes.py
    test_masking.py
    test_sampling.py
    test_gradients.py
    test_state.py
    test_imports.py
```

## Build (buck2)

`trainer/BUCK`:
- `python_library(name = "trainer", srcs = glob(["trainer/**/*.py"]), base_module = "",
  deps = ["//rl_env:rl_env"], visibility = ["PUBLIC"])`. torch and numpy are not
  buck2 targets: the repo Python toolchain (`toolchains/BUCK`) runs the venv
  interpreter (`.venv`, `requirements-rl.txt`: `torch==2.15.0`, `numpy==2.5.5`), so
  `import torch` works in every target, as numpy does in rl_env. `//arena` is not a
  dependency yet (later stages: fixed opponents, evaluator).
- `python_test(name = "trainer_test", srcs = glob(["tests/*.py"]), base_module =
  "trainer_tests", deps = [":trainer", "//rl_env:rl_env", "//arena:arena"])` —
  `//arena:arena` only so that `test_imports` can check arena's modules. `unittest`,
  like the other projects. All tests run on CPU in seconds.

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
trainer that handles the privileged input, and its docstring says so; `PolicyNet`
never receives it.

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

Round trip: `b.load_state_dict(a.state_dict())` for two networks of the same config
gives `torch.equal` outputs for the same inputs.

### Public exports (`trainer/__init__.py`)
`PolicyNet`, `CriticNet`, `mlp`, `masked_logits`, `critic_input`, `MASK_VALUE`,
`LOGIT_LIMIT`, `NetConfigError`.

## 2. Tests

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
- `SamplingController(net, seed)` — test-only rl_env `Controller` (`wants_views =
  False`): `logits = net(torch.from_numpy(batch.obs), torch.from_numpy(batch.mask))`,
  samples from `Categorical(logits=logits)` under `torch.no_grad()` (a
  `torch.Generator` seeded with `seed`, via `torch.multinomial(probs, 1,
  generator=...)`), returns the actions. (The real acting policy is a later stage.)

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
- backward of `entropy().sum() + log_prob(legal_actions).sum()` gives finite gradients.

### `test_sampling`
- 100,000 samples from `Categorical` over masked logits of real decisions (seeded):
  no illegal action ever; every legal action appears (initial policy ≈ uniform).
- A `TableBatch` of 128 tables, 2–10 players, every seat a `SamplingController`
  over a fresh `PolicyNet`, for ≥ 2,000 hands: `TableBatch.submit` re-checks every
  action against its mask and raises on an illegal one, so finishing without error
  proves no illegal action was chosen; the test also checks each returned action
  against its row's mask itself before returning it.

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

### `test_imports`
- Static: for every module of the `rl_env` and `arena` packages (found with
  `pkgutil.walk_packages` over `rl_env.__path__` / `arena.__path__`, source via
  `importlib.util.find_spec(...).origin`), parse with `ast` and assert no `import` /
  `from` of `torch` or `trainer` (top-level or nested).
- Runtime: a fresh interpreter (`subprocess.run([sys.executable, "-c", ...])` with
  `PYTHONPATH` = the test's `sys.path`) imports `rl_env`, `arena` and all their
  submodules and reports that neither `torch` nor `trainer` is in `sys.modules`.
- trainer's own modules import only the standard library, numpy, torch and `rl_env`.

Regression: `buck2 test //...` — entities, engine, rl_env, arena, cli_server and
cli_client tests unchanged and green.

## Required capabilities from other projects

Used as they are, no change requested:
- `//rl_env`: `OBS_DIM`, `ENCODER_VERSION`, `N_ACTIONS`, `FOLD` / `CHECK_CALL` /
  `BET_RAISE`, `PRIV_DIM`, `expand_critic`; for tests `TableBatch`, `DecisionBatch`,
  `Recorder` / `TrainingRecords` (`obs` float32, `mask` bool, `critic` int8 (N, 9, 4),
  `reward` float32 big bets), `HandSampler`, `RandomController`.
- `//arena`: only as a test dependency of `test_imports`.
- Repo venv: `torch==2.15.0` is already pinned in `requirements-rl.txt`; buck2 already
  runs the venv interpreter.
