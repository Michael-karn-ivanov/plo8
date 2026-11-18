# Limit O8 self-play training cluster: the uber-plan

How to build a reinforcement-learning bot for fixed-limit Omaha Hi-Lo 8-or-better on top of
what already exists in this repo (`//entities`, `//engine`, `//cli_server`, `//cli_client`),
from a laptop prototype up to a Ray cluster on AWS.

This revises the original plan,
[Limit O8 self-play training cluster: implementation plan](https://claude.ai/artifact/CTQcB6Kv5v8eev3iYu9eX1)
(2026-11-17). The RL design is the same: synchronous PPO, a critic that sees all cards,
batched multi-table rollout workers, an opponent pool with MMD regularisation, duplicate
evaluation, Ray, ONNX export. What changed is everything that touches code that already
exists. The original assumed a stand-alone `o8rl` repo wrapping a black-box engine. This
plan builds inside the plo8 monorepo, with buck2, `spec.md` / `plan.md` per project, and
the engine and server as they are today.

---

## 0. Review: where the original plan and the repo disagree

| # | Original plan assumed | What plo8 actually has | Change in this plan |
|---|---|---|---|
| 1 | A step-style engine: `to_act()`, `apply(action)`, so one process can interleave hundreds of tables | `//engine` is **callback-driven**: `Dealer.play_hand()` runs a whole hand and blocks inside `agent.on_turn(player, view, legal)`. A table cannot be paused mid-hand to batch decisions across tables | **New engine capability: step mode** (spec change in `engine/spec.md`). The dealer's hand flow becomes a generator; the callback mode is rebuilt on top of it, so the server and all existing tests keep working. This is the main new piece of work (Phase 1A, before any RL project exists) |
| 2 | Separate `o8rl` repo, `pyproject.toml`, engine as a pip dep / submodule, `vendor/engine` in the Docker image | Monorepo, buck2, one folder per project with `spec.md` + `plan.md`, planner/coder agents, `[SPEC]` → plan → code commits | New projects `//rl_env`, `//arena`, `//trainer`, `//bot_client` in the repo, each with spec/plan/BUCK, each created only when its phase starts. The Docker image copies the repo |
| 3 | Own card numbering `id = rank*4 + suit` | `entities.Card.index` = `(rank-2)*4 + suit_index`, `0..51`, plus `Card.from_index` | Use `Card.index` everywhere; no second numbering |
| 4 | "Add a test that deck order depends only on the seed, if your engine draws lazily" | Deal order is a **documented contract** (`engine/plan.md`, "Deck use and deal order"), covered by `test_replay`; `Environment(deck=...)` takes any `Deck`; `cli_server` already has a `StackedDeck` | Duplicate deals = a fresh `Deck(random.Random(seed))` per rotation. Nothing new is needed in the engine |
| 5 | Adapter returns chip results; reward in "big bets" | Engine amounts are in **BB** (floats over integer chips), `HandResult.net[seat]`; limit structure is fixed: SB 0.5, BB 1, small bet 1 BB, big bet **2 BB** | `return = HandResult.net[seat] / 2` |
| 6 | 3-action legal mask from the adapter | `LegalActions(can_check, call_amount, can_raise, raise_to, bet_unit, raises_left)`; the engine allows folding when checking is free | `action_mask(legal)` = `[not can_check, True, can_raise]` (fold masked when check is free) |
| 7 | Encoder "raises this street, one-hot 0–4" | 4-bet cap, **lifted heads-up** (`raises_left = None`) | Last bucket is "4 or more" |
| 8 | Encoder has no stack features | Real games through `cli_server` have any buy-in; short stacks go all-in, side pots exist | Add my stack and per-seat stacks (in big bets, clipped). Train mostly deep, sometimes short (Phase 3) |
| 9 | A second adapter `seat_view_from_server(msg)` for the bot client, plus a test that it matches the engine path | `cli_server.protocol` already decodes `turn` messages into the engine's own `TableView` and `LegalActions` dataclasses | The encoder takes `TableView` + `LegalActions`, so the server path needs no new adapter. The parity test becomes a protocol round-trip test plus replaying `e2e_test_sets` |
| 10 | "If the server holds rules logic, make it call the same engine" | `cli_server` already plays through `//engine`; `cli_client` only renders | Done. Rules live only in `//engine`, and any speed-up of the evaluator happens there |
| 11 | Engine speed unknown; target 2,000 hands/s/core | Measured on this Mac (Python 3.14, random agents, no NN): **2-max 4.0k hands/s, 6-max 1.2k, 9-max 0.8k; ~30k decisions/s** in all three. `best_omaha` ≈ **125 µs** per player on the river. At 6-max, evaluator ≈ 36% and `table_view` ≈ 33% of the time | Speed-ups go into `//engine`: lazy views, faster evaluator (Phase 1A). Target set from the measured baseline |
| 12 | `engine.remote` style IPC not mentioned | `engine/remote.py` runs agents in other processes over `multiprocessing` pipes | Not used for training: one IPC round trip per decision is far too slow. Ray moves *batches*, never single decisions |
| 13 | Hand-written bots only inside the trainer | The engine's `Agent` protocol (`on_turn`) is what `cli_server` drives | Arena bots implement **both** the batched `Controller` and the engine `Agent` protocol. They can play inside training and also sit at `cli_server` (via `//bot_client`) against you in `cli_client` |
| 14 | Python 3.12 venv, `pip install torch ray ...` | System Python 3.14; buck2 `system_demo_toolchains()`, which downloads its own CPython 3.13 and ignores any venv; no third-party packages anywhere | One Python 3.14 venv (`plo8/.venv`) for the RL projects; `toolchains/BUCK` points buck2's Python toolchain at it (Phase 0, done). Pure engine projects stay dependency-free |
| 15 | Rake as an adapter option (Phase 7) | No rake in the engine; e2e cases have `"rake": 0` | Rake is a rules change, so it goes in `//engine` (spec change) if you ever need it |
| 16 | Bot client and export at the very end | `cli_server` + `cli_client` already work | `//bot_client` comes right after Phase 2 (RuleBot at the real table) and Phase 3 (first neural bot). It's the fastest sanity check there is: you play it yourself |

What I agree with and kept unchanged: phase gates with "Done when" tests; synchronous PPO
first; CPU learner first; a privileged critic; whole-hand Monte-Carlo returns (no GAE);
batched inference over many tables per process; Ray as the fleet manager; a league with
anchors, PFSP sampling and MMD; duplicate evaluation with confidence intervals;
checkpoints in S3; ONNX export with an encoder version. The hyperparameters are also
unchanged.

---

## 1. Overview

Phases 0–7, with Phase 1 split into 1A (engine) and 1B (`rl_env`), plus an export and bot-client track that runs alongside them. Phases
0–5 run on your Mac. AWS only comes in at Phase 6, once training already works locally.
Don't start a phase until the previous one passes its "Done when" test. Most RL bugs are
silent, and those tests are how you catch them.

What you end up with:

- **`//engine` with step mode**: the same rules, now drivable one decision at a time.
- **`//rl_env`**: the RL view of the engine. It has the step-mode table, the observation
  encoder, `TableBatch` (hundreds of tables in one process) and the benchmark. Needs numpy only.
- **`//arena`**: baseline bots and the duplicate-match evaluator (bb/100 with a
  confidence interval). Needs numpy only.
- **`//trainer`**: networks, PPO, rollout workers, league, Ray actors, `train.py`,
  configs, Docker and cluster files. Needs torch and ray.
- **`//bot_client`**: a `cli_server` client that plays with an arena bot or an exported
  `.onnx` model. Needs onnxruntime.
- Checkpoints in S3 that you can resume from, and `.onnx` models you can play against in
  `cli_client`.

How work flows in this repo (unchanged): for each phase, edit the project's `spec.md`
(`[SPEC]: ...` commit). The `planner` agent then updates `plan.md`, the `coder` agent
implements it, `buck2 test //<project>/...` passes, and the `commit-author` skill makes
the commits. After any engine change, also run the `e2e-testing` skill
(`buck2 run //cli_server:cli_server -- --e2e_test`), so the server and the 26 recorded
hands still agree with the engine.

Each phase below is sized to be one spec change for one project, with its tests as the
acceptance criteria.

Because committing a spec starts plan generation and implementation, **a project's
folder and `spec.md` are created only when its phase starts**, never earlier as an
empty skeleton. The order is:

1. Phase 0: venv and tooling (no specs).
2. Phase 1A: `//engine` (step mode, then speed). No new projects yet.
3. Phase 1B: `//rl_env` is created.
4. Phase 2: `//arena`, then `//bot_client`.
5. Phase 3: `//trainer`.

---

## 2. Machine learning in plain language

(Condensed from the original. Every later section uses these terms.)

The bot is a function. What a seat can see goes in, and a probability for each of fold,
call and raise comes out. Training adjusts about a million numbers inside that function
so that actions that won chips become more likely.

1. **Observation.** A network reads a fixed-length vector. The **encoder** turns "seat 3,
   turn dealt, 6 players, my cards A♠2♦K♥3♣, pot 6 big bets…" into about 480 numbers,
   mostly 0/1. It must only use what that seat is allowed to see. If opponents' cards leak
   in, the bot learns to cheat and becomes useless in real games.
2. **Policy network.** A few matrix multiplications with non-linearities between them.
   It outputs three **logits**. Illegal actions are masked out, then a softmax gives the
   probabilities. The bot **samples** from them instead of taking the top one, because
   good poker needs mixed strategies (bluff frequencies). Sampling also makes the bot
   try alternatives while it learns.
3. **Reward and return.** The chips a seat won or lost in the hand, in big bets (2 BB).
   Every decision in the hand gets the hand's result as its return. One hand is one
   episode.
4. **Critic and advantage.** Results are mostly luck. A second network, the critic,
   predicts the expected result from the current situation, and the **advantage** is the
   return minus that prediction. During training only, the critic sees every player's
   cards. That makes its predictions much sharper, so more of the luck cancels. The
   policy never sees them, and the critic isn't used at play time.
5. **PPO.** Nudge every weight so that positive-advantage actions become more likely,
   but cap how far one update can move any probability (about ±20%). The learning rate
   sets the size of each nudge. An entropy bonus stops the bot from becoming predictable
   too early.
6. **Iteration.** Play about 250k decisions with the current weights. Shuffle them into
   minibatches of about 8k and take a gradient step on each. Go over the data about 3
   times (each pass is an **epoch**), then throw the data away and play again. PPO is
   **on-policy**, which is why playing and training alternate.
7. **Weights, checkpoints, export.** A model is a class plus a `state_dict` (about 4 MB).
   A checkpoint adds the optimizer state and the iteration number. To play outside
   training, export the policy to ONNX.
8. **CPU vs GPU.** One network call on 1 decision costs almost the same as on 1,000.
   That's why each worker process plays hundreds of tables at once and asks the network
   about all of them in one call. CPUs are enough for v1.
9. **Self-play and the pool.** First train against hand-written bots. Then train against
   copies of itself plus frozen past versions (the **league**), because strategies can
   cycle like rock-paper-scissors. **MMD** adds a small penalty for drifting from a
   slowly refreshed copy of the policy (the "magnet").
10. **Measuring progress.** Self-play reward is about 0 by definition. The evaluator
    plays frozen snapshots against fixed opponents in **duplicate** (the same cards
    replayed with seats rotated) and reports **bb/100** (big bets per 100 hands) with a
    95% confidence interval.

---

## 3. Your questions, answered for this repo

**What happens to `//engine`?** It becomes the core of training. Each rollout worker
imports it and calls it in memory, with no network. Two engine changes are needed:
**step mode** (Phase 1A.1), so a worker can hold many tables and pause each one at a
decision, and **speed** (Phase 1A.2), mainly the evaluator and lazy view building.
Everything else the engine has (joining, leaving, sitting out, rebuys, timeouts, events,
locks) stays, and training simply doesn't use it. The engine stays the single source of
rules truth: training, the server, the e2e tests and the bot client all run the same code.

**What happens to `//cli_server` and `//cli_client`?** They stay out of the training
loop, because a network round trip per decision is hundreds of times slower than a
function call. They become the play and inspection front end:
- `//bot_client` connects to `cli_server` like any player and plays with an arena bot or
  an `.onnx` model.
- You play against it in `cli_client`, or watch bot-vs-bot tables.
- `cli_server --e2e_test` keeps guarding the engine against regressions while you change
  it for speed and step mode.

**Do you need a fleet manager to run 1,000 games in parallel?** No. Inside one process,
`TableBatch` holds 256–1,024 tables in step mode. Each step collects every table waiting
for a decision, makes one batched network call per controller, and applies the actions,
with no threads. Across cores and machines, Ray runs one such worker per core, moves
batches, restarts crashed workers, and starts and stops EC2 instances from a YAML file.

**How are weights exported and applied?** Unchanged from the original:

| Situation | What moves | How |
|---|---|---|
| Learner → workers, every iteration | Policy `state_dict` (~4 MB) | `ray.put(weights)`; workers call `load_state_dict` |
| Saving progress | Policy, critic, optimizer, magnet, league index, iteration | `torch.save` locally, then S3 every N iterations |
| Adding a past version to the pool | Frozen policy weights | File in S3 plus an entry in the league index; workers cache by id |
| Using the bot outside training | Policy only | `torch.onnx.export` → `.onnx`, loaded by `//bot_client` with onnxruntime |

"Applying" a model is always three steps: encode the engine `TableView` with the
**same** `rl_env` encoder, run the network, sample an action. The encoder and model must
stay in sync, so version them together (`ENCODER_VERSION`).

---

## 4. Target architecture (v1)

Unchanged from the original. A driver loop sends the current weights out, collects
played hands back, trains, and every so often freezes a snapshot as a future opponent.

1. The driver gets the learner's policy weights and `ray.put`s them.
2. It gets the league's opponent list: snapshot ids, sampling probabilities and fixed bots.
3. Every rollout worker plays hands on its tables until it has its share of decisions,
   finishes any hands in progress, and returns a batch of arrays.
4. The learner computes advantages with the critic, runs PPO for a few epochs, and
   returns metrics.
5. Every 25 iterations the driver checkpoints to S3 and sends the policy to the evaluator
   as a candidate.
6. The evaluator plays the candidate in duplicate against fixed bots and the pool. The
   league admits it if it clearly wins.

| Component | Runs where | Holds | Lives in |
|---|---|---|---|
| Driver (`train.py`) | Ray head | loop, config, run id | `//trainer` |
| Rollout worker | 1 process per core, many machines | 256–1,024 step-mode engine tables, policy copy, opponent cache | `//trainer` (uses `//rl_env`) |
| Learner | 1 process (CPU first, GPU later) | policy, critic, optimizer, magnet | `//trainer` |
| League | 1 lightweight actor | pool index: id, path, rating, results | `//trainer` |
| Evaluator | a few processes | fixed bots, duplicate seeds | `//arena` (match runner), wrapped by `//trainer` |
| Storage | S3 (local disk on the Mac) | checkpoints, snapshots, eval results, logs | `//trainer` |

On the Mac all of these are processes on one machine. On AWS the same code runs across
machines and only the config changes.

---

## 5. Repository layout

New projects sit next to the existing ones. Every new project has `spec.md`, `plan.md`
and `BUCK`, like the rest of the repo. This is the **end state**: each folder is created
in the phase noted next to it, not up front.

```
plo8/
  CLAUDE.md
  uber-plan.md                 # this file
  requirements-rl.txt          # pinned third-party deps for the RL projects (numpy, torch, ray, onnx, ...)
  entities/                    # existing: Card (index 0..51), Hand, Board, Street, Deck (+ put_on_top)
  engine/                      # existing + Phase 1A: step mode, speed, bench/bench.py (engine-only)
  cli_server/                  # existing: TCP server, protocol, e2e mode + e2e_test_sets/
  cli_client/                  # existing: curses client
  rl_env/                      # created in Phase 1B — deps: //engine, numpy
    rl_env/
      actions.py               # FOLD / CHECK_CALL / BET_RAISE, action_mask, mask_array, to_engine
      table.py                 # Table: one step-mode Environment per hand
      stacks.py                # StackConfig, HandSpec, HandSampler
      tracker.py               # HandTracker: public state of a hand from engine events
      hints.py                 # hand-strength hints (engine evaluator), cached per street
      encoder.py               # encode_view(view, legal); OBS_DIM, OBS_LAYOUT, ENCODER_VERSION
      critic.py                # critic_cards, expand_critic, PRIV_DIM (critic only)
      batch.py                 # TableBatch, DecisionBatch, FinishedHand, Controller protocol
      controllers.py           # RandomController
      recorder.py              # Recorder, TrainingRecords (obs, mask, action, logp, critic, reward)
    bench/bench.py             # decisions/s with encoder + mask over TableBatch
    tests/
  arena/                       # created in Phase 2 — deps: //rl_env
    arena/
      bots.py                  # RandomBot, CallingStation, Maniac
      rules.py                 # RuleBot (tight-aggressive), uses engine.best_omaha
      agent.py                 # BotAgent: any arena bot as an engine Agent (on_turn) for cli_server
      match.py                 # duplicate matches, bb/100 with CI, sample hand histories
    tests/
  trainer/                     # created in Phase 3, grows through Phase 6 — deps: //rl_env, //arena, torch, ray
    trainer/
      nets.py                  # PolicyNet, CriticNet
      policy.py                # NeuralPolicy: Controller over a PolicyNet
      lineup.py                # who sits where
      rollout.py               # RolloutWorker.collect(): the play loop, no Ray inside
      ppo.py                   # ppo_loss(), Learner
      league.py                # pool index, PFSP sampling, admission
      storage.py               # checkpoints locally or in S3
      actors.py                # thin @ray.remote wrappers
      export.py                # checkpoint -> .onnx + metadata
    train.py                   # driver loop (python_binary main)
    configs/{local,aws}.yaml
    infra/{Dockerfile,cluster.yaml}
    tests/
  bot_client/                  # created in Phase 2 (2c), model support in 3g — deps: //cli_server:protocol, //cli_client (net), //arena, //rl_env, onnxruntime
    bot_client/
      main.py                  # --server-port --port --name --stack --bot rule|... | --model x.onnx
      players.py               # BotPlayer (arena bot) / OnnxPlayer (encoder + onnxruntime)
    tests/
```

The rule that keeps this manageable: `rollout.py`, `ppo.py`, `league.py` and
`arena/match.py` are plain Python classes you can call in a unit test. `actors.py` only
wraps them. A bug then shows up first in a fast `buck2 test`, not in a cluster log.

Dependency direction: `entities ← engine ← rl_env ← arena ← trainer`, and
`bot_client ← {cli_server:protocol, cli_client, arena, rl_env}`. `//rl_env` must **not**
depend on torch, so arena, the bot client and the encoder tests stay light.

---

## Phase 0: tooling and learning the loop

**Goal:** a working Python + buck2 setup with third-party packages, and a known-good PPO
taken apart on a toy problem before you write your own.

### 0a. Python and buck2

Done (2026-11-17). What was set up:

- **Python 3.14.** torch 2.15, ray 2.60 and onnxruntime 1.32 all publish cp314 wheels for
  macOS arm64 and Linux x86_64, so the venv uses the system Python 3.14 rather than 3.12.
  Re-check wheel availability for all three before moving to a newer Python.
- **One venv for the repo**, `plo8/.venv` (in `.gitignore`):
  `python3.14 -m venv .venv && .venv/bin/pip install -r requirements-rl.txt`.
  `requirements-rl.txt` pins exact versions of numpy, torch, ray[default], onnx,
  onnxruntime, boto3, PyYAML, tensorboard, and gymnasium (Phase 0 only).
- **buck2 runs the venv interpreter.** The prelude's `system_demo_toolchains()` does *not*
  use `python3` from `PATH`: its Python toolchain is `remote_python_toolchain`, which
  downloads a standalone CPython 3.13, so activating a venv has no effect. `toolchains/BUCK`
  therefore replaces the demo macro with an explicit, minimal set: `system_python_toolchain`
  and `system_python_bootstrap_toolchain` with `interpreter = ".venv/bin/python"`, plus
  `python_wheel`, `cxx`, `genrule`, `remote_test_execution` and `test`. Run `buck2 kill`
  after recreating the venv. If a future project needs a language the demo set covered
  (Rust, Go, Java, ...), add that toolchain to `toolchains/BUCK` explicitly.
- `entities`, `engine`, `cli_server` and `cli_client` still pass `buck2 test //...` on this
  interpreter and import no third-party packages.
- Third-party packages are **not** modelled as buck2 targets in v1. The venv is the
  contract, and `requirements-rl.txt` is the single place versions are pinned. If you
  later want hermetic builds, wrap wheels with `prebuilt_python_library`.

### 0b. Learn the loop

- PyTorch "Learn the Basics" (tensors, `nn.Module`, the optimizer loop).
- Spinning Up: "Intro to Policy Optimization" and the PPO page. Keep the intuition.
- Clone CleanRL and run `ppo.py --env-id CartPole-v1` (it also needs `tyro`; install it
  outside the repo venv, e.g. `pip install --target <scratch>/deps tyro` and
  `PYTHONPATH=<scratch>/deps`). Watch `charts/episodic_return` climb towards 500 in
  TensorBoard.
- CleanRL targets gymnasium 0.29; with gymnasium 1.x it trains but logs no returns.
  Pass `autoreset_mode=gym.vector.AutoresetMode.SAME_STEP` to `SyncVectorEnv` (keeps
  CleanRL's done handling correct) and read episode stats from
  `infos["final_info"]["episode"]["r"][i]` for each `i` where
  `infos["final_info"]["_episode"][i]` is true.
- Read `ppo.py` top to bottom and break it on purpose, one change at a time: learning
  rate ×10, clip 10, entropy coefficient 0. These are the failure modes you'll later
  diagnose in poker. On CartPole none of the three visibly breaks the return curve (all
  reach ~500 by 500k steps); the signatures show only in the diagnostics: lr ×10 gives
  ~19× the `approx_kl` and a non-zero `clipfrac`, clip 10 never clips, entropy 0 lets
  entropy fall fastest. Watch those, not just reward.

| CleanRL `ppo.py` | plo8 |
|---|---|
| `SyncVectorEnv` | `rl_env.TableBatch` (hundreds of step-mode engine tables) |
| `agent.get_action_and_value` | `NeuralPolicy.act` on workers; `CriticNet` on the learner |
| rollout loop over `num_steps` | `RolloutWorker.collect` |
| GAE | `advantage = return − critic(obs, all cards)` (a hand is short) |
| clipped surrogate + value loss + entropy | `trainer/ppo.py: ppo_loss` |
| `update_epochs`, `num_minibatches` | same names in `configs/*.yaml` |

Phase 0 commits only tooling (`requirements-rl.txt`, `.gitignore`, `toolchains/BUCK` if
you change it). It doesn't create any new project folder or `spec.md`.

**Done when:** CartPole reaches about 500, you can say what each block of CleanRL's
`ppo.py` does, `buck2 test //...` passes with the venv interpreter, and
`python -c "import torch, ray, numpy"` works in that interpreter.

---

## Phase 1A: engine readiness (`//engine` only)

**Goal:** an engine that can be driven one decision at a time and is fast enough to
train on. Phase 1A only touches `//engine`. No RL project exists yet, and
`cli_server`, `cli_client` and the e2e cases are the regression net.

It's done as two separate engine spec changes, each with its own plan and
implementation: first step mode (1A.1), then speed (1A.2).

### 1A.1 Step mode (engine spec change #1)

The engine already has every rule. What it lacks is a way for the caller to drive a hand
one decision at a time. Today `Dealer.play_hand()` loops through
`_play_streets → _betting_round → _take_turn → agent.on_turn(...)` and only returns at
the end of the hand.

Proposed addition to `engine/spec.md` (wording for the `[SPEC]` commit):

> The environment can also play a hand in **step mode**. Instead of calling the acting
> player's `on_turn` callback, it stops at each decision and hands it to the caller: which
> player is to act, their legal actions and, on request, their table view. It continues
> when the caller submits that player's action. Callback mode and step mode produce
> identical events, stacks and results for the same deck, seating and actions. Step mode
> is used to run many tables in one process (reinforcement learning).

Implementation sketch, for the planner:

- Turn the hand flow into a generator. `_play_streets`, `_betting_round` and `_take_turn`
  become generators (`yield from`). `_take_turn` yields a `TurnRequest(player, legal)`
  and receives the `Action` through `send()`. The rest (`_apply`, events, `PlayerActed`,
  showdown, `_settle`, `_end_hand`) is unchanged.
- `play_hand()` (callback mode) becomes a thin driver. It runs the generator, and for
  each `TurnRequest` it builds the view, calls `agent.on_turn` exactly as today and sends
  the action back. Timeouts, leave-as-fold and `IllegalActionError` handling stay in this
  driver, so the server's behaviour doesn't change.
- Public step API on `Environment` (as built):

  ```python
  d = env.start_hand()        # -> Decision | None. Like start_round(), but stops at the first decision
  while d is not None:
      d.seat, d.legal          # Decision(hand_number, seat, name, player, legal)
      view = env.view()        # TableView of the acting player, built only when asked (lazy)
      d = env.step(ActionType.CALL)   # Action or ActionType; runs to the next decision or the end
  result = env.last_result    # HandResult; end-of-hand processing (on_hand_end, button move) done
  ```

  `env.decision` is the pending decision. An illegal action in `step` raises
  `IllegalActionError` and leaves the hand
  unchanged. The engine never sits a player out for this in step mode, because the
  caller is in control.
- Locks: the generator must not yield while holding the table lock.
- Lazy views: in callback mode, `on_hole_cards` currently builds a `TableView` per
  participant even when the agent has no `on_hole_cards` (the arguments to `_notify` are
  evaluated eagerly). Build the view only when the callback exists.

Tests (`engine/tests/test_step.py`):
- For random seeds, seatings and action scripts (2–10 players, with all-ins), callback
  mode and step mode produce identical event streams (ignoring nothing but object
  identity), identical `HandResult`s and identical final stacks.
- The step-mode legal actions equal the ones callback mode passes to `on_turn`.
- An illegal `step` raises and doesn't change the state.
- `start_hand` during a hand raises (`HandInProgressError`). A hand where no decision is
  possible (for example both blinds all-in) returns `None` from `start_hand` with a
  complete result.
- Everything existing still passes: `buck2 test //engine/...`, plus
  `//cli_server` and `//cli_client` tests and `cli_server --e2e_test` (all 26 cases).

(Fallback if you want training to start before the refactor: run each table's
`play_hand()` in a greenlet and switch back to the scheduler from `on_turn`. This needs
no engine change but adds a third-party dependency and hides the control flow. Prefer
the generator.)

### 1A.2 Speed (engine spec change #2)

Measured baseline (this Mac, Python 3.14, random agents, callback mode, no encoder):

| Players | Hands/s | Decisions/s | Showdown rate |
|---|---|---|---|
| 2 | 4,000 | 27,600 | 41% |
| 6 | 1,230 | 32,000 | 80% |
| 9 | 800 | 30,900 | 89% |

At 6-max, `best_omaha` (≈36%) and `table_view` (≈33%) dominate. In order:

1. **Benchmark first:** `engine/bench/bench.py` (`//engine:bench`, a `python_binary`,
   standard library only). It plays random legal actions on 512 tables in step mode in
   one process and reports hands/s and decisions/s at 2, 6 and 9 players. Run it under
   py-spy. Record the numbers in `engine/plan.md` before and after each change.
2. **Lazy views** (mostly already done by 1A.1): step mode builds a view only when
   `env.view()` is called, and callback mode only for callbacks that exist.
3. **A faster evaluator** inside `//engine`, keeping `rank_high` / `rank_low` /
   `best_omaha` and their tests as the oracle. Work on card indices instead of
   `Card`/`Counter`/enum hashing, use a rank bitmask for lows, and reuse the 10 board
   triples across all players at showdown. Add an exhaustive check against the current
   implementation on random hands.
4. Only if still too slow: a native evaluator (Rust + PyO3) behind the same functions.
   buck2 can build Rust, but the Python extension packaging adds work, so do it last
   (or defer it to Phase 7).

Target: **at least 2,000 hands/s per core at 6-max**, engine-only, in step mode.

### 1A.3 Tests and gate

- `engine/tests/test_step.py` (above) and the evaluator oracle check.
- Determinism: the same deck seed, seating and actions give an identical event stream
  in step mode.
- Fuzz: 100k random step-mode hands at random table sizes (2–10) and stacks, with no
  exceptions and `sum(HandResult.net) == 0` per hand (no rake). A million overnight.
- All existing suites: `buck2 test //entities/... //engine/... //cli_server/... //cli_client/...`
  and the `e2e-testing` skill (`cli_server --e2e_test`, all 26 cases).

**Done when:** all of the above pass, and the benchmark meets the target or the gap is
written down in `engine/plan.md` with the next step chosen.

---

## Phase 1B: the RL environment (`//rl_env`, new project)

**Goal:** a fast, well-tested layer between the engine and everything else. Mistakes
here are invisible later: the bot still trains, it just learns the wrong thing. This
phase creates `rl_env/` with its first `spec.md` and `plan.md`, which cover the table,
the observation, `TableBatch`, the `Controller` protocol, the training records and the
benchmark. It depends only on
`//engine` (step mode from 1A) and numpy from the venv.

### 1B.1 Actions and one table (`rl_env/actions.py`, `rl_env/table.py`)

The only modules in the RL stack that drive the engine. A `Table` is one engine
`Environment` in step mode, rebuilt for each hand. (`rl_env/plan.md` is the authoritative
design; the code below shows its shape.)

```python
# rl_env/actions.py
FOLD, CHECK_CALL, BET_RAISE = 0, 1, 2
N_ACTIONS = 3

def action_mask(legal: LegalActions) -> tuple[bool, bool, bool]:
    # Folding when you could check is never correct, so it is masked out.
    return (not legal.can_check, True, legal.can_raise)

def mask_array(legal: LegalActions) -> np.ndarray: ...       # the same as a bool row, shape (3,)

def to_engine(action: int, legal: LegalActions) -> ActionType:
    # FOLD -> FOLD, CHECK_CALL -> CHECK or CALL, BET_RAISE -> RAISE.
    # An action outside the mask raises IllegalMaskedActionError before the engine is touched.

# rl_env/table.py
class Table:
    def __init__(self, chips_per_bb: int = 100): ...
    def start_hand(self, n_seats, stacks_bb, button, deck_seed, *, deck=None) -> bool:
        self.env = Environment(
            players=[Player(f"s{i}", stacks_bb[i]) for i in range(n_seats)],
            button=button, chips_per_bb=self.chips_per_bb,
            seed=deck_seed,                 # deal order is an engine contract -> duplicate-safe
            deck=deck,                      # tests / recorded-hand replays only
            action_timeout=None,
        )
        return self.env.start_hand() is not None   # False: the hand needed no decision
    # during a hand
    decision_pending: bool; seat_to_act: int; legal: LegalActions; mask: tuple[bool, bool, bool]
    def view(self) -> TableView: ...                # env.view(), only when asked
    def observation(self, out=None) -> np.ndarray: ...   # training path, no view built (1B.2)
    def act(self, action: int) -> None: ...         # to_engine + env.step()
    # after the hand
    hand_over: bool; result: HandResult; net_bb: np.ndarray; rewards: np.ndarray   # rewards = net_bb / 2
```

Notes:
- A fresh `Environment` per hand is the simplest way to vary table size and stacks. It
  needs no rebuys, sit-outs or joins. The benchmark profile shows its cost; if it goes
  above ~10%, reuse one environment per table size.
- Stacks (`rl_env/stacks.py`: `StackConfig`, `draw_stacks`, `HandSpec`, `HandSampler`):
  80% of hands start everyone at 100 BB (50 big bets, effectively deep in limit); the
  rest give each seat a random stack from 5 to 150 BB, so all-ins and side pots appear in
  training. `HandSampler(seed).sample()` draws size, stacks, button and deck seed.
- Seats are engine list indices `0..n-1`, clockwise. `button` is any of them.
- Zero-sum is checked in chips, never on floats.

Engine spec change #3 (small, requested by `rl_env/plan.md` and already in
`engine/spec.md`): read the current hand's public events and the acting player's hole
cards **without** building a view, a separate accessor for every participant's hole
cards (critic only), exact starting stacks at any chip unit (recorded-hand replays), and
the engine benchmark's workload as a library function (`//engine:bench_lib`).
`cli_server/spec.md` makes the `:e2e` library (recorded hands, stacked deck) usable by
other projects' tests.

### 1B.2 The observation (`rl_env/tracker.py`, `rl_env/hints.py`, `rl_env/encoder.py`, `rl_env/critic.py`)

`encode_view(view: TableView, legal: LegalActions) -> np.ndarray` (float32, `OBS_DIM`).
It reads only the engine's `TableView` (its `history`, `seat`, `hole_cards`) and the
legal actions, so the bot client can call it on a view decoded from `cli_server`
protocol messages. Training uses `Table.observation()`, which builds the **same bytes**
without a view: a `HandTracker` is fed only the engine events added since the
previous decision. A test checks the two paths are byte-identical at every decision.

Seats are renumbered relative to the acting player over the hand's participants: slot 0
is me, slot 1 the next participant clockwise, and so on. Empty slots stay zero. Amounts
are in big bets (BB / 2), scaled and clipped; every value is in [0, 2].

| Block | Size | Encoding |
|---|---|---|
| My hole cards | 52 | multi-hot over `Card.index` |
| Board, by street | 156 | flop / turn / river, 52 each |
| Street | 4 | one-hot |
| Players dealt | 9 | one-hot of 2..10 |
| My position | 10 | offset from the button, one-hot |
| Betting state | 17 | pot, to call, pot odds, bets this street one-hot 0–4+ (heads-up is uncapped), my chips in pot (total, this street), players still in, players able to act, **my stack**, can raise, can check, cap lifted, raises left |
| My action history | 6 | my bets per street (4), last aggressor this street, last aggressor before |
| Each other seat (9 slots) | 135 | 15 each: occupied, still in, all-in, chips in pot, chips this street, bets per street (4), last aggressor (this street, before), acts after me, has acted this street, **stack**, is button |
| Last 8 actions | 144 | 18 each: present, who (slot, 10), what (fold / check-call / bet-raise), street (4) |
| Hand-strength hints | 21 | from `engine.best_omaha(hole, board)` (flop onwards): high category one-hot (9), has qualifying low, highest low card one-hot 5–8 (4), low strength; on every street: distinct A–8 ranks in my hand one-hot 0–4 (5), has A+2 |
| **Total** | **554** | `OBS_DIM`, plus `ENCODER_VERSION = 1`; offsets in `OBS_LAYOUT` and `rl_env/plan.md` |

- The hand-strength hints aren't strictly needed, but they make early learning much
  faster and the evaluator gives them almost for free. On the river `best_omaha` costs
  about 125 µs, more than the rest of the decision. They are cached per
  `(table, seat, street)` and use the faster evaluator from 1A.2.
- The critic's extra input: `critic_cards(table, seat)` gives the other seats' hole
  cards as a `(9, 4)` int8 array of `Card.index` in the observation's slot order,
  `NO_CARD = 52` for empty slots. `expand_critic` turns it into a `(PRIV_DIM = 468)`
  multi-hot (9 slots × 52), the reference for the learner's expansion. `encoder.py` and
  `tracker.py` never import `critic.py`.
- The encoder writes sparsely into preallocated per-group arrays. It runs once per
  decision in Python, so it can easily cost more than the engine. The benchmark
  measures it.

### 1B.3 `TableBatch`, the `Controller` protocol and the `Recorder` (`rl_env/batch.py`, `rl_env/recorder.py`)

```python
# rl_env/batch.py
@dataclass
class DecisionBatch:                       # one per controller per step
    key: Hashable                          # controller key
    tables: np.ndarray; seats: np.ndarray  # int32, (B,)
    obs: np.ndarray                        # float32, (B, OBS_DIM)
    mask: np.ndarray                       # bool, (B, 3)
    legal: list[LegalActions]
    views: list[TableView] | None          # only built when the controller wants_views
    @staticmethod
    def single(view, legal, key=None) -> "DecisionBatch": ...   # one engine on_turn / server turn

class Controller(Protocol):
    wants_views: bool
    def act(self, batch: DecisionBatch) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """One action per row; optionally also each chosen action's log-probability.
        Neural policies read obs/mask; rule bots read views/legal."""

class TableBatch:
    def __init__(self, n_tables, *, chips_per_bb=100, recorder: Recorder | None = None): ...
    def start_hand(self, table, spec: HandSpec, controllers: Sequence[Hashable]) -> int: ...  # -> hand_id
    def pending(self, controllers) -> list[DecisionBatch]: ...  # grouped by controller, <= 1 row per table
    def submit(self, batch: DecisionBatch, actions, logp=None) -> None: ...
    def step(self, controllers: Mapping[Hashable, Controller]) -> list[FinishedHand]: ...
    def collect_finished(self) -> list[FinishedHand]: ...
    # FinishedHand(table, hand_id, spec, controllers, net_bb, rewards, result)
    def idle_tables(self) -> list[int]: ...
    def any_in_hand(self) -> bool: ...

# rl_env/recorder.py
class Recorder:
    def __init__(self, keys=None): ...     # record only these controller keys (e.g. {"learner"})
    n_ready: int; n_pending: int
    def drain(self) -> TrainingRecords: ...
    # columns: table, hand_id, seat, controller, obs, mask, action, logp, critic, reward
    # rows are released only when their hand finishes, with reward = that seat's net / 2
```

`rl_env/controllers.py` has only `RandomController(seed)`, for rl_env's own bench, fuzz
and tests. The baseline bots belong to arena (Phase 2).

### 1B.4 Speed with the encoder (`rl_env/bench/bench.py`, `buck2 run //rl_env:bench`)

The engine's own speed was handled in 1A.2. This benchmark measures what a rollout worker
will actually pay per decision: `TableBatch` over 512 tables, with the observation and
mask built for every decision and `RandomController` (random actions, no network). It
reports decisions/s at 2, 6 and 9 players next to the engine-only number from
`//engine:bench_lib`, so you can see the encoder's share.

Target before Phase 6: **at least 5,000 decisions/s per core with the encoder** (the
network adds little per decision when batched). Every 2× here roughly halves the AWS
bill. If the encoder dominates, optimise it (preallocated arrays, cached hints, fewer
Python loops over the hand's events) before touching the engine again.

### 1B.5 Tests (`//rl_env/tests`)

- **Determinism:** the same seeds give an identical `TableBatch` run (actions,
  results).
- **Fuzz through the batch:** 100k random hands at random table sizes and stacks via
  `TableBatch`, with no exceptions and nets summing to 0 in chips.
- **Legal mask:** matches `LegalActions`; fold is masked exactly when check is free;
  `to_engine` never produces an action the engine rejects; a masked action raises and
  changes nothing.
- **Leakage (the most important test):** swap opponents' hole cards, keep everything
  public the same, and the observation must be byte-identical on both paths
  (`Table.observation()` and `encode_view`).
- **Training path = view path:** at every decision, `Table.observation()` equals
  `encode_view(table.view(), legal)` byte for byte.
- **Rotation:** the same situation seen from different physical seats encodes identically.
- **Shape and range:** `OBS_DIM` is constant; every value is finite and in [−1, 2]
  (the layout uses [0, 2]).
- **Hand hints:** known hands give the expected categories and lows (reuse cases from
  `engine/tests/test_evaluator.py`).
- **Recorded hands:** replay every `cli_server/e2e_test_sets/*.json` case through
  a `Table`. Build the deck with the e2e `deal_script` / `StackedDeck` and play the
  recorded actions. Final stacks and pot splits must match the recording. This reuses 26
  hand-checked hands, including side pots, quartered lows and odd chips.
- **Server path:** for every decision in a few thousand fuzz hands,
  `encode_view(view, legal) == encode_view(decode_table_view(encode_table_view(view)), decode_legal(encode_legal(legal)))`,
  using `cli_server.protocol`. This replaces the original plan's
  `seat_view_from_server` parity test.
- **Training records:** every recorded decision is matched to exactly one finished hand;
  its reward is that seat's net in big bets; `logp` is kept when the controller returns it.

**Done when:** all `rl_env` tests pass (and the 1A suites still do), and the benchmark
numbers are recorded in `rl_env/plan.md`.

---

## Phase 2: baseline bots and the evaluator (`//arena`)

**Goal:** fixed opponents and a trustworthy way to measure "bot A beats bot B by X
bb/100". You need this before any training, or you can't tell learning from noise.

This phase creates `arena/` (2a, 2b). Once arena passes its tests, it creates
`bot_client/` (2c) as a second, separate spec. `//trainer` doesn't exist yet.

### 2a. The bots

Every bot implements the `Controller` protocol (batched, for training and evaluation)
**and**, through `arena.agent.BotAgent`, the engine `Agent` protocol (`on_turn`), so
the same bot can sit at `cli_server`. `BotAgent.on_turn(player, view, legal)` calls the
bot with `rl_env`'s `DecisionBatch.single(view, legal)` and submits `to_engine(...)`. Rule
bots set `wants_views = True`; RandomBot can wrap `rl_env`'s `RandomController`.

| Bot | Plays | Why you want it |
|---|---|---|
| RandomBot | uniformly random legal action (fold masked when check is free) | weakest possible opponent; any learner must crush it |
| CallingStation | always checks or calls | catches bots that only win by bluffing |
| Maniac | bets or raises whenever allowed | catches bots that fold too much under pressure |
| RuleBot | tight-aggressive: a preflop score decides raise/call/fold; after the flop it raises strong highs and made nut lows (`engine.best_omaha`), calls good draws, folds the rest | a sensible human-like baseline; your own O8 experience is the spec |

Keep RuleBot to about 100 lines. It's a yardstick, not a product.

### 2b. The evaluator (`arena/match.py`)

`play_match(controllers, table_size, n_deals, seed)` plays duplicate. For each deal it
replays the same deck seed `table_size` times, with the same button, rotating the lineup
by one seat each time. Every controller holds every seat's cards once. Because the
engine's deal order is a fixed contract, the same seed gives the same cards per seat
whatever the actions are, so this works without engine changes.

Report per controller and per table size:
- **bb/100** = 100 × mean result per hand in big bets.
- **95% CI** = ±1.96 × std / √(number of deals), with each rotation group counted as
  one sample.
- About 20 **sample hand histories** per match as readable text, built from
  `HandResult.history` (the engine's events), using `Card`/`Board` `__str__`.

Keep playing until the interval is narrower than the difference you care about.

### 2c. RuleBot at the real table (`//bot_client`, new project, first version)

`buck2 run //bot_client:bot_client -- --server-port 8008 --port 9101 --name RuleBot --bot rule`.
It joins `cli_server` and answers `turn` messages by decoding `TableView` +
`LegalActions` with `cli_server.protocol` and calling the bot. On `hand_end` it rebuys
up to its starting stack. Reuse `cli_client.net` (`ServerConnection`,
`CallbackListener`) for the sockets. Then start `cli_client` and play against it.

### Tests and sanity checks

- RandomBot against copies of itself gives about 0 bb/100, with a CI that includes 0.
- A deterministic bot (CallingStation, Maniac, RuleBot) against copies of itself in
  duplicate gives **exactly 0** per rotation group. This catches seat-assignment bugs.
- RuleBot beats RandomBot and CallingStation 6-handed, with a CI that excludes 0.
- The same deal seed gives the same hole cards per seat across rotations (a direct
  check of the deal-order contract from the arena side).
- `bot_client --bot rule` plays 200 hands at `cli_server` against scripted clients
  without errors (in-process `GameServer`, as in `cli_client/tests/test_integration.py`).

**Done when:** those pass, and the evaluator prints a results table
(agent, opponents, players, hands, bb/100, 95% CI).

---

## Phase 3: first learning bot, one process (`//trainer`)

**Goal:** a network that learns to beat the Phase 2 bots, in a single Python process
with no Ray. Training against fixed bots gives a reward that means something, so bugs
show up as "it isn't improving".

This phase creates `trainer/` with its first `spec.md`, which covers the networks,
`NeuralPolicy`, the rollout loop, PPO and the single-process driver. Ray (Phase 4), the
league (Phase 5) and AWS (Phase 6) are later spec changes to the same project.

### 3a. The networks

```python
# trainer/nets.py
import torch, torch.nn as nn
from rl_env.encoder import OBS_DIM
from rl_env.critic import PRIV_DIM

def mlp(inp, hidden=512, layers=3):
    mods, d = [], inp
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    return nn.Sequential(*mods)

class PolicyNet(nn.Module):                      # sees only the seat's view
    def __init__(self, obs_dim=OBS_DIM, n_actions=3):
        super().__init__()
        self.body, self.head = mlp(obs_dim), nn.Linear(512, n_actions)
    def forward(self, obs, mask):                 # mask: True = legal
        return self.head(self.body(obs)).masked_fill(~mask, -1e9)

class CriticNet(nn.Module):                      # also sees everyone's cards
    def __init__(self, in_dim=OBS_DIM + PRIV_DIM):
        super().__init__()
        self.body, self.head = mlp(in_dim), nn.Linear(512, 1)
    def forward(self, obs_priv):
        return self.head(self.body(obs_priv)).squeeze(-1)
```

The mask uses −1e9, not −inf, so the probability math never produces NaN.

### 3b. Acting on the workers

```python
# trainer/policy.py — an rl_env Controller
class NeuralPolicy:
    wants_views = False                           # reads obs/mask only: no views are built
    def __init__(self):
        torch.set_num_threads(1)                  # one core per worker
        self.net = PolicyNet().eval()
    def load(self, weights): self.net.load_state_dict(weights)
    @torch.no_grad()
    def act(self, batch: DecisionBatch):
        dist = torch.distributions.Categorical(
            logits=self.net(torch.from_numpy(batch.obs), torch.from_numpy(batch.mask)))
        a = dist.sample()
        return a.numpy(), dist.log_prob(a).numpy()   # the Recorder stores logp for PPO
```

### 3c. The play loop

```python
# trainer/rollout.py
class RolloutWorker:
    def collect(self, learner_weights, league_view, n_decisions):
        # self.tables = TableBatch(n_tables, recorder=Recorder(keys={"learner"}))
        self.learner.load(learner_weights)
        controllers = self.controllers(league_view)          # key -> Controller ("learner", bots, snapshots)
        rec = self.tables.recorder
        while rec.n_ready + rec.n_pending < n_decisions or self.tables.any_in_hand():
            if rec.n_ready + rec.n_pending < n_decisions:    # stop dealing once we have enough
                for i in self.tables.idle_tables():
                    spec, lineup = sample_lineup(league_view, self.sampler, self.rng)  # HandSpec, key per seat
                    self.tables.start_hand(i, spec, lineup)
            # pending -> ONE batched act() per controller -> submit; the Recorder keeps the
            # learner's obs, mask, action, logp and critic cards, and attaches each seat's
            # reward (net BB / 2) when its hand finishes
            for hand in self.tables.step(controllers):
                self.stats.add(hand)                          # learner's result vs bots, etc.
        return self.tables.recorder.drain(), self.stats.flush()   # TrainingRecords: finished hands only
```

In Phase 3, `sample_lineup` picks a table size from 2 to 10 (the engine's `MAX_PLAYERS`),
puts the learner in one to three seats, fills the rest with random arena bots, picks a
random button and a fresh deck seed, and draws stacks as in 1B.1 (`HandSampler`).

### 3d. The learner and the PPO loss

```python
# trainer/ppo.py
def ppo_loss(policy, critic, d, priv, adv, idx, cfg, magnet=None):
    logits = policy(d.obs[idx], d.mask[idx])
    dist = torch.distributions.Categorical(logits=logits)
    ratio = torch.exp(dist.log_prob(d.action[idx]) - d.logp[idx])   # new prob / old prob
    a = adv[idx]; a = (a - a.mean()) / (a.std() + 1e-8)
    pg = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
    v_loss = 0.5 * (critic(priv[idx]) - d.reward[idx]).pow(2).mean()
    ent = dist.entropy().mean()
    loss = pg + cfg.vf_coef * v_loss - cfg.ent_coef * ent
    if magnet is not None:                                       # Phase 5
        with torch.no_grad():
            m = torch.log_softmax(magnet(d.obs[idx], d.mask[idx]), -1)
        lp = torch.log_softmax(logits, -1)
        loss = loss + cfg.magnet_coef * (lp.exp() * (lp - m)).sum(-1).mean()
    return loss, dict(pg=pg.item(), v_loss=v_loss.item(), entropy=ent.item(),
                      approx_kl=((ratio - 1) - ratio.log()).mean().item(),
                      clipfrac=((ratio - 1).abs() > cfg.clip).float().mean().item())

class Learner:
    def update(self, *batches):
        d = to_tensors(concat(batches))
        priv = torch.cat([d.obs, expand_critic(d.critic)], dim=1)   # torch twin of rl_env.critic.expand_critic
        with torch.no_grad():
            adv = d.reward - self.critic(priv)                       # luck subtracted out
        for _ in range(self.cfg.epochs):
            for idx in shuffled_minibatches(len(d.reward), self.cfg.minibatch):
                loss, info = ppo_loss(self.policy, self.critic, d, priv, adv, idx, self.cfg,
                                      magnet=self.magnet)
                self.opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params, self.cfg.max_grad_norm)
                self.opt.step()
        return info
```

### 3e. The Phase 3 driver

```python
# trainer/train.py (single-process mode)
worker, learner = RolloutWorker(cfg, seed=0), Learner(cfg)
for it in range(cfg.iterations):
    batch, stats = worker.collect(learner.get_weights(), BOTS_ONLY, cfg.decisions_per_iter)
    info = learner.update(batch)
    log(it, stats, info)                                   # TensorBoard
    if it % 20 == 0: arena.play_match(...)                 # learner vs bots, duplicate
    if it % 50 == 0: save_checkpoint(learner, it)
```

Use `decisions_per_iter = 65,536` locally. Expect slow progress in one process. Phase 4
adds the parallelism.

### 3f. What healthy training looks like

| Metric | Healthy | If not |
|---|---|---|
| Learner's average result vs bots | starts negative, rises, goes positive | reward sign, seat mapping, the leakage test |
| Entropy | starts near 1.1 (ln 3), falls slowly | drops to 0 fast: raise `ent_coef` or lower lr |
| approx_kl per update | about 0.005–0.03 | much higher: lower lr or epochs |
| clipfrac | about 0.05–0.3 | near 0: too timid; near 0.5: too aggressive |
| Critic explained variance | rises well above 0 | stuck near 0: critic input or targets wrong |
| Decisions per second | stable | falls: rows piling up in the `Recorder` (`n_pending` should drop to 0 between iterations) |

### 3g. First export and play it yourself

Right after the first run that beats RandomBot, export the policy (see "Exporting and
playing through cli_server") and play it in `cli_client` via `//bot_client --model`.
Reading what the bot actually does catches problems no metric shows.

**Done when:** the evaluator shows the learner beating all four bots heads-up and
6-handed, with CIs that exclude 0, and its hand histories look sensible: it raises strong
scoop hands and folds trash.

---

## Phase 4: parallelise with Ray on your Mac

**Goal:** the Phase 3 run with one rollout worker per core through Ray. The same code then
runs on AWS with only a config change.

**Ray in five sentences.** `ray.init()` starts a local Ray runtime.
`ray.remote(SomeClass)` turns a class into an actor, a process that keeps its state
between calls. `actor.method.remote(...)` returns at once with an `ObjectRef`, a handle
to a future result, and `ray.get(ref)` waits for it. `ray.put(x)` stores `x` once in
shared memory, so passing the ref to 100 workers doesn't copy it 100 times.

```python
# trainer/actors.py
import ray
from trainer.rollout import RolloutWorker
from trainer.ppo import Learner
from trainer.league import League

RolloutWorkerActor = ray.remote(num_cpus=1, max_restarts=-1)(RolloutWorker)  # restart if it dies
LearnerActor = ray.remote(num_cpus=4)(Learner)     # num_gpus=1 on a GPU node
LeagueActor = ray.remote(num_cpus=0)(League)
```

Mark `RolloutWorker.collect` with `@ray.method(num_returns=2)` (big batch + small stats).
The decorator does nothing without Ray, so Phase 3 tests keep working.

```python
# trainer/train.py
ray.init()                                         # on AWS: ray.init(address="auto")
workers = [RolloutWorkerActor.remote(cfg, seed=cfg.seed + i) for i in range(cfg.num_workers)]
learner, league = LearnerActor.remote(cfg), LeagueActor.remote(cfg)
per_worker = cfg.decisions_per_iter // cfg.num_workers

for it in range(start_iter, cfg.iterations):
    w_ref = ray.put(ray.get(learner.get_weights.remote()))
    view = ray.get(league.view.remote())
    calls = [w.collect.remote(w_ref, view, per_worker) for w in workers]
    good, stats = [], []
    for batch_ref, stats_ref in calls:
        try:
            stats.append(ray.get(stats_ref))       # small; tells us the worker succeeded
            good.append(batch_ref)
        except ray.exceptions.RayActorError:
            pass                                   # worker died; Ray restarts it, we skip its share
    info = ray.get(learner.update.remote(*good))   # refs go straight to the learner
    log(it, stats, info)
```

**Ray and buck2.** Ray workers import modules by name in fresh processes, so they need the
repo on their import path. Locally, run `train.py` with the venv interpreter and set
`PYTHONPATH=entities:engine:rl_env:arena:trainer` (each project uses `base_module = ""`,
so `<project>/<package>/` imports as `<package>`). Or pass the same through Ray's
`runtime_env={"env_vars": {"PYTHONPATH": ...}}`. Don't rely on buck2's in-place
`python_binary` layout reaching Ray workers.

Local config (`configs/local.yaml`): `num_workers` = cores − 2, `tables_per_worker` = 256,
`decisions_per_iter` = 131,072, `minibatch` = 4,096. The Ray dashboard is at
http://127.0.0.1:8265.

Tests:
- With `num_workers = 1`, results match the Phase 3 driver for the same seed and settings.
- Throughput is close to `num_workers` × the Phase 3 decisions/s. Much lower means the
  learner or driver is the bottleneck, so time each step.
- `kill -9` one worker mid-run. Training continues and the worker comes back.

**Done when:** a Ray run reproduces the Phase 3 result several times faster in
wall-clock time.

---

## Phase 5: self-play with an opponent pool

**Goal:** the bot keeps improving after beating the hand-written bots, by playing itself
and its past versions. This is the phase that makes it strong, and the reason you need the
cluster.

### 5a. Who sits at each table

- Table size: start uniform over 2–10, then weight towards the sizes you care about.
- Per seat: about 50% current learner, 35% pool snapshot, 15% arena bot. Every table
  gets at least one learner seat, and every learner seat produces training data.
- The bots stay in the mix so the learner never forgets how to punish simple,
  exploitable play.

### 5b. The league

```python
# trainer/league.py
@dataclass
class Snapshot:
    id: str                 # "it00450"
    path: str               # local file, or s3://...
    weights_ref: object     # Ray ObjectRef, so workers don't all hit S3
    ema_result: float = 0.0 # learner's recent result vs this snapshot, big bets per hand

class League:
    def view(self):
        s = np.array([p.ema_result for p in self.pool])
        w = np.exp(-s / self.cfg.pfsp_temp)          # opponents we beat least get picked more
        w = np.maximum(w / w.sum(), self.cfg.min_prob)
        return [(p.id, p.weights_ref, q) for p, q in zip(self.pool, w / w.sum())]
    def record(self, per_opponent_results): ...      # update ema_result from worker stats
    def admit(self, snapshot): ...                   # add; if over max_pool, drop the oldest non-anchor
```

The pool starts with the final Phase 3 bot as `anchor-0`, never dropped. Workers keep the
last 16 opponent policies in an LRU cache keyed by snapshot id.

### 5c. Snapshots and the evaluator gate

Every 25 iterations the driver sends the current weights to an Evaluator actor (a Ray
wrapper around `arena.play_match`) as a candidate, without waiting. It polls with
`ray.wait([ref], timeout=0)` each iteration. The evaluator plays the candidate in
duplicate at 2, 6 and 9 players against the four arena bots, `anchor-0` and the five
most recently admitted snapshots. The candidate is admitted if it beats the most recent
snapshot with a CI above 0. Also admit one every 200 iterations regardless, so the pool
keeps moving.

### 5d. MMD regularisation

The learner keeps a frozen **magnet** (a copy of its policy, refreshed every 50
iterations) and adds `magnet_coef × KL(policy ‖ magnet)` to the loss (already in
`ppo_loss`). Start with `magnet_coef = 0.05`. If learning looks frozen, lower it. If
results against past snapshots swing up and down, raise it.

### 5e. What to watch

| Signal | Healthy |
|---|---|
| Learner vs itself | about 0 by definition; ignore |
| Learner vs pool (training stats) | positive, shrinking slowly as the pool improves |
| Evaluator: candidate vs anchor-0 and RuleBot | rises over hundreds of iterations, then flattens |
| Evaluator: candidate vs older snapshots | never significantly negative (a drop means forgetting: raise pool share or `magnet_coef`) |
| Entropy | falls slowly, stays clearly above 0 on the river, where bluffing lives |

**Done when:** in a long local run each admitted snapshot beats the one before it, the
result against `anchor-0` rises, and a round-robin of all snapshots shows newer beating
older.

---

## Phase 6: run on AWS

**Goal:** the Phase 5 run on a Ray cluster in AWS. Spot CPU machines play and one
on-demand GPU machine runs the driver, learner, league and evaluator. No training-code
changes, only config and infrastructure (`trainer/configs/aws.yaml`, `trainer/infra/`).

### 6a. Account preparation (start during Phase 3: quotas take days)

- `aws configure sso`; pick one region (eu-west-1 usually has more Spot capacity than
  eu-west-2; check both).
- AWS Budget with alerts at 50/80/100% of a monthly limit.
- Service Quotas: raise "All Standard Spot Instance Requests" vCPUs (for example 512) and
  "Running On-Demand G and VT instances" (at least 16).
- An S3 bucket `o8rl-<unique>` and an ECR repository `o8rl`.

### 6b. The machines

| Role | Instance | Market | Count |
|---|---|---|---|
| Head: driver, learner (GPU), league, evaluator | g6.4xlarge (16 vCPU, 1 L4, 64 GiB) | on-demand (losing it stops the run) | 1 |
| Rollout | c7a.16xlarge or c8a.16xlarge (64 vCPU, full cores) | Spot | 4 to start |

Keep everything on x86 for v1. Graviton rollout nodes are a Phase 7 optimisation.

### 6c. The image

The image carries the **whole monorepo** (the engine is not a pip package) and the pinned
RL requirements:

```dockerfile
# trainer/infra/Dockerfile — build context is the repo root
FROM rayproject/ray:<same-ray-version-as-requirements-rl.txt>-py314-gpu  # match the venv's Python; if no py314 image exists, use the nearest one and build the venv there
COPY --chown=ray:users requirements-rl.txt /home/ray/plo8/
RUN pip install --no-cache-dir -r /home/ray/plo8/requirements-rl.txt
COPY --chown=ray:users . /home/ray/plo8
ENV PYTHONPATH=/home/ray/plo8/entities:/home/ray/plo8/engine:/home/ray/plo8/rl_env:/home/ray/plo8/arena:/home/ray/plo8/trainer
```

Add a `.dockerignore` for `buck-out/`, `.venv/`, `.git/` and `__pycache__/`. Run the
pure-Python test suites inside the image once (`python -m unittest discover` per project)
as a smoke test, since buck2 isn't in the image. Build for x86 from the ARM Mac and push:

```
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin <acct>.dkr.ecr.eu-west-1.amazonaws.com
docker buildx build --platform linux/amd64 -f trainer/infra/Dockerfile \
  -t <acct>.dkr.ecr.eu-west-1.amazonaws.com/o8rl:$(git rev-parse --short HEAD) --push .
```

Tag every image with the git commit, so each run records exactly which engine, encoder
and trainer it ran. If a native evaluator makes the emulated build slow, build in CI
instead.

### 6d. The cluster file

Start from Ray's AWS examples (`python/ray/autoscaler/aws/`) and check fields against the
cluster YAML reference.

```yaml
# trainer/infra/cluster.yaml
cluster_name: o8rl
max_workers: 8
idle_timeout_minutes: 10
provider: {type: aws, region: eu-west-1, cache_stopped_nodes: false}   # terminate idle nodes
auth: {ssh_user: ubuntu}
docker:
  image: <acct>.dkr.ecr.eu-west-1.amazonaws.com/o8rl:<git-sha>
  container_name: ray_container
  pull_before_run: true
initialization_commands:
  - aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin <acct>.dkr.ecr.eu-west-1.amazonaws.com
head_node_type: head
available_node_types:
  head:
    node_config: {InstanceType: g6.4xlarge}
    resources: {CPU: 16, GPU: 1}
  rollout:
    min_workers: 4
    max_workers: 8
    node_config:
      InstanceType: c7a.16xlarge
      InstanceMarketOptions: {MarketType: spot}
    resources: {CPU: 64, rollout: 64}       # custom resource: only rollout workers go here
```

Create workers with `RolloutWorkerActor.options(resources={"rollout": 1})` and the learner
with `num_gpus=1`. On the Mac, `ray.init(resources={"rollout": n})` makes the same code
run. Make sure the node IAM role can read and write the bucket and pull from ECR
(`AmazonEC2ContainerRegistryReadOnly`).

### 6e. Running a job

```
ray up trainer/infra/cluster.yaml
ray dashboard trainer/infra/cluster.yaml          # http://127.0.0.1:8265
ray job submit --address http://127.0.0.1:8265 --no-wait -- \
    python /home/ray/plo8/trainer/train.py --config /home/ray/plo8/trainer/configs/aws.yaml --run-id r001
ray job logs --address http://127.0.0.1:8265 -f <job-id>
ray down trainer/infra/cluster.yaml
aws ec2 describe-instances --filters Name=tag:ray-cluster-name,Values=o8rl   # nothing left running
```

`configs/aws.yaml`: `num_workers` 256, `tables_per_worker` 128, `decisions_per_iter`
524,288, `minibatch` 16,384, `device` cuda, `checkpoint_every` 25 to
`s3://<bucket>/runs/<run-id>/`.

### 6f. Spot interruptions and resuming

Dead workers are already skipped (Phase 4), Ray restarts them and the autoscaler replaces
nodes. Every checkpoint includes policy, critic, optimizer, magnet, league index and
iteration, so `train.py --resume s3://<bucket>/runs/<run-id>/latest` continues exactly.
Test both on purpose: terminate a rollout instance mid-run, and resume after `ray down`.

### 6g. Monitoring and cost

Log to Weights & Biases (API key via environment variable) or TensorBoard on the head
through an SSH tunnel. The Ray dashboard shows per-node CPU and memory.

hours = decisions needed / (decisions/s per vCPU × rollout vCPUs × 3600), and cost =
hours × (rollout nodes × Spot price + head on-demand price). Take decisions/s from your
Phase 4 measurement, not from the original plan's guess. For example, at 3,000
decisions/s/vCPU on 256 vCPUs, a billion decisions take about 22 minutes of collection
plus learner time. Start with a few-hour budget and extend runs that are still improving.
The evaluator's curve against `anchor-0` flattening is your signal to stop.

**Done when:** a cluster run reproduces your best local result faster, survives a killed
Spot node, resumes from S3 after `ray down`, and leaves no instances running.

---

## Phase 7: scale up and harden

The upgrades are independent. Do each one only when its trigger shows up in your metrics.

| Upgrade | Do it when | What it involves |
|---|---|---|
| Exploiter agents | you want exploitability, not just "beats old versions" | freeze a snapshot, train a fresh learner only against it for a fixed budget; its bb/100 is the exploitability estimate, and it should shrink across snapshots; add good exploiters to the pool |
| Overlap play and training | the dashboard shows workers idle during updates | collect iteration t+1 with old weights while training on t (a lag of one iteration is fine for PPO); first step towards IMPALA |
| Faster engine | decisions/s per core is the main cost driver | native evaluator (PyO3) in `//engine` behind the same functions; then, only if needed, a native step-mode core behind the same `Environment` step API, with the Python engine kept as the reference that `test_step`, `e2e_test_sets` and fuzz compare against |
| Graviton rollout nodes | the Spot bill matters and everything builds on ARM | multi-arch image, a second node type (for example c8g) with the `rollout` resource; compare cost per million decisions |
| Bigger network | results plateau while critic explained variance is still low | wider MLP (1,024) or attention over the 10 seat slots; bump `ENCODER_VERSION` only if the observation changes |
| Table-size weighting | some sizes lag in the evaluator breakdown | raise their weight in `sample_lineup` |
| Rake | before real games with rake | a rules change in `//engine` (spec, plan, e2e cases with rake), then fine-tune a converged bot under it |
| Deep CFR comparison (heads-up) | heads-up exploiters keep finding large leaks | a Deep CFR baseline on the same step-mode engine; rollout workers become tree-traversal workers |

---

## Exporting and playing through `cli_server` (`//trainer` export, `//bot_client`)

A trained bot leaves the cluster as one `.onnx` file plus a metadata file. `//bot_client`
plays it at `cli_server` with only numpy, onnxruntime, the `rl_env` encoder and the
existing protocol code.

| | Checkpoint (`.pt`) | Export (`.onnx`) |
|---|---|---|
| Contains | policy, critic, optimizer, magnet, league index, iteration | policy only |
| Used by | `train.py --resume`, evaluator, opponent pool | `//bot_client`, anything outside training |
| Needs | torch + `//trainer` | onnxruntime + the same `ENCODER_VERSION` |

```python
# trainer/export.py
class Exportable(torch.nn.Module):
    def __init__(self, policy):
        super().__init__(); self.policy = policy
    def forward(self, obs, mask):                       # mask as 0/1 floats keeps ONNX simple
        return torch.softmax(self.policy(obs, mask > 0.5), dim=-1)

ckpt = torch.load(ckpt_path, map_location="cpu")
policy = PolicyNet(); policy.load_state_dict(ckpt["policy"]); policy.eval()
torch.onnx.export(Exportable(policy), (torch.zeros(1, OBS_DIM), torch.ones(1, 3)), out_path,
                  input_names=["obs", "mask"], output_names=["probs"],
                  dynamic_axes={"obs": {0: "batch"}, "mask": {0: "batch"}, "probs": {0: "batch"}})
```

Next to the model, write metadata: run id, iteration, git commit, `ENCODER_VERSION`,
`OBS_DIM`. The bot client refuses a model whose encoder version differs from its own. A
mismatched encoder doesn't crash; it just plays badly.

```python
# bot_client/players.py
from cli_server.protocol import decode_table_view, decode_legal
from rl_env.encoder import encode_view
from rl_env.actions import mask_array, to_engine

class OnnxPlayer:
    def __init__(self, model_path):
        self.sess = onnxruntime.InferenceSession(model_path); self.rng = np.random.default_rng()
    def on_turn(self, msg) -> str:                                   # msg: decoded `turn` message
        view, legal = decode_table_view(msg["view"]), decode_legal(msg["legal"])   # engine dataclasses
        obs  = encode_view(view, legal)[None]                         # SAME bytes as training
        mask = mask_array(legal)[None].astype(np.float32)
        probs = self.sess.run(["probs"], {"obs": obs, "mask": mask})[0][0]
        return to_engine(int(self.rng.choice(3, p=probs)), legal).value   # sample, don't argmax
```

The networking (join with name, stack and callback port; answering `turn` and `hand_end`
as requests on the server port) reuses `cli_client.net` and the message helpers in
`cli_server.protocol`. On `hand_end` the bot rebuys back to its buy-in, or leaves after a
configured number of hands. `buck2 run //bot_client:bot_client -- --server-port 8008 --port 9102 --name O8bot --model o8_r001_it04500.onnx`.

Sample from the probabilities at play time, as in training. Always taking the most likely
action removes the bluffing frequencies the bot learned.

Tests:
- ONNX and PyTorch outputs match (`np.allclose`) on 1,000 random observations.
- Protocol round trip: `encode_view` of an in-process `TableView` equals `encode_view` of
  the same view after `encode_table_view` → JSON → `decode_table_view` (from Phase 1B.5;
  repeat it here with a real model).
- The bot playing through an in-process `cli_server` `GameServer` against `bot_client --bot rule`
  scores the same bb/100 as `arena.play_match` reports for the same pairing, within the
  CI. This proves the server path matches training.

---

## Reference

### Starting hyperparameters

Reasonable first values, not tuned ones. Change one at a time, and judge every change with
the evaluator, not with the training curves.

| Parameter | Mac (Phases 3–5) | AWS (Phase 6) | Controls |
|---|---|---|---|
| decisions_per_iter | 131,072 | 524,288 | data per update |
| minibatch | 4,096 | 16,384 | decisions per gradient step |
| epochs | 3 | 3 | passes over each batch |
| lr (Adam) | 3e-4 | 3e-4 → 1e-4 | size of each nudge |
| clip | 0.2 | 0.2 | max probability change per update |
| ent_coef | 0.01 | 0.01 → 0.003 | pressure to stay unpredictable |
| vf_coef | 0.5 | 0.5 | weight of the critic loss |
| max_grad_norm | 1.0 | 1.0 | caps very large updates |
| Network | 3 × 512 | 3 × 512 | policy and critic size |
| magnet_coef / refresh | 0.05 / 50 it | 0.05 / 50 it | MMD (Phase 5 on) |
| Snapshot candidate | every 25 it | every 25 it | evaluator cadence |
| Seat mix learner / pool / bots | 50 / 35 / 15% | 50 / 35 / 15% | who the learner plays |
| pfsp_temp / min_prob | 1.0 / 0.02 | 1.0 / 0.02 | focus on toughest opponents |
| Max pool size | 50 | 100 | oldest non-anchor drops first |
| Stacks | 80% at 100 BB, 20% uniform 5–150 BB | same | exposure to all-ins and side pots |

### Debugging checklist

| Symptom | Likely cause | First thing to check |
|---|---|---|
| Can't beat RandomBot | reward sign, seat mapping, mask | print one hand's decisions with their returns (`HandResult.history`) |
| Beats bots fast, plays badly through `cli_server` | leakage, or server-path mismatch | leakage test; protocol round-trip test; `ENCODER_VERSION` |
| Entropy near 0 within a few iterations | lr too high, entropy bonus too low | lr, `ent_coef`, advantage normalisation |
| approx_kl spikes, then collapse | updates too large | lower lr or epochs |
| Critic explained variance stuck near 0 | critic input or targets wrong | privileged card expansion; returns in big bets (BB / 2) |
| NaN loss | −inf in the mask, zero std | the −1e9 mask, the +1e-8 |
| Decisions/s fall during a run | memory growth | `Recorder.n_pending`, opponent cache, engine `history` kept by finished tables |
| Results differ between engine and server path | engine change not mirrored in protocol | `cli_server --e2e_test`, protocol round trip |
| Self-play results swing | strategy cycling | raise pool share or `magnet_coef` |
| AWS slower than expected | workers idle during updates, or the reverse | dashboard CPU per node; Phase 7 overlap |

### Glossary

| Term | Meaning |
|---|---|
| Step mode | Engine mode where the caller receives each decision and submits the action (Phase 1A.1), as opposed to callback mode (`on_turn`) |
| Observation / encoder | fixed-length vector of what one seat can see / code that builds it from a `TableView` |
| Policy / logits / mask | function from observation to action probabilities / raw scores / zeroes illegal actions |
| Reward / return | chips won or lost in the hand, in big bets, given to every decision in it |
| Critic / advantage | predicts the expected return, sees all cards in training only / return minus prediction |
| PPO | push probabilities by advantage, capped per update |
| Iteration / epoch / minibatch | one collect-and-train cycle / one pass over its data / one chunk per gradient step |
| Entropy / KL | how random the policy is / how different two distributions are |
| On-policy | learns only from data played by the current policy |
| Self-play / league / anchor | training against itself / pool of frozen past versions / a pool member never dropped |
| MMD / magnet | regularisation towards a slowly refreshed copy of the policy |
| Exploiter | a bot trained only to beat one frozen snapshot |
| bb/100 | big bets won per 100 hands |
| Duplicate | same deck seed replayed with seats rotated, so card luck cancels |
| Checkpoint / ONNX | saved training state / portable policy file for onnxruntime |
| Ray actor / ObjectRef | long-lived remote process / handle to a future result |
| Spot instance | spare EC2 capacity, cheaper, can be reclaimed at short notice |

### Sources

- Original plan: https://claude.ai/artifact/CTQcB6Kv5v8eev3iYu9eX1
- Repo: `engine/spec.md`, `engine/plan.md` (deal-order contract, units, events),
  `entities/plan.md` (`Card.index`, `Deck.put_on_top`), `cli_server/plan.md`
  (protocol, e2e mode), `cli_server/e2e_test_sets/`
- CleanRL `ppo.py`; Spinning Up: Intro to Policy Optimization and PPO; PyTorch Learn the
  Basics; Ray Core walkthrough and cluster YAML reference; Sokota et al., magnetic mirror
  descent (MMD); Brown et al., Deep CFR
