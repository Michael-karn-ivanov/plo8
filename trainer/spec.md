Python library (`trainer`): the neural networks and, later, the training loop for a fixed-limit Omaha Hi-Lo 8-or-better bot.
It sits on top of plo8/rl_env (observation vectors, action masks, critic input, the table batch and the training records) and plo8/arena (the fixed opponent bots now; the evaluator in a later stage). The project grows in stages; this spec currently covers the first stages: the networks, the acting policy, the rollout loop that plays the learner against the arena bots and collects its training records, and the PPO learner that updates the policy and critic from those records. The single-process driver, Ray, the opponent league (including the MMD magnet term of the loss), AWS and ONNX export are later changes to this spec.
Mistakes here rarely crash: a wrong mask, a wrong input size or hidden information reaching the policy gives a network that trains and plays badly, or cheats. So the shapes, the masking and the separation between what the policy and the critic may see matter more than speed.

Dependencies
* plo8/rl_env: `OBS_DIM` and `ENCODER_VERSION` (the observation vector), `N_ACTIONS` and the action order fold / check-or-call / bet-or-raise, the action mask (True = legal), `PRIV_DIM` and `expand_critic` (the critic's extra input).
* plo8/arena: the four baseline bots (RandomBot, CallingStation, Maniac, RuleBot), created by name and seed.
* rl_env's table batch (`TableBatch`, the `Controller` protocol, `HandSampler` / `HandSpec` for table size, stacks, button and deck seed) and its `Recorder` / `TrainingRecords` (per decision: observation, mask, action, log-probability, critic input, and the seat's reward once the hand is over).
* torch and numpy, from the repo venv (requirements-rl.txt). No other third-party package for now.
* All poker rules stay in plo8/engine and all table driving and encoding stay in plo8/rl_env. trainer never re-implements the encoder or the critic-input expansion; it only consumes their arrays.
* Dependency direction: entities <- engine <- rl_env <- arena <- trainer. Nothing in engine, rl_env or arena may depend on trainer or on torch.

1. Networks
1.1 Policy network (`PolicyNet`)
* Input: a batch of observation vectors (float32, shape (B, `OBS_DIM`)) and a batch of action masks (bool, shape (B, `N_ACTIONS`), True = legal), exactly as rl_env produces them.
* Output: one logit per action (shape (B, `N_ACTIONS`)), in rl_env's action order.
* Illegal actions are masked by setting their logits to a large negative finite value (-1e9), never to -inf, so softmax, log-probabilities and entropy never produce NaN, also when only one action is legal. A softmax over the masked logits gives (numerically) zero probability to every illegal action.
* The policy sees only the observation and the mask: it never receives critic input or any other hidden information.
* Body: a multi-layer perceptron, by default 3 hidden layers of 512 units with ReLU, then a linear head. The hidden size and number of layers are constructor parameters, so later configs can change them.

1.2 Critic network (`CriticNet`)
* Input: a batch of observation vectors concatenated with the expanded critic input (rl_env's `expand_critic`, which shows the other seats' hole cards): float32, shape (B, `OBS_DIM` + `PRIV_DIM`). Only the critic ever receives this privileged part, and only during training; it is not used at play time.
* Output: one value per row (shape (B,)): the predicted result of the hand for the acting seat, in big bets (the unit of rl_env's rewards).
* Body: the same kind of multi-layer perceptron as the policy (by default 3 × 512, ReLU), then a linear head with one output. The policy and the critic share no weights.

1.3 Common requirements
* Input sizes default to rl_env's `OBS_DIM` and `PRIV_DIM`, so a change of the encoder changes the networks with it; the networks record the `ENCODER_VERSION` they were built for, so later checkpoints and exports can refuse a mismatched encoder.
* Both networks accept numpy arrays converted with `torch.from_numpy` from rl_env's outputs without copying or reshaping them (dtypes float32 and bool).
* Both networks work for any batch size including 1, on CPU. Construction is reproducible: with the same torch seed, two networks get the same initial weights.
* Weights round-trip through `state_dict` / `load_state_dict` (a network loaded from another's state dict gives identical outputs).

2. Acting policy (`NeuralPolicy`)
* `NeuralPolicy` is an rl_env `Controller` over a `PolicyNet`: it can sit at any number of seats of a `TableBatch` (as the learner, or later as a frozen past version of it) and decides a whole batch of decisions with one network call.
* It reads only the observation vectors and action masks of the decision batch; it declares that it needs no table views, so the batch never builds views for it. It never receives the critic input.
* For each decision it samples an action from the categorical distribution over the masked logits (it does not take the most likely action: mixed strategies and exploration need sampling). It returns the actions (one per row, in rl_env's action order) together with the log-probability of each chosen action under the current weights (float32), in the form the rl_env `Controller` protocol defines, so rl_env's `Recorder` stores them for PPO.
* Every returned action is allowed by its row's mask.
* Inference runs on CPU without gradient tracking, with the network in evaluation mode. By default it uses one torch thread, because each rollout worker owns one core; the thread count is a constructor parameter.
* Weights are loaded from a `PolicyNet` state dict (as produced by `state_dict()` of the learner's policy); a state dict built for another network config or `ENCODER_VERSION` is refused (section 1.3). Loading new weights replaces the old ones completely; the policy keeps no other state between batches.
* Sampling is reproducible: the policy has its own random generator, seeded when it is created (and re-seedable), independent of torch's global generator. The same seed, weights and decision batches give the same actions and log-probabilities.
* A batch of any size works, including a single decision (as in rl_env's `DecisionBatch.single`, used when a bot plays one decision at a time) and an empty batch.

3. Rollout loop (`RolloutWorker`)
* A `RolloutWorker` plays many hands side by side in one rl_env `TableBatch` (the number of tables is a constructor parameter, default 256) and returns the learner's training records. It is the piece that later runs once per core (Ray); here it runs in the caller's process, single-threaded.
* The learner is a `NeuralPolicy` owned by the worker. At the start of every collection the worker loads the learner weights it is given (a `PolicyNet` state dict, refused as in section 2 if it does not match), so every decision of one collection is made with exactly those weights.
* Opponents: in this stage the learner plays only against the arena bots. The set of opponent bots to draw from is a parameter (default: all four, `random`, `calling`, `maniac`, `rule`); each bot is created once per worker (random bots seeded from the worker's seed) and sits at any number of tables at once. Later stages (frozen snapshots of the learner, the league) add other kinds of opponents through the same mechanism: a controller key per seat.
* Lineups: every new hand gets a fresh lineup:
    * the table size is drawn uniformly from 2 to 10 (the range is a parameter);
    * the learner sits in 1 to 3 seats (uniform, but always leaving at least one opponent seat: heads-up it is exactly one seat), at random positions;
    * every other seat gets a bot drawn uniformly from the opponent set, independently per seat;
    * stacks, button and deck seed come from rl_env's `HandSampler` (its default stack distribution: mostly 100 big blinds, sometimes short).
* Collection: `collect(weights, n_decisions)` keeps starting hands on idle tables while fewer than `n_decisions` learner decisions have been recorded (finished or still in a running hand); once that many exist it starts no new hands but plays every running hand to its end, so the records only ever come from complete hands. It returns the learner's training records (rl_env's `TrainingRecords`: observation, mask, action, log-probability, critic input, reward in big bets) and the statistics below. The number of records is at least `n_decisions` and exceeds it by at most the decisions of the hands that were already running.
* Only the learner's decisions are recorded; the bots' decisions are not. Every recorded decision belongs to exactly one finished hand and carries that seat's reward (net result in big bets); no record of an unfinished hand is ever returned, and after `collect` returns nothing is left pending in the recorder or the tables, so consecutive collections are independent.
* All decisions of one controller pending at the same time are decided in one batched call (one network call for the learner per step), as rl_env's batch provides.
* Statistics per collection: hands played, learner decisions, the learner's total and mean result per hand in big bets (bb/100, counting all learner seats of a hand), broken down by table size and by opponent bot name (a hand counts for every bot name present at the table), and the wall time with decisions per second. These are for logging and the health checks of later stages; they are not an evaluation (that is arena's duplicate match).
* Reproducible: the worker has a seed; the same seed, weights and `n_decisions` give exactly the same lineups, hands, records and statistics (apart from wall time), also over a sequence of collections; another seed gives different hands.

4. PPO learner (`Learner`)
4.1 What it does
* The `Learner` owns the trainable `PolicyNet` and `CriticNet` (sections 1.1, 1.2), one Adam optimiser over both, and the PPO hyperparameters. It turns one or more `TrainingRecords` (as returned by `RolloutWorker.collect`, possibly from several workers) into one PPO update of both networks, and hands out its policy weights for the next collection.
* Each record is one learner decision of a finished hand: the hand is a single episode with one terminal reward (the seat's net result in big bets) and no discounting, so the return of every decision is the hand's reward. There is no bootstrapping and no GAE in this stage.
* Advantage: for each decision, its reward minus the critic's value of that decision (observation plus expanded critic input), computed once per update with the critic as it was before the update and without gradients. Because the critic sees all hole cards, this subtracts most of the card luck out of the reward.
* The critic input is built with rl_env's `expand_critic` from the records' critic cards (the same critic input as section 1.2); trainer never re-implements it.

4.2 The loss (per minibatch)
* Policy: the PPO clipped surrogate. The ratio is exp(new log-probability of the recorded action under the current policy − recorded log-probability). Advantages are normalised within the minibatch (mean 0, std 1, with a small epsilon). The policy loss is −mean(min(ratio · adv, clip(ratio, 1 − clip, 1 + clip) · adv)).
* Critic: 0.5 · mean squared error between the critic's value and the reward.
* Entropy bonus: the mean entropy of the masked policy distribution.
* Total: policy loss + `vf_coef` · critic loss − `ent_coef` · entropy.
* New log-probabilities and entropy use the same masked logits and log-softmax as `NeuralPolicy` (sections 1.1, 2), so a policy that has not changed gives a ratio of exactly 1 for every record, and illegal actions contribute nothing (no NaN or inf, also when only one action is legal).
* The loss is a separate function from the `Learner` (given the networks, the tensors of the batch, the advantages, the minibatch indices and the config), so it can be tested on its own and extended later (the MMD magnet term of a later stage) without changing the update loop.

4.3 The update
* `update(*records)` concatenates the given records, computes the advantages (4.1), then runs `epochs` passes; each pass shuffles the decisions and splits them into minibatches of `minibatch` decisions (the last one may be smaller). For each minibatch: compute the loss, zero the gradients, back-propagate, clip the global gradient norm of all parameters to `max_grad_norm`, take one optimiser step.
* Records with a missing log-probability (NaN, i.e. not from a `NeuralPolicy`), with a different observation width than `OBS_DIM`, or an empty batch are refused before any weight changes.
* It returns statistics of the update for logging and the health checks of later stages: the policy loss, critic loss, entropy, approximate KL divergence between the old and new policy (mean of (ratio − 1) − log ratio), clip fraction (share of decisions with |ratio − 1| > clip), the critic's explained variance on the batch (computed with the pre-update values; undefined, reported as NaN, when every reward of the batch is the same), the gradient norm before clipping, and the number of decisions and minibatch steps. Loss, entropy, KL and clip-fraction values are averaged over all minibatches of the last epoch.
* The policy weights used for collection are the learner's current `PolicyNet` state dict (`get_weights()`), in the form `NeuralPolicy` and `RolloutWorker.collect` accept (section 2), detached and on CPU, so changing them later does not change the learner and vice versa.
* Everything runs on CPU in this stage; the number of torch threads of the learner is a constructor parameter.

4.4 Configuration
* The hyperparameters are one config object with these defaults: lr 3e-4 (Adam), clip 0.2, epochs 3, minibatch 4,096, vf_coef 0.5, ent_coef 0.01, max_grad_norm 1.0, network 3 hidden layers of 512 units for both networks. All are constructor parameters.
* Reproducible: the learner has a seed that sets the initial weights of both networks and the minibatch shuffling; the same seed, config and records give identical weights and statistics after any number of updates.

4.5 Checkpoints
* The learner saves and restores its full state (both networks, the optimiser state, the config and the number of updates done) as one file; a restored learner continues exactly as the original would have (same weights and statistics after the next update with the same records). A checkpoint built for another `ENCODER_VERSION` or network config is refused (section 1.3).

5. Acceptance
* Shapes: for batch sizes 1 and several hundred, the policy returns (B, `N_ACTIONS`) and the critic (B,), both float32, from real rl_env observations, masks and critic inputs of hands played through a `TableBatch` (2 to 10 players).
* Masking: for every mask rl_env can produce (including only check-or-call legal, and fold masked when checking is free), illegal actions get probability ~0, the legal probabilities sum to 1, and log-probabilities and entropy of a categorical distribution over the logits are finite (no NaN or inf), also for extreme inputs.
* Sampling from the masked policy over many decisions never picks an illegal action.
* Gradients flow: one optimisation step on a small batch changes the weights of every layer, and gradients are finite.
* State dict round trip gives identical outputs; same seed gives identical initial weights.
* NeuralPolicy at the table: playing complete hands through a `TableBatch` (2 to 10 players, a few hundred hands, NeuralPolicy at every seat or mixed with an rl_env random controller) works without errors; every action is legal; the batch builds no views for it; the recorded log-probabilities equal the log-probabilities of the recorded actions under the policy's network (recomputed from the recorded observations and masks); with the same seed and weights two runs are identical, and with another seed they differ.
* NeuralPolicy sampling: over many decisions from a fixed batch, the frequencies of the actions match the network's probabilities; illegal actions never appear.
* NeuralPolicy weights: after loading another policy's state dict its log-probabilities match that network's; a mismatched state dict is refused.
* Rollout lineups: over many hands, table sizes cover 2 to 10, the learner holds 1 to 3 seats and never every seat, every other seat holds a bot from the configured opponent set, and all four default bots appear.
* Rollout collection: `collect` returns at least `n_decisions` records, all from the learner and all from finished hands; every record's reward equals the learner seat's net result of its hand in big bets; every recorded action is allowed by its mask; the recorded log-probabilities equal the log-probabilities of the recorded actions under the loaded weights (recomputed from the recorded observations and masks); nothing is pending afterwards.
* Rollout weights: after `collect` with another policy's weights, the recorded log-probabilities match that policy's network; mismatched weights are refused before any hand is played.
* Rollout statistics: the hand and decision counts match the records, and the learner's total result equals the sum of the rewards of its records' hands per learner seat (each hand's learner result counted once per learner seat, not once per decision).
* Rollout determinism: two workers with the same seed and weights give identical records and statistics (apart from wall time) over two consecutive collections; another seed gives different records.
* Import boundary: rl_env and arena do not import torch or trainer (checked by a test).
* PPO loss: with the policy unchanged since the records were collected (by a `RolloutWorker` with the learner's weights), every ratio is 1 (to float precision), approx_kl is ~0 and clip fraction is 0; the loss, its parts and all gradients are finite, including for records where only one action is legal.
* PPO clipping: when the ratio of a decision is outside [1 − clip, 1 + clip] in the direction its advantage favours, that decision contributes no policy gradient.
* Advantages: computed once per update from the pre-update critic, without gradients; equal to reward minus critic value.
* Learner update: one update on real records changes the weights of both networks, keeps them finite, and never changes the records; the gradient norm is clipped to `max_grad_norm`; the returned statistics are finite (except the explained variance, which is NaN when every reward of the batch is the same) and the decision and step counts match the batch size, `minibatch` and `epochs`. Records with missing log-probabilities, a wrong observation width or no rows are refused and leave the weights unchanged.
* Learner weights: `get_weights()` loads into a `NeuralPolicy` and a `RolloutWorker`, and the policy then gives the learner's log-probabilities; the returned weights do not change when the learner updates afterwards.
* Learning: on a small synthetic problem (for example a fixed batch of decisions where one legal action always has a higher reward), repeated updates raise that action's probability and lower the critic loss. Against the arena bots, a short run of collect-and-update iterations (a few thousand decisions each) on CPU finishes in reasonable time with finite statistics, entropy starting near ln 3 for three legal actions and not collapsing within those iterations.
* Learner determinism and checkpoints: two learners with the same seed and records give identical weights and statistics over two updates; a learner restored from a checkpoint after one update gives the same weights and statistics after the second as the original; mismatched checkpoints are refused.
