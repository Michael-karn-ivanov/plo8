Python library (`trainer`): the neural networks and, later, the training loop for a fixed-limit Omaha Hi-Lo 8-or-better bot.
It sits on top of plo8/rl_env (observation vectors, action masks, critic input) and, in later phases, plo8/arena (fixed opponents and the evaluator). The project grows in stages; this spec currently covers the first stages: the networks and the acting policy. The rollout loop, PPO learner, single-process driver, Ray, the opponent league, AWS and ONNX export are later changes to this spec.
Mistakes here rarely crash: a wrong mask, a wrong input size or hidden information reaching the policy gives a network that trains and plays badly, or cheats. So the shapes, the masking and the separation between what the policy and the critic may see matter more than speed.

Dependencies
* plo8/rl_env: `OBS_DIM` and `ENCODER_VERSION` (the observation vector), `N_ACTIONS` and the action order fold / check-or-call / bet-or-raise, the action mask (True = legal), `PRIV_DIM` and `expand_critic` (the critic's extra input).
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

3. Acceptance
* Shapes: for batch sizes 1 and several hundred, the policy returns (B, `N_ACTIONS`) and the critic (B,), both float32, from real rl_env observations, masks and critic inputs of hands played through a `TableBatch` (2 to 10 players).
* Masking: for every mask rl_env can produce (including only check-or-call legal, and fold masked when checking is free), illegal actions get probability ~0, the legal probabilities sum to 1, and log-probabilities and entropy of a categorical distribution over the logits are finite (no NaN or inf), also for extreme inputs.
* Sampling from the masked policy over many decisions never picks an illegal action.
* Gradients flow: one optimisation step on a small batch changes the weights of every layer, and gradients are finite.
* State dict round trip gives identical outputs; same seed gives identical initial weights.
* NeuralPolicy at the table: playing complete hands through a `TableBatch` (2 to 10 players, a few hundred hands, NeuralPolicy at every seat or mixed with an rl_env random controller) works without errors; every action is legal; the batch builds no views for it; the recorded log-probabilities equal the log-probabilities of the recorded actions under the policy's network (recomputed from the recorded observations and masks); with the same seed and weights two runs are identical, and with another seed they differ.
* NeuralPolicy sampling: over many decisions from a fixed batch, the frequencies of the actions match the network's probabilities; illegal actions never appear.
* NeuralPolicy weights: after loading another policy's state dict its log-probabilities match that network's; a mismatched state dict is refused.
* Import boundary: rl_env and arena do not import torch or trainer (checked by a test).
