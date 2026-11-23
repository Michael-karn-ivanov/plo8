"""trainer: neural networks and (in stages) the training loop for a fixed-limit
Omaha Hi-Lo 8-or-better bot, on top of `//rl_env` and `//arena`.

This stage: the policy and critic networks, the acting policy, the rollout loop
and the PPO learner.

- `PolicyNet(obs, mask) -> logits` sees only rl_env's observation and action
  mask; illegal actions get the finite logit `MASK_VALUE` (never -inf).
- `CriticNet(critic_input(obs, critic_cards)) -> value` (big bets) is the only
  network that receives the privileged critic input.
- `NeuralPolicy(net, seed=...)` is an rl_env Controller over a private copy of a
  `PolicyNet`: samples with its own generator, returns `(actions, logp)`.
- `RolloutWorker(seed=...).collect(weights, n_decisions)` plays the learner
  against arena's bots (lineups from `draw_lineup`) in one TableBatch and returns
  the learner's `TrainingRecords` of finished hands and `RolloutStats`.
- `Learner(PPOConfig(), seed=...).update(*records)` turns those records into PPO
  updates of both networks (`make_batch`, `compute_advantages`, `ppo_loss`) and
  returns `UpdateStats`; `get_weights()` feeds the next collection.

trainer never re-implements rl_env's encoder, mask or critic expansion; it only
consumes their arrays.
"""

from .lineup import DEFAULT_OPPONENTS, LEARNER_KEY, draw_lineup
from .nets import (
    LOGIT_LIMIT,
    MASK_VALUE,
    CriticNet,
    NetConfigError,
    PolicyNet,
    check_state_dict,
    critic_input,
    log_probs,
    masked_logits,
    mlp,
)
from .policy import NeuralPolicy
from .ppo import (
    Learner,
    LossParts,
    PPOBatch,
    PPOConfig,
    UpdateStats,
    compute_advantages,
    make_batch,
    ppo_loss,
)
from .rollout import ResultStats, RolloutStats, RolloutWorker

__all__ = [
    "PolicyNet",
    "CriticNet",
    "NeuralPolicy",
    "RolloutWorker",
    "RolloutStats",
    "ResultStats",
    "draw_lineup",
    "LEARNER_KEY",
    "DEFAULT_OPPONENTS",
    "Learner",
    "PPOConfig",
    "PPOBatch",
    "UpdateStats",
    "LossParts",
    "make_batch",
    "compute_advantages",
    "ppo_loss",
    "mlp",
    "masked_logits",
    "log_probs",
    "critic_input",
    "check_state_dict",
    "MASK_VALUE",
    "LOGIT_LIMIT",
    "NetConfigError",
]
