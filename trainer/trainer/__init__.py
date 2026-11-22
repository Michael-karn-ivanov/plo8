"""trainer: neural networks (and, in later stages, the training loop) for a
fixed-limit Omaha Hi-Lo 8-or-better bot, on top of `//rl_env`.

This stage: the policy and critic networks and the acting policy.

- `PolicyNet(obs, mask) -> logits` sees only rl_env's observation and action
  mask; illegal actions get the finite logit `MASK_VALUE` (never -inf).
- `CriticNet(critic_input(obs, critic_cards)) -> value` (big bets) is the only
  network that receives the privileged critic input.
- `NeuralPolicy(net, seed=...)` is an rl_env Controller over a private copy of a
  `PolicyNet`: samples with its own generator, returns `(actions, logp)`.

trainer never re-implements rl_env's encoder, mask or critic expansion; it only
consumes their arrays.
"""

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

__all__ = [
    "PolicyNet",
    "CriticNet",
    "NeuralPolicy",
    "mlp",
    "masked_logits",
    "log_probs",
    "critic_input",
    "check_state_dict",
    "MASK_VALUE",
    "LOGIT_LIMIT",
    "NetConfigError",
]
