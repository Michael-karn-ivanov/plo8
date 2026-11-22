"""trainer: neural networks (and, in later stages, the training loop) for a
fixed-limit Omaha Hi-Lo 8-or-better bot, on top of `//rl_env`.

This stage: the policy and critic networks.

- `PolicyNet(obs, mask) -> logits` sees only rl_env's observation and action
  mask; illegal actions get the finite logit `MASK_VALUE` (never -inf).
- `CriticNet(critic_input(obs, critic_cards)) -> value` (big bets) is the only
  network that receives the privileged critic input.

trainer never re-implements rl_env's encoder, mask or critic expansion; it only
consumes their arrays.
"""

from .nets import (
    LOGIT_LIMIT,
    MASK_VALUE,
    CriticNet,
    NetConfigError,
    PolicyNet,
    critic_input,
    masked_logits,
    mlp,
)

__all__ = [
    "PolicyNet",
    "CriticNet",
    "mlp",
    "masked_logits",
    "critic_input",
    "MASK_VALUE",
    "LOGIT_LIMIT",
    "NetConfigError",
]
