"""arena: baseline bots and the duplicate-match evaluator for fixed-limit Omaha
Hi-Lo 8-or-better, on top of `//rl_env`.

- Bots (rl_env `Controller`s): `RandomBot`, `CallingStation`, `Maniac`,
  `RuleBot`; `make_bot(name, seed)` / `BOT_NAMES`.

No poker rule lives here: betting, pots, showdown and hand evaluation stay in
the engine, table driving in rl_env.
"""

from .bots import BOT_NAMES, Bot, CallingStation, Maniac, RandomBot, make_bot
from .rules import (
    NUT_LOW_PAIRS,
    RULE_PARAMS,
    PostflopClass,
    RuleBot,
    RuleParams,
    decide,
    nut_low_pairs,
    postflop_class,
    preflop_score,
)

__all__ = [
    # bots
    "Bot",
    "RandomBot",
    "CallingStation",
    "Maniac",
    "RuleBot",
    "BOT_NAMES",
    "make_bot",
    # rules
    "RuleParams",
    "RULE_PARAMS",
    "PostflopClass",
    "NUT_LOW_PAIRS",
    "nut_low_pairs",
    "preflop_score",
    "postflop_class",
    "decide",
]
