"""arena: baseline bots and the duplicate-match evaluator for fixed-limit Omaha
Hi-Lo 8-or-better, on top of `//rl_env`.

- Bots (rl_env `Controller`s): `RandomBot`, `CallingStation`, `Maniac`,
  `RuleBot`; `make_bot(name, seed)` / `BOT_NAMES`.
- `BotAgent`: any bot as an engine agent (callback mode, later bot_client).
- `play_match(lineup, n_deals, seed, ...)`: duplicate match, results per report
  name in big bets per 100 hands with a 95% interval; `combine` for more deals.

No poker rule lives here: betting, pots, showdown and hand evaluation stay in
the engine, table driving in rl_env.
"""

from .agent import BotAgent
from .bots import BOT_NAMES, Bot, CallingStation, Maniac, RandomBot, make_bot
from .history import format_hand
from .match import (
    AccountingError,
    Entrant,
    HandRecord,
    MatchResult,
    NameStats,
    combine,
    play_match,
)
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
from .stats import CHIPS_PER_BIG_BET, Summary, summarize

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
    # agent
    "BotAgent",
    # match
    "Entrant",
    "NameStats",
    "HandRecord",
    "MatchResult",
    "AccountingError",
    "play_match",
    "combine",
    # stats / history
    "CHIPS_PER_BIG_BET",
    "Summary",
    "summarize",
    "format_hand",
]
