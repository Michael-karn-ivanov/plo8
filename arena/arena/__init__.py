"""arena: baseline bots and the duplicate-match evaluator for fixed-limit Omaha
Hi-Lo 8-or-better, on top of `//rl_env`.

- Bots (rl_env `Controller`s): `RandomBot`, `CallingStation`, `Maniac`;
  `make_bot(name, seed)` / `BOT_NAMES`.

No poker rule lives here: betting, pots, showdown and hand evaluation stay in
the engine, table driving in rl_env.
"""

from .bots import BOT_NAMES, Bot, CallingStation, Maniac, RandomBot, make_bot

__all__ = [
    # bots
    "Bot",
    "RandomBot",
    "CallingStation",
    "Maniac",
    "BOT_NAMES",
    "make_bot",
]
