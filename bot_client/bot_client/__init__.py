"""bot_client: seats one arena bot at a running cli_server table and plays it there.

- `players`: the `Player` abstraction (`BotPlayer` = an arena bot behind `BotAgent`).
- `policy`: the pure rebuy / sit-in / leave reply for a `hand_end`.
- `client`: `BotClient`, the message loop (turns, hand ends, lifecycle).
- `stats`: per-hand log lines and the final summary.
- `main`: command line, startup / join, signals, exit codes.

No poker rule and no bot logic lives here: legality, pots and showdown are the
engine's, the table is run by cli_server, decisions come from arena.
"""
