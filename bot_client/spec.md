Python binary (`bot_client`): a headless client that seats a bot at the real table, plo8/cli_server, and plays it there with no human in the loop.
It is the bridge between the bots of plo8/arena (and, later, the trained bots exported by plo8/trainer) and the same server that human players reach through plo8/cli_client. Its purpose is that a person can start `cli_server`, start one or more bot clients and then sit down with `cli_client` and play against them, and that the bots behave at the server exactly as they do in arena's matches.
This first version plays the arena bots (`--bot rule` and the others). Playing an exported neural network (`--model x.onnx`) is a later extension (uber-plan Phase 3g); this version must leave room for it but must not implement it.
Mistakes here mostly show up as a bot that times out, sits out, stops rebuying or plays differently from arena. So correctness of the protocol handling and "same decision as in arena" matter more than features.

Dependencies
* plo8/arena (the bots and the way to create one from its name, `BOT_NAMES`, and `BotAgent` as the reference for how a bot answers an engine turn), plo8/rl_env (`DecisionBatch.single`, the action constants, the mask and `to_engine`), plo8/cli_server's wire protocol library (`//cli_server:protocol`: message reading and writing, request builders, decoding of table views, legal actions, hand results and table settings), the client networking of plo8/cli_client (`ServerConnection` for requests to the server port, `CallbackListener` for the server's callback channel), plo8/engine and plo8/entities (the types those decode into).
* numpy, from the repo venv. No other third-party package in this version (onnxruntime arrives with model support later). bot_client must not depend on torch, and must not depend on the curses user interface of cli_client.
* No poker rule lives here: betting, legality, pots and showdown stay in plo8/engine and the table is run by plo8/cli_server. No bot logic lives here either: decisions come from the arena bot, unchanged.
* Dependency direction: bot_client depends on cli_server's protocol, cli_client's networking, arena and rl_env; nothing depends on bot_client.
* Changes needed in other projects (to be requested in their specs, see section 6): cli_client's networking must be usable as a library by bot_client without pulling in its user interface, and cli_server's in-process `GameServer` must be usable by bot_client's tests, as cli_client's integration test already does.

1. Command line
* A buck2 binary, for example: `buck2 run //bot_client:bot_client -- --server-port 8008 --port 9101 --name RuleBot --bot rule`.
* Arguments:
    * `--server-port` - port of the running cli_server (default 8008, as the server's default);
    * `--port` - the bot's own callback port, on which it listens for server messages (0 = pick a free port; the chosen port is logged);
    * `--name` - the player name at the table (default: derived from the bot name, e.g. `rule-bot`); names must be unique at the server, and a rejected join is reported with the server's reason;
    * `--stack` - buy-in in big blinds (default 100, as cli_client), also the target of rebuys;
    * `--bot` - the arena bot to play, one of arena's bot names (`random`, `calling`, `maniac`, `rule`); default `rule`; an unknown name fails at startup and lists the known names;
    * `--seed` - seed for bots that use randomness (RandomBot); default 0;
    * `--rebuy` - rebuy policy (section 3): `topup` (default), `bust` or `never`;
    * `--hands` - optional: leave the table after this many hands played (dealt in), then exit; default: play until stopped;
    * `--log-level` - logging verbosity; default: one line per hand (section 4).
* Invalid arguments (bad ports, non-positive stack, unknown bot) fail before anything connects, with a clear message and a non-zero exit code.
* `--bot` and the later `--model` are two ways of choosing the player. The command line, the startup and the player abstraction must be shaped so that adding `--model` later (mutually exclusive with `--bot`) needs no change to the networking or the hand-end handling.

2. Joining and playing
* At startup bot_client opens its callback listener first, then sends `join` with its name, stack and callback port, as cli_client does. The join response is checked: rejected joins (duplicate name, table full, buy-in above the table maximum, unreachable callback port) end the program with the server's error and a non-zero exit code. An accepted join may be seated or queued (game in progress); both are fine, the bot simply waits.
* The table settings in the join response (blinds, bet sizes, maximum seats, maximum buy-in, time to act) are logged. If the requested stack is above the table's maximum buy-in the join fails as above; bot_client does not silently lower it.
* On every `turn` message the bot client decodes the table view and legal actions with the protocol library, builds a one-row decision (rl_env's `DecisionBatch.single`, exactly as arena's `BotAgent` does), asks the bot, turns the chosen action into an engine action with rl_env's `to_engine`, and sends an `action` request carrying the turn's `seq`.
    * The decision must be identical to the decision the same bot makes behind arena's `BotAgent`, and inside an rl_env `TableBatch`, for the same view and legal actions: the server path adds nothing and loses nothing.
    * The bot never sends an action the legal actions forbid, and in particular never folds when checking is free (the mask forbids it).
    * A bot answers in a small fraction of the server's time to act; bot_client never relies on the timeout. If the server nevertheless rejects an action (e.g. a stale `seq`), the rejection is logged; bot_client does not retry with a different action for a stale `seq`, and for a current `seq` it falls back to check-or-call, which is always legal.
* Every other server message (`seated`, `hole_cards`, `event`, `acting`, `lobby`, `sat_out`, `returned`, unknown types and unknown fields) needs no answer. bot_client tracks only what it needs: its own stack and status (seated, waiting, sitting out, left) and the hand count. Unknown message types and fields are ignored, as the protocol requires of clients.
* Bots are stateless between decisions (arena requirement), so bot_client keeps no per-hand memory for the bot; everything a decision needs is in the `turn` message.

3. End of hand: rebuy, return, leave
* Every `hand_end` message is answered with a `hand_end` request carrying its `seq`, promptly (a missing answer means no rebuy, no sit-in and no leave for that window, see cli_server).
* Rebuy policy, using the bot's stack after the hand (from the hand result) and the table's maximum buy-in:
    * `topup` (default): if the stack is below `--stack`, rebuy the difference, so the bot starts every hand with its buy-in (never above the table maximum);
    * `bust`: rebuy `--stack` only when the stack is zero;
    * `never`: never rebuy; a busted bot then stays out of the game (engine rule) and bot_client leaves the table and exits, since it can no longer play.
* If the bot has been sat out (e.g. after a missed turn, which should not happen but is an engine rule), it asks to sit in again in its `hand_end` reply.
* With `--hands N`, once N hands have been played the `hand_end` reply asks to leave; bot_client then waits for `left` / `close` and exits normally.
* A rebuy the server rejects is logged and does not stop the bot; if it leaves the bot busted under `topup` or `bust`, it is tried again at the next `hand_end`.

4. Lifecycle, logging and errors
* bot_client runs until one of: `--hands` reached, busted with `--rebuy never`, the server sends `left` / `close`, the callback channel or the server port is lost, or the user stops it (Ctrl-C / SIGTERM). On a user stop it sends `leave` and exits once the server confirms or a short timeout passes.
* Exit code 0 for a normal end (hands reached, left, server closed the table), non-zero for a failed join, invalid arguments or a lost connection.
* It never prints a full-screen interface; it writes log lines. At the default level: the join result and table settings, then one line per hand the bot played: hand number, the bot's net result for the hand in big blinds, its stack, any rebuy; and a final summary on exit: hands played, total net result in big blinds and in big bets per 100 hands (as arena reports), total rebought.
* The bot must never see hidden information: bot_client passes it only what the server sends to that player (its own hole cards, public state). It needs no change in cli_server to work.
* Several bot clients (different names and ports) may sit at the same table, in separate processes; a person with cli_client can join the same table.

5. Acceptance
* Arguments: invalid ports, a non-positive stack and an unknown `--bot` fail before connecting with a non-zero exit code and a clear message; every arena bot name is accepted.
* Same decisions as arena: for many views and legal actions taken from random hands (including ones that went through the protocol's encode and decode), the action bot_client sends equals the action of the same bot behind `BotAgent` and in a `TableBatch`.
* Real table, in process: an in-process cli_server `GameServer` (short grace period, no pause between hands, as in cli_client's integration test) with `--bot rule` against scripted clients plays at least 200 hands without errors; the bot never sends an illegal or rejected action, never times out, is never sat out, and answers every `hand_end`.
* Bot against bot: two or more bot clients (e.g. `rule` against `calling` and `random`) at one in-process server play at least 200 hands; total chips at the table stay equal to buy-ins plus rebuys minus chips taken away by leaving.
* Rebuy policies: with short stacks, `topup` keeps the bot at its buy-in before every hand (never above the table maximum), `bust` rebuys only at zero, `never` leaves and exits once busted.
* `--hands N` leaves after exactly N hands played and exits with code 0; the server sees the seat emptied.
* Join failures: a duplicate name and a stack above the table maximum are reported with the server's error and a non-zero exit code.
* Lost server: if the server shuts down mid-game, bot_client exits with a non-zero code and a log line, without hanging.
* Manual check (documented in plan.md, not automated): `cli_server`, two `bot_client --bot rule` processes and one `cli_client` at one table; a person plays a few hands against the bots.
* All existing tests of plo8/entities, plo8/engine, plo8/rl_env, plo8/arena, plo8/cli_server and plo8/cli_client still pass, including the cli_server e2e mode.

6. Required changes in other projects
* plo8/cli_client: its networking (`ServerConnection`, `CallbackListener`) is available to bot_client as a library that does not include the curses user interface; cli_client's own behaviour does not change.
* plo8/cli_server: its in-process `GameServer` is available to bot_client's tests (today it is visible only to cli_client); the server's behaviour does not change.
* No change in plo8/engine, plo8/rl_env or plo8/arena is expected. If one turns out to be needed, it is requested in that project's spec, not worked around here.
