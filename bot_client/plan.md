# bot_client — implementation plan

A Python binary that seats one arena bot at a running `//cli_server` table and plays it
there with no human in the loop. To the server it is an ordinary client. It joins over
the JSON-lines protocol (`cli_server/plan.md`, "Wire protocol"), listens on its own
callback port, answers every `turn` with the bot's decision and every `hand_end` with
its rebuy / sit-in / leave choice, and writes log lines. The point is that a person can
start `cli_server`, start a few bot clients, sit down with `cli_client` and play against
bots that behave exactly as they do in arena's matches.

Guiding rules:
- **Same decision as in arena.** The decision path is arena's own:
  `arena.BotAgent(bot).decide(view, legal)`, which is `DecisionBatch.single` →
  `bot.act` → `to_engine`. bot_client calls that method and has no decision code of its
  own. The view and legal actions come from `cli_server.protocol`'s decoders, whose
  round trip is exact (`cli_server/plan.md`, "Round trip is exact"). So the server path
  adds nothing and loses nothing.
- **No rules, no bot logic here.** Legality, pots and showdown stay in `//engine`, the
  table is run by `//cli_server`, and decisions come from `//arena`. bot_client only
  moves messages and applies a rebuy policy.
- **Never rely on the timeout.** Every `turn` and `hand_end` is answered as soon as it
  is read. One thread reads the inbox and answers. Nothing on that thread blocks except
  the request to the server port, which gets an immediate response.
- **Room for `--model`.** Choosing the player (`--bot` now, `--model` in uber-plan Phase
  3g) is isolated in `players.py`. The networking, the turn handling and the hand-end
  handling only see a `Player` with `decide(view, legal) -> ActionType`.

## Layout

```
bot_client/
  spec.md
  plan.md
  BUCK
  bot_client/
    __init__.py
    main.py        # command line, logging, signals, startup / join, exit codes
    players.py     # Player protocol, BotPlayer (arena bot), make_player(args)
    policy.py      # pure: rebuy / sit-in / leave reply for a hand_end
    client.py      # BotClient: message loop, turn and hand_end answers, status, stats
    stats.py       # pure: per-hand lines and the final summary (BB, big bets / 100)
  tests/
    helpers.py     # in-process server, scripted check/call client, bot clients on threads
    test_args.py
    test_players.py
    test_policy.py
    test_client.py
    test_stats.py
    test_integration.py
    test_lost_server.py
    test_imports.py
```

## Build (buck2)

`bot_client/BUCK`:
- `python_library(name = "bot_client_lib", srcs = glob(["bot_client/*.py"]),
  base_module = "", deps = ["//arena:arena", "//rl_env:rl_env",
  "//cli_server:protocol", "//cli_client:net", "//engine:engine",
  "//entities:entities"])`. `//cli_client:net` is the networking-only library of
  cli_client (see Required changes in other projects). bot_client does not depend on
  `//cli_client:cli_client_lib`, which holds the curses UI and its controller.
- `python_binary(name = "bot_client", main_module = "bot_client.main",
  deps = [":bot_client_lib"])`.
- `python_test(name = "bot_client_test", srcs = glob(["tests/*.py"]),
  base_module = "bot_client_tests", deps = [":bot_client_lib",
  "//cli_server:cli_server_lib"])`. The integration tests run an in-process
  `GameServer`, and `test_lost_server` starts `cli_server.main` as a child process. Both
  need `//cli_server:cli_server_lib` to be visible to `//bot_client/...`.
- Third-party: numpy only, from the repo venv (`toolchains/BUCK` runs `.venv/bin/python`,
  as for rl_env and arena). It is reached through arena and rl_env; bot_client's own
  modules do not import it. No torch and no onnxruntime in this version.
  `test_imports` checks this.

Run: `buck2 run //bot_client:bot_client -- --server-port 8008 --port 9101 --name RuleBot --bot rule`.
Test: `buck2 test //bot_client/...`.

## 1. Command line (`main.py`)

```
bot_client [--server-port PORT] [--port PORT] [--name NAME] [--stack BB]
           [--bot NAME] [--seed N] [--rebuy topup|bust|never] [--hands N]
           [--log-level LEVEL]
```
- `--server-port`: default `8008`, the server's default. Must be 1..65535 (the server
  port cannot be 0).
- `--port`: callback port, default `0`, which picks a free port. Must be 0..65535. The
  port actually bound is logged and sent in `join`.
- `--name`: default `f"{bot}-bot"` (`rule-bot`, `random-bot`, ...). It must not be empty
  after stripping. Uniqueness is checked by the server, and a rejected join reports the
  server's reason.
- `--stack`: buy-in in BB, default `100`. Must be finite and > 0. It is also the rebuy
  target.
- `--bot`: one of `arena.BOT_NAMES` (`random`, `calling`, `maniac`, `rule`), default
  `rule`. The argparse `choices` come from `BOT_NAMES`, so an unknown name fails with
  the list of known names and every arena bot is accepted with no change here.
- `--seed`: int, default `0`, passed to `arena.make_bot(name, seed)` (used by `random`,
  ignored by the others).
- `--rebuy`: `topup` (default), `bust` or `never` (section 3).
- `--hands`: optional int ≥ 1. With it, the bot leaves after that many hands it was
  dealt into. Without it, the bot plays until stopped.
- `--log-level`: `warning`, `info` (default) or `debug`. `info` gives the join result,
  the table settings, one line per hand and the summary (section 4). `debug` adds every
  message sent and received and every decision (view summary, legal actions, chosen
  action). `warning` keeps only problems and the summary.

`--bot` sits in an argparse mutually exclusive group that has only one member today.
Adding `--model PATH` later means adding it to that group and a branch in
`make_player`. Nothing else in the command line, the startup or `client.py` changes.

Invalid arguments are rejected by argparse (exit status 2, message on stderr) before
anything is bound or connected. That covers bad ports, a non-positive or non-finite
stack, `--hands 0`, and an unknown bot or rebuy policy. The bot is then created
(`make_player`) before the listener is opened. A bot whose construction fails, and the
RuleBot import that builds its nut-low tables, therefore also fail or cost their time
before joining, not during the first turn.

### Startup
1. Parse the arguments and set up logging (stderr, `%(asctime)s %(levelname)s %(message)s`).
2. `player = make_player(args)`.
3. Open `CallbackListener(name, inbox, port=args.port).start()`. It must be listening
   before `join`, because the server connects to it while handling the join. A port that
   cannot be bound gives exit status 2 with "cannot listen on port N: ...".
4. Open `ServerConnection(args.server_port)` and send
   `join_request(name, stack, listener.port, seat=None)`. `seat=None` means joining as
   the big blind, as cli_client does by default.
   - `ConnectionLost` → "cannot reach the server on port N: ..." and exit 2.
   - `ok: false` → log the server's `error` (duplicate name, table full, buy-in above
     the maximum, unreachable callback port, server shutting down, ...) and exit 2. The
     stack is never lowered to fit the table maximum.
   - `ok: true` → log `joined as NAME: seat S` or `joined as NAME: queued until the
     current hand ends`, then the table settings from `decode_join_settings(response)`:
     blinds, small and big bet, max seats, max buy-in (or "none"), time to act. If the
     server sent no `table` (an older server), log "table settings: not sent".
5. Install the SIGINT / SIGTERM handlers (section 4) and run `BotClient.run()`. Its
   return value is the exit status.
6. On exit, always: `listener.close()` and `conn.close()`, then log the summary.

The startup is in `start_bot(args, ...) -> BotClient`, and `main(argv) -> int` wraps it,
so tests run the whole binary in-process (`main([...])` on a thread) as well as the
pieces.

## 2. Players (`players.py`)

```python
class Player(Protocol):
    label: str                                          # for logs: "rule", later "model x.onnx"
    def decide(self, view: TableView, legal: LegalActions) -> ActionType: ...

class BotPlayer:                                        # an arena bot
    def __init__(self, bot_name: str, seed: int = 0):
        self.label = bot_name
        self.agent = arena.BotAgent(arena.make_bot(bot_name, seed))   # rebuy_to_bb=None
    def decide(self, view, legal) -> ActionType:
        return self.agent.decide(view, legal)           # DecisionBatch.single → act → to_engine

def make_player(args) -> Player:                        # --bot now; --model later
    return BotPlayer(args.bot, args.seed)
```
- `BotAgent.decide` is the exact code arena's engine-agent path runs, so "the same
  decision as behind `BotAgent`" holds by construction. The parity with a `TableBatch`
  row is arena's guarantee (`DecisionBatch.single` equals the batch row, which rl_env and
  arena test). bot_client tests both again on decoded views (section 6).
- `BotAgent`'s own `on_hand_end` rebuy is not used. Rebuys go through the `hand_end`
  reply (section 3), which is the only way a remote player can rebuy during play.
- The bot is stateless between decisions (arena requirement). RandomBot's generator is
  its only state. One `BotPlayer` lives for the whole process, so a seeded RandomBot
  makes the same sequence of choices given the same sequence of turns.
- A later `OnnxPlayer(model_path)` (uber-plan, "Exporting and playing through
  cli_server") implements the same `decide`: `encode_view` → onnxruntime → sample →
  `to_engine`. It is not part of this version. `players.py` imports nothing for it.

## 3. The client (`client.py`)

```python
class BotClient:
    def __init__(self, name, player, conn, inbox, *, stack_bb, rebuy, max_hands,
                 settings: TableSettings | None, queued: bool,
                 clock=time.monotonic, leave_timeout=5.0): ...
    def run(self) -> int                 # message loop until an end condition; exit status
    def request_stop(self) -> None       # signal handlers: thread-safe, puts STOP on the inbox
    # state, readable by tests
    status: str                          # "waiting" | "seated" | "sitting_out" | "leaving" | "left"
    stack: float | None                  # own stack in BB, from views, events and results
    seat: int | None                     # own seat in the current hand (seats are renumbered)
    hands_played: int                    # hand_end results in which the bot was dealt in
    stats: Stats                         # per-hand nets, rebuys (stats.py)
    rejected_actions: int; fallbacks: int; sat_out_count: int; problems: list[str]
```

### Threads
- The `CallbackListener` reader thread only puts messages on `inbox` (a
  `queue.Queue`). It drops messages addressed to another name, skips malformed lines,
  and puts `DISCONNECTED` when the channel ends (cli_client's networking).
- The main thread runs `run()`: `inbox.get(timeout=0.2)`, handle the message, repeat.
  Every request to the server goes out from this thread through the one
  `ServerConnection`, so requests never interleave. The timeout lets the loop notice a
  stop request and the leave deadline.
- Signal handlers only call `request_stop()`, which sets a flag and puts a `STOP`
  marker on the inbox. The loop does the leaving.

### Message handling
Messages are dispatched on `type`. Any message type not in the table below, and any
field not read, is ignored (forward compatibility, as the protocol requires). A message
that fails to decode (`ProtocolError`) is logged as a warning and added to `problems`,
and the loop goes on, except for a `turn` (see below).

| message | effect |
|---|---|
| `seated` | `status = "seated"` (was `"waiting"` if the join was queued). `stack` from the view's own seat. Log "seated at seat N" |
| `hole_cards` | `stack`, `seat` from the view (debug log of the cards) |
| `event` | `decode_event`. Events about the bot's own name update `stack` from `stack_bb`. `player_rebought` for the bot adds to `stats.rebought` and confirms a pending rebuy. `hand_started` marks a hand in progress and checks a rebuy requested at the previous `hand_end` (below). An unknown event type is ignored |
| `turn` | answer it (Turns) |
| `hand_end` | answer it (Hand end) |
| `sat_out` | `status = "sitting_out"`, `sat_out_count += 1`, warning "sat out by the server" |
| `returned` | `status = "seated"` |
| `acting`, `lobby` | `lobby` is used only to retry a failed rebuy / sit-in between hands (Hand end). `acting` needs nothing |
| `left` | `status = "left"`, log the final stack from `final_stack_bb`. End: exit 0 once `close` / `DISCONNECTED` arrives, or after 1 s |
| `close` | the server closed this channel: end with exit 0 (the server closed the table, or it follows `left`) |
| `DISCONNECTED` (marker) | the channel ended without `close`: if `left` was received, exit 0; otherwise log "connection to the server lost" and exit 1 |
| `STOP` (marker) | user stop (Lifecycle) |

### Turns
On `("turn", seq, view, legal)` (`decode_server_message`):
1. `t = player.decide(view, legal)`. If `decide` raises (a bot bug, or
   `IllegalMaskedActionError` from `to_engine`), log the exception as an error, add it
   to `problems`, and use `check` if `legal.can_check` else `call`. Check-or-call is
   always legal.
2. `response = conn.request(action_request(name, seq, t.value))`.
3. `ok: true` → done (debug log). `ok: false` → `rejected_actions += 1`, log a warning
   with the server's error, then:
   - stale `seq` (the server's "turn N is not the pending turn of NAME" / "has no
     pending turn" errors: the turn already timed out or was answered): no retry;
   - any other rejection for this `seq` (the server's "X is not legal now (legal: ...)"):
     send check-or-call once for the same `seq`, `fallbacks += 1`. If that is rejected
     too, log it and give up on the turn. The server's timeout then applies, as for any
     client.

   The two kinds are told apart by `is_stale_turn_error(text)`, the one place that knows
   the server's wording. `test_integration` pins it against the real `GameServer`
   (an action with a wrong `seq`), so a change of wording fails the bot_client tests
   instead of silently changing the behaviour.
4. A `turn` whose view or legal actions do not decode cannot be decided. It is logged as
   an error and not answered. The server then times the bot out, which the acceptance
   tests would catch.

The bot sees only what the server sent to this player: the `turn` view (own hole cards,
public state, public history) and the legal actions. bot_client keeps no per-hand memory
for the bot. `stack`, `seat` and `status` are client bookkeeping that the bot never
reads.

### Hand end
On `("hand_end", seq, seat, result)`:
1. Read the bot's line in the result: `seat` from the message (fallback: look up `name`
   in `result.names`), `dealt_in = result.dealt_in.get(seat, False)`, `net =
   result.net.get(seat, 0.0)`, `stack = result.stacks[seat]`.
2. If dealt in: `hands_played += 1` and `stats.add_hand(result.hand_number, net)`.
3. Compute the reply with the pure policy function:
   ```python
   reply = hand_end_reply(stack=stack, sitting_out=status == "sitting_out",
                          target=args.stack, max_buy_in=settings.max_buy_in_bb if settings else None,
                          policy=args.rebuy, hands_played=hands_played, max_hands=args.hands,
                          stopping=stop_requested)
   # -> HandEndChoice(rebuy_bb: float | None, sit_in: bool, leave: bool, reason: str)
   ```
4. Send `hand_end_request(name, seq, reply.rebuy_bb, reply.sit_in, reply.leave)` at
   once. A rejection (e.g. stale `seq`) is logged as a warning. A missing reply would
   mean no rebuy, no sit-in and no leave for that window (cli_server).
5. Log the hand line (section 4): `hand 12: net -2.00 BB, stack 98.00 BB, rebuy 2.00 BB`.
6. If `reply.leave`: `status = "leaving"`, log the reason ("2000 hands played",
   "busted, --rebuy never", "stopping"), and wait for `left` / `close`.

`policy.py` — `hand_end_reply` (pure, no I/O):
- `leave` = `stopping` or (`max_hands` set and `hands_played >= max_hands`) or
  (`policy == "never"` and `stack == 0`). When leaving: no rebuy, no sit-in.
- `cap = max_buy_in if max_buy_in is not None else inf`. `missing(to) = round(min(to,
  cap) - stack, 2)` (BB with two decimals = whole chips at the server's 100 chips per BB,
  as `BotAgent` rounds), and only a positive amount is a rebuy:
  - `topup`: rebuy `missing(target)` when `stack < target`. The bot then starts every
    hand at its buy-in, never above the table maximum;
  - `bust`: rebuy `missing(target)` only when `stack == 0`;
  - `never`: no rebuy.
- `sit_in = sitting_out` (and not leaving). A bot that was sat out, for example after a
  missed turn, asks to come back. The engine checks whether it can: a busted bot comes
  back only with a rebuy, and the rebuy and the sit-in go in the same reply.

Rebuy outcome: the `hand_end` response only says that the reply was delivered. The
engine applies the rebuy after the hand, and it shows as a `player_rebought` event for
the bot before the next `hand_started`. If the next `hand_started` (or the next
`hand_end`, for a bot that was not dealt in) comes without that event, the rebuy was
refused. bot_client then logs "rebuy of X BB was not applied" and does nothing more. The
next `hand_end` computes the reply again from the stack in its result, so `topup` /
`bust` simply try again. The bot never stops because of a refused rebuy.

Between hands: if a `lobby` message arrives (no hand running) while the bot is busted
(stack 0 under `topup` / `bust`) or sitting out, there may be no further `hand_end` to
retry in. For example, heads-up, the table cannot start again without this bot. So the
bot sends the same choice as direct requests, `rebuy_request(name, amount)` and/or
`sit_in_request(name)`, which the engine accepts between hands. It sends at most one
attempt per lobby status change. Rejections are logged as warnings.

### `--rebuy never` and bust
A busted bot under `never` replies `leave=True` at that `hand_end` and exits 0 once
`left` / `close` arrives. The engine would otherwise keep it seated without chips and
out of the game.

### Waiting and sitting out
An accepted but queued join (`queued: true`) sets `status = "waiting"`. The bot gets no
messages until `seated` and needs to do nothing. `sat_out` should never happen, since
the bot answers every turn at once. It is logged as a warning and counted
(`sat_out_count`, which the acceptance tests require to stay 0). The next `hand_end`
reply then asks to sit in.

## 4. Lifecycle, logging, exit codes

End conditions of `run()`:
| condition | how | exit |
|---|---|---|
| `--hands N` reached | `hand_end` reply with `leave=True`, then `left` / `close` | 0 |
| busted with `--rebuy never` | same | 0 |
| server sends `left` / `close` | e.g. the server's `shutdown()` sends `close` to every session | 0 |
| callback channel lost | `DISCONNECTED` without `left` | 1 |
| server port lost | `ConnectionLost` from any request (cli_client's connection reconnects once first) | 1 |
| user stop (Ctrl-C, SIGTERM) | see below | 0 if the leave was confirmed (`left` / `close`) or the leave request was accepted, 1 if the server could not be reached |

Failed startup (invalid arguments, a port that cannot be bound, a server that cannot be
reached, a rejected join) exits with 2, before `run()`. The codes are the same as
cli_client's.

User stop: the first SIGINT / SIGTERM sets `stop_requested` and puts `STOP` on the
inbox. The loop then sends `leave_request(name)` and keeps handling messages:
- turns are still answered. The engine normally force-folds a leaving player without
  asking, but answering is harmless;
- a `hand_end` is answered with `leave=True` (the policy's `stopping`).

It exits on `left` / `close`, or when `leave_timeout` (5 s) has passed. A human at the
table may hold a hand open for up to the action timeout. If the timeout passes, closing
the callback channel on exit makes the server treat the bot as disconnected, and the
engine then folds it and removes it after the hand. A second SIGINT exits at once with
the same code.

Logging (stderr, never a full-screen UI):
- startup: `listening on port 9101`, the join result, the table settings;
- per hand the bot was dealt into, at `info`:
  `hand 12: net +1.50 BB, stack 101.50 BB` and, when a rebuy is requested,
  `, rebuy 3.00 BB`. Also, when they happen: `seated at seat 3`, `sat out by the server`,
  `rebuy of X BB was not applied`, `leaving: <reason>`;
- warnings: rejected actions and hand-end replies, fallbacks, undecodable messages;
- summary on exit (`stats.summary()`), also at `warning` level:
  `RuleBot (rule): 200 hands, net +34.50 BB = +8.63 big bets/100 hands, rebought 120.00 BB, final stack 100.00 BB`.
  Big bets per 100 hands = `100 * (total_net_bb / 2) / hands`, the unit arena reports
  (one big bet = 2 BB). With 0 hands the rate is printed as `n/a`.

`stats.py` keeps nets in integer chips (`round(net_bb * 100)`), as arena does, so the
totals do not drift over thousands of hands. It converts to BB / big bets only for
output.

Several bot clients are simply several processes with different `--name` and `--port`.
Nothing in bot_client is shared between them, and a person with `cli_client` can join
the same table.

## 5. Manual check (not automated)

1. `buck2 run //cli_server:cli_server -- --port 8008`
2. `buck2 run //bot_client:bot_client -- --port 9101 --name RuleBot1 --bot rule`
3. `buck2 run //bot_client:bot_client -- --port 9102 --name RuleBot2 --bot rule --seed 1`
4. `buck2 run //cli_client:cli_client -- --port 9001 --name Me`

Play a few hands. Check that the bots act within a fraction of a second, raise strong
low/scoop hands preflop and fold trash to raises, show sensible hands at showdown, top
up after losing chips, and that their log lines match the hand results on the
cli_client screen. Ctrl-C one bot: it leaves after the hand, its seat empties, and it
prints its summary.

## 6. Tests

All tests use `unittest`. Server-based tests use an in-process `GameServer(port=0,
grace_period=0.05, hand_pause=0, env_kwargs={"seed": S, "action_timeout": 5.0, ...})`,
as cli_client's integration test does. Bot clients run in-process on threads:
`main([...])` or `start_bot(...)` + `run()`, one thread each. The bot's own process-wide
state (signal handlers) is installed only by `main` when it runs on the main thread.

- `helpers.py`:
  - `server(**env)`;
  - `run_bot(argv) -> (thread, result box, client)`;
  - `ScriptedClient(name, stack, rebuy_to)`, a minimal check/call client built on
    `ServerConnection` / `CallbackListener` that answers every `turn` with check or call
    and every `hand_end` with a rebuy back to its stack. It is the "scripted client" of
    the spec. It does not use cli_client's controller, so no UI code is involved;
  - `wait_for(cond, timeout)`.
- `test_args`: defaults (server port 8008, port 0, stack 100, bot `rule`, seed 0, rebuy
  `topup`, no hand limit, name `rule-bot`, level `info`); every name in
  `arena.BOT_NAMES` is accepted; an unknown bot exits 2 and lists the known names; ports
  outside the range, server port 0, stack 0 / negative / `nan`, `--hands 0` and an
  unknown `--rebuy` exit 2. None of these opens a socket: `CallbackListener` and
  `ServerConnection` are patched to fail the test if they are constructed.
- `test_players`: same decisions as arena. Views and legal actions come from many random
  hands (`rl_env.TableBatch` with views at 2..10 seats, `StackConfig()` stacks so
  all-ins and side pots occur, plus every recorded hand of `cli_server`'s e2e set
  through rl_env tables where convenient). A recording controller wraps each arena bot
  and stores `(view, legal, action)` for every row it answers. For every recorded row:
  - `BotPlayer(name).decide(view, legal)` and
    `BotPlayer(name).decide(decode_table_view(json round trip of encode_table_view(view)),
    decode_legal(json round trip of encode_legal(legal)))` both equal
    `to_engine(action, legal)` of the batch row, and equal
    `BotAgent(make_bot(name)).decide(view, legal)`;
  - the result is allowed by `legal`, and it is never `FOLD` when `legal.can_check`.

  RandomBot draws from its generator in call order, so its comparison uses equally
  seeded bots fed the same rows in the same order, one row per call (a `TableBatch` with
  one table, so every batch has one row). The three deterministic bots are compared on
  multi-table batches.
- `test_policy` (`hand_end_reply`, table-driven): `topup` below / at / above the target,
  capped by the max buy-in, the amount rounded to cents; `bust` only at 0; `never` never
  rebuys and leaves at 0; `sit_in` only when sitting out; `--hands` reached → leave with
  no rebuy, even when busted; stopping → leave.
- `test_stats`: hand lines and the summary for hand-made nets (sign, two decimals, big
  bets per 100 = 100 · net / 2 / hands, `n/a` for 0 hands, integer-chip totals).
- `test_client` (fake `ServerConnection` that records requests and returns scripted
  responses, inbox fed with encoded messages built with `cli_server.protocol`, fake
  clock):
  - a `turn` is answered with `action_request(name, seq, ...)` carrying its `seq`, and
    with the player's decision;
  - a stale-`seq` rejection is logged and not retried; an illegal-action rejection
    is followed by one check-or-call for the same `seq`; a raising player falls back to
    check-or-call without contacting the server first;
  - every `hand_end` is answered once with its `seq`; the reply follows the policy;
    `dealt_in = false` results do not count as hands;
  - a missing `player_rebought` before the next `hand_started` logs "not applied", and
    the next `hand_end` asks again;
  - a `lobby` while busted / sitting out sends one `rebuy` / `sit_in` request per status
    change;
  - `seated`, `hole_cards`, `event`, `acting`, `lobby`, `sat_out`, `returned`, an unknown
    type (`{"type": "future_thing"}`) and unknown fields need no request, except the
    lobby retry above; a malformed `event` is skipped;
  - `left` then `close` → exit 0; `close` alone → 0; `DISCONNECTED` without `left` → 1;
    `ConnectionLost` on a request → 1;
  - `request_stop` → `leave_request`, a `hand_end` during the stop replies `leave=True`,
    `left` → 0; no answer within `leave_timeout` → exit (fake clock).
- `test_integration` (real in-process server):
  - rule bot vs scripted clients: `--bot rule` with two `ScriptedClient`s plays at
    least 200 hands (`--hands 200`, exit 0). `rejected_actions == 0`, `fallbacks == 0`,
    `sat_out_count == 0`, `problems == []`. No `player_acted` for the bot in
    `server.results` histories has a `reason` (timeout / leave), except the last hand's
    leave if one is forced. Every `hand_end` was answered: the server never logged a
    missing reply, and `hands_played` equals the number of server results in which the
    bot was dealt in;
  - bot against bot: `rule`, `calling`, `random` (and a `maniac` run) as separate bot
    clients at one server, at least 200 hands. Chips are conserved: for every
    consecutive pair of `server.results`, the starting stacks of hand k+1 (its
    `hand_started` participants) sum to the final stacks of hand k plus the rebuys
    between them (`player_rebought` events) minus the stacks of players who left plus
    joiners' buy-ins. Over the run, `Σ final stacks = Σ buy-ins + Σ rebuys − Σ taken away
    by leavers`, in integer chips. Each bot's summary net equals the sum of its nets in
    `server.results`;
  - rebuy policies with short stacks (`--stack 3`, Maniac opponents so stacks move
    quickly): `topup` starts every hand it is dealt into with exactly 3 BB (from
    `hand_started` participants), and with `max_buy_in` set below a requested top-up it
    never goes above the maximum; `bust` rebuys only after a hand that ends at 0;
    `never` replies leave on the first bust and exits 0, and its name disappears from
    `server.session_names()`;
  - `--hands 5` exits 0 after exactly 5 dealt-in hands; the server's next result has no
    seat with the bot's name, and `session_names()` no longer lists it;
  - join failures: a duplicate name and `--stack` above `max_buy_in` exit 2 with the
    server's error in the log (`assertLogs`);
  - a queued join (a bot joins while a hand is waiting on a scripted client's turn)
    waits, is seated after the hand and plays;
  - `is_stale_turn_error` matches the real server's response to an `action` with a
    wrong `seq` and does not match its response to an illegal action.
- `test_lost_server`: start `cli_server` as a child process (`sys.executable -m
  cli_server.main --port 0 --grace-period 0.05 --hand-pause 0`, `PYTHONPATH` from
  `sys.path` as cli_server's e2e driver does, port read from its log line). Run two bot
  clients in-process, wait until a few hands are played, then `SIGKILL` the server. Both
  bot clients return 1 within a few seconds with a "connection to the server lost" log
  line, and no thread is left hanging.
- `test_imports`: scans every import in `bot_client/bot_client/`. Allowed: the standard
  library, `arena`, `rl_env`, `engine`, `entities`, `cli_server.protocol`,
  `cli_client.net`. Forbidden: `torch`, `onnxruntime`, `curses`, `cli_client.ui` /
  `controller` / `render` / `state` / `main`, `cli_server.server` / `session` / `main`.
  It also checks in a child interpreter that importing `bot_client.main` loads neither
  `curses` nor `torch` (`sys.modules`).
- Regression: `buck2 test //...` (entities, engine, rl_env, arena, cli_server, cli_client)
  stays green, and `buck2 run //cli_server:cli_server -- --e2e_test` passes all cases.

## Required changes in other projects

Requested in their specs. The behaviour of both projects stays the same.
- **`//cli_client`** (`cli_client/spec.md`): the networking as its own library target.
  `python_library(name = "net", srcs = ["cli_client/__init__.py", "cli_client/net.py"],
  base_module = "", deps = ["//cli_server:protocol"], visibility = ["//bot_client/..."])`,
  and `cli_client_lib` takes the remaining sources (`glob(["cli_client/*.py"], exclude
  = [the two files of :net])`) plus a dependency on `:net`. `net.py` already imports only
  the standard library and `cli_server.protocol`, and the package `__init__.py` is a
  docstring, so no code moves. A `net_test` built against `:net` only checks that
  `cli_client.net` imports without `curses` or the controller. `ServerConnection`,
  `CallbackListener`, `ConnectionLost`, `DISCONNECTED` and `HOST` are the API bot_client
  relies on.
- **`//cli_server`** (`cli_server/spec.md`): `//cli_server:cli_server_lib` visible to
  `//bot_client/...` as well as `//cli_client/...` (`visibility = ["//cli_client/...",
  "//bot_client/..."]`), so bot_client's tests can run an in-process `GameServer` and
  start `cli_server.main` as a child process. bot_client also relies on the error
  wording of rejected `action` requests ("... is not the pending turn ..." vs "... is
  not legal now ..."), pinned by its integration test.
- No change in `//engine`, `//rl_env` or `//arena`. Used as they are:
  `arena.BOT_NAMES`, `make_bot`, `BotAgent.decide`; `rl_env.DecisionBatch.single`,
  `to_engine`, the action constants, `TableBatch` / `StackConfig` (tests); the engine's
  `ActionType`, `LegalActions`, `TableView`, `HandResult` and events through
  `cli_server.protocol`.
