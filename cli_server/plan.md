# cli_server — implementation plan

A Python binary that hosts a single fixed-limit Omaha Hi-Lo 8-or-better table on
localhost. Clients (separate processes, possibly not written in Python) join over TCP.
The server runs the game hand by hand using `//engine`. It waits until at least two
players are seated, then waits a grace period after the last join before the first
hand. After that it plays hands for as long as at least two players are active, with a
short pause between hands so players can see the showdown.

The same binary also has an end-to-end test mode (`--e2e_test`). It replays recorded
hands from `cli_server/e2e_test_sets/*.json` through a real server and real client
processes, and checks what every client receives against the recording (see End-to-end
test mode).

The server holds no poker rules of its own. Table logic (seating, waiting queue,
blinds, betting, showdown, rebuy window, sitting out, leaving, action timeout) is the
engine's `Environment` / `Dealer`. Remote play goes through the engine's `RemoteAgent`
and `handle_join`. This project adds:
- a TCP transport with the topology the spec asks for (client → server on the server
  port, server → client on the client's callback port);
- a language-neutral JSON wire format;
- the lobby / grace-period loop and the pause between hands;
- the extra information clients need to draw the table like an online poker room:
  table settings on join, "whose turn it is" notifications, and lobby status;
- the e2e replay engine: case loader, stacked deck, scripted client processes and the
  checker.

## Layout

```
cli_server/
  spec.md
  plan.md
  BUCK
  e2e_test_sets/
    12_heads_up_three_raises_preflop_bb_scoops.json
    ...                # one recorded hand per file
  cli_server/
    __init__.py
    main.py          # command line parsing, logging, signal handling, entry point
    protocol.py      # JSON-lines framing; encode/decode of engine types and messages
    session.py       # ClientSession: one joined client; the `conn` given to RemoteAgent
    server.py        # GameServer: request listener, session registry, lobby, game loop
    e2e/
      __init__.py
      case.py        # Case: load and validate a test-set file
      deck.py        # StackedDeck: deals the case's cards in the engine's deal order
      expect.py      # builds what every client is expected to receive
      client.py      # the scripted client process (`python -m cli_server.e2e.client`)
      driver.py      # runs one case: server + client processes; runs all cases
      check.py       # compares client transcripts with the expectations
  tests/
    fake_client.py   # test helper: a scripted TCP client with its own callback port
    test_protocol.py
    test_session.py
    test_server.py
    test_main.py
    test_e2e_case.py
    test_e2e_deck.py
    test_e2e_check.py
    test_e2e_cases.py  # runs every file in e2e_test_sets through the driver
```

## Build (buck2)

`cli_server/BUCK`:
- `python_library(name = "protocol", srcs = ["cli_server/__init__.py", "cli_server/protocol.py"],
  base_module = "", deps = ["//engine:engine"], visibility = ["PUBLIC"])`. This is a
  separate target so clients (e.g. `//cli_client`) can share the wire format without
  depending on the server.
- `python_library(name = "cli_server_lib", srcs = ["cli_server/session.py",
  "cli_server/server.py", "cli_server/main.py"], base_module = "",
  deps = [":protocol", ":e2e", "//engine:engine"], visibility = ["//cli_client/..."])`. The
  client's integration test runs an in-process `GameServer`.
- `python_library(name = "e2e", srcs = glob(["cli_server/e2e/*.py"]), base_module = "",
  resources = glob(["e2e_test_sets/*.json"]), deps = [":protocol", "//engine:engine",
  "//entities:entities"])`. The JSON files are resources, so they sit next to the
  `cli_server` package in the built binary. `driver.py` finds them at
  `Path(__file__).resolve().parents[2] / "e2e_test_sets"`. `driver.py` imports
  `GameServer` lazily, inside the function that runs a case, so `:e2e` does not depend on
  `:cli_server_lib` at import time and there is no dependency cycle.
- `python_binary(name = "cli_server", main_module = "cli_server.main",
  deps = [":cli_server_lib"])`.
- `python_test(name = "cli_server_test", srcs = glob(["tests/*.py"]),
  base_module = "cli_server_tests", deps = [":cli_server_lib"])`. Tests use `unittest`
  and the standard library only (`socket`, `socketserver`, `threading`, `json`,
  `subprocess`).

Run: `buck2 run //cli_server:cli_server -- --port 8008`,
`buck2 run //cli_server:cli_server -- --e2e_test [--case NAME]`, `buck2 test //cli_server/...`.

## Command line (`main.py`)

```
cli_server [--port PORT] [--grace-period SECONDS] [--action-timeout SECONDS]
           [--hand-pause SECONDS] [--max-seats N] [--max-buy-in BB] [--seed N]
           [--log-level LEVEL] [--e2e_test [--case NAME]]
```
- `--port` (spec: "port number"): TCP port the server listens on for client calls.
  Default `8008`. `0` picks a free port. The actual port is logged at startup.
- `--grace-period` (spec: "graceful period at the start of the game"): default `15`
  seconds. Must be ≥ 0. Tests use a fraction of a second.
- `--action-timeout` (spec: "time to act for client"): default `180` seconds, passed to
  `Environment(action_timeout=...)` (overriding the engine's 600 s default). Must be
  > 0. The same limit applies to the `hand_end` reply (rebuy / sit-in / leave window),
  because `RemoteAgent` uses one timeout for both.
- `--hand-pause` (spec: "time between hands"): default `3` seconds. Must be ≥ 0; `0`
  disables the pause. Tests use `0` or a fraction of a second.
- `--max-seats`, `--max-buy-in`, `--seed`: passed to `Environment`.
- `--e2e_test` (spec): a flag, false when not passed. Runs the end-to-end test mode
  instead of serving a game. `--e2e-test` is accepted as an alias.
- `--case NAME` (spec): a file in `e2e_test_sets`, with or without `.json` (e.g.
  `12_heads_up_three_raises_preflop_bb_scoops`). Without `--case`, every `*.json` file
  in `e2e_test_sets` is run, sorted by name. `--case` without `--e2e_test`, or a name
  that has no file, is an argument error (exit status 2; the unknown-name message lists
  the available cases).
- In e2e mode `--port`, `--grace-period`, `--hand-pause`, `--max-seats` and
  `--max-buy-in` are not used: every case gets its own server on a free port, with the
  settings the case needs (see Running a case). `--action-timeout` defaults to 10 s in
  this mode, so a stuck replay fails quickly; an explicit value is used as given.
  `--seed` (default `0` in this mode) seeds the shuffle that fills the cards the case
  does not fix, so runs are reproducible. `--log-level` applies as usual.
- Invalid values (negative times, non-numbers, a port outside `0..65535`) are rejected
  by `argparse` with exit status 2.
- The server binds to `127.0.0.1` only, because server and clients share a host (spec).
- Logging goes to stderr: listening address and the table settings, joins and leaves,
  lobby changes, each hand start (hand number, button, players), each hand result
  (winners, pots, stacks).
- SIGINT / SIGTERM: stop accepting requests. If a hand is running, the interrupt
  aborts it in the main thread (the engine's `_abort_hand` refunds the bets). A lobby
  wait or a pause between hands is woken up and ends at once. Then send `close` to
  every session and exit 0. In e2e mode an interrupt stops the current case, kills its
  client processes, prints the report so far and exits 1.

## Wire protocol (`protocol.py`)

### Framing
- Every message is one UTF-8 JSON object on one line (newline-delimited JSON). Every
  object has a `"type"` field.
- `read_message(file) -> dict | None` / `write_message(file, dict)` helpers. A line
  that is not valid JSON, or has no `type`, raises `ProtocolError`.
- Why JSON and not pickle: any local process can connect to the port, and unpickling
  untrusted data runs arbitrary code. JSON also lets clients be written in any
  language and can be debugged with `nc`.
- Forward compatibility: clients must ignore message types and fields they do not
  know. The informational messages below (`acting`, `lobby`) and the `table` field
  rely on this, so a client that ignores them still plays correctly (spec).

### Channels
There are two kinds of TCP connection, matching the spec:

1. **Client → server** (the server's listening port). A client may keep one
   connection open for all its calls or open one per call. Each request line gets
   exactly one response line: `{"type": "response", "ok": true, ...}` or
   `{"type": "response", "ok": false, "error": "<reason>"}`. Requests are processed in
   order per connection. Every request carries the player's `name`. The server uses
   it to tell clients apart (spec), not the connection.
2. **Server → client** (the client's callback port, given in `join`). The server
   opens one connection when the client joins and keeps it for the player's whole
   time at the table. It only sends notifications and requests. The client never
   writes on it; answers go back as calls on channel 1. Each message has
   `"to": <name>`, so one client process may host several players on one callback
   port.

### Client → server requests
| `type` | fields | effect |
|---|---|---|
| `join` | `name` (required, non-empty), `stack_bb`, `seat` (optional offset from the button; null = join as big blind), `callback_port` | joins the table (see Joining). The response carries `queued` (true if a hand is in progress and the player waits to be seated), `name`, `seat` (null when queued) and `table` (the table settings, see below) |
| `action` | `name`, `seq`, `action`: `fold` / `check` / `call` / `raise` / `leave` | answers the `turn` request with that `seq` |
| `hand_end` | `name`, `seq`, `rebuy_bb` (optional), `sit_in` (bool), `leave` (bool) | answers the `hand_end` request with that `seq` (rebuy / sit-in / leave window) |
| `rebuy` | `name`, `amount_bb` | rebuy between hands (e.g. a busted player while no game is running) |
| `sit_in` | `name` | return from sitting out between hands |
| `leave` | `name` | leave the table at any time |

Errors (`ok: false`): unknown player name; malformed request; for `action`, a `seq`
that is not the pending turn (late or duplicate) or an action that the turn's
`LegalActions` does not allow; for `hand_end`, no pending `hand_end` with that `seq`;
for `rebuy` / `sit_in`, the engine's `IllegalActionError` / `ValueError` message
(e.g. "during betting use the hand_end reply"). A rejected `action` does not use up
the turn. The client may send a corrected action until `action_timeout` runs out.

### Table settings (`table` in the join response)
`TableSettings` dataclass with `encode_settings` / `decode_settings`:
```
{"small_blind_bb": 0.5, "big_blind_bb": 1.0, "small_bet_bb": 1.0, "big_bet_bb": 2.0,
 "max_seats": 10, "max_buy_in_bb": 100.0 | null, "action_timeout": 180.0,
 "grace_period": 15.0, "hand_pause": 3.0}
```
- Blinds come from the dealer (`to_bb(dealer.small_blind)`, `to_bb(dealer.big_blind)`).
  The bet sizes follow the engine's fixed-limit structure: small bet = 1 BB (preflop,
  flop), big bet = 2 BB (turn, river).
- `max_seats`, `max_buy_in_bb` (null when there is no cap) and `action_timeout` come
  from the `Environment`. `grace_period` and `hand_pause` come from the server.
- Built once when the server starts (the settings do not change) and attached to every
  accepted join response. Rejected joins have no `table`.

### Server → client messages
Messages that map 1:1 to the engine's `RemoteAgent` messages, plus hole cards:
| `type` | fields | engine source |
|---|---|---|
| `seated` | `view` | `on_seated`: the player has been seated (at once, or after the hand they waited through) |
| `hole_cards` | `hand_number`, `cards`, `view` | `on_hole_cards`: the player's own 4 cards, sent after the blinds and the deal, before any `turn` |
| `event` | `event` | `on_event`: every public `TableEvent` (hand started, blinds, fold/check/call/raise, board cards, shown hands, returned bets, pot awards, player joined / sat out / returned / rebought / left). Sent to every seated player, including the one it is about |
| `turn` | `seq`, `view`, `legal`, `timeout` | `on_turn`: the client must answer with an `action` request carrying `seq`. `timeout` is the seconds the player has to answer (the action timeout), so the client can show its own countdown |
| `hand_end` | `seq`, `seat`, `result` | `on_hand_end`: the round summary. The client must answer with a `hand_end` request carrying `seq` |
| `sat_out` / `returned` | — | `on_sit_out` / `on_return` |
| `left` | `final_stack_bb` | `on_left`. The server closes the connection after it (`close`) |
| `close` | — | the server is closing this channel |

Informational messages generated by the server itself (no reply, optional for clients):
| `type` | fields | when |
|---|---|---|
| `acting` | `hand_number`, `seat`, `name`, `timeout` | a player has been asked to act. Sent to every other seated player (active and sitting out), not to the actor, who gets `turn`. `timeout` is the seconds the actor has, counted from now |
| `lobby` | `state`, `active`, `needed`, `starts_in` | no hand is running (see Lobby status). `state` is `"waiting"` (not enough players), `"starting"` (grace period ticking) or `"next_hand"` (pause between hands). `active` = number of `ACTIVE` seated players with chips, `needed` = the server's `required_players` (2, the engine's `MIN_PLAYERS`, outside e2e mode), `starts_in` = seconds until the next hand starts, or null while waiting |

Clients must answer `turn` and `hand_end` promptly. The dealer waits for each answer,
up to `action_timeout`. A missing answer to `turn` folds the hand and sits the player
out (engine rule). A missing answer to `hand_end` means no rebuy, sit-in or leave.

Timers are sent as relative seconds, not absolute times, because the client's clock
is not the server's. Clients count down from the moment the message arrives.

### Encoding of engine types
`protocol.py` has `encode_*` and `decode_*` functions for each type, so a Python
client can rebuild the engine dataclasses. Unit tests check that every type survives
a round trip.
- `Card` → `"As"`. `Hand` → `["As", "Kd", "7h", "2c"]`. `Board` → `{"flop": [...] | null,
  "turn": "9s" | null, "river": ... }`. Text forms come from `entities`
  (`Card.parse` / `str`).
- Enums (`Street`, `ActionType`, `PlayerStatus`) → their `.value` strings.
- `TableEvent` subclasses → `{"type": <snake_case class name>, "hand_no", "seq",
  ...the dataclass fields}`, e.g. `{"type": "player_acted", "hand_no": 3, "seq": 41,
  "seat": 1, "name": "B", "street": "flop", "action": "raise", "amount_bb": 1.0,
  "to_bb": 1.0, "all_in": false, "stack_bb": 37.0, "reason": null}`. Types:
  `hand_started`, `blind_posted`, `player_acted`, `board_dealt`, `hand_shown`,
  `bet_returned`, `pot_awarded`, `player_joined`, `player_sat_out`,
  `player_returned`, `player_rebought`, `player_left`. `stack_bb` is the player's
  stack after the event. `participants` in `hand_started` is a list of
  `[seat, name, stack_bb]`. `decode_event` picks the dataclass by `type`.
- `LegalActions` → `{"can_check", "call_amount", "can_raise", "raise_to", "bet_unit",
  "raises_left"}`.
- `TableView` → all its fields. `seats` is a list of `SeatView` objects, `side_pots`
  a list of `{amount, eligible}`, `history` a list of events, and `hole_cards` is the
  recipient's own hand or null.
- `HandResult` → `hand_number`, `button`, `sb_seat`, `bb_seat`, `board`, `showdown`,
  `pots` (each `{amount, eligible, high: [[seat, amount]...], low: [...],
  uncontested}`), `history`. The per-seat dicts (`shown`, `net`, `stacks`, `names`,
  `dealt_in`) are merged into `seats: [{seat, name, dealt_in, net, stack, shown}]`,
  because JSON object keys cannot be ints. This message is the spec's "round summary":
  winners and pot amounts per pot, shown hands, and the updated stack of every player.
- `TableSettings`, `acting` and `lobby` have `encode_*` / `decode_*` helpers too
  (`decode_*` returns small frozen dataclasses `Acting`, `LobbyStatus`), so Python
  clients do not parse them by hand.
- Amounts are BB floats, as in the engine views.

## ClientSession (`session.py`)

One per joined client. It is the `conn` object handed to the engine's `RemoteAgent`
(which only needs `send(msg)`, `recv()` and `close()`), so `RemoteAgent` keeps
handling sequence numbers, the action timeout, discarding late replies and
disconnects.

State: `name`, `player` (the engine `Player`, set once `handle_join` returns it), the
outbound socket to `127.0.0.1:callback_port`, an outbound `queue.Queue` with a writer
thread, an `inbox` `queue.Queue` read by `recv()`, `pending_turn: (seq, LegalActions)
| None`, `pending_hand_end: seq | None`, `join_reply: JoinReply | None`, `closed`, and
two hooks given by the server: `on_turn(session, view)` and `on_status(session, kind)`.

- `send(msg)` is called by `RemoteAgent` on the dealer thread (or on a listener thread
  for `seated` after a join between hands). It must never block the dealer on a slow
  client. It converts the engine tuple message to JSON and puts it on the outbound
  queue:
  - `("joined", JoinReply)` is not forwarded. It is stored in `join_reply` and returned
    as the response to the `join` request. No hook is called for it (it is sent while
    `handle_join` holds the table lock).
  - `("turn", seq, view, legal)` sets `pending_turn`, adds `timeout` (the server's
    action timeout) to the JSON, enqueues it, then calls the `on_turn` hook, which
    announces the turn to the other players (see Turn announcements).
  - `("hand_end", seq, seat, result)` sets `pending_hand_end`.
  - `seated`, `sat_out`, `returned` and `left` call the `on_status` hook after being
    enqueued, because they change the number of active players (lobby) or end the
    session.
  - `("close",)` is forwarded. The writer then flushes and closes the socket, and
    `inbox` gets an EOF sentinel.
  - If the session is already broken, it raises `OSError`, which `RemoteAgent` treats
    as a closed connection.
- `notify(dict)`: enqueue a server-generated message (`acting`, `lobby`), adding
  `"to": name`. Never blocks and never raises; a broken or closed session drops it.
- Writer thread: writes queued lines to the socket. On a socket error the session is
  marked broken and the EOF sentinel goes to `inbox`. `RemoteAgent`'s reader then
  sees a lost connection and calls `player.leave()` (engine: folded at the next turn,
  removed after the hand).
- `recv()` blocks on `inbox` and raises `EOFError` on the sentinel.
- Request handlers (called from listener threads, see below) feed `inbox`:
  - valid `action` → `("action", seq, Action(type) | Leave())`, clears `pending_turn`;
  - valid `hand_end` → `("hand_end", seq, HandEndReply(rebuy_bb, sit_in, leave))`,
    clears `pending_hand_end`;
  - `leave` → `Leave()` (`RemoteAgent`'s reader calls `player.leave()`).

Hooks run on whatever thread called `send`. They must not call into the engine or wait
on the lobby condition while that thread may hold the table lock; they only enqueue
messages and poke the lobby (see Concurrency).

## GameServer (`server.py`)

`GameServer(port, grace_period, hand_pause, env_kwargs, required_players=MIN_PLAYERS,
max_hands=None)` owns the `Environment`, the `TableSettings`, the request listener,
the session registry (`dict[name, ClientSession]`, guarded by `registry_lock`) and the
lobby `threading.Condition` with a `generation` counter.
- `env_kwargs` is passed to `Environment(...)` as is. Besides the command-line values
  it may carry `chips_per_bb` and `deck`; the e2e driver uses both.
- `required_players` (≥ `MIN_PLAYERS`): the number of active players needed before the
  grace period starts. The command line always uses `MIN_PLAYERS` (spec: 2 players).
  The e2e driver sets it to the number of players in the case, so the hand starts only
  once all of them have joined, whatever the join timing.
- `max_hands`: when set, `serve_forever()` returns after that many hands have been
  played (the sessions are left open; `shutdown()` closes them). The command line never
  sets it; the e2e driver uses `1`.
- `port` / `settings` are readable after construction (the actual port when `0` was
  given).

### Request listener
A `socketserver.ThreadingTCPServer` on `127.0.0.1:port` (`allow_reuse_address`,
daemon threads). The handler reads request lines, dispatches on `type` and writes one
response per request. A `ProtocolError` gets an error response, and the connection
stays open.

### Joining
1. Validate the fields: `name` is a non-empty string not already in the registry,
   `callback_port` is an int, `stack_bb` is a number. The engine re-checks the stack,
   offset, duplicate names and table size.
2. Connect to `127.0.0.1:callback_port` (5 s timeout). On failure respond with
   "cannot reach callback port".
3. Create a `ClientSession` and register it under `name` before joining. Otherwise a
   `turn` could reach the client, and its `action` arrive, before the name is known.
4. Put `Join(name, stack_bb, seat)` into the session's `inbox` and call the engine's
   `handle_join(env, session, timeout=action_timeout)`. It reads the `Join`, calls
   `env._join` under the table lock, attaches a `RemoteAgent(session)` and sends
   `("joined", JoinReply)`, which the session captures. If a hand is in progress the
   player waits in the engine's queue and is seated after the hand ("put as waiting
   player"). Store the returned `Player` in `session.player`.
5. Respond with the `JoinReply` (`accepted`, `queued`, `name`, `seat`, or `error`),
   plus `table` (the encoded `TableSettings`) when accepted. On rejection
   `handle_join` closes the session: unregister it and close the socket.
6. On acceptance, poke the lobby with `arrival=True` (restarts the grace period, see
   Lobby). If no hand is running, the new session also gets the current `lobby`
   status at once, even if it has not changed for the others.

The `seated` notification (and the first `lobby`) may reach the callback port before
the `join` response reaches the client. Clients must accept either order (as the
engine's `join_table` does).

### Other requests
- `action` / `hand_end`: look up the session by `name`. Check that `seq` matches the
  pending request. For `action`, also check `legal.allows(type)` (`leave` is always
  allowed). Then feed `inbox` as above.
- `rebuy` / `sit_in`: call `player.rebuy(amount_bb)` / `player.sit_in()` directly from
  the handler thread. The engine guards this bookkeeping with its lock and rejects it
  during betting. On success, poke the lobby with `arrival=True`: the player is back,
  the game may now be able to start, and the grace period counts from them.
- `leave`: feed `Leave()` to the session. If the player is still waiting in the
  engine's queue they are dropped at once. Poke the lobby.
- Session cleanup: when the engine removes a player (`on_left` → `left` + `close`),
  or a waiting player leaves, the session is closed and unregistered, and the name can
  be used again. The `on_status` hook for `left` does this and pokes the lobby.

### Turn announcements (`acting`)
The session's `on_turn` hook runs on the dealer thread right after the actor's `turn`
message is enqueued:
1. Take a snapshot of the registered sessions under `registry_lock`, then release it.
2. For every other session whose `player` is seated (`ACTIVE` or `SITTING_OUT`; not
   `WAITING` or `LEFT`), call `notify({"type": "acting", "hand_number":
   view.hand_number, "seat": view.seat, "name": view.me.name, "timeout":
   action_timeout})`.

Ordering: the engine delivers every event of the hand synchronously, in seat order,
before it calls the next `on_turn`, so each client's outbound queue already holds the
events that led to this turn. The `acting` line is queued after them and before the
next `player_acted` event, so a client highlights the right seat at the right time.
A forced fold (leave requested) does not call `on_turn`, so no `acting` is sent for it.
The `acting` state ends with the actor's `player_acted` event, which clients use to
clear the highlight.

### Lobby status (`lobby`)
`_lobby_status() -> LobbyStatus(state, active, needed, deadline)` is computed by the
main thread from `env` (never under the lobby condition):
- `active = len(env.active_players)`, `needed = required_players`;
- LOBBY with `active < needed` → `"waiting"`, no deadline;
- LOBBY with `active >= needed` → `"starting"`, deadline = `last_arrival +
  grace_period`;
- pause between hands → `"next_hand"`, deadline = hand end + `hand_pause`.

`_publish_lobby(status)` sends `lobby` (with `starts_in = max(0, deadline -
monotonic())`, or null) to every registered session, but only if `(state, active,
needed, deadline)` differs from the last status published. So it is re-sent whenever
the number of active players changes, the grace period restarts, or the state changes,
and not on every wake-up. Clients count `starts_in` down locally.

No `lobby` messages are sent while a hand is running; clients drop the lobby display
on `hand_started`.

### Lobby and game loop (main thread)
```
hands = 0
while not stopping:
    while not stopping:                          # LOBBY: no hand is running
        status = lobby_status()                  # reads env, outside cond
        publish_lobby(status)
        if status.state == "starting" and status.deadline <= monotonic():
            break
        wait_for_poke(gen, timeout=deadline - monotonic() or None)
    log("game starts")
    while not stopping and env.can_start():      # PLAYING
        try:
            env.start_round()                    # blocks while clients act
        except NotEnoughPlayersError:
            break
        hands += 1
        if max_hands is not None and hands >= max_hands:
            return                               # e2e: one hand per case
        if hand_pause > 0 and env.can_start():   # PAUSE between hands
            pause_until = monotonic() + hand_pause
            publish_lobby(next_hand status)
            sleep_until(pause_until)             # interrupted only by stopping
    log("waiting for players")                   # < 2 active players: back to LOBBY
```
- At startup the server is in LOBBY. Nothing happens until two players are seated
  (`env.can_start()`: two or more `ACTIVE` players with chips) and, with
  `required_players` above two, until that many are active. Clients see `lobby`
  `"waiting"` with `active` 0/1.
- `last_arrival` is set on every accepted join, and on a rebuy or sit-in that brings
  a player back while in LOBBY, so the grace period is always counted from the last
  player who arrived. Each new arrival restarts the grace period and publishes a new
  `"starting"` status with a fresh `starts_in`. If a leave or a sit-out drops the
  table below two players, the loop publishes `"waiting"` and waits with no timeout.
- Joins while PLAYING: between hands (including during the pause) they are seated at
  once by the engine. During a hand they are queued and seated after it. Joins while
  PLAYING do not restart any timer.
- Pause between hands: after `start_round()` returns (all `hand_end` replies are in,
  the button has moved, leavers are removed and queued joiners seated), the server
  waits `hand_pause` seconds before the next `start_round()`, so the showdown and the
  pot awards stay on the clients' screens. Clients get `lobby` `"next_hand"` with
  `starts_in = hand_pause`. Players leaving during the pause can drop the table below
  two; `can_start()` is checked again after the pause. No pause is made after the last
  hand before returning to LOBBY, nor before the first hand (the grace period covers
  that).
- The grace period applies again whenever play stops (fewer than two active players)
  and later becomes possible again.
- `start_round()` raising `NotEnoughPlayersError` (players left between the check and
  the start) sends the loop back to LOBBY.

### Concurrency
- Pokes: `_poke(arrival=False)` takes the condition, sets `last_arrival =
  monotonic()` if `arrival` is true and the loop is in LOBBY, increments `generation`
  and `notify_all()`s. `wait_for_poke(gen, timeout)` waits on the condition only while
  `generation == gen`, so a poke that happens between reading `env` and waiting is not
  lost. Poke sources: accepted join, rebuy, sit-in, leave request, and the session's
  `on_status` hook (`seated`, `sat_out`, `returned`, `left`, which also covers
  disconnects and timeouts handled inside the engine).
- The main thread never holds the condition while calling into the engine, and hooks
  never call into the engine. So the table lock and the lobby condition are never held
  together and there is no lock-order deadlock.
- `registry_lock` is held only to read or change the registry, never while sending or
  calling into the engine.
- `stopping` is set by the signal handler, which also pokes the condition so a lobby
  wait or a pause ends at once (the pause sleeps with `wait_for_poke` too).
- `serve_forever()` may run on a thread other than the main thread (the e2e driver
  does this). Then signals are not involved and `stop()` sets `stopping` and pokes.

## How the spec's example maps to messages
- Join: response with `queued`, `seat` and `table` (blinds, bet sizes, seats, max
  buy-in, time to act, grace period). While the grace period ticks, every client gets
  `lobby` `"starting"` with `starts_in`, re-sent after each new join ("graceful period
  starts ticking").
- Blinds: `event` `blind_posted` (`blind` `"SB"` / `"BB"`) with `stack_bb`, sent to
  every seated client, the poster included.
- "hand for player X": `hole_cards` to client X only. No client ever receives another
  player's hole cards before showdown.
- Client D joins mid-hand: `join` response `queued: true`. D gets no messages for that
  hand (engine: not seated, no events, no `acting`, no `hand_end`). After the hand D
  gets `seated`, everyone gets `event` `player_joined`, and D is dealt into the next
  hand.
- "Call for action": `turn` (with `timeout`) to the actor, `acting` (seat, name,
  timeout) to every other seated client. The actor's reply is an `action` call on the
  server port.
- fold / check / call / raise: `event` `player_acted` with `amount_bb`, `to_bb` and
  `stack_bb`, sent to everyone. The spec lists only the other players for fold and
  check. Sending the same event to the actor too is harmless and keeps one code path.
- Flop / turn / river: `event` `board_dealt` with the new cards and the whole board.
- Round summary: `hand_end` with the encoded `HandResult` (winners per pot, amounts,
  shown hands, stacks of all players).
- "Server starts 2nd round": after the `hand_end` replies, `lobby` `"next_hand"` with
  `starts_in = hand_pause`, then after the pause `event` `hand_started`.

## End-to-end test mode (`cli_server/e2e/`)

`cli_server --e2e_test [--case NAME]` replays recorded hands through the real server
(`GameServer`, the wire protocol, `ClientSession`, `RemoteAgent`) and the real engine.
Each player in the hand is a separate, very simple client process. It plays the
recorded decisions and records everything it receives. The cards come from the
recording: the deck is stacked. Afterwards the transcripts are checked against what
the recording says every client must have seen. One case = one file = one hand,
played on a fresh server.

### Case files (`e2e_test_sets/*.json`, `case.py`)
A case is one recorded hand, as in `12_heads_up_three_raises_preflop_bb_scoops.json`:
- `hand_id`, `description`: used only for logging and the report.
- `game`: `variant` (`"omaha_hi_lo_8"`), `betting` (`"fixed_limit"`), `unit`
  (`"chips"`), `blinds {sb, bb}`, `small_bet`, `big_bet`, `cap_raises`, `rake`.
- `players`: `id` (the player's name), `seat` (table seat number; only the clockwise
  order matters), `stack` (chips), `hole` (4 cards or null when unknown).
- `button`: `id` of the button player. `hero` is ignored (every player is replayed).
- `streets`: `street` (`preflop` / `flop` / `turn` / `river`), `board` (the cards
  dealt on that street: 0 / 3 / 1 / 1), `actions` (`player`, `action` in `post_sb`,
  `post_bb`, `fold`, `check`, `call`, `bet`, `raise`; `amount` = chips put in by this
  action, `to` = the player's street bet after it; both optional for fold / check),
  `pot_after`.
- `showdown`: `player`, `hole`, `mucked`, `hi` / `lo` (category and cards; for the
  reader, not checked: the engine reports hands, not hand names).
- `pots`: `type` (`main` / `side`), `amount`, `eligible`, `hi {winners, amount}`, `lo
  {winners, amount} | null`, `odd_chips`, `awards {id: chips}`.
- `result`: `ended_by` (`"showdown"` / `"fold"`), `net {id: chips}`, `final_stacks {id:
  chips}`.

`load_case(path) -> Case` parses the file into frozen dataclasses (`Case`,
`CasePlayer`, `CaseStreet`, `CaseAction`, `CasePot`, `CaseResult`) and validates it.
A file that fails validation is reported as an ERROR (a bad test, not a failed test)
with the reason. Checks:
- the game is the one the engine plays: `variant` `omaha_hi_lo_8`, `betting`
  `fixed_limit`, `unit` `chips`, `rake` 0, `sb * 2 == bb`, `small_bet == bb`,
  `big_bet == 2 * bb`, `cap_raises == 4` (read as the engine's cap of four bets per
  street: a bet and three raises, the big blind counting as the bet preflop);
- 2 to 10 players with unique ids and seats and positive stacks; `button` is one of
  them;
- all cards parse (`Card.parse`), each known hole has 4 cards, the boards have 3 / 1
  / 1 cards, and no card appears twice in the case (a hole listed both in `players`
  and in `showdown` must be the same hand);
- every player at showdown has known hole cards (unknown cards are random, and a
  random hand could change the result);
- actions name known players; preflop starts with `post_sb` and `post_bb`; `bet` only
  when nobody has bet on the street yet; the running sums of `amount` agree with `to`,
  `pot_after`, the stacks, `net` and `final_stacks` (`final_stack = stack + net`, the
  nets sum to 0); for each pot `awards` sum to `amount`.

### Seating: case seats → engine seats
The engine numbers seats clockwise from 0 and has no gaps. The driver orders the case
players clockwise by `seat`, starting with the button player. The k-th of them (k = 0
is the button) gets engine seat k:
- the button player joins first, at an empty table, and takes the button (engine
  rule);
- the second joins next and is seated after them (engine rule for a one-player
  table);
- the k-th (k ≥ 2) joins with `seat = k`, the offset "right of the button" with k
  players seated, so it lands at the end of the clockwise list.

The driver waits for each join response before starting the next client, so the join
order is fixed. With `required_players = n` and `grace_period = 0`, the hand starts
right after the last join, with the button on engine seat 0. The engine then picks the
blinds: heads-up the button is SB, otherwise the next two players. Case validation
checks that the `post_sb` / `post_bb` players agree.

### Stacked deck (`deck.py`)
`StackedDeck(script: Sequence[Card | None], rng)` is an `entities.Deck` subclass. It
is passed to the engine as `Environment(deck=...)`. `script` lists the cards in the
order the dealer takes them from the deck, and `None` marks a card the case does not
fix. `shuffle()` calls `Deck.shuffle()` and then puts the script on top with
`Deck.put_on_top(...)` (see Required entities capabilities). Each `None` is replaced
by the next shuffled card that the script does not use. Unknown cards are therefore
random (seeded by `--seed`) but never duplicate a known card.

The script follows the engine's documented deal order (engine plan, Dealer → Hand
flow; no burn cards):
1. hole cards: four rounds, one card per round to each participant, starting with the
   first player left of the button (engine seats 1, 2, …, n−1, 0);
2. flop (3 cards), turn (1), river (1). A street the hand does not reach is `None`.

If the engine's deal order ever changes, the `hole_cards` / `board_dealt` checks fail
for every case, which makes the cause easy to see.

### Units
The engine runs with `chips_per_bb = case.game.blinds.bb`, so engine chips are case
chips. Odd-chip splits then happen exactly as in the recording (with the default 100
chips per BB a 1-chip odd chip of the case would be split differently). On the wire
everything is in BB, as usual: the driver joins with `stack_bb = stack / bb`. The
checker converts every BB float it receives back to chips with `round(x * bb)` and
compares integers.

### Scripted client process (`client.py`)
A very simple client (not `cli_client`), started as `python -m cli_server.e2e.client`
with `sys.executable`. The parent's `sys.path` is passed as `PYTHONPATH`, so the child
finds the same modules inside the buck2-built binary or test. It uses only
`protocol.py` (`read_message` / `write_message`).
- Its config is one JSON line on stdin: server port, `name`, `stack_bb`, `seat`
  offset (or null), `action_timeout`, and `script`: this player's actions in order,
  each `{street, action}`, where `action` is the engine action (`bet` and `raise` are
  both `raise`; blinds are not in the script, the engine posts them).
- It listens on a free callback port, connects to the server port, sends `join` and
  prints `{"type": "joined", "response": ...}` on stdout. The driver waits for this
  line before starting the next client.
- It reads its callback channel and appends every message, in arrival order, to its
  transcript. It answers:
  - `turn`: the next scripted action, if its street matches `view.street` and
    `legal` allows it. Otherwise it records a problem (`unexpected turn`, `street
    mismatch`, `not legal: ...`) and sends `fold`, so the hand still finishes. A
    rejected `action` response is recorded as a problem too.
  - `hand_end`: no rebuy, no sit-in, no leave.
  - `close` (or EOF): prints `{"type": "transcript", "messages": [...], "responses":
    [...], "problems": [...], "unused_script": [...]}` on stdout and exits 0.
- Any exception is reported in `problems` (the transcript is still printed) and the
  process exits 1.

### Running a case (`driver.py`)
`run_case(case, seed, action_timeout) -> CaseReport`:
1. Build the seating and the `StackedDeck` script from the case.
2. Create `GameServer(port=0, grace_period=0, hand_pause=0, env_kwargs={action_timeout,
   chips_per_bb: bb, deck: StackedDeck(script, Random(seed)), seed}, required_players=n,
   max_hands=1)` and run `serve_forever()` on a background thread.
3. Start the client processes one at a time in seating order. Each waits for its
   `joined` line.
4. Wait until the server thread returns (the hand was played) and call `shutdown()`.
   This sends `close` to every session. Collect each client's `transcript` line and
   wait for the process to exit.
5. Overall deadline per case: `30 s + 4 × action_timeout`. When it passes, the driver
   stops the server, kills the clients and reports FAIL "timed out" with whatever
   transcripts it has.
6. Check (below) and return the report: PASS / FAIL (with the list of mismatches) /
   ERROR (invalid case, or the server or a client crashed).

`run_cases(names | None, seed, action_timeout) -> int` runs the selected cases
sequentially, each on its own server. It prints one line per case (`PASS`, `FAIL` or
`ERROR`, the name, the description). Under each failed case it prints the mismatches:
client, message index, expected and actual, at most 20 per case. Then a summary line
`N passed, M failed, K errors`. It returns the exit status: 0 if every case passed,
otherwise 1. Every check is collected, not just the first failure.

### What is checked (`expect.py`, `check.py`)
`expect.py` derives from the case, with the engine seats from Seating, the expected
public events of the hand and the expected result. `check.py` compares every client's
transcript with them.

1. **Joins**: every join was accepted, not queued, at the planned engine seat, and its
   `table` shows the blinds 0.5 / 1 BB. Every client got `seated`.
2. **Public information** (spec: "all clients receive correct updates on
   information"):
   - The `event` messages from `hand_started` to the last award are identical for all
     clients (same events, same order, same `seq`).
   - They match the expected stream. Exact match, in order:
     `hand_started` (button 0, SB / BB seats, participants with their starting
     stacks), `blind_posted` × 2 (amount, `all_in`, `stack_bb`), then for each street
     `board_dealt` (exactly the case's cards; not for preflop) followed by one
     `player_acted` per case action (seat, name, street, action with `bet` → `raise`,
     `amount_bb`, `to_bb` when the case gives it, `stack_bb` = the stack tracked from
     the case, `all_in` = that stack is 0, `reason` null).
   - Settlement events, which a hand history does not list one by one, are checked by
     content: the `hand_shown` events are exactly the case's showdown players with
     their holes (any order); the `pot_awarded` amounts, summed per pot and player,
     equal the case's `awards`, and summed per share they equal `hi.amount` /
     `lo.amount` (`whole` counts as hi); the winners per share equal `hi.winners` /
     `lo.winners`; `bet_returned` amounts plus awards give the final stacks.
   - No `hand_shown` when `ended_by` is `"fold"`.
   - Every client got `hole_cards` exactly once, before its first `turn`, with its own
     case hole when known (4 cards that are not in the case otherwise). No client got
     another player's cards before showdown, in `hole_cards` or in any `view`.
   - Turns: each client got one `turn` per scripted action, in order. Every other
     client got one `acting` for it (actor's seat and name) after the preceding events
     and before the actor's `player_acted`. The actor got no `acting` for its own turn.
   - No `lobby` message between `hand_started` and `hand_end`.
3. **Stacks** (spec: "stacks are correctly updated through the hand and at the end of
   it"): every `stack_bb` in the events equals the stack tracked from the case at that
   point. The seats in each `turn` view show the same stacks. `hand_end` seats have
   `stack` = `final_stacks` and `net` = `net`, for every client.
4. **Winners and pot split** (spec): `hand_end` `result` on every client has
   `showdown` = (`ended_by == "showdown"`). It has one pot per case pot, main first,
   then side pots in order: `amount`, `eligible` (ids → seats), `high` / `low` winners
   and amounts equal to the case (per winner as in `awards`, odd chips included),
   `uncontested` when the hand ended by fold. The sum of the final stacks equals the
   sum of the starting stacks.
5. **Script fully played**: no client has `problems` or `unused_script`. No `action`
   request was rejected. Every client exited 0.

## Required engine capabilities
Used as they are: `Environment` (incl. `max_seats`, `action_timeout`,
`active_players`, `can_start`, and for e2e mode `chips_per_bb` and `deck`),
`Dealer.small_blind` / `big_blind` / `max_buy_in` / `to_bb`, `MIN_PLAYERS`,
`handle_join`, `RemoteAgent`, `Join`, `JoinReply`, `Leave`, `HandEndReply`, `Action`,
`ActionType`, `LegalActions`, `IllegalActionError`, `NotEnoughPlayersError`, the views
(`TableView.seat` / `me` identify the actor for `acting`) and the `TableEvent` classes
from `engine.events`.

Already in the engine and relied on:
- **Hole cards on deal.** `on_hole_cards(player, hand, view)` is called for each
  dealt-in player after the blinds and the deal, before the first turn. `RemoteAgent`
  forwards it as the one-way message `("hole_cards", hand, view)`. The session encodes
  it as the `hole_cards` message, taking `hand_number` from `view`.
- **Stack in events.** Every event about a player carries `stack_bb`, the player's
  stack after the event. `RemoteAgent` forwards events as the one-way message
  `("event", event)`.
- **Synchronous event delivery before the next turn**, which gives `acting` its
  ordering.
- **Injected deck and chip unit** (e2e mode): `Environment(deck=..., chips_per_bb=...)`
  uses the given `Deck` for every hand (`reset()` and `shuffle()` once per hand, then
  `deal_one()` only), and `chips_per_bb` sets the blinds (SB = `chips_per_bb // 2`).
  Both are part of the engine's contract, together with the deal order (see
  `engine/plan.md`, "Dealer → Deck use and deal order (contract)"); `chips_per_bb`
  must be an even int >= 2.
- **Seating of the first joiners** (empty table: first joiner takes the button,
  second sits after them; offset `n` = right of the button) and the blind order, used
  by the e2e seating.

## Required entities capabilities
- `Deck.put_on_top(cards)` (see `entities/plan.md`, "Stacking"): moves
  the given cards, which must all still be in the deck and be distinct, to the top of
  the deck in the given order, so they are dealt first. The other cards keep their
  order. All-or-nothing like `remove`. `StackedDeck` needs this because `Deck` keeps
  its card list private.

## Tests
- `test_protocol`: encode/decode round trip for `Card`, `Hand`, `Board`, every
  `TableEvent` type, `LegalActions`, `TableView`, `HandResult`, `TableSettings`
  (with and without max buy-in), `acting`, `lobby` (with null `starts_in`) and every
  request and message type; invalid JSON, missing `type` and unknown action names
  raise `ProtocolError`.
- `test_session`: engine messages map to the right JSON lines; `turn` carries
  `timeout` and calls the `on_turn` hook after it is queued; `seated` / `sat_out` /
  `returned` / `left` call `on_status`; `joined` is captured, not forwarded, and calls
  no hook; `notify` adds `to` and never raises on a broken session; `send` does not
  block on a client that does not read; a broken callback socket makes `recv` raise
  `EOFError`; `close` flushes, then closes.
- `test_server` (an in-process `GameServer` on port 0 with a 0.2 s grace period, a
  short `action_timeout`, `hand_pause` 0 unless the test is about the pause, a fixed
  seed, and `fake_client` clients; `fake_client` ignores message types it does not
  handle):
  - no hand starts with one client; with two, the first hand starts only after the
    grace period;
  - a third client that joins within the grace period is dealt into the first hand,
    and each join restarts the timer;
  - the join response carries `table` with the blinds 0.5/1, bets 1/2, the
    configured `max_seats`, `max_buy_in_bb`, `action_timeout`, `grace_period` and
    `hand_pause`; a rejected join has no `table`;
  - lobby: the first client gets `"waiting"` with `active` 1; after the second join
    both get `"starting"` with `starts_in` close to the grace period; a third join
    re-sends `"starting"` with a new `starts_in`; a leave that drops the table to one
    player sends `"waiting"`; a rebuy that brings a busted player back sends
    `"starting"`; no `lobby` is sent while a hand is running; unchanged status is not
    re-sent;
  - `required_players = 3`: two joined players stay `"waiting"` with `needed` 3; the
    third join starts the grace period;
  - `max_hands = 1`: `serve_forever()` returns after one hand, and `shutdown()` then
    sends `close` to every client;
  - `env_kwargs` with `chips_per_bb` and a `deck` reach the `Environment` (blinds in
    events are still 0.5 / 1 BB; the dealt cards come from the given deck);
  - `acting`: for every `turn`, each other seated client (sitting-out ones included)
    gets one `acting` with the actor's seat, name and timeout, after the events that
    preceded the turn and before the actor's `player_acted`; the actor and a queued
    (waiting) client get none; no `acting` for a forced fold of a leaving player;
  - pause between hands: with `hand_pause` 0.3, clients get `lobby` `"next_hand"`
    after their `hand_end`, and the next `hand_started` comes no earlier than the
    pause after the last `hand_end` reply; a client joining during the pause is seated
    at once and dealt into the next hand; players leaving during the pause send the
    server back to LOBBY;
  - a client that joins mid-hand gets `queued: true`, no messages for that hand, then
    `seated`, and is dealt into the next hand;
  - every client receives the blind events with stacks, its own `hole_cards` (and
    never another player's) before any `turn`, the action and board events, and a
    `hand_end` whose stacks match the engine;
  - actions are routed by name to the right player; wrong `seq`, an illegal action
    and an unknown name get errors, and a corrected action within the timeout is
    accepted;
  - a duplicate name, an unreachable callback port and a malformed join are rejected;
  - a client that never answers `turn` is folded and sits out; a client whose callback
    socket closes is removed; the `leave` request works mid-hand and between hands,
    and the name can be reused afterwards;
  - a busted player's `rebuy` request between hands brings the table back to two
    players, and play resumes after the grace period;
  - a stop request ends a lobby wait and a pause at once;
  - the total chips stay the same across hands, apart from joins, rebuys and leaves.
- `test_main`: argument defaults (port 8008, grace period 15 s, action timeout 180 s,
  hand pause 3 s, `e2e_test` false, `case` none), `--port 0`, `--hand-pause 0`, invalid
  values (negative times, zero action timeout, out-of-range port) rejected; `--case`
  without `--e2e_test` and an unknown case name exit 2; `--e2e_test` defaults
  (action timeout 10 s, seed 0); `--case NAME` and `--case NAME.json` pick the same
  file.
- `test_e2e_case`: the sample case loads into the expected dataclasses. Invalid cases
  are rejected with a clear reason: wrong variant / betting / blinds / bet sizes / cap,
  rake, duplicate card, unknown player in an action, showdown player with unknown hole,
  `players` and `showdown` holes that differ, inconsistent `pot_after` / `to` / `net` /
  `final_stacks`.
- `test_e2e_deck`: the seating gives the button engine seat 0 and keeps the clockwise
  order, including seats with gaps (1, 4, 6) and a button that is not the lowest
  seat. The script puts the hole cards in the engine's round-robin order from the left
  of the button and the board after them. A dealer using `StackedDeck` deals exactly
  the case's holes and board. `None` slots get cards not used elsewhere, and the same
  seed gives the same filler. Every hand reshuffles and restacks.
- `test_e2e_check`: synthetic transcripts. A correct one passes. Each kind of
  mismatch is reported with client and message index: a changed stack, a missing
  `acting`, an extra `lobby` during the hand, a wrong pot split, another player's hole
  cards, clients seeing different event streams, unused script.
- `test_e2e_cases`: `run_cases(None, seed=0, action_timeout=10)` over every file in
  `e2e_test_sets` returns 0. This makes the recorded hands part of `buck2 test`
  (server, wire protocol and engine together).
