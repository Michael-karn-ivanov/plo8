# cli_server — implementation plan

A Python binary that hosts a single fixed-limit Omaha Hi-Lo 8-or-better table on
localhost. Clients (separate processes, possibly not written in Python) join over TCP.
The server runs the game hand by hand using `//engine`. It waits until at least two
players are seated, then waits a grace period after the last join before the first
hand. After that it plays hands for as long as at least two players are active.

The server holds no poker rules of its own. Table logic (seating, waiting queue,
blinds, betting, showdown, rebuy window, sitting out, leaving, action timeout) is the
engine's `Environment` / `Dealer`. Remote play goes through the engine's `RemoteAgent`
and `handle_join`. This project adds three things:
- a TCP transport with the topology the spec asks for (client → server on the server
  port, server → client on the client's callback port);
- a language-neutral JSON wire format;
- the lobby / grace-period loop.

## Layout

```
cli_server/
  spec.md
  plan.md
  BUCK
  cli_server/
    __init__.py
    main.py          # command line parsing, logging, signal handling, entry point
    protocol.py      # JSON-lines framing; encode/decode of engine types and messages
    session.py       # ClientSession: one joined client; the `conn` given to RemoteAgent
    server.py        # GameServer: request listener, session registry, lobby, game loop
  tests/
    fake_client.py   # test helper: a scripted TCP client with its own callback port
    test_protocol.py
    test_session.py
    test_server.py
    test_main.py
```

## Build (buck2)

`cli_server/BUCK`:
- `python_library(name = "protocol", srcs = ["cli_server/__init__.py", "cli_server/protocol.py"],
  base_module = "", deps = ["//engine:engine"], visibility = ["PUBLIC"])`. This is a
  separate target so future clients can share the wire format without depending on
  the server.
- `python_library(name = "cli_server_lib", srcs = ["cli_server/session.py",
  "cli_server/server.py", "cli_server/main.py"], base_module = "",
  deps = [":protocol", "//engine:engine"])`.
- `python_binary(name = "cli_server", main_module = "cli_server.main",
  deps = [":cli_server_lib"])`.
- `python_test(name = "cli_server_test", srcs = glob(["tests/*.py"]),
  base_module = "cli_server_tests", deps = [":cli_server_lib"])`. Tests use `unittest`
  and the standard library only (`socket`, `socketserver`, `threading`, `json`).

Run: `buck2 run //cli_server:cli_server -- --port 8008`, `buck2 test //cli_server/...`.

## Command line (`main.py`)

```
cli_server [--port PORT] [--grace-period SECONDS] [--action-timeout SECONDS]
           [--max-seats N] [--max-buy-in BB] [--seed N] [--log-level LEVEL]
```
- `--port` (optional, the spec's "port number"): TCP port the server listens on for
  client calls. Default `8008`. `0` picks a free port. The actual port is logged at
  startup.
- `--grace-period`: default `15` seconds (spec). Tests use a fraction of a second.
- `--action-timeout`: passed to `Environment(action_timeout=...)`, default the
  engine's `DEFAULT_ACTION_TIMEOUT` (600 s).
- `--max-seats`, `--max-buy-in`, `--seed`: passed to `Environment`.
- The server binds to `127.0.0.1` only, because server and clients share a host (spec).
- Logging goes to stderr: listening address, joins and leaves, each hand start (hand
  number, button, players), each hand result (winners, pots, stacks).
- SIGINT / SIGTERM: stop accepting requests. If a hand is running, the interrupt
  aborts it in the main thread (the engine's `_abort_hand` refunds the bets). Then
  send `close` to every session and exit 0.

## Wire protocol (`protocol.py`)

### Framing
- Every message is one UTF-8 JSON object on one line (newline-delimited JSON). Every
  object has a `"type"` field.
- `read_message(file) -> dict | None` / `write_message(file, dict)` helpers. A line
  that is not valid JSON, or has no `type`, raises `ProtocolError`.
- Why JSON and not pickle: any local process can connect to the port, and unpickling
  untrusted data runs arbitrary code. JSON also lets clients be written in any
  language and can be debugged with `nc`.

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
| `join` | `name` (required, non-empty), `stack_bb`, `seat` (optional offset from the button; null = join as big blind), `callback_port` | joins the table (see Joining). The response carries `queued` (true if a hand is in progress and the player waits to be seated), `name` and `seat` (null when queued) |
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

### Server → client messages
These map 1:1 to the engine's `RemoteAgent` messages, plus hole cards:
| `type` | fields | engine source |
|---|---|---|
| `seated` | `view` | `on_seated`: the player has been seated (at once, or after the hand they waited through) |
| `hole_cards` | `hand_number`, `cards`, `view` | `on_hole_cards`: the player's own 4 cards, sent after the blinds and the deal, before any `turn` |
| `event` | `event` | `on_event`: every public `TableEvent` (hand started, blinds, fold/check/call/raise, board cards, shown hands, returned bets, pot awards, player joined / sat out / returned / rebought / left). Sent to every seated player, including the one it is about |
| `turn` | `seq`, `view`, `legal` | `on_turn`: the client must answer with an `action` request carrying `seq` |
| `hand_end` | `seq`, `seat`, `result` | `on_hand_end`: the round summary. The client must answer with a `hand_end` request carrying `seq` |
| `sat_out` / `returned` | — | `on_sit_out` / `on_return` |
| `left` | `final_stack_bb` | `on_left`. The server closes the connection after it (`close`) |
| `close` | — | the server is closing this channel |

Clients must answer `turn` and `hand_end` promptly. The dealer waits for each answer,
up to `action_timeout`. A missing answer to `turn` folds the hand and sits the player
out (engine rule). A missing answer to `hand_end` means no rebuy, sit-in or leave.

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
- Amounts are BB floats, as in the engine views.

## ClientSession (`session.py`)

One per joined client. It is the `conn` object handed to the engine's `RemoteAgent`
(which only needs `send(msg)`, `recv()` and `close()`), so `RemoteAgent` keeps
handling sequence numbers, the action timeout, discarding late replies and
disconnects.

State: `name`, the outbound socket to `127.0.0.1:callback_port`, an outbound
`queue.Queue` with a writer thread, an `inbox` `queue.Queue` read by `recv()`,
`pending_turn: (seq, LegalActions) | None`, `pending_hand_end: seq | None`,
`join_reply: JoinReply | None`, `closed`.

- `send(msg)` is called by `RemoteAgent` on the dealer thread. It must never block the
  dealer on a slow client. It converts the engine tuple message to JSON and puts it on
  the outbound queue:
  - `("joined", JoinReply)` is not forwarded. It is stored in `join_reply` and returned
    as the response to the `join` request.
  - `("turn", seq, view, legal)` also sets `pending_turn`. `("hand_end", seq, seat,
    result)` sets `pending_hand_end`.
  - `("close",)` is forwarded. The writer then flushes and closes the socket, and
    `inbox` gets an EOF sentinel.
  - If the session is already broken, it raises `OSError`, which `RemoteAgent` treats
    as a closed connection.
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

## GameServer (`server.py`)

`GameServer(port, grace_period, env_kwargs)` owns the `Environment`, the request
listener, the session registry (`dict[name, ClientSession]`, guarded by a lock) and a
`threading.Condition` that the lobby waits on.

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
   player").
5. Respond with the `JoinReply` (`accepted`, `queued`, `name`, `seat`, or `error`). On
   rejection `handle_join` closes the session: unregister it and close the socket.
6. On acceptance, notify the lobby condition and record `last_join = monotonic()`.

The `seated` notification may reach the callback port before the `join` response
reaches the client. Clients must accept either order (as the engine's `join_table`
does).

### Other requests
- `action` / `hand_end`: look up the session by `name`. Check that `seq` matches the
  pending request. For `action`, also check `legal.allows(type)` (`leave` is always
  allowed). Then feed `inbox` as above.
- `rebuy` / `sit_in`: call `player.rebuy(amount_bb)` / `player.sit_in()` directly from
  the handler thread. The engine guards this bookkeeping with its lock and rejects it
  during betting. On success, notify the lobby condition, because the game may now be
  able to start.
- `leave`: feed `Leave()` to the session. If the player is still waiting in the
  engine's queue they are dropped at once. Notify the lobby condition.
- Session cleanup: when the engine removes a player (`on_left` → `left` + `close`),
  or a waiting player leaves, the session is closed and unregistered, and the name can
  be used again.

### Lobby and game loop (main thread)
```
while not stopping:
    with cond:                                   # LOBBY: no hand is running
        while not stopping:
            if env.can_start():
                remaining = last_change + grace_period - monotonic()
                if remaining <= 0:
                    break
                cond.wait(remaining)
            else:
                cond.wait()
    log("game starts")
    while not stopping and env.can_start():      # PLAYING
        env.start_round()                        # blocks while clients act
    log("waiting for players")                   # < 2 active players: back to LOBBY
```
- At startup the server is in LOBBY. Nothing happens until two players are seated
  (`env.can_start()`: two or more `ACTIVE` players with chips).
- `last_change` is set on every accepted join. A rebuy or sit-in that brings a player
  back while in LOBBY also sets it, so the grace period is always counted from the
  last player who arrived. Each new arrival restarts the grace period. If a leave
  drops the table below two players, the loop goes back to waiting with no timeout.
- Joins while PLAYING: between hands they are seated at once by the engine. During a
  hand they are queued and seated after it. There is no grace period between hands.
  Hands follow each other immediately.
- The grace period applies again whenever play stops (fewer than two active players)
  and later becomes possible again.
- `start_round()` raising `NotEnoughPlayersError` (players left between the check and
  the start) is caught and sends the loop back to LOBBY.

## How the spec's example maps to messages
- Blinds: `event` `blind_posted` (`blind` `"SB"` / `"BB"`) with `stack_bb`, sent to
  every seated client, the poster included.
- "hand for player X": `hole_cards` to client X only. No client ever receives another
  player's hole cards before showdown.
- Client D joins mid-hand: `join` response `queued: true`. D gets no messages for that
  hand (engine: not seated, no events, no `hand_end`). After the hand D gets `seated`,
  everyone gets `event` `player_joined`, and D is dealt into the next hand.
- "Call for action": `turn`. The client's reply is an `action` call on the server port.
- fold / check / call / raise: `event` `player_acted` with `amount_bb`, `to_bb` and
  `stack_bb`, sent to everyone. The spec lists only the other players for fold and check. Sending the same
  event to the actor too is harmless and keeps one code path.
- Flop / turn / river: `event` `board_dealt` with the new cards and the whole board.
- Round summary: `hand_end` with the encoded `HandResult` (winners per pot, amounts,
  shown hands, stacks of all players).

## Required engine capabilities
Used as they are: `Environment`, `handle_join`, `RemoteAgent`, `Join`, `JoinReply`,
`Leave`, `HandEndReply`, `Action`, `ActionType`, `LegalActions`, `IllegalActionError`,
`NotEnoughPlayersError`, the views and the `TableEvent` classes from `engine.events`.

The two additions requested in `engine/spec.md` are now in the engine:
- **Hole cards on deal.** `on_hole_cards(player, hand, view)` is called for each
  dealt-in player after the blinds and the deal, before the first turn. `RemoteAgent`
  forwards it as the one-way message `("hole_cards", hand, view)`. The session encodes
  it as the `hole_cards` message, taking `hand_number` from `view`.
- **Stack in events.** Every event about a player carries `stack_bb`, the player's
  stack after the event. `RemoteAgent` forwards events as the one-way message
  `("event", event)`.

## Tests
- `test_protocol`: encode/decode round trip for `Card`, `Hand`, `Board`, every `TableEvent` type,
  `LegalActions`, `TableView`, `HandResult` and every request and message type;
  invalid JSON, missing `type` and unknown action names raise `ProtocolError`.
- `test_session`: engine messages map to the right JSON lines; `joined` is captured,
  not forwarded; `send` does not block on a client that does not read; a broken
  callback socket makes `recv` raise `EOFError`; `close` flushes, then closes.
- `test_server` (an in-process `GameServer` on port 0 with a 0.2 s grace period, a
  short `action_timeout`, a fixed seed, and `fake_client` clients):
  - no hand starts with one client; with two, the first hand starts only after the
    grace period;
  - a third client that joins within the grace period is dealt into the first hand,
    and each join restarts the timer;
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
  - the total chips stay the same across hands, apart from joins, rebuys and leaves.
- `test_main`: argument defaults (port 8008, grace 15 s), `--port 0`, invalid values
  rejected.
