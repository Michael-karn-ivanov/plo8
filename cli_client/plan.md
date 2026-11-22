# cli_client — implementation plan

A Python binary that lets one person play at the `//cli_server` Omaha Hi-Lo
(fixed-limit, 8-or-better) table from a terminal. It joins the table over the server's
JSON-lines protocol (see `cli_server/plan.md`, "Wire protocol"). It listens on its own
port for the server's callbacks and draws a full-screen, coloured table, similar to an
online poker room: seats around an oval felt, a board and pot in the middle, a log of
table actions on the right, and the player's own cards and an action menu at the
bottom.

The client holds no poker rules. Legal actions, bet sizes, pots and winners all come
from the server. The client only:
- keeps a local model of the table, built from the server's views and events;
- draws that model;
- turns key presses into protocol requests.

Its networking (`net.py`) is also a library of its own. Other projects use it to talk
to the server without the table model or the curses UI: `//bot_client` uses it to seat
bots at the table (see Networking library).

## Layout

```
cli_client/
  spec.md
  plan.md
  BUCK
  cli_client/
    __init__.py     # package marker only: no imports (owned by :net)
    main.py         # command line, startup / join, run the UI, shutdown
    net.py          # CallbackListener (server -> client), ServerConnection (client -> server)
    state.py        # TableState: local table model updated from server messages
    controller.py   # Client: message handling, menus, replies, presets (no curses)
    cards.py        # card / suit text and colours, hand descriptions
    render.py       # pure: TableState + controller UI state -> Frame (styled text grid)
    ui.py           # curses backend: colour pairs, drawing a Frame, key input loop
  tests/
    test_cards.py
    test_state.py
    test_controller.py
    test_render.py
    test_net.py
    test_net_library.py   # built against :net only
    test_integration.py
```

## Build (buck2)

`cli_client/BUCK` has three targets. The networking library is split out so that
projects using it do not get the UI.
- `python_library(name = "net", srcs = ["cli_client/__init__.py",
  "cli_client/net.py"], base_module = "", deps = ["//cli_server:protocol"],
  visibility = ["PUBLIC"])`. This is the reusable networking library (see Networking
  library). It owns the package file `cli_client/__init__.py`. That file must stay
  empty of imports, so importing `cli_client.net` loads nothing else from the package.
  This follows the same pattern as `//cli_server:protocol`.
- `python_library(name = "cli_client_lib", srcs = glob(["cli_client/*.py"],
  exclude = ["cli_client/__init__.py", "cli_client/net.py"]), base_module = "",
  deps = [":net", "//cli_server:protocol", "//engine:engine",
  "//entities:entities"])`. This is the table model, controller, rendering, curses UI
  and `main`. The wire format, its encode/decode functions and the engine dataclasses
  are reused from `//cli_server:protocol`. No code is copied.
- `python_binary(name = "cli_client", main_module = "cli_client.main",
  deps = [":cli_client_lib"])`.
- `python_test(name = "cli_client_test", srcs = glob(["tests/*.py"], exclude =
  ["tests/test_net_library.py"]), base_module = "cli_client_tests", deps =
  [":cli_client_lib", "//cli_server:cli_server_lib"])`. The integration test runs a
  real `GameServer` in-process. `cli_server/BUCK` makes `//cli_server:cli_server_lib`
  visible to `//cli_client/...` (and to `//bot_client/...`, per `cli_server/spec.md`).
- `python_test(name = "net_library_test", srcs = ["tests/test_net_library.py"],
  base_module = "cli_client_tests", deps = [":net"])`. It is built against `:net` only,
  so the test fails if `net.py` starts importing a module that `:net` does not contain.
- Standard library only: `curses`, `socket`, `threading`, `queue`, `argparse`,
  `unittest`. The demo toolchain has no third-party packages. `curses` is available on
  macOS and Linux. Only `ui.py` imports `curses`.

Run: `buck2 run //cli_client:cli_client -- --server-port 8008 --port 9001 --name Alice`.
Test: `buck2 test //cli_client/...`.

## Command line (`main.py`)

```
cli_client --server-port PORT --port PORT --name NICK [--stack BB] [--seat N]
           [--ascii] [--hand-end-timeout SECONDS] [--log-file PATH]
```
- `--server-port` (spec: "port of the running server"): default `8008`, the server's
  default.
- `--port` (spec: "its own port … to listen for server events"): required. `0` picks a
  free port. The port actually bound is the one sent in `join`.
- `--name` (spec: "player name (nick)"): required, non-empty.
- `--stack` (spec: "player's stack, default 100B"): buy-in in big blinds, default
  `100`. It is also the default amount offered for a rebuy.
- `--seat`: optional join offset from the button (the protocol's `seat`). The default
  is null, which means joining as the big blind.
- `--ascii`: draw suits as letters (`h d s c`) and boxes with ASCII characters, for
  terminals without Unicode. Colours are kept.
- `--hand-end-timeout`: how long the rebuy / sit-in menu waits for the user before
  the default answer is sent. Default `20` s. Other players at the table wait for this
  answer, so it must be short.
- `--log-file`: debug log of every message sent and received. There is no logging by
  default, because curses owns the terminal and writing to stderr would corrupt the
  screen.

Startup:
1. Start the `CallbackListener` on `127.0.0.1:--port`. It must be listening before
   `join` is sent, because the server connects to it while handling the join.
2. Open the `ServerConnection` and send `join_request(name, stack, callback_port,
   seat)`. If the server cannot be reached, or the join is rejected (duplicate name,
   full table, invalid stack, …), print the error to stderr and exit with status 2.
   Curses is not started in that case.
3. Start the curses UI (`ui.run(client)`). Messages that arrived before the join
   response (for example `seated`) are already waiting in the queue.
4. Before exiting (the player left, the server closed, or Ctrl-C), restore the
   terminal. Then print a one-line summary: hands played and the final stack, from the
   `left` message.

Exit codes: `0` after a normal leave, `1` if the server connection is lost, and `2` if
startup or the join fails. Exit with an error if stdin/stdout is not a TTY.

## Networking (`net.py`)

### Networking library
`net.py` is the `:net` target. It is the only part of cli_client that other projects
may depend on (`//bot_client` today). The contract is:
- **Public API** (`__all__`): `HOST`, `DISCONNECTED`, `ConnectionLost`,
  `ServerConnection`, `CallbackListener`. Changing their signatures or behaviour
  affects `//bot_client` too.
- **Imports**: only the standard library (`socket`, `threading`, `queue`, `logging`)
  and `cli_server.protocol` (`read_message`, `write_message`, `ProtocolError`,
  `LineTooLongError`). It never imports `curses`, `state`, `controller`, `render`,
  `cards`, `ui` or `main`. Importing it does not import `curses`.
- **No UI or table assumptions**:
  - The inbox is any `queue.Queue` the caller supplies. The library puts raw message
    dicts on it, plus the `DISCONNECTED` marker. Decoding (`protocol.decode_*`) is
    left to the caller.
  - It holds no player state other than the name used for the `to` filter.
  - It writes nothing to stdout or stderr. It logs to the `cli_client.net` logger at
    debug / warning level and installs no handlers. The caller decides where the log
    goes: cli_client uses `--log-file`, and bot_client uses its own log.
- **Callers do their own protocol steps**: building requests (`protocol.*_request`),
  sending them in order (the listener before `join`), checking `ok`, and answering
  `turn` / `hand_end`. All of this happens above the library. cli_client does it in
  `main.py` / `controller.py`.
- **Threads**:
  - `ServerConnection.request` is safe to call from any thread, because a lock
    serialises calls.
  - `CallbackListener`'s threads are daemon threads, so they never keep a process
    alive.
  - `close()` on either object can be called more than once, from any thread. A
    `request` after `close()` raises `ConnectionLost`.

### `ServerConnection` (client → server, the server's port)
- One persistent TCP connection to `127.0.0.1:server_port`, opened lazily on the
  first request. In cli_client, that is the `join` at startup.
- `request(msg) -> dict` writes one line with `protocol.write_message` and reads one
  `response` line. A lock serialises calls. Timeout is 10 s (constructor argument).
- A response with `ok: false` is returned to the caller; cli_client's controller shows
  the `error` text. A malformed response line comes back as an `ok: false` response
  whose error is "bad response: …". The stream is still in sync after it.
- A socket error, a timeout or an over-long line drops the connection. It then
  reconnects and resends once, before raising `ConnectionLost`. The server tells
  clients apart by `name`, not by connection. cli_client's UI shows "Connection to
  server lost" when `ConnectionLost` is raised.
- In cli_client it is called only from the UI thread, never from the callback reader
  thread. Responses are immediate, so the UI never waits long.

### `CallbackListener` (server → client, own port)
- `CallbackListener(name, inbox, host=HOST, port=0)` binds and listens in the
  constructor. Bind errors are raised there, so the caller can report them before
  `join`. `port` 0 picks a free port, and the bound port is in `.port`. `start()`
  launches the accept thread and returns `self`.
- The server opens one connection per joined player and keeps it open. If a new
  connection arrives, it replaces the old one, and the old one is closed without a
  `DISCONNECTED` marker.
- A reader thread runs `protocol.read_message` in a loop. Each message is put on the
  inbox. Messages whose `to` is not this player's name are dropped and counted in
  `.dropped`.
- End of stream, a socket error or an over-long line on the current connection puts
  the `DISCONNECTED` marker (`("disconnected",)`) on the inbox. Nothing is put after
  `close()`.
- A malformed line must not kill the reader thread. It is logged, counted in
  `.malformed`, and skipped. Unknown message types are passed through unchanged.
  Decoding is done by the consumer (cli_client's controller), not in the reader.

## Table model (`state.py`)

`TableState` is the client's picture of the table. It is updated only from server
messages, on the UI thread.

- Table: `hand_number`, `street`, `board` (`Board`), `button`, `sb_seat`, `bb_seat`,
  `pot`, `current_bet`, `bet_unit`, `waiting` (names queued for a seat), and the
  table settings from the join response (see Server capabilities used).
- `seats: list[SeatState]` in clockwise order. Each `SeatState` has `name`, `stack`,
  `status` (active / sitting out), `street_bet`, `total_bet`, `folded`, `all_in`,
  `leaving`, `in_hand`, `last_action` (text label such as "Raise 2", cleared on a new
  street), `shown` (`Hand | None`) and `won` (amount won this hand, for highlighting).
- Me: `my_name`, `my_seat` (looked up by name), `my_cards` (`Hand | None`),
  `queued` (waiting for a seat), `sitting_out`, `left`.
- `acting: (seat, deadline) | None`: whose turn it is.
- `log: list[LogLine]`. Each line is a list of `(text, style)` segments, so cards in
  the log keep their suit colours. At most 500 lines are kept.

Seat numbers change between hands: the engine renumbers seats when players join or
leave. The model therefore identifies players by **name**, which is unique at the
table, and treats seat indices as positions in the current list:
- `apply_view(view)` (from `seated`, `hole_cards`, `turn`) replaces the whole table
  part of the model with the snapshot. This keeps the model in sync with the server.
- `PlayerJoined(seat, name, stack_bb)` inserts the player at index `seat`.
  `PlayerLeft` removes the player by name.
- `HandStarted` sets the hand number, button, SB and BB. It resets the per-hand state
  (bets, folds, shown cards, board, pot, last actions) and marks the participants as
  in the hand with their stacks.
- `BlindPosted` / `PlayerActed` set `stack` (from `stack_bb`), add `amount_bb` to
  `total_bet` and the pot, and set `street_bet = to_bb` (`amount_bb` for a blind),
  plus `folded`, `all_in` and `last_action`. A forced fold shows its `reason` ("Fold
  (timeout)", "Fold (left)").
- `BoardDealt` sets the board and street, and moves street bets into the pot display
  (street bets go to 0, `last_action` is cleared).
- `HandShown` sets `shown`. `BetReturned` / `PotAwarded` set `stack` and subtract the
  amount from the pot. `PotAwarded` also adds to `won`.
- `PlayerSatOut` / `PlayerReturned` / `PlayerRebought` set the status and stack.
- `apply_result(result)` (from `hand_end`) reconciles names, stacks and shown hands
  with `result.seats`. This is the only full resync a sitting-out player gets, since
  sitting-out players are sent no `hole_cards` / `turn` views.
- Every event also adds a line to `log` (see Action log).
- The pot shown is the sum of `total_bet` over all seats, minus the amounts already
  returned or awarded. It always agrees with `view.pot` when a view arrives.

## Controller (`controller.py`)

`Client(name, conn, inbox, clock, settings)` contains the UI logic but no curses code,
so tests can drive it headless. The UI loop calls three methods:
- `drain()`: take every message from `inbox` and run `handle_message` on it;
- `handle_key(key)`;
- `tick(now)`: timers.

### Message handling
| message | effect |
|---|---|
| `seated` | `apply_view`. `queued = False`. Log "You sit down at seat N" |
| `hole_cards` | `apply_view`, `my_cards = cards` |
| `event` | `decode_event`, then `state.apply_event`. An unknown event type is ignored (forward compatible) |
| `turn` | `apply_view`. Store `pending_turn = (seq, legal)`. Set `acting` to me. Build the action menu. If a preset action matches `legal`, send it at once (see Presets). Ring the terminal bell |
| `hand_end` | `apply_result`. Store `pending_hand_end = seq`. Answer it (see Hand end) |
| `sat_out` / `returned` | set `sitting_out` and log it |
| `left` | `left = True`, store `final_stack_bb`, log it. The UI exits after a short delay |
| `close` / disconnected | if `left` was not received, show "Server closed the table" and exit on the next key |
| `acting` (proposed, see below) | `state.acting = (seat, now + timeout)` |
| `lobby` (proposed, see below) | store the lobby status shown in the centre of the table |

### Action menu (my turn)
Built from `LegalActions` and the view. Only legal options are listed, numbered in a
fixed order like an online room:
```
[1] Fold
[2] Check                  | [2] Call 1.5
[3] Bet 1 / Raise to 4     (shown only if can_raise; "Bet" when current_bet == 0)
[4] Leave table
```
- The amounts come from `legal.call_amount` and `legal.raise_to`, and are marked
  "(all-in)" when they use up the stack.
- When a check is possible, Fold asks for confirmation ("Fold instead of checking?
  [y/N]"), as real rooms do.
- Keys: `1`–`9` select an option directly. `↑`/`↓` move the highlight and `Enter`
  selects it. Hotkeys `f`, `c` (check or call), `r` (bet or raise) also work.
- When an option is selected, the controller sends `action_request(name, seq, action)`
  and clears the menu.
  - If the response is `ok: false` and the turn is still pending (an illegal action),
    the error is shown and the menu stays.
  - If the error says the turn is no longer pending (timeout), `pending_turn` is
    cleared.
- If the turn deadline is known, a countdown is shown next to the menu.

### Presets (when it is not my turn)
These are the "pre-action" checkboxes of online rooms. While I am in the hand and it
is not my turn, keys toggle one preset: `F2` / `x` Check/Fold, `F3` / `a` Call any.
When the next `turn` arrives:
- Check/Fold sends `check` if `legal.can_check`, otherwise `fold`.
- Call any sends `check` or `call`.

A preset is cleared after it is used and at the end of the hand.

### Hand end (`hand_end` reply)
The dealer waits for every seated player's reply before the next hand, so the client
answers at once unless it needs the user to decide:
- I am **busted** (my stack in the result is 0): show the spec's menu
  ```
  [1] Rebuy 100 BB      (default --stack. [e] edits the amount; capped by max buy-in when known)
  [2] Sit out
  [3] Leave table
  ```
  Each option is sent as `hand_end_request(name, seq, rebuy_bb=… | sit_in=False |
  leave=True)`.
- I am **sitting out with chips** (I timed out earlier): menu `[1] Sit in`,
  `[2] Keep sitting out`, `[3] Leave table`.
- Otherwise: reply at once. The reply includes any queued choices: a top-up (`t`, see
  below) as `rebuy_bb`, and a pending "leave after this hand" as `leave=True`.
- The menu shows a countdown. After `--hand-end-timeout` it sends the default: sit out
  (no rebuy, no sit-in), or keep sitting out.

### Between hands and at any time
- `t`: top up. Asks for an amount, default "back to `--stack`". It is queued and sent
  with the next `hand_end` reply, because the engine accepts a rebuy only outside
  betting. If no hand is running (lobby), it is sent at once as a `rebuy` request.
- While sitting out and no hand is running: `r` sends a `rebuy` request (busted) and
  `s` sends a `sit_in` request. If one of these is rejected because a hand is running
  ("during betting …"), the choice is kept and offered again in the next `hand_end`
  menu.
- `l` / `q` / Ctrl-C: leave, after a confirmation if I am in a hand. This sends
  `leave_request(name)`. The engine folds me at my next turn and removes me after the
  hand, so the status line shows "Leaving after this hand…" until `left` arrives.
  A second Ctrl-C quits without waiting.
- `PgUp` / `PgDn` scroll the action log. `?` shows a help overlay with the keys.

## Cards and colours (`cards.py`)
- `card_text(card, ascii=False)`: rank (`A K Q J T 9 … 2`) and suit symbol
  (`♥ ♦ ♠ ♣`, or `h d s c` with `--ascii`).
- `suit_style(suit)` gives the spec's four-colour deck: hearts **red**, diamonds
  **blue**, spades **yellow**, clubs **green**. Cards are drawn bold on a white or
  black card face so that the colours stand out from the green felt.
- `card_segments(cards)` returns the `(text, style)` segments for a list of cards, in
  the form `[A♥][K♦]`. It is used by the table view, the hole cards and the log. Face
  down cards are drawn as `[▒▒]` (or `[##]`).
- `describe_hand(hand, board)` is used for "wins with …" in the log and the showdown
  labels. It calls the engine evaluator (`engine.evaluator.best_omaha`), so the
  exactly-two-hole-cards rule is applied, and names the high category ("Full house",
  "Flush", "Two pair", …) and the low ("7-5-4-2-A low", or no low). The category
  names map the category index of the `rank_high` tuple.

## Rendering (`render.py`)
`render(state, ui, width, height) -> Frame`. A `Frame` is a `height × width` grid of
`(char, style)` cells, with `put(x, y, text, style)` and `box(...)` helpers. Styles
are names (`felt`, `seat`, `seat_me`, `seat_acting`, `folded`, `hearts`, `diamonds`,
`spades`, `clubs`, `card_back`, `chip`, `button`, `pot`, `winner`, `log_me`,
`log_board`, `menu`, `menu_selected`, `error`, `dim`). Rendering is a pure function,
so it is unit-tested without a terminal. `ui.py` only maps style names to curses
attributes.

Screen layout (minimum 100 × 30). If the terminal is smaller, only a message is
shown: "Please enlarge the terminal to at least 100×30".
```
┌ Omaha Hi-Lo 8/b · Limit 1/2 (blinds 0.5/1) · Hand #12 ───────────────┬ Table log ─────────────┐
│              ┌Bob ────┐        ┌Carol ──┐                            │ ── Hand #12 ──         │
│              │ 42.5 BB│        │ 18 BB  │                            │ Bob posts SB 0.5       │
│              │[▒▒][▒▒]│   ●1   │ FOLD   │                            │ Carol posts BB 1       │
│        .-''''''''''''''''''''''''''''''''-.                          │ Dave raises to 2       │
│ ┌Dave┐ (    [A♥][7♦][2♣]   [K♠]  [ ]    )  ┌Erin┐                    │ Carol folds            │
│ │ D  │ (         Pot: 9.5 BB              ) │ SB │                    │ Flop: [A♥][7♦][2♣]     │
│        '-................................-'                          │ Bob checks             │
│                       ┌You (Alice) ┐  ◉ D                            │ Turn: [K♠]             │
│                       │  37.5 BB   │                                 │ ...                    │
├──────────────────────────────────────────────────────────────────────┤                        │
│ Your hand: [A♠][2♦][K♥][3♣]   ⏱ 0:24                                 │                        │
│ [1] Fold   [2] Call 2   [3] Raise to 4   [4] Leave      F2 Check/Fold │                        │
└──────────────────────────────────────────────────────────────────────┴────────────────────────┘
```
- **Header**: the game name, the limits and blinds (from the table settings, or the
  engine's fixed 0.5/1 blinds and 1/2 bets when the server does not send them), the
  hand number, and the street.
- **Felt**: an ellipse filled with the `felt` style (green background). The centre
  shows the board (five slots: flop, turn and river cards in suit colours, and empty
  `[ ]` slots), the total pot, and the side pots when there are any. Between hands it
  shows the lobby status instead: "Waiting for players (1/2)", "Game starts in 12 s",
  or "Waiting for the current hand to finish" while queued.
- **Seats**: up to `max_seats` boxes on the ellipse. Seats are rotated so that **my
  seat is always at the bottom centre**, as in online rooms. Each box shows:
  - the name, and the stack in BB;
  - a badge: `D` (button), `SB`, `BB`, `ALL-IN`, `SITTING OUT`, `LEAVING`;
  - the last action ("Raise 2", "Call", "Check", "Fold");
  - cards: face-down backs for players still in the hand, shown cards at showdown, and
    my own cards face up.

  The current street bet is drawn as a chip (`●2`) between the seat and the centre.
  The dealer button is drawn as `◉ D` next to its seat. Styles: folded players are
  dimmed, my seat is bold, the acting seat is highlighted (reverse video), and winners
  are drawn in the `winner` style with "+3.5" until the next hand starts.
- **My panel** (bottom): my hole cards, large and in suit colours; my stack; the
  action menu or hand-end menu with the countdown; the active preset; and a status /
  error line.
- **Action log** (right column): the spec's "highlight actions that happened on the
  table". It is newest-at-bottom and auto-scrolls unless the user has scrolled up.
  Lines, with my own lines bold:
  - a separator `── Hand #12 ──` with the button and the blinds;
  - "Bob posts SB 0.5";
  - "Dave raises to 2", "Carol folds", "Bob checks", "Erin calls 2 (all-in)",
    "Carol folds (timeout)";
  - "Flop: [A♥][7♦][2♣]", "Turn: [K♠]", "River: [5♣]", with the cards in suit
    colours;
  - "Bob shows [..][..][..][..] – Flush, A high; 7-5-4-2-A low";
  - "Bob wins 4.5 (high)", "Dave wins 4.5 (low)", "Bob wins 9 (uncontested)", "5 returned to Bob";
  - "Erin joins the table (100 BB)", "Carol sits out", "Carol rebuys 50", "Carol leaves".

## Curses backend (`ui.py`)
- `curses.wrapper(main_loop)`. Set up `curs_set(0)`, `keypad`, `use_default_colors`,
  and one colour pair per style. On terminals with fewer than 8 colours, styles fall
  back to bold / dim / reverse.
- Loop: `getch()` with a 100 ms timeout, then `client.drain()`, then key handling, then
  `client.tick(time.monotonic())`. Redraw only when something changed or a countdown
  moved. `KEY_RESIZE` redraws at the new size.
- Drawing writes the `Frame` cell by cell, in rows of equal style. Writing the
  bottom-right cell raises `curses.error`, which is ignored.
- Ctrl-C (`KeyboardInterrupt`) is passed to the controller as the leave key, as
  described above.
- This is the only module that imports `curses`. It is part of `:cli_client_lib`,
  never of `:net`.

## Server capabilities used
The client needs only what the protocol already provides: `join` / `action` /
`hand_end` / `rebuy` / `sit_in` / `leave` requests; `seated`, `hole_cards`, `event`,
`turn`, `hand_end`, `sat_out`, `returned`, `left` and `close` notifications; and the
engine types encoded by `//cli_server:protocol`.

To look like a real poker room, the client also uses the additions proposed in
`cli_server/spec.md`. The client must still work without them; without each one, the
fallback below is used:
- **Table settings in the join response** (`table`: blinds, small and big bet, max
  seats, max buy-in, action timeout, grace period). These feed the header, the empty
  seat count, the rebuy cap and the countdowns. Fallback: the engine's fixed
  structure, 10 seats, no countdown.
- **`acting` notification** (`seat`, `name`, `timeout`), sent to the other players
  when someone is asked to act. It highlights the acting seat and shows their timer.
  Fallback: only my own turn is highlighted.
- **`lobby` notification** (`active`, `needed`, `starts_in`). It drives the "Waiting
  for players" and "Game starts in N s" text. Fallback: "Waiting for players", shown
  when no hand is running and fewer than two players are active.
- **Pause between hands.** The server waits a few seconds after a hand, so the
  showdown and the winner highlight stay on screen before the next `hand_started`.
  Fallback: the result stays visible in the log and in the seat "+amount" labels until
  the next `HandStarted`.

## Tests
All tests use `unittest` and the standard library.
- `test_cards`: card text in Unicode and ASCII; the four suit styles (hearts red,
  diamonds blue, spades yellow, clubs green); `describe_hand` for high categories,
  lows, no low, and the exactly-two-hole-cards rule.
- `test_state`: event sequences built from engine dataclasses with
  `protocol.encode_event` give the right stacks, bets, pot, board, folds, last
  actions and log lines. Also:
  - stacks always equal the events' `stack_bb`;
  - the pot matches `view.pot` when the next view arrives;
  - joins and leaves between hands renumber the seats correctly (players are
    identified by name);
  - `apply_view` and `apply_result` resync a model that has drifted;
  - an unknown event type is ignored.
- `test_controller` (fake `ServerConnection`, fake clock):
  - menus are built from `LegalActions`: check vs call, bet vs raise, no raise
    when `can_raise` is false, all-in labels;
  - number keys, arrow keys and hotkeys send the right `action_request` with the
    turn's `seq`;
  - an illegal action error keeps the menu, and a stale-turn error clears it;
  - Check/Fold and Call any presets send the right action at the next turn;
  - `hand_end` is answered immediately when no decision is needed;
  - busted → rebuy / sit-out / leave menu; sitting out → sit-in menu; the timeout
    sends the default; a queued top-up and a queued leave go into the reply;
  - `leave` sends `leave_request`, and `left` ends the client;
  - `close` without `left` shows the server-closed message.
- `test_render`: on a fixed state, the frame contains every seat's name, stack and
  badges, the board cards with their suit styles, the pot, my hole cards face up and
  other players' cards face down, the menu items and the log lines. Also: my seat is
  drawn at the bottom centre for every seat index; the acting seat and the winners
  are highlighted; a too-small terminal shows the enlarge message.
- `test_net`: `CallbackListener` delivers lines in order, drops other players'
  messages, survives a malformed line and reports disconnects; a replaced connection
  does not report a disconnect, and nothing is reported after `close()`.
  `ServerConnection` pairs requests with responses, reconnects once after a dropped
  connection, and raises `ConnectionLost` on a closed socket and after `close()`.
- `test_net_library` (`:net_library_test`, built against `:net` only):
  - `import cli_client.net` works without the rest of cli_client;
  - after the import, `curses` is not in `sys.modules`, and none of
    `cli_client.ui`, `controller`, `state`, `render`, `cards` or `main` is loaded or
    importable;
  - every name in `net.__all__` exists;
  - a round trip, with a stub server on port 0, a `ServerConnection` and a
    `CallbackListener` feeding a plain `queue.Queue`, works with no other cli_client
    module.

  This is the test that guards the library contract that `//bot_client` relies on.
- `test_integration`: an in-process `GameServer` on port 0 with a short grace period
  and a fixed seed, and two or three headless `Client`s driven by scripted key
  presses. They join, play hands to the end, and the hand ends with consistent
  stacks. Each client's model matches the server's `HandResult`. A third client that
  joins mid-hand is queued, then seated. A busted client rebuys through the hand-end
  menu. A client that leaves receives `left`.
