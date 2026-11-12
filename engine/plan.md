# Engine — implementation plan

Python library (`engine`) providing an Omaha Hi-Lo 8-or-better game engine for a
**fixed-limit** table (the repo is named "plo8", but the game is limit, not pot-limit).
The engine is the library as a whole, not one class: there is no `Engine` class. It
exports `Player`, `Dealer`, `Environment` and the classes they depend on (`Action`,
`LegalActions`, views, events, ...). Card types (`Card`, `Rank`, `Suit`, `Hand`, `Board`,
`Street`, `Deck`) come from `//entities:entities` and are re-exported for convenience. Agents drive players through callbacks; the engine enforces rules,
moves chips, handles all-ins/side pots, decides who wins and reports every public
table event to the seated players.

## Layout

```
engine/
  spec.md
  plan.md
  BUCK
  engine/
    __init__.py        # public exports
    evaluator.py       # high hand ranking, 8-or-better low ranking, Omaha 2+3 selection
    actions.py         # ActionType, Action, LegalActions
    player.py          # Player, PlayerStatus, agent callback protocol
    pots.py            # Pot, side-pot construction, pot splitting
    dealer.py          # Dealer: dealing, betting rounds, showdown, button movement
    environment.py     # Environment: table setup, joining/leaving players, start_round()
    views.py           # read-only snapshots passed to agents (TableView, HandResult)
    events.py          # public table events broadcast to seated players
    remote.py          # adapter for agents running in separate processes
  tests/
    test_evaluator.py
    test_pots.py
    test_betting.py
    test_dealer.py
    test_environment.py
    test_events.py
    test_join.py
    test_sit_out.py
    test_leave.py
    test_remote.py
```

## Build (buck2)

`engine/BUCK`:

- `python_library(name = "engine", srcs = glob(["engine/**/*.py"]), base_module = "", deps = ["//entities:entities"], visibility = ["PUBLIC"])`
- one `python_test` per test file (or a single one with `srcs = glob(["tests/*.py"])`),
  depending on `:engine`. Tests use the standard `unittest` module, so no third-party
  dependencies are needed with the system demo toolchain.

Run: `buck2 build //engine:engine`, `buck2 test //engine/...`.

## Units / chips

Stacks are expressed in big blinds as the spec requires, but stored internally as
integer chips to avoid float rounding in pot splits:

- `CHIPS_PER_BB = 100` (configurable on `Environment`); small blind = 50 chips, big blind = 100.
- `Player.stack` (float, in BB) is a property over `Player._chips` (int).
- All public amounts (`rebuy`, `join`, views, events, `LegalActions`) accept/return BB;
  conversion is done in one place (`to_chips` / `to_bb` helpers) and amounts are rounded
  to whole chips.
- Bet sizes are fixed by the limit structure (see Actions), so agents never pass a bet
  amount.

## Cards, hands, board and deck (from `//entities`)

The engine does not define its own card types; it imports them from the `entities`
library (`from entities import Card, Rank, Suit, Hand, Board, Street, Deck`):
- `Card`, `Rank`, `Suit` — immutable cards; `Rank` values 2..14 give high-hand order and
  `Rank.low_value` (ace = 1) gives 8-or-better low order.
- `Hand` — a player's 4 hole cards. The dealer deals one card at a time to each player
  (`Deck.deal_one()`), then builds a `Hand` from each player's 4 cards.
- `Board` / `Street` — the dealer's board is an immutable `Board`, replaced street by
  street via `with_flop` / `with_turn` / `with_river`; `Board.street` is the street
  reported in `TableView`.
- `Deck` — the dealer owns one `Deck(rng)` (seeded `random.Random` from `Environment`).
  A new hand calls `deck.reset()` then `deck.shuffle()` (`reset()` does not shuffle).
  No burn cards are dealt: with an RNG-shuffled deck that no agent can see, burning has
  no effect on fairness or odds.
- All of these are picklable, so they can be sent to agent processes as-is.

No changes to `entities` are needed beyond guaranteed pickling (see
`entities/plan.md`, "Changes requested by engine").

## Hand evaluation (`evaluator.py`)

- `rank_high(five_cards) -> tuple` — comparable tuple (category, tiebreak ranks...),
  categories high card … straight flush; wheel A-2-3-4-5 is a straight.
- `rank_low(five_cards) -> tuple | None` — 8-or-better: five distinct ranks all ≤ 8
  (Ace counts as 1); returns tuple of ranks sorted descending (lower compares better),
  `None` if not qualifying.
- `best_omaha(hand: Hand, board: Board) -> (high_rank, low_rank | None)` — iterate over all
  C(4,2)·C(5,3) = 60 combinations using **exactly two** hole cards and **exactly three**
  board cards; high and low are chosen independently.

## Actions (`actions.py`)

- `ActionType`: `FOLD`, `CHECK`, `CALL`, `RAISE` (a bet into an unopened round is a
  `RAISE` from 0).
- `Action(type)` — no amount: in limit poker every bet/raise is exactly one bet unit.
- Betting structure is **fixed limit**:
  - Small bet = 1 BB on preflop and flop; big bet = 2 BB on turn and river.
  - Blinds: SB = 0.5 BB, BB = 1 BB (one small bet).
  - A bet or raise increases the current bet by exactly one bet unit for the street.
  - Cap: one bet plus three raises per street (4 bets total; the BB counts as the
    opening bet preflop). The cap is lifted when only two players are left in the hand.
  - A player who cannot cover a full bet/call may go all-in for less. An all-in raise of
    at least half a bet unit counts as a full raise (reopens betting and counts toward
    the cap); less than half acts as a call and does not reopen betting for players who
    already acted.
- `LegalActions` — computed by the dealer for the acting player: `can_check`,
  `call_amount`, `can_raise`, `raise_to` (the fixed total after raising, capped by the
  player's stack), `bet_unit` and `raises_left` on this street.

## Player (`player.py`)

State:
- `name` (nick, see below), `seat`, `status`, `_chips` (stack),
  `hole_cards` (`Hand | None`, private), `street_bet`, `total_bet` (this hand),
  `folded`, `all_in`, `has_acted`, `leave_requested`.
- `PlayerStatus`:
  - `WAITING` — joined, queued for a seat (see Environment → Joining); `seat is None`.
  - `ACTIVE` — seated with chips; dealt into the next hand.
  - `SITTING_OUT` — seated but not playing: either 0 chips after the rebuy window of a
    hand, or the agent failed to act on its turn (see Sitting out). Keeps the seat but
    is not dealt cards, posts no blinds, never gets `on_turn` / `on_hole_cards`.
  - `LEFT` — removed from the table; terminal, the object is no longer usable for play.
- `name` is set when the player is created / joins (`Player(name=None, agent=None)`);
  if omitted the environment assigns `"Player<N>"`. Names must be unique at the table
  (seated — active or sitting out — and waiting players); `name` is what views and
  events show for the player. A name becomes free again once its player has left.
- Each player keeps a back-reference to its `Environment` (set on seating / join), used
  by `leave()` and by `rebuy()` validation.

Agent-facing API (the "managing agent" calls these):
- `fold()`, `check()`, `call()`, `raise_()` — called during its turn; each records the
  pending action; validated by the dealer against `LegalActions`. Invalid actions raise
  `IllegalActionError` to the agent, which may try again within the same turn. If the
  turn ends without a valid action (see No action / timeout), the hand is folded and
  the player sits out from the next hand.
- `rebuy(amount_bb)` — adds to the stack; amount > 0, resulting stack ≤ table max buy-in
  if set. Allowed only outside betting: inside `on_hand_end` or while the dealer is not
  in a hand (raises `IllegalActionError` during betting, for `WAITING` and for `LEFT`
  players). A `SITTING_OUT` player that rebuys becomes `ACTIVE` and is dealt into the
  next hand.
- `sit_in()` — return to play after sitting out for not acting: a `SITTING_OUT`
  player with chips becomes `ACTIVE` and is dealt into the next hand. Allowed inside
  `on_hand_end` or between hands, like `rebuy`. (A busted player returns by rebuying.)
- `leave()` — leave the table, emptying the seat (delegates to `Environment.leave`, see
  Leaving). Idempotent.
- `show_cards() -> Hand` — called by the dealer at showdown; the hole cards
  then become public (`HandShown` event and the hand result). (Mucking is not
  modelled: a player required to show at showdown always shows.)

Callbacks (set via constructor or `Player.set_agent(agent)`; agent implements
an `Agent` protocol). Optional callbacks are looked up with `getattr` and skipped when
the agent does not define them:
- `on_hole_cards(player, hand: Hand, view: TableView) -> None` — called once per hand
  for every participant, as soon as all hole cards are dealt and before the first
  `on_turn` of the hand. `hand` is that player's own `Hand` only; `view` is the same
  per-player snapshot as in `on_turn` (blinds already posted, own cards only). Not
  called for sitting-out or waiting players. Optional; the agent can also read the
  cards later from `view.hole_cards` in `on_turn`.
- `on_turn(player, view: TableView, legal: LegalActions) -> None` — invoked by the
  dealer when it is this player's turn; the agent must call exactly one of
  `fold/check/call/raise_` on `player` before returning, or call `player.leave()`
  (which counts as a fold). Returning without a valid action counts as not acting (see
  No action / timeout).
- `on_event(player, event: TableEvent) -> None` — receives every public table event
  (see Events) for the table the player is seated at, including events caused by the
  player itself. Optional; must not act on the player (actions are only accepted in
  `on_turn`) — rebuy / leave from here follow the usual rules (leave is allowed any
  time, rebuy only between hands).
- `on_hand_end(player, result: HandResult) -> None` — invoked after stacks are
  updated, whether or not there was a showdown; the agent may call `player.rebuy(...)`
  (and/or `player.leave()`). Called for every seated player — those dealt into the hand
  and those `SITTING_OUT` — so a busted player keeps getting a rebuy opportunity after
  every hand. Not called for `WAITING` players or players that already left.
- Optional `on_seated(player, view: TableView)` — called when a waiting player is
  actually seated (immediately on `join` between hands, or after the current hand).
- Optional `on_sit_out(player)` / `on_return(player)` — notify when the player starts
  sitting out (0 chips and no rebuy, or not acting on a turn) and when a rebuy or
  `sit_in()` brings them back.
- Optional `on_left(player, final_stack_bb)` — called once when the player has been
  removed from the table.

Exceptions raised from the notification callbacks (`on_hole_cards`, `on_event`,
`on_seated`, `on_sit_out`, `on_return`, `on_left`) are caught and logged by the engine so
a faulty observer cannot break the hand for the others.

Default agent (no callbacks set): check if possible, else fold; never rebuys; ignores
hole-card and event notifications — useful in tests.

## Events (`events.py`)

All public events are frozen, picklable dataclasses deriving from `TableEvent`, which
carries `hand_no` (number of the hand in progress, `None` for table events between
hands) and `seq` (monotonic per table, so agents can order / de-duplicate them).
Events about a player carry that player's `seat`, `name` and `stack_bb` — the player's
stack **after** the event.

Hand events (emitted by the dealer):
- `HandStarted(button, sb_seat, bb_seat, participants: list[(seat, name, stack_bb)])` —
  before blinds are posted.
- `BlindPosted(seat, name, blind: "SB" | "BB", amount_bb, all_in, stack_bb)`.
- `PlayerActed(seat, name, street, action: ActionType, amount_bb, to_bb, all_in,
  stack_bb)` — one per applied action; `amount_bb` is the chips put in by this action
  (0 for fold/check), `to_bb` the player's street bet after it. Forced folds (leave
  requested, no action / timeout) are reported as `FOLD` with a `reason`
  (`"leave"`, `"timeout"`, or `None` for a voluntary fold).
- `BoardDealt(street, cards: tuple[Card, ...], board: Board)` — flop (3 cards), turn,
  river; also for cards run out after all-ins. No stack (no player concerned).
- `HandShown(seat, name, hand: Hand, stack_bb)` — at showdown, one per non-folded
  player, in showing order (first non-folded player left of the button, clockwise).
- `BetReturned(seat, name, amount_bb, stack_bb)` — uncalled bet / single-eligible pot
  given back.
- `PotAwarded(seat, name, pot_index, share: "high" | "low" | "whole", amount_bb,
  stack_bb)` — one per winner per share of each pot (a scoop of a split pot is two
  events, high and low). Pots are credited one share at a time, so `stack_bb` is the
  stack right after that share is added. When all but one player fold, the remaining
  player gets `share = "whole"` without any `HandShown`.

Table events (emitted by the environment): `PlayerJoined(seat, name, stack_bb)`,
`PlayerSatOut(seat, name, stack_bb)`, `PlayerReturned(seat, name, stack_bb)`,
`PlayerRebought(seat, name, amount_bb, stack_bb)`, `PlayerLeft(seat, name, stack_bb)`.

Delivery (`Environment._broadcast(event)`):
- Every event goes to every **seated** player (`ACTIVE` and `SITTING_OUT`, including
  players with `leave_requested` until they are removed and the player the event is
  about). `WAITING` and `LEFT` players get no events; `PlayerJoined` is delivered to the
  joiner too, after `on_seated`.
- Delivery is synchronous and in seat order, and happens right after the state change
  is applied (chips moved / card dealt) and before the dealer moves on, so the event
  stream an agent sees is always consistent with the next `TableView` it gets.
- The same events are appended to the hand's history, which `TableView.history` exposes,
  so an agent that ignores `on_event` can still read everything that happened.

## Views (`views.py`)

Snapshots handed to agents so they cannot mutate engine state or see others' cards:
- `TableView`: hand number, street (`Street`), board (`Board`), pot total, side pots,
  current bet, bet unit and raises made on this street, button/SB/BB seats, per-seat
  public info (name, stack, status — active / sitting out, street_bet, total_bet,
  folded, all_in, leaving), names of players waiting to be seated, `history` (the list
  of public events of the current hand, see Events), the receiving player's own
  `hole_cards` (only theirs; `None` if not dealt in).
- `HandResult`: hand number, final `Board`, shown hands (seat → `Hand`) if showdown,
  winners per pot (high / low shares with amounts), net chip change per seat, final
  stacks, `showdown: bool`; per-seat entries also carry the player's name and whether the
  player was dealt in (sitting-out seats appear with `dealt_in = False`, net change 0).
  The same result object is sent to all seated players, including sitting-out ones. It
  summarises the `HandShown` / `BetReturned` / `PotAwarded` events of the hand.

## Pots (`pots.py`)

- Contributions are tracked per player via `total_bet`.
- `build_pots(players) -> list[Pot]`: sort distinct contribution levels of all players
  (including folded ones — their chips stay in the pot), create one pot per level;
  each `Pot` has `amount` and `eligible` = non-folded players who contributed at that
  level. Pots with a single eligible player are returned to that player (uncalled bet,
  reported as `BetReturned`).
- `split_pot(pot, hands) -> list[(player, share, chips)]`:
  - If any eligible player has a qualifying low: half to best high (ties split),
    half to best low (ties split); a player can scoop both halves.
  - Otherwise the whole pot goes to best high.
  - Odd chips: the odd chip from halving goes to the high half; odd chips within a
    tied half go to the first winner(s) left of the button.
  - Returns the individual shares so the dealer can credit them one by one and emit a
    `PotAwarded` per share.

## Dealer (`dealer.py`)

Holds the `Deck`, the `Board`, pots, button seat, the list of seated players, the hand
counter, the current hand's event history and an `in_hand` flag (true from the start
of `play_hand()` until end-of-hand processing is done). The dealer sees all seated
players but deals in only `ACTIVE` ones; the participants of a hand are fixed when it
starts (`hand_players`). Waiting players are kept by the `Environment`. The dealer
emits events through a broadcast function supplied by the `Environment`.

Hand flow (`Dealer.play_hand()`):
1. **Start**: emit `HandStarted`.
2. **Preflop**: post SB and BB (short stacks post all-in), emitting `BlindPosted` for
   each; deal 4 hole cards to each participant one at a time starting left of the
   button; once all are dealt, call `on_hole_cards` for each participant (seat order,
   starting left of the button) — before any betting. Betting starts left of BB
   (heads-up: button is SB and acts first preflop, last postflop). SB/BB are the first
   and second `ACTIVE` players clockwise from the button; sitting-out seats are skipped
   for blinds, dealing and action.
3. **Flop**: deal 3 board cards (`BoardDealt`), betting starts at first active player
   left of the button.
4. **Turn**: deal 1 card (`BoardDealt`), betting round.
5. **River**: deal 1 card (`BoardDealt`), betting round.
6. **Showdown**: if ≥ 2 non-folded players remain.

Betting round (`_betting_round(street)`):
- Reset `street_bet`/`has_acted` (except blinds preflop which count as bets).
- Iterate participants in seat order, skipping folded/all-in players; for each, if
  `leave_requested` is set, fold them without calling the agent; otherwise compute
  `LegalActions`, invoke the player's `on_turn` callback, read and validate the action
  (a `leave()` inside `on_turn` is a fold), apply it (move chips from stack to
  `street_bet`/`total_bet`), update the current bet and the street's raise count,
  reset `has_acted` for others on a full raise, then emit `PlayerActed` (forced folds
  included, with their `reason`).
- Round ends when every non-folded, non-all-in player has acted and matched the
  current bet (or is all-in).
- If at any point only one non-folded player remains, the hand ends immediately: the
  uncalled part of the last bet is returned (`BetReturned`) and that player wins all
  pots without showing (`PotAwarded`, `share = "whole"`); remaining streets are skipped
  (no `BoardDealt` for them).
- If ≤ 1 player can still act (others all-in), skip further betting and run out the
  remaining board cards (one `BoardDealt` per street), then go to showdown.

Showdown:
- Ask every non-folded player to `show_cards()` in showing order and emit `HandShown`
  for each; evaluate with `best_omaha`, build side pots, return uncalled bets
  (`BetReturned`), split each pot as described above and credit stacks share by share,
  emitting `PotAwarded` after each credit. An all-in player who requested to leave
  stays in the hand and is paid normally before being removed.

End of hand (in this order):
1. Build `HandResult`, call `on_hand_end` for every seated player that has not left —
   participants and `SITTING_OUT` players (rebuy / leave window). Rebuys done here emit
   `PlayerRebought`.
2. Update statuses: an `ACTIVE` player with 0 chips, or one who did not act on a turn
   this hand, becomes `SITTING_OUT` (`on_sit_out`, `PlayerSatOut`); a `SITTING_OUT`
   player who rebought or called `sit_in()` (and has chips) becomes `ACTIVE`
   (`on_return`, `PlayerReturned`). (A rebuy or `sit_in()` outside the hand-end window,
   between hands, also returns the player immediately.)
3. Move the button to the next `ACTIVE`, non-leaving player clockwise from the current
   button (SB/BB shift accordingly). Sitting-out players never hold the button.
4. Clear `in_hand`; the environment then removes players that requested to leave and
   seats queued joiners (see below).

## Environment (`environment.py`)

- `Environment(players=(), button=0, chips_per_bb=100, seed=None, max_buy_in=None,
  max_seats=10, action_timeout=600.0)` — owns the `Dealer`, the seated player list and the join queue.
  Seated players (`ACTIVE` and `SITTING_OUT`) are kept in clockwise order;
  `Player.seat` is the index in that list. Initial `players` are seated directly
  (each must have a positive stack; unnamed ones get a default name). With no initial
  players there is no button and no game until two players have joined (see Joining);
  `button` is then ignored, and `action_timeout` (default 600 s) sets the turn timeout.
- Owns event broadcasting (`_broadcast`, see Events) and the per-table `seq` counter;
  the dealer and the join / leave / rebuy / sit-out bookkeeping all go through it.
- `start_round()` — begins a new hand: resets the deck (shuffle), clears hole cards
  and per-hand player state and the hand's event history, empties flop/turn/river and
  pots, has SB/BB players post blinds, then lets the dealer run the hand with the
  `ACTIVE` players. Returns the `HandResult`. After the hand, processes leave requests
  and the join queue.
- Requires 2–`max_seats` `ACTIVE` players to start a hand; otherwise raises
  `NotEnoughPlayersError` (sitting-out players do not count).
- `run(n_hands)` convenience loop; stops early when fewer than two `ACTIVE` players
  remain after a hand's rebuy window.
- `max_seats` counts seated (active + sitting out) and waiting players; a sitting-out
  player keeps occupying a seat until they leave.

### Sitting out

- A player busted to 0 chips who does not rebuy in `on_hand_end` becomes `SITTING_OUT`:
  not dealt cards, no blinds, no `on_turn` / `on_hole_cards`, not in pots, shown as
  sitting out in views. They still receive all public events (`on_event`).
- They keep their seat position in the clockwise order (relevant for join offsets, which
  count all seated players), but are skipped for button / blinds / action.
- After every subsequent hand they still receive `on_hand_end` (with `dealt_in = False`
  for themselves) and may rebuy there; a rebuy (there, or between hands) makes them
  `ACTIVE` again and they are dealt into the next hand without posting extra blinds.
- A player who did not act on a turn (see No action / timeout) also becomes
  `SITTING_OUT` at the end of that hand, keeping their chips. They stay sitting out until
  their agent calls `sit_in()` (in `on_hand_end` or between hands), so an unresponsive
  agent does not hold up every later hand with another timeout.
- A sitting-out player may `leave()` at any time.

### No action / timeout

- Each turn has a timeout, `action_timeout` on `Environment`, default 600 seconds
  (deliberately long: it only exists to catch agents that have hung or disconnected).
- If the agent gives no valid action in time, or its `on_turn` returns without one, the
  dealer folds the player's hand (chips already committed stay in the pot; reported as
  `PlayerActed(FOLD, reason="timeout")`) and the player becomes `SITTING_OUT` at the end
  of the hand.
- For in-process agents the dealer cannot interrupt the callback, so for them the rule
  applies when `on_turn` returns without a valid action; the timeout is enforced by
  `RemoteAgent` for agents in other processes.

### Joining (`join`)

`join(stack_bb, seat=None, name=None, agent=None) -> None`:
- `stack_bb` — buy-in in big blinds; must be > 0 and ≤ `max_buy_in` if set.
- `name` — the player's nick; must be unique among seated and waiting players
  (`ValueError` otherwise); default name assigned if omitted.
- `seat` — position **relative to the button** (clockwise offset): `1` = first seat left
  of the button, `2` = next, …, up to `n` = right of the button (button's seat itself
  is not available; the button stays with its current player). `n` counts all seated
  players, including sitting-out ones. If `None`, the player joins **as big blind**:
  directly after the first `ACTIVE` player clockwise from the button (the player who
  will post SB next hand), so the newcomer posts BB next hand. Offsets outside `1..n`
  raise `ValueError`.
- **Empty or one-player table** (no game is running): `seat` is ignored. The first
  player to join an empty table is seated and takes the button. Heads-up the button
  posts SB, so the first joiner is SB and the second joiner is seated after them as BB;
  the first hand can then start. The same applies when only one `ACTIVE` player is
  left at the table.
- Raises `TableFullError` if seated + waiting players would exceed `max_seats`.
- Returns nothing. The joiner's agent gets its `Player` through `on_seated` and the
  later callbacks. What `join` guarantees: if it does not raise, the player is dealt
  cards at the start of the next hand and acts in it, whether they joined between
  hands (seated immediately) or during a hand (queued, seated after it ends).

Seating:
- **Between hands** (`dealer.in_hand` is false): the player is seated immediately —
  inserted into the clockwise list at the requested offset from the current button,
  seats are renumbered, `status = ACTIVE`, `on_seated` / `PlayerJoined` are fired. They
  are dealt in on the next `start_round()`.
- **During a hand**: the request is appended to a FIFO `pending_joins` queue; the
  player does not take part in the current hand (not dealt, not in pots, not asked to
  act, no events, no `on_hand_end`). After the hand's end processing (stack updates,
  rebuy callbacks, status updates, button move, removal of leavers), the environment
  seats queued players in order, resolving each offset relative to the **new** button
  position, so a default joiner posts the big blind in the hand that follows. Each
  queued request is re-validated when seated (offset clamped to the new `1..n` range).
- No missed-blind / dead-button rules: a newly seated player is dealt in the next hand
  regardless of position, without posting extra blinds.

### Leaving (`leave`)

`Environment.leave(player)` (also via `Player.leave()`):
- **`WAITING` player**: removed from `pending_joins` immediately; status `LEFT`.
- **Between hands** (seated, `in_hand` false): removed from the seat list immediately,
  seats renumbered. If the leaver held the button, the button passes to the next
  `ACTIVE` player clockwise (the one that would have received it next).
- **During a hand**: sets `leave_requested`; removal is deferred to the end of the hand
  so seat indices stay stable:
  - a participant still in the hand is folded at their next turn (or immediately, when
    called from their own `on_turn`), reported as `PlayerActed(FOLD, reason="leave")`;
    chips already committed stay in the pot;
  - an all-in participant stays in the hand until it ends and is paid any winnings;
  - a sitting-out player or one who already folded is simply removed after the hand.
  The button move at end of hand skips leaving players; they are removed afterwards
  (before queued joiners are seated). Until removed they keep receiving events.
- On removal: `status = LEFT`, `seat = None`, the player's remaining stack is reported
  in `on_left(player, final_stack_bb)`, `PlayerLeft(seat, name, stack_bb)` is
  broadcast to the remaining seated players, and the name becomes available. Chips of
  a leaving player leave the table with them.
- Leaving a player that already left is a no-op.

Concurrency:
- Seat renumbering only happens between hands, so seat indices in a `TableView` /
  `HandResult` / events are stable for the duration of a hand.
- `join`, `leave`, `rebuy` bookkeeping, queue processing and `seq` assignment are
  guarded by a `threading.Lock`, so a host thread that accepts agent connections can
  call `join` / `leave` while the dealer is running a hand. Callbacks are invoked
  outside the lock.

## Multi-agent / separate processes (`remote.py`)

The engine itself is single-threaded and runs in one host process that owns the
`Environment` and `Dealer`; agents in other processes play through proxies:
- `RemoteAgent(conn)` implements the `Agent` protocol: `on_turn` sends the
  (picklable) `TableView` + `LegalActions` with a turn id over a
  `multiprocessing.Connection`, blocks for a reply message (`Action` or `Leave`) with
  that id, and applies it via `player.fold()/call()/.../leave()`;
  `on_hand_end` sends the `HandResult` and receives a reply with an optional rebuy
  amount, a sit-in flag and/or a leave flag — sent to sitting-out players too, so remote agents can
  rebuy back in; `on_seated`, `on_sit_out`, `on_return` and `on_left` notify the agent
  process of status changes (`on_left` closes the connection afterwards).
- `on_hole_cards` and `on_event` are **one-way** messages (`HoleCards(hand, view)`,
  `Event(event)`): sent in order on the same connection, no reply awaited, so a slow
  observer never delays the hand. Because the connection is ordered, the agent process
  sees its hole cards and all preceding events before the `on_turn` request.
- Joining from another process: the agent process sends a `Join(name, stack_bb, seat)`
  message on a new connection; a host-side listener thread calls
  `env.join(..., agent=RemoteAgent(conn))` and replies with the result (accepted /
  queued / error such as duplicate name or full table).
- Leaving from another process outside its turn: an unsolicited `Leave` message (or a
  closed connection) is picked up by the listener thread, which calls
  `env.leave(player)`.
- `agent_loop(conn, agent)` helper runs in the agent process, receiving messages and
  dispatching to a local agent object with the same callback interface (including
  `on_hole_cards` and `on_event`, skipped if the local agent does not define them).
- The `action_timeout` (see No action / timeout) is enforced here: no reply in time →
  the hand is folded and the player sits out from the next hand. A late reply that
  arrives after the timeout (stale turn id) is discarded.
- Each agent only ever receives its own hole cards (in `HoleCards` and in its own
  `TableView`s); other players' cards reach it only through `HandShown` at showdown, so
  information is isolated per process.

## Tests

- Card, hand, board and deck behaviour is tested in `//entities`; engine tests only
  cover how the dealer uses them (no card dealt twice in a hand,
  same seed gives the same hand).
- `test_evaluator`: category ordering, wheel straight, exactly-2-hole-cards rule
  (e.g., four flush cards on board + one in hand is not a flush), low qualification,
  counterfeited lows, A-2 low vs A-3 low.
- `test_pots`: side pots with multiple all-ins of different sizes, folded players'
  contributions, uncalled bet return, hi/lo split, scoop, quartering, odd-chip rules,
  individual shares returned by `split_pot`.
- `test_betting`: small bet on preflop/flop and big bet on turn/river, raise cap of
  4 bets and its removal heads-up, all-in for less than half a bet not reopening action
  while half or more does, heads-up blind/action order.
- `test_dealer`: hand ends early when all but one fold, all-in run-out, chip
  conservation (sum of stacks constant across hands apart from rebuys, joins and
  leaves), button moves.
- `test_environment`: `start_round` resets deck/hands/board and posts blinds; rebuy
  callback is called after every hand and rebuy increases stack; rebuy during betting
  is rejected; default and duplicate names; `NotEnoughPlayersError` with < 2 active
  players.
- `test_events`: every participant gets `on_hole_cards` with exactly its own 4 cards
  after the blinds and before the first `on_turn` of the hand, sitting-out players do
  not; every seated player (actor included, sitting-out included, waiting excluded)
  receives the same event sequence in the same order: `HandStarted`, `BlindPosted` ×2,
  `PlayerActed` per action, `BoardDealt` per street (also on all-in run-out),
  `HandShown` at showdown, `BetReturned` / `PotAwarded`; `stack_bb` in each event equals
  the player's stack right after it, and the last stack per player equals the
  `HandResult` final stack; fold-out hand has no `HandShown` and a single `whole` award;
  forced folds carry `reason`; `seq` is strictly increasing; `TableView.history`
  matches the events delivered so far; an `on_event` that raises does not break the
  hand.
- `test_sit_out`: busted player without rebuy becomes `SITTING_OUT`, is not dealt,
  posts no blind and is skipped by the button; keeps receiving `on_hand_end` and events
  after later hands; rebuy in `on_hand_end` (or between hands) returns them and they are
  dealt in the next hand; busted player who rebuys immediately never sits out; an
  agent whose `on_turn` returns without acting is folded and sits out from the next
  hand with its chips, keeps receiving `on_hand_end`, and returns after `sit_in()`;
  `run()` stops when fewer than two active players remain.
- `test_join`: join between hands with no seat → player posts BB next hand (also with a
  single active player → heads-up BB, and with sitting-out seats around the button);
  explicit offset places player at that position relative to the button; join from
  inside an `on_turn` callback is queued — player is not dealt, gets no events and no
  `on_hand_end`, and is dealt in the next hand at the offset from the moved button;
  multiple queued joins seated in FIFO order; on an empty table the first joiner takes
  the button and is SB and the second is BB (any `seat` ignored), and the same with one
  active player left; `join` returns `None` and, both between hands and
  mid-hand, the joiner's agent gets `on_seated`, then `on_hole_cards` and `on_turn` in
  the next hand (dealt 4 cards); invalid stack/offset, duplicate name and full table
  (counting sitting-out players) are rejected.
- `test_leave`: leave between hands empties the seat and renumbers; leaving button
  passes the button on; `leave()` inside `on_turn` folds and removes after the hand;
  leave requested mid-hand by another player folds them at their turn; all-in leaver
  is paid before removal; sitting-out player leaves; waiting player leaves the queue;
  `on_left` reports final stack; `PlayerLeft` reaches the remaining players but not the
  leaver; name reusable afterwards; leave is idempotent.
- `test_remote`: a hand played with agents in child processes via `RemoteAgent`, each
  receiving its own `HoleCards` before its first turn and the full event stream; an
  agent process joining mid-hand via a `Join` message; a remote sitting-out agent
  rebuying from `on_hand_end`; a remote agent leaving via `Leave`; a remote agent that
  does not reply within a short test `action_timeout` is folded, sits out, and its late
  reply is discarded.
