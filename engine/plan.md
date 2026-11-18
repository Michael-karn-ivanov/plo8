# Engine — implementation plan

Python library (`engine`) providing an Omaha Hi-Lo 8-or-better game engine for a
**fixed-limit** table (the repo is named "plo8", but the game is limit, not pot-limit).
The engine is the library as a whole, not one class: there is no `Engine` class. It
exports `Player`, `Dealer`, `Environment` and the classes they depend on (`Action`,
`LegalActions`, `Decision`, views, events, ...). Card types (`Card`, `Rank`, `Suit`, `Hand`, `Board`,
`Street`, `Deck`) come from `//entities:entities` and are re-exported for convenience.
A hand can be played in two interchangeable ways:
- **callback mode** (`Environment.start_round()`): agents drive players through
  callbacks;
- **step mode** (`Environment.start_hand()` / `step()`): the caller gets each decision
  and submits the action, so one process can run many tables and decide for all of
  them in batches (reinforcement learning, `//rl_env`).

Either way the engine enforces rules, moves chips, handles all-ins/side pots, decides
who wins and reports every public table event to the seated players. Both modes run
the same hand code (see Dealer → One hand flow, two drivers), which is how they are
kept identical.

## Layout

```
engine/
  spec.md
  plan.md
  BUCK
  engine/
    __init__.py        # public exports
    evaluator.py       # high hand ranking, 8-or-better low ranking, Omaha 2+3 selection (table-driven)
    actions.py         # ActionType, Action, LegalActions
    player.py          # Player, PlayerStatus, agent callback protocol
    pots.py            # Pot, side-pot construction, pot splitting
    dealer.py          # Dealer: dealing, betting rounds, showdown, button movement; hand generator
    environment.py     # Environment: table setup, joining/leaving, start_round(), step mode
    step.py            # Decision, HandHistory, step-mode errors
    views.py           # read-only snapshots passed to agents (TableView, HandResult)
    events.py          # public table events broadcast to seated players
    remote.py          # adapter for agents running in separate processes
    benchmark.py       # step-mode self-play workload + report (library function, stdlib only)
  bench/
    bench.py           # command-line wrapper around engine.benchmark (buck2 binary)
  tests/
    reference_evaluator.py  # frozen copy of the original evaluator, used as the oracle
    test_evaluator.py
    test_evaluator_equivalence.py
    test_pots.py
    test_betting.py
    test_dealer.py
    test_environment.py
    test_stacks.py
    test_replay.py
    test_events.py
    test_join.py
    test_sit_out.py
    test_leave.py
    test_remote.py
    test_step.py
    test_step_reads.py
    test_step_equivalence.py
    test_lazy_views.py
    test_benchmark.py
  fuzz/
    test_fuzz_step.py  # 100,000 random step-mode hands (own test target)
```

## Build (buck2)

`engine/BUCK`:

- `python_library(name = "engine", srcs = glob(["engine/**/*.py"]), base_module = "", deps = ["//entities:entities"], visibility = ["PUBLIC"])`
  — includes `engine/benchmark.py`, so other projects (`//rl_env`'s benchmark) can
  call the engine-only workload from the engine's own code.
- `python_test(name = "engine_test", srcs = glob(["tests/*.py"]), base_module = "engine_tests", deps = [":engine"])`.
  Tests use the standard `unittest` module, so no third-party dependencies are needed
  with the system demo toolchain.
- `python_test(name = "engine_fuzz_test", srcs = glob(["fuzz/*.py"]), base_module = "engine_fuzz", deps = [":engine"])`
  — the 100,000-hand fuzz test, kept apart because it runs for about a minute.
  The hand count can be lowered with the `ENGINE_FUZZ_HANDS` environment variable for
  quick local runs; the default (and CI) is 100,000.
- `python_binary(name = "bench", main = "bench/bench.py", deps = [":engine"])` — the
  benchmark tool (see Speed).

Run: `buck2 build //engine:engine`, `buck2 test //engine/...`,
`buck2 run //engine:bench -- [options]`.

## Units / chips

Stacks are expressed in big blinds as the spec requires, but stored internally as
integer chips to avoid float rounding in pot splits:

- The chip unit is `chips_per_bb` on `Environment` (default `100`): BB = `chips_per_bb`
  chips, SB = `chips_per_bb // 2`, small bet = `chips_per_bb`, big bet =
  `2 * chips_per_bb`. It must be an even `int` ≥ 2 (`ValueError` otherwise), so the SB
  is exactly half the BB and half a bet unit (the all-in reopening threshold) is a whole
  number of chips.
- Every chip computation — blinds, bets, pots, side pots, hi/lo halves and odd chips —
  is done in these integer chips. A caller that replays a hand recorded in chips sets
  `chips_per_bb` to the recording's big blind, so engine chips are the recording's
  chips and odd-chip splits come out exactly as recorded (with the default 100 a
  1-chip odd chip of the recording would not exist or land elsewhere).
- `Player.stack` (float, in BB) is a property over `Player._chips` (int).
- All public amounts (`rebuy`, `join`, views, events, `LegalActions`) accept/return BB;
  conversion is done in one place (`Dealer.to_chips` / `Dealer.to_bb`, using the table's
  `chips_per_bb`) and incoming amounts are rounded to whole chips
  (`round(amount_bb * chips_per_bb)`), so a BB amount that is a whole number of the
  table's chips (e.g. `1.234` at 1,000 chips/BB, despite float representation) converts
  to exactly those chips. An amount that rounds to 0 chips is rejected like any
  non-positive amount.
- `Environment.chips_per_bb`, `Dealer.small_blind` and `Dealer.big_blind` (in chips) are
  readable, so hosts can report the table settings.
- Bet sizes are fixed by the limit structure (see Actions), so agents never pass a bet
  amount.

### Starting stacks are seated exactly

A starting stack in BB is converted to chips **once, with the table's chip unit**,
whichever way the player arrives (`//rl_env` replays recorded hands with players
passed to `Environment(players=...)`, not through `join`):
- **Players created directly** (`Player(name, stack, agent)`): a player that is not yet
  at a table has no chip unit, so it keeps the amount as given in BB
  (`Player._unseated_bb`; a `rebuy()` before seating adds to it, and to
  `_unseated_rebuy_bb`). `stack` / `chips` of an unseated player report that amount
  (chips at the default 100/BB, for display only). On the first `_attach` to a dealer
  `_chips = to_chips(_unseated_bb, dealer.chips_per_bb)` (and `rebuy_chips` likewise
  from `_unseated_rebuy_bb`) — no intermediate conversion at 100 chips/BB, so e.g.
  `Player("A", 1.234)` at `chips_per_bb=1000` gets 1,234 chips (not 1,230), and
  `Player("A", 0.005)` at 1,000 gets 5 chips (not rejected as 0). A player is attached
  to one dealer for its whole life, so there is no later rescaling. At the default unit
  this gives the same chips as before.
- `Environment.__init__` validates each initial player's stack with the table's unit:
  `to_chips(p._unseated_bb, chips_per_bb) <= 0` → `ValueError` (negative stacks are
  rejected by `Player` itself).
- **Joiners**: `join(stack_bb)` converts with `Dealer.to_chips` (the table's unit) and
  sets the chips on the new player directly — already exact; covered by the same tests.
- A stack that is not a whole number of the table's chips is rounded once to the
  nearest chip (e.g. 1.237 BB at 10 chips/BB → 12 chips).

## Cards, hands, board and deck (from `//entities`)

The engine does not define its own card types; it imports them from the `entities`
library (`from entities import Card, Rank, Suit, Hand, Board, Street, Deck`):
- `Card`, `Rank`, `Suit` — immutable cards; `Rank` values 2..14 give high-hand order and
  `Rank.low_value` (ace = 1) gives 8-or-better low order. `Card.index` (0..51) is the
  compact integer the fast evaluator works with.
- `Hand` — a player's 4 hole cards. The dealer deals one card at a time to each player
  (`Deck.deal_one()`), then builds a `Hand` from each player's 4 cards, in the order
  they were received.
- `Board` / `Street` — the dealer's board is an immutable `Board`, replaced street by
  street via `with_flop` / `with_turn` / `with_river`; `Board.street` is the street
  reported in `TableView`.
- `Deck` — the dealer owns one deck for the life of the table: either the deck passed
  to `Environment(deck=...)` (any `Deck` instance, including subclasses such as a
  stacked deck) or, by default, `Deck(random.Random(seed))`. How the dealer uses it is
  a fixed contract, see Dealer → Deck use and deal order.
  No burn cards are dealt: with an RNG-shuffled deck that no agent can see, burning has
  no effect on fairness or odds, and it keeps the deal order simple for replays.
- All of these are picklable, so they can be sent to agent processes as-is.

`entities` already provides pickling (see `entities/plan.md`, "Changes requested by
engine") and `Deck.put_on_top`, which callers use to build stacked decks (the engine
itself does not call it). For speed the engine relies on two cheap operations from
`entities` (requested in `entities/spec.md`): `Card.index` is a stored integer (not
recomputed from the enums on every access) and `Deck.deal_one()` is a constant-time
pop from the top. Stacked cards are caller-built `Card` objects that need not be the
interned `ALL_CARDS` instances, so the engine never keys on `id(card)`; it always goes
through `card.index`.

## Hand evaluation (`evaluator.py`)

Public functions (signatures and results unchanged from the original implementation):
- `rank_high(five_cards) -> tuple` — comparable tuple (category, tiebreak ranks...),
  categories high card … straight flush; wheel A-2-3-4-5 is a straight.
- `rank_low(five_cards) -> tuple | None` — 8-or-better: five distinct ranks all ≤ 8
  (Ace counts as 1); returns tuple of ranks sorted descending (lower compares better),
  `None` if not qualifying.
- `best_omaha(hand: Hand, board: Board) -> (high_rank, low_rank | None)` — best over all
  C(4,2)·C(5,3) = 60 combinations using **exactly two** hole cards and **exactly three**
  board cards; high and low are chosen independently. Accepts 3–5 board cards, raises
  `ValueError` for anything else (as before).

Fast implementation (pure Python, lookup tables built once at import):
- **Per-card tables** indexed by `card.index`: `PRIME[i]` (a distinct prime per rank,
  2..41), `SUIT[i]` (0..3), `LOWBIT[i]` (`1 << low_value` if `low_value ≤ 8`, else 0).
- **High ranks as integers.** Within one category the original tuples all have the
  same length (quads/full house 2 tiebreaks, flush/high card 5, trips/two pair 3,
  pair 4, straights 1), so encoding a tuple as
  `category·16⁵ + Σ tiebreak_k·16^(4−k)` preserves its order exactly. `HIGH_INT` maps a
  rank multiset to the integer of its non-flush rank, `FLUSH_INT` maps five distinct
  ranks to the integer of its flush / straight-flush rank, both keyed by the product of
  the five rank primes (unique per multiset). `DECODE_HIGH[int] -> tuple` gives the
  public tuple back. All tables are built at import by running the **original**
  `rank_high` logic over the 6,175 rank multisets (and the 1,287 flushable rank sets),
  so the tuples are the original ones by construction.
- **Low as a bitmask.** A qualifying low is five distinct low values ≤ 8; for sets of
  equal size, comparing the descending tuples equals comparing the bitmasks
  `Σ 1 << v` as integers, so the best low is the minimum mask. `DECODE_LOW[mask]`
  gives the public tuple.
- `rank_high` / `rank_low` (5 cards): one prime product + one suit test + one dict
  lookup, then decode.
- `best_omaha`: precompute the 6 hole pairs (prime product, suit if suited else −1,
  low mask if two distinct low values else 0) and the 10 (or 1 / 4) board triples
  (same three fields). For each pair × triple: high key = product; a flush exists only
  if pair and triple have the same suit; take the max integer. Low: a combination
  qualifies iff pair and triple masks are both non-zero and disjoint (popcount 5 is
  then automatic); take the minimum `pair_mask | triple_mask`. Decode both at the end.
- `Dealer._settle` computes the board triples once per showdown and evaluates every
  shown hand against them (`best_omaha_many(hands, board)` internal helper), instead
  of rebuilding them per player.
- Results must stay exactly those of the original implementation: the original code
  is kept verbatim in `tests/reference_evaluator.py` and is the oracle for
  `test_evaluator_equivalence` (see Tests). It is not used at runtime.

## Actions (`actions.py`)

- `ActionType`: `FOLD`, `CHECK`, `CALL`, `RAISE` (a bet into an unopened round is a
  `RAISE` from 0).
- `Action(type)` — no amount: in limit poker every bet/raise is exactly one bet unit.
  The four possible `Action`s are module-level singletons (`Action.fold()` etc. return
  them), so applying an action allocates nothing.
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
  player's stack), `bet_unit` and `raises_left` on this street; `actions` lists the
  legal `ActionType`s (computed once, stored), `allows(type)`.

## Player (`player.py`)

State:
- `name` (nick, see below), `seat`, `status`, `_chips` (stack),
  `hole_cards` (`Hand | None`, private), `street_bet`, `total_bet` (this hand),
  `folded`, `all_in`, `has_acted`, `leave_requested`; before the player is first
  attached to a table, `_unseated_bb` / `_unseated_rebuy_bb` (see Units / chips →
  Starting stacks are seated exactly).
- `PlayerStatus`:
  - `WAITING` — joined, queued for a seat (see Environment → Joining); `seat is None`.
  - `ACTIVE` — seated with chips; dealt into the next hand.
  - `SITTING_OUT` — seated but not playing: either 0 chips after the rebuy window of a
    hand, or the agent failed to act on its turn (see Sitting out). Keeps the seat but
    is not dealt cards, posts no blinds, never gets `on_turn` / `on_hole_cards`.
  - `LEFT` — removed from the table; terminal, the object is no longer usable for play.
- `name` is set when the player is created / joins (`Player(name=None, stack=0.0,
  agent=None)`); if omitted the environment assigns `"Player<N>"`. Names must be unique
  at the table (seated — active or sitting out — and waiting players); `name` is what
  views and events show for the player. A name becomes free again once its player has
  left.
- Each player keeps a back-reference to its `Environment` (set on seating / join), used
  by `leave()` and by `rebuy()` validation.

Agent-facing API (the "managing agent" calls these):
- `fold()`, `check()`, `call()`, `raise_()` — called during its turn in callback mode;
  each records the pending action; validated by the dealer against `LegalActions`.
  Invalid actions raise `IllegalActionError` to the agent, which may try again within
  the same turn. If the turn ends without a valid action (see No action / timeout), the
  hand is folded and the player sits out from the next hand. In step mode these raise
  `IllegalActionError` ("not your turn"): actions are submitted through
  `Environment.step()` only.
- `rebuy(amount_bb)` — adds to the stack; amount > 0, resulting stack ≤ table max buy-in
  if set. Allowed only outside betting: inside `on_hand_end` or while the dealer is not
  in a hand (raises `IllegalActionError` during betting — in step mode also while a
  decision is pending — for `WAITING` and for `LEFT` players). A `SITTING_OUT` player
  that rebuys becomes `ACTIVE` and is dealt into the next hand.
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
the agent does not define them. The lookups are cached per player (`Player._callbacks`,
a small dict filled on first use) and the cache is cleared by `set_agent`, so the hot
path does not repeat `getattr` for every event:
- `on_hole_cards(player, hand: Hand, view: TableView) -> None` — called once per hand
  for every participant, as soon as all hole cards are dealt and before the first
  decision of the hand. `hand` is that player's own `Hand` only; `view` is the same
  per-player snapshot as in `on_turn` (blinds already posted, own cards only). Not
  called for sitting-out or waiting players. Optional; the view is built only for
  agents that define this callback. Called in both modes.
- `on_turn(player, view: TableView, legal: LegalActions) -> None` — callback mode only:
  invoked by the dealer when it is this player's turn; the agent must call exactly one of
  `fold/check/call/raise_` on `player` before returning, or call `player.leave()`
  (which counts as a fold). Returning without a valid action counts as not acting (see
  No action / timeout). Never called in step mode.
- `on_event(player, event: TableEvent) -> None` — receives every public table event
  (see Events) for the table the player is seated at, including events caused by the
  player itself. Optional; must not act on the player (actions are only accepted in
  `on_turn` / `step()`) — rebuy / leave from here follow the usual rules (leave is
  allowed any time, rebuy only between hands). Called in both modes.
- `on_hand_end(player, result: HandResult) -> None` — invoked after stacks are
  updated, whether or not there was a showdown; the agent may call `player.rebuy(...)`
  (and/or `player.leave()`). Called for every seated player — those dealt into the hand
  and those `SITTING_OUT` — so a busted player keeps getting a rebuy opportunity after
  every hand. Not called for `WAITING` players or players that already left. Called in
  both modes.
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

Default agent (no agent set, `agent is None`): check if possible, else fold; never
rebuys; ignores hole-card and event notifications — useful in tests. The dealer plays
it directly (`fallback_action(legal)`) without building a `TableView`. In step mode a
player's agent (default or not) is never asked for actions.

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
  (`"leave"`, `"timeout"`, or `None` for a voluntary fold). Step-mode actions are
  reported exactly like callback-mode ones (`reason = None`, or `"leave"`).
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

Stamping: `seq` and `hand_no` are set on the freshly built event before anyone else
has seen it (`object.__setattr__` on the new frozen instance), instead of copying it
with `dataclasses.replace`. The resulting objects are equal to what the copy produced.

Delivery (`Environment._broadcast(event)`):
- Every event goes to every **seated** player (`ACTIVE` and `SITTING_OUT`, including
  players with `leave_requested` until they are removed and the player the event is
  about). `WAITING` and `LEFT` players get no events; `PlayerJoined` is delivered to the
  joiner too, after `on_seated`.
- Delivery is synchronous and in seat order, and happens right after the state change
  is applied (chips moved / card dealt) and before the dealer moves on, so the event
  stream an agent sees is always consistent with the next `TableView` it gets.
- Fast path: if no recipient defines `on_event`, nothing is queued for delivery (there
  is nothing to order); the event is still stamped and recorded. The set of recipients
  with `on_event` is cached and invalidated when seats change (join / leave / status
  change) or an agent is replaced.
- The same events are appended to the hand's history (`Dealer.history`, a list created
  fresh by `start_hand()` for each hand, append-only during the hand), which
  `TableView.history` and `HandResult.history` expose as tuples, and which a step-mode
  caller can read without a view and without a copy (`Environment.history`, see
  Environment → Step mode). So an agent or caller that ignores `on_event` can still
  read everything that happened.

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

Views are built **only when they are consumed** (`Dealer.table_view(p)` is the only
builder):
- callback mode: for `on_turn` of players with an agent (the default agent gets none),
  for `on_hole_cards` and `on_seated` only if the agent defines them;
- step mode: when the caller calls `Environment.view()`; the view of the pending
  decision is cached until the decision is answered, so asking twice is free. The
  view-free reads (`history`, `hole_cards()`, `privileged_hole_cards()`) never build
  one.
No view is built to be dropped (checked by `test_lazy_views`).

## Pots (`pots.py`)

- Contributions are tracked per player via `total_bet` (in chips).
- `build_pots(players) -> list[Pot]`: sort distinct contribution levels of all players
  (including folded ones — their chips stay in the pot), create one pot per level;
  each `Pot` has `amount` and `eligible` = non-folded players who contributed at that
  level. Pots with a single eligible player are returned to that player (uncalled bet,
  reported as `BetReturned`).
- `split_pot(pot, hands) -> list[(player, share, chips)]`:
  - If any eligible player has a qualifying low: half to best high (ties split),
    half to best low (ties split); a player can scoop both halves.
  - Otherwise the whole pot goes to best high.
  - Odd chips (in the table's chip unit, see Units / chips): the odd chip from halving
    goes to the high half; odd chips within a tied half go to the first winner(s) left
    of the button. These rules are deterministic, so a replayed hand splits exactly as
    recorded.
  - Returns the individual shares so the dealer can credit them one by one and emit a
    `PotAwarded` per share.

## Dealer (`dealer.py`)

Holds the `Deck`, the `Board`, pots, button seat, the list of seated players, the hand
counter, the current hand's event history and an `in_hand` flag (true from the start
of a hand until end-of-hand processing is done, in both modes — so in step mode it is
true while a decision is pending). The dealer sees all seated
players but deals in only `ACTIVE` ones; the participants of a hand are fixed when it
starts (`hand_players`). Waiting players are kept by the `Environment`. The dealer
emits events through a broadcast function supplied by the `Environment`. It also holds
the chip unit (`small_blind`, `big_blind` in chips, `to_chips` / `to_bb`).

Hand flow:
1. **Start**: prepare the deck (`reset()` + `shuffle()`, see below), emit `HandStarted`.
2. **Preflop**: post SB and BB (short stacks post all-in), emitting `BlindPosted` for
   each; deal the hole cards in the documented deal order (below); once all are dealt,
   call `on_hole_cards` for each participant whose agent defines it (seat order,
   starting left of the button) — before any betting. Betting starts left of BB
   (heads-up: button is SB and acts first preflop, last postflop). SB/BB are the first
   and second `ACTIVE` players clockwise from the button; sitting-out seats are
   skipped for blinds, dealing and action.
3. **Flop**: deal 3 board cards (`BoardDealt`), betting starts at first active player
   left of the button.
4. **Turn**: deal 1 card (`BoardDealt`), betting round.
5. **River**: deal 1 card (`BoardDealt`), betting round.
6. **Showdown**: if ≥ 2 non-folded players remain.

### One hand flow, two drivers

The whole hand (steps 1–6 and settlement) is one generator, `Dealer._hand_steps()`.
It runs until a player has to decide, then yields `(player, legal)` and waits for the
answer `(action, reason)` sent back into it; when the hand is settled it returns the
`HandResult` (`StopIteration.value`). `_betting_round` is a generator used with
`yield from`. Everything that is not a decision — blinds, dealing, the forced fold of a
player with `leave_requested`, skipping folded / all-in players, run-outs, showdown,
pots — happens inside the generator, so it is the same code in both modes.

At each decision the generator computes `legal = self.legal_actions(p)` once and
yields it; the driver must answer with an action that `legal` allows (or a forced
fold). Nothing in the hand state changes between the yield and the answer.

- **Callback driver** (`Dealer.play_hand()`, used by `Environment.start_round()`):
  `start_hand()`, then loop `req = gen.send(answer)`; for each `(p, legal)` it runs the
  turn exactly as before: `p._begin_turn(legal)`, call `on_turn(p, view, legal)` (view
  built only for a real agent; the default agent is answered with
  `fallback_action(legal)`), `p._end_turn()`; a `leave()` during the turn gives
  `(FOLD, "leave")`, no / illegal action gives `(FOLD, "timeout")` and sets
  `missed_turn`. On `StopIteration` it runs `_end_hand(result)`. Any other exception
  aborts the hand (`_abort_hand`, bets refunded) and is re-raised, as today.
- **Step driver** (`Dealer.begin_step_hand()` / `Dealer.submit(action)`, used by the
  environment's step mode, see Environment → Step mode): `begin_step_hand()` runs
  `start_hand()` and advances the generator to the first yield, storing it as the
  pending decision; `submit` validates the action against the stored `legal` and
  sends it in, advancing to the next yield or to the end. On `StopIteration` it runs
  `_end_hand(result)` (same end-of-hand processing) and clears the pending decision.
  It never calls `on_turn`, never sets `missed_turn` and has no timeout.

Because the hand logic, the order of events, the `legal_actions` computation and the
end-of-hand processing are shared, the two modes produce identical events, legal
actions, results and stacks for the same deck, seating and actions.

### Deck use and deal order (contract)

How the dealer consumes the deck is part of the engine's public contract, so that a
caller-supplied deck (e.g. a stacked deck replaying a recorded hand) produces exactly
the intended hole cards and board (identical in both modes):
- **Once per hand**, before any card is dealt, the dealer calls `deck.reset()` and then
  `deck.shuffle()` — exactly once each, in that order, on the deck object itself (no
  copy), so a subclass that overrides `shuffle()` (e.g. `super().shuffle()` followed by
  `self.put_on_top(script)`) controls the top of the deck for that hand.
- **During the hand** the dealer only calls `deck.deal_one()`, one card at a time from
  the top. It never calls `deal(n)`, `remove`, `put_on_top`, `shuffle` or `reset`
  mid-hand, and never looks at cards without dealing them.
- **Hole cards**: four rounds; in each round one card to every participant, clockwise,
  starting with the first participant left of the button and ending with the button.
  Sitting-out seats are skipped. Heads-up this means the BB (left of the button)
  receives the first card and the button/SB the second. A player's `Hand` lists the
  cards in round order (card *k* of the hand is the card from round *k*).
- **Board**: then the flop (next 3 cards, in that order as the `Board.flop` tuple), the
  turn (next card) and the river (next card). **No burn cards.** The same order is used
  when the board is run out after all-ins.
- Streets that are not reached (hand ends by folds) are not dealt; their cards stay in
  the deck.
- Resulting positions, with `n` participants counted from the left of the button
  (participant `0` = first left of the button, `n-1` = the button): deck card `i`
  (0-based, top first) for `i < 4n` goes to participant `i % n` as hole card `i // n`;
  cards `4n`, `4n+1`, `4n+2` are the flop, `4n+3` the turn, `4n+4` the river.

The deal order is documented in the module docstring of `dealer.py` and covered by
`test_replay`; changing it is a breaking change for callers that stack decks
(e.g. the end-to-end tests of `//cli_server`).

Betting round (`_betting_round(order)`, generator):
- Reset `street_bet`/`has_acted` (except blinds preflop which count as bets).
- Iterate participants in seat order, skipping folded/all-in players; for each, if
  `leave_requested` is set, fold them without a decision (`PlayerActed(FOLD,
  reason="leave")`); otherwise compute `LegalActions`, yield the decision, take the
  answer, apply it (move chips from stack to `street_bet`/`total_bet`), update the
  current bet and the street's raise count, reset `has_acted` for others on a full
  raise, then emit `PlayerActed` (forced folds included, with their `reason`).
- Round ends when every non-folded, non-all-in player has acted and matched the
  current bet (or is all-in).
- If at any point only one non-folded player remains, the hand ends immediately: the
  uncalled part of the last bet is returned (`BetReturned`) and that player wins all
  pots without showing (`PotAwarded`, `share = "whole"`); remaining streets are skipped
  (no `BoardDealt` for them).
- If ≤ 1 player can still act (others all-in), skip further betting and run out the
  remaining board cards (one `BoardDealt` per street), then go to showdown. This is
  also the "no decision possible" case: e.g. when everyone left is all-in after the
  blinds the generator runs to the end without yielding once.
- Bookkeeping kept incrementally for speed (same results as recomputing): counts of
  live (non-folded) and can-act (non-folded, non-all-in) participants are updated in
  `_apply` / `_put` instead of rebuilding lists each iteration; the participant order
  left of the button is computed once per hand (rotation of `_active`) and reused for
  dealing, postflop streets and showdown; the bet unit is set once per street.

Showdown:
- Ask every non-folded player to `show_cards()` in showing order and emit `HandShown`
  for each; evaluate all shown hands against the board (one set of board triples, see
  Hand evaluation), build side pots, return uncalled bets (`BetReturned`), split each
  pot as described above and credit stacks share by share, emitting `PotAwarded`
  after each credit. An all-in player who requested to leave stays in the hand and is
  paid normally before being removed.

End of hand (in this order, identical in both modes):
1. Build `HandResult`, store it as `last_result`, call `on_hand_end` for every seated
   player that has not left — participants and `SITTING_OUT` players (rebuy / leave
   window). Rebuys done here emit `PlayerRebought`.
2. Update statuses: an `ACTIVE` player with 0 chips, or one who did not act on a turn
   this hand (callback mode only), becomes `SITTING_OUT` (`on_sit_out`,
   `PlayerSatOut`); a `SITTING_OUT` player who rebought or called `sit_in()` (and has
   chips) becomes `ACTIVE` (`on_return`, `PlayerReturned`). (A rebuy or `sit_in()`
   outside the hand-end window, between hands, also returns the player immediately.)
3. Move the button to the next `ACTIVE`, non-leaving player clockwise from the current
   button (SB/BB shift accordingly). Sitting-out players never hold the button.
4. Clear `in_hand`; the environment then removes players that requested to leave and
   seats queued joiners (see below).

## Environment (`environment.py`)

- `Environment(players=(), button=0, chips_per_bb=100, seed=None, deck=None,
  max_buy_in=None, max_seats=10, action_timeout=600.0)` — owns the `Dealer`, the seated
  player list and the join queue.
  Seated players (`ACTIVE` and `SITTING_OUT`) are kept in clockwise order;
  `Player.seat` is the index in that list. Initial `players` are seated directly
  (each must have a stack of at least one chip of the table, and gets exactly the
  chips its BB stack is worth at the table's unit, see Units / chips → Starting stacks
  are seated exactly; unnamed ones get a default name). With no initial
  players there is no button and no game until two players have joined (see Joining);
  `button` is then ignored, and `action_timeout` (default 600 s) sets the turn timeout
  (callback mode only).
- `chips_per_bb` — the chip unit (see Units / chips): even `int` ≥ 2, `ValueError`
  otherwise. Fixed for the life of the table.
- `deck` — optional `entities.Deck` instance (subclasses allowed, e.g. a stacked deck)
  used by the dealer for every hand of this table, following Dealer → Deck use and deal
  order; anything that is not a `Deck` raises `TypeError`. When omitted, the
  environment creates `Deck(random.Random(seed))`. When a deck is given, `seed` is not
  used for dealing (the supplied deck brings its own RNG); the deck is not copied, so
  the caller keeps control over it.
- Together, `deck` and `chips_per_bb` let a caller replay a recorded hand exactly: same
  hole cards and board (stacked deck + documented deal order), the same starting
  stacks in chips (exact seating) and the same odd-chip splits (chip unit = the
  recording's big blind). Seat order and button come from the usual seating rules
  (initial `players` + `button`, or the join rules below).
- Owns event broadcasting (`_broadcast`, see Events) and the per-table `seq` counter;
  the dealer and the join / leave / rebuy / sit-out bookkeeping all go through it.
- `start_round()` — callback mode: begins a new hand: resets and shuffles the deck
  (once, see Dealer → Deck use and deal order), clears hole cards and per-hand player
  state and the hand's event history, empties flop/turn/river and pots, has SB/BB
  players post blinds, then lets the dealer run the hand with the `ACTIVE` players.
  Returns the `HandResult`. After the hand, processes leave requests and the join
  queue. Raises `HandInProgressError` if a hand (e.g. a step-mode hand) is in progress.
- Requires 2–`max_seats` `ACTIVE` players to start a hand (either mode); otherwise
  raises `NotEnoughPlayersError` (sitting-out players do not count).
- `run(n_hands)` convenience loop (callback mode); stops early when fewer than two
  `ACTIVE` players remain after a hand's rebuy window.
- `max_seats` counts seated (active + sitting out) and waiting players; a sitting-out
  player keeps occupying a seat until they leave.

### Step mode

Types (`step.py`, exported from `engine`):
- `Decision` — frozen dataclass describing the pending decision: `hand_number`, `seat`,
  `name`, `player` (the `Player` object, for convenience) and `legal: LegalActions`.
  It carries no view and no cards.
- `HandHistory` — read-only `collections.abc.Sequence[TableEvent]` over the dealer's
  history list of one hand (`__slots__ = ("_events",)`; `__len__`, `__getitem__` — an
  int gives the event, a slice gives a tuple of just those events — `__iter__`; no
  mutating methods, the list itself is not reachable through public attributes). It
  is a live window, not a copy: events appended to the hand later show up in it.
- `HandInProgressError(RuntimeError)` — a hand is started (either mode) while another
  one is in progress.
- `NoDecisionError(RuntimeError)` — `step()` / `view()` / `hole_cards()` called while
  no decision is pending.
- Illegal actions raise the existing `IllegalActionError`.

`Environment` methods:
- `start_hand() -> Decision | None` — starts a hand in step mode: everything
  `start_round()` does at the start (deck reset + shuffle, `HandStarted`, blinds, deal,
  `on_hole_cards`), then runs to the first decision and returns it. Returns `None` if
  the hand needed no decision at all (e.g. everyone left in the hand is all-in after
  the blinds): the hand has then been played to the end, including end-of-hand
  processing, and `last_result` holds its result. Raises `HandInProgressError` if a
  hand is in progress and `NotEnoughPlayersError` like `start_round()`.
- `step(action: Action | ActionType) -> Decision | None` — answers the pending
  decision. Validation happens **before** anything changes: not an `Action` /
  `ActionType` → `IllegalActionError`; `not decision.legal.allows(type)` →
  `IllegalActionError` naming the legal actions; no pending decision →
  `NoDecisionError`. A rejected action leaves the hand exactly as it was (same pending
  decision, same events, chips, history, `seq`), so the caller can submit again; the
  player is never folded or sat out for it. On success the action is applied
  (`PlayerActed`) and the hand runs on to the next decision, which is returned, or to
  the end (end-of-hand processing done, `last_result` set), returning `None`.
- `decision` (property) — the pending `Decision` or `None`.
- `view() -> TableView` — the acting player's table view for the pending decision
  (same as `on_turn` would have received: only that player's hole cards). Built on the
  first call and cached until the decision is answered. `NoDecisionError` if none is
  pending.
- `last_result` (property) — `HandResult` of the last finished hand (either mode),
  `None` before the first hand.
- `hand_in_progress` (property) — `dealer.in_hand`.

View-free reads (for callers such as `//rl_env` that build their own observations and
must not pay for a `TableView` at every decision; none of them builds a view):
- `history` (property) — `HandHistory` of the current hand: the public events stamped
  so far, the same events, in the same order, as `view().history` would hold. The
  `HandHistory` object is created once per hand by `Dealer.start_hand()` together with
  the fresh history list (`Dealer.history_view`), so reading it costs nothing per
  decision. A caller processes only new events with a cursor:
  `new = env.history[seen:]; seen = len(env.history)`, resetting `seen` to 0 when
  `decision.hand_number` changes (a new hand gives a new `HandHistory`). After the hand
  ends it still shows that hand's events (equal to `last_result.history`) until the
  next hand starts; before the first hand it is empty.
- `hole_cards() -> Hand` — the acting player's own hole cards for the pending decision
  (`decision.player.hole_cards`, identical to `view().hole_cards`). `NoDecisionError`
  if no decision is pending.
- `privileged_hole_cards() -> Mapping[int, Hand]` — seat → `Hand` for **every
  participant** of the current hand (folded ones included), or of the last hand once it
  is over (until the next hand starts); empty before the first hand. Built once per
  hand on first call (hole cards do not change during a hand) and returned as a
  read-only `types.MappingProxyType`; the cache is cleared by `Dealer.start_hand()`.
  This is the only engine accessor that reveals other players' hidden cards: it is
  meant for training tools allowed to see everything (the critic of `//rl_env`). It is
  deliberately separate from everything an agent receives: it is not part of
  `Decision`, `TableView`, `HandResult` (beyond the cards shown at showdown), events or
  any callback, and `RemoteAgent` / `agent_loop` never call or send it. Its docstring
  states this.

Behaviour details:
- Nothing else differs from callback mode: `on_hole_cards`, `on_event`, `on_hand_end`,
  `on_seated`, `on_sit_out`, `on_return`, `on_left` are called if the agent has them;
  events are stamped, recorded in the hand history and delivered the same way;
  `on_turn` is never called; there is no turn timeout.
- `leave()` of the acting player while their decision is pending sets
  `leave_requested` (as during a callback-mode turn); the next `step()` — whatever
  legal action is submitted — is applied as `FOLD` with `reason="leave"`, exactly as a
  `leave()` inside `on_turn` is in callback mode. Other players' leave requests are
  handled at their turn inside the generator (forced fold, no decision yielded).
- `join()` while a step-mode hand is in progress queues the joiner (`in_hand` is
  true); they are seated during end-of-hand processing, as in callback mode.
- `rebuy()` / `sit_in()` while a decision is pending raise `IllegalActionError`
  (betting is in progress); inside `on_hand_end` (run during the final `step()` /
  `start_hand()`) or between hands they work as usual.
- Determinism: a step-mode hand uses only the deck (its RNG or stacked cards), the
  seating and the submitted actions; the engine itself draws no random numbers.
- Step mode is meant for a single controlling thread per table (one process may own
  many tables); `join` / `leave` from other threads remain guarded by the table lock
  as before. The view-free reads are meant for that controlling thread (the history is
  only appended to by it).

Typical batched loop:

```python
pending = {env: env.start_hand() for env in envs}
seen = {env: 0 for env in envs}
while True:
    batch = [(env, d) for env, d in pending.items() if d is not None]
    for env, _ in batch:                              # view-free observation input
        update_obs(env, env.history[seen[env]:], env.hole_cards())
        seen[env] = len(env.history)
    views = [env.view() for env, _ in batch]          # only if the policy needs them
    actions = policy(views, [d.legal for _, d in batch])
    for (env, _), a in zip(batch, actions):
        pending[env] = env.step(a)
    for env, d in pending.items():
        if d is None:
            consume(env.last_result)
            pending[env] = env.start_hand()           # next hand
            seen[env] = 0
```

### Sitting out

- A player busted to 0 chips who does not rebuy in `on_hand_end` becomes `SITTING_OUT`:
  not dealt cards, no blinds, no `on_turn` / `on_hole_cards` / decisions, not in pots,
  shown as sitting out in views. They still receive all public events (`on_event`).
- They keep their seat position in the clockwise order (relevant for join offsets, which
  count all seated players), but are skipped for button / blinds / action.
- After every subsequent hand they still receive `on_hand_end` (with `dealt_in = False`
  for themselves) and may rebuy there; a rebuy (there, or between hands) makes them
  `ACTIVE` again and they are dealt into the next hand without posting extra blinds.
- A player who did not act on a turn (see No action / timeout; callback mode only)
  also becomes `SITTING_OUT` at the end of that hand, keeping their chips. They stay
  sitting out until their agent calls `sit_in()` (in `on_hand_end` or between hands),
  so an unresponsive agent does not hold up every later hand with another timeout.
- A sitting-out player may `leave()` at any time.

### No action / timeout (callback mode)

- Each turn has a timeout, `action_timeout` on `Environment`, default 600 seconds
  (deliberately long: it only exists to catch agents that have hung or disconnected).
- If the agent gives no valid action in time, or its `on_turn` returns without one, the
  dealer folds the player's hand (chips already committed stay in the pot; reported as
  `PlayerActed(FOLD, reason="timeout")`) and the player becomes `SITTING_OUT` at the end
  of the hand.
- For in-process agents the dealer cannot interrupt the callback, so for them the rule
  applies when `on_turn` returns without a valid action; the timeout is enforced by
  `RemoteAgent` for agents in other processes.
- None of this applies in step mode: the caller is in control, an illegal action is
  just rejected (see Step mode).

### Joining (`join`)

`join(stack_bb, seat=None, name=None, agent=None) -> None`:
- `stack_bb` — buy-in in big blinds; must be > 0 (at least one chip of the table) and
  ≤ `max_buy_in` if set. Converted with the table's chip unit, so a buy-in that is a
  whole number of the table's chips is seated with exactly those chips.
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
  are dealt in on the next `start_round()` / `start_hand()`.
- **During a hand** (either mode): the request is appended to a FIFO `pending_joins`
  queue; the player does not take part in the current hand (not dealt, not in pots,
  not asked to act, no events, no `on_hand_end`). After the hand's end processing
  (stack updates, rebuy callbacks, status updates, button move, removal of leavers),
  the environment seats queued players in order, resolving each offset relative to the
  **new** button position, so a default joiner posts the big blind in the hand that
  follows. Each queued request is re-validated when seated (offset clamped to the new
  `1..n` range).
- No missed-blind / dead-button rules: a newly seated player is dealt in the next hand
  regardless of position, without posting extra blinds.

### Leaving (`leave`)

`Environment.leave(player)` (also via `Player.leave()`):
- **`WAITING` player**: removed from `pending_joins` immediately; status `LEFT`.
- **Between hands** (seated, `in_hand` false): removed from the seat list immediately,
  seats renumbered. If the leaver held the button, the button passes to the next
  `ACTIVE` player clockwise (the one that would have received it next).
- **During a hand** (either mode): sets `leave_requested`; removal is deferred to the
  end of the hand so seat indices stay stable:
  - a participant still in the hand is folded at their next turn (or immediately, when
    called from their own `on_turn`; in step mode, by the `step()` answering their
    pending decision), reported as `PlayerActed(FOLD, reason="leave")`;
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
  `HandResult` / events / `Decision` / `privileged_hole_cards()` are stable for the
  duration of a hand.
- `join`, `leave`, `rebuy` bookkeeping, queue processing and `seq` assignment are
  guarded by a `threading.RLock`, so a host thread that accepts agent connections can
  call `join` / `leave` while the dealer is running a hand. Callbacks are invoked
  outside the lock. On the hot path the lock is taken once per event (stamp + queue)
  and `_flush()` returns immediately when the outbox is empty.
- Only the dealer's thread touches the deck (one reset + shuffle per hand, then
  `deal_one()`), so a supplied deck does not need its own locking. In step mode "the
  dealer's thread" is the thread calling `start_hand()` / `step()`.

## Multi-agent / separate processes (`remote.py`)

The engine itself is single-threaded and runs in one host process that owns the
`Environment` and `Dealer`; agents in other processes play through proxies (callback
mode; in step mode `RemoteAgent.on_turn` is simply never called, its notifications
still are):
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
  information is isolated per process. `Environment.privileged_hole_cards()` is a
  host-side, step-mode-caller accessor and is never used by this module.

## Speed

Target: ≥ 2,000 hands/s on one CPU core at a 6-player table, random legal actions in
step mode, engine only. Speed-ups never change rules, events, results or public
behaviour: every one of them must leave `test_step_equivalence`,
`test_evaluator_equivalence`, the fuzz test and all existing tests (engine, entities,
cli_server, cli_client, cli_server end-to-end cases) green.

### Benchmark workload (`engine/benchmark.py`, library)

The workload lives in the library so that `//rl_env`'s benchmark can print the
engine-only number from the engine's own code next to its own. Standard library only.

- `run_benchmark(players: int, *, tables=512, hands=20_000, stack=100.0, seed=0,
  views=False, callback=False, warmup=1_000) -> BenchResult` — one run at one table
  size:
  - builds `tables` environments of `players` seats, players without agents (no
    callbacks), each table with its own `Deck(random.Random(seed + t))`; one
    `random.Random(seed)` chooses actions;
  - loop as in Environment → Step mode: for every table with a pending decision pick
    `rng.choice(decision.legal.actions)` and `step()` it; for every table whose hand
    ended, top up any player below the starting stack with `rebuy()` (between hands,
    so the table size stays constant and nobody sits out) and `start_hand()` again.
    Views are not requested (engine-only cost); `views=True` requests one per decision
    to measure that cost separately; `callback=True` runs the same workload in
    callback mode with a random agent, for comparison;
  - timing with `time.perf_counter()`, the first `warmup` hands excluded.
- `BenchResult` (frozen dataclass): `players`, `hands`, `decisions`, `seconds`,
  `hands_per_s`, `decisions_per_s`, `showdown_share`, and the settings used (`tables`,
  `mode` — `"step"` / `"step+views"` / `"callback"`).
- `format_report(results: Sequence[BenchResult]) -> str` — the report table: one line
  per table size with hands, decisions, wall time, hands/s, decisions/s and the share
  of hands reaching showdown. `//rl_env` prints these lines (or the `BenchResult`
  numbers) as its engine-only reference.
- Deterministic for given arguments (same hands and decisions; only the timings vary).

### Benchmark tool (`bench/bench.py`, `buck2 run //engine:bench`)

A thin command-line wrapper: parses the options, calls `run_benchmark` once per table
size and prints `format_report(...)`. Options: `--tables N` (default 512),
`--players 2,6,9` (default; one run per size), `--hands H` (hands per run, default
20,000), `--stack BB` (default 100), `--seed S` (default 0), `--views`, `--callback`,
`--profile` (wrap the runs in `cProfile` and print the top functions by cumulative and
total time — profiling stays in the tool, not in the library function).

### Speed-ups (in order of expected gain; each measured with the benchmark)

1. **Lazy views**: no `TableView` for agents without `on_hole_cards`, for the default
   agent's turns, or in step mode unless `view()` is called (see Views). In the
   baseline about a third of the time went into building views. The view-free reads
   (`history` without a copy, `hole_cards()`, cached `privileged_hole_cards()`) let
   step-mode callers such as `//rl_env` keep this saving.
2. **Table-driven evaluator** (see Hand evaluation): prime-product / bitmask lookups
   over the 60 combinations, board triples shared by all players at a showdown. In the
   baseline about a third of the time went into evaluation.
3. **Event path**: in-place stamping instead of `dataclasses.replace`; no outbox
   entry when nobody listens; one lock acquisition per event; cached callback lookups.
4. **Betting loop**: incremental live / can-act counts, participant rotation computed
   once per hand, per-street bet unit, singleton `Action`s, `LegalActions.actions`
   precomputed; `build_pots` only at settlement (and for views on request).
5. **entities**: stored `Card.index` and O(1) `Deck.deal_one()` (requested in
   `entities/spec.md`).

### Results (recorded after each step, on the development Mac)

| Version | 2 players hands/s | 6 players hands/s | 9 players hands/s | decisions/s (6p) |
|---|---|---|---|---|
| Baseline (callback mode, before this change) | ~4,000 | ~1,230 | ~800 | ~30,000 |
| + step mode, lazy views | _to measure_ | _to measure_ | _to measure_ | _to measure_ |
| + table-driven evaluator | _to measure_ | _to measure_ | _to measure_ | _to measure_ |
| + event path | _to measure_ | _to measure_ | _to measure_ | _to measure_ |
| + betting loop, entities | _to measure_ | _to measure_ | _to measure_ | _to measure_ |

Each row is filled in from `buck2 run //engine:bench` output when that step lands.

### If pure Python falls short

From the baseline profile, removing view building (≈⅓) and most evaluation cost (≈⅓)
should bring 6-player throughput to roughly 2.5–3× the baseline, i.e. above the
target. If the measured result after step 5 is still below 2,000 hands/s, the gap
and the profile are recorded in the table above, and the next step is a **native
evaluator** behind the same `rank_high` / `rank_low` / `best_omaha` functions: a small
C extension (`cxx_python_extension` in buck2) taking card indices, loaded with a
fallback to the pure-Python tables when it is not built, and checked by the same
equivalence test. If evaluation is no longer the bottleneck, the next step is instead
moving the betting-state bookkeeping into flat integer arrays per table (players as
indices, not objects) inside the generator, keeping the public objects as views over
it.

## Tests

- Card, hand, board and deck behaviour (including `put_on_top`) is tested in
  `//entities`; engine tests only cover how the dealer uses them (no card dealt twice
  in a hand, same seed gives the same hand, the deal-order contract in `test_replay`).
- Stacked-deck test helpers: the `StackedDeck` classes in `test_dealer`, `test_events`,
  `test_leave`, `test_sit_out` and `test_remote` currently override only `deal()` and
  rely on `Deck.deal_one()` delegating to it. Since `//entities` made `deal_one()` a
  direct O(1) pop (it no longer calls `deal()`), those helpers no longer serve their
  scripted cards and 4 tests fail. Rewrite each helper to the contract form used in
  `test_replay`: `shuffle()` calls `super().shuffle()` then `self.put_on_top(self.top)`,
  with no `deal()` / `deal_one()` override. Test helpers must not override the deal
  methods, so the dealer's use of `deal_one()` stays the only path exercised.
- `test_evaluator`: category ordering, wheel straight, exactly-2-hole-cards rule
  (e.g., four flush cards on board + one in hand is not a flush), low qualification,
  counterfeited lows, A-2 low vs A-3 low.
- `test_evaluator_equivalence` (fast vs `tests/reference_evaluator.py`):
  - `rank_high` / `rank_low`: every one of the 6,175 rank multisets in a non-flush suit
    assignment and every flushable rank set as a flush, plus 200,000 random 5-card
    hands — identical tuples / `None`;
  - `best_omaha`: 100,000 random (hand, board) pairs with 5 board cards and 10,000
    each with 3 and 4 — identical `(high, low)`;
  - ordering: for random pairs of hands, comparing fast results gives the same
    winner as comparing reference results; error cases (wrong card counts) raise the
    same `ValueError`.
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
  players; `chips_per_bb` that is odd, < 2 or not an `int`, and a `deck` that is not a
  `Deck`, are rejected.
- `test_stacks` (exact seating, acceptance "a starting stack that is a whole number of
  chips is seated exactly"): at `chips_per_bb` 2, 10, 1,000 and 10,000 (and 100 as a
  control), for many stacks that are whole numbers of the table's chips (including
  ones that are not whole at 100/BB, e.g. 1.234 BB at 1,000, 0.005 BB at 1,000, 0.5 BB
  at 2, and values with inexact float representations such as 0.29, 1.1, 2.3 BB),
  players passed to `Environment(players=...)` and players added with `join()` (both
  between hands and queued mid-hand) have `chips == round(stack_bb · chips_per_bb)`
  exactly and report the same `stack_bb` in `PlayerJoined` / `HandStarted`; a
  `Player` rebought before seating is seated with stack + rebuy exactly; a stack below
  one chip of the table is rejected, while one that is ≥ 1 chip of the table but < 1
  chip at 100/BB is accepted; a non-whole stack is rounded once to the nearest chip;
  at the default unit seating is unchanged from before (same chips as
  `to_chips(stack, 100)`).
- `test_replay` (supplied deck and chip unit):
  - a recording `Deck` subclass sees exactly one `reset()` followed by one `shuffle()`
    per hand, before any card is dealt, and only `deal_one()` calls during the hand;
    the same deck object is used for every hand of the table;
  - a stacked deck (subclass whose `shuffle()` calls `super().shuffle()` then
    `put_on_top(script)`) deals exactly the scripted cards: for 2 (heads-up: BB gets
    the first card), 3 and 6 participants, with the button not on seat 0 and with a
    sitting-out seat that is skipped, every player's `Hand` (in round order) and the
    flop / turn / river equal the positions given by the deal-order formula; the same
    holds for an all-in run-out; a hand that ends preflop deals no board cards;
  - over several hands with the same stacked deck, each hand deals its own script
    (the overridden `shuffle()` runs once per hand and restacks);
  - with `chips_per_bb = 2` (and e.g. 10): blinds are 1 / 2 chips (0.5 / 1 BB in
    events), bet sizes scale accordingly, a split pot with an odd chip and a tied low
    awards the odd chips per the odd-chip rules in those units, and stacks in events
    and `HandResult` equal `chips / chips_per_bb`;
  - replaying the same scripted hand (same deck script, chip unit, seating and agent
    actions) on two fresh environments gives identical event streams and identical
    `HandResult`s; the same holds in step mode, and the deck sees the same calls;
  - a recorded hand with players created directly (`Environment(players=...)`, as
    `//rl_env` does) at the recording's chip unit, with starting stacks that are not
    whole at 100/BB, reproduces the recorded final stacks in chips.
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
  hand; stamped events equal `dataclasses.replace(original, seq=…, hand_no=…)`.
- `test_sit_out`: busted player without rebuy becomes `SITTING_OUT`, is not dealt,
  posts no blind and is skipped by the button; keeps receiving `on_hand_end` and events
  after later hands; rebuy in `on_hand_end` (or between hands) returns them and they are
  dealt in the next hand; busted player who rebuys immediately never sits out; an
  agent whose `on_turn` returns without acting is folded and sits out from the next
  hand with its chips, keeps receiving `on_hand_end`, and returns after `sit_in()`;
  `run()` stops when fewer than two active players remain; in step mode a busted
  player sits out the same way, and an illegal `step()` never makes anyone sit out.
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
  (counting sitting-out players) are rejected; a join while a step-mode decision is
  pending is queued and seated when the final `step()` ends the hand.
- `test_leave`: leave between hands empties the seat and renumbers; leaving button
  passes the button on; `leave()` inside `on_turn` folds and removes after the hand;
  leave requested mid-hand by another player folds them at their turn; all-in leaver
  is paid before removal; sitting-out player leaves; waiting player leaves the queue;
  `on_left` reports final stack; `PlayerLeft` reaches the remaining players but not the
  leaver; name reusable afterwards; leave is idempotent; in step mode, the acting
  player leaving makes the next `step()` a `FOLD` with `reason="leave"`, and another
  player's leave request folds them at their turn without a decision.
- `test_remote`: a hand played with agents in child processes via `RemoteAgent`, each
  receiving its own `HoleCards` before its first turn and the full event stream; an
  agent process joining mid-hand via a `Join` message; a remote sitting-out agent
  rebuying from `on_hand_end`; a remote agent leaving via `Leave`; a remote agent that
  does not reply within a short test `action_timeout` is folded, sits out, and its late
  reply is discarded.
- `test_step`:
  - `start_hand()` returns the first decision (preflop, first to act left of the BB;
    heads-up the button/SB) with the same `LegalActions` as `legal_actions(p)`; blinds
    and deal events are already in the history; `on_hole_cards` was called;
  - `view()` shows only the acting player's hole cards and equals what `on_turn` would
    get; repeated `view()` returns the cached object; `decision` property matches the
    last returned decision;
  - `step()` with each legal action type; with an illegal type, a non-`Action`, and
    while no decision is pending: raises (`IllegalActionError` / `NoDecisionError`) and
    the state is unchanged — same `decision`, `history`, every player's chips and bets,
    `seq`, deck remaining cards — and a legal action afterwards proceeds normally;
  - no possible decision: heads-up with a 0.5 BB SB and 1 BB BB (both all-in on the
    blinds), and 3+ players all all-in after the blinds → `start_hand()` returns
    `None`, the hand ran to a full result (board of 5 cards, `HandShown`, awards),
    end-of-hand processing done (`in_hand` false, button moved, `on_hand_end` called);
  - `start_hand()` / `start_round()` while a step-mode hand is in progress →
    `HandInProgressError`; `start_hand()` with < 2 active players →
    `NotEnoughPlayersError`;
  - end of hand: stacks updated, `on_hand_end` / rebuy window, leavers removed, queued
    joiners seated, button moved, `last_result` set; `on_turn` never called;
  - determinism: two fresh environments with the same seed (and with the same stacked
    deck), seating and actions give identical events and results.
- `test_step_reads` (view-free reads, acceptance "events and hole cards read without a
  view equal the view's, and reading them builds no view"):
  - over many random step-mode hands (sizes 2–10, short stacks and all-ins included),
    at every decision: `tuple(env.history) == env.view().history` and
    `env.hole_cards() == env.view().hole_cards`, with the view-free reads done first on
    a fresh decision; and on a second set of hands with views never requested, a
    `Dealer.table_view` call counter stays at 0 while `history`, `hole_cards()` and
    `privileged_hole_cards()` are read at every decision;
  - cursor reading: concatenating `env.history[seen:]` slices taken at each decision
    (and once after the hand ends) gives exactly `last_result.history`; after the hand
    `env.history` equals `last_result.history`; a new hand gives a new, initially
    short `HandHistory`, and the old one still shows the old hand;
  - `history` is not a copy: the same `HandHistory` object is returned at every
    decision of a hand, and its length grows as events are added; it has no mutating
    methods (`append`, `__setitem__`, `__delitem__` → `AttributeError` / `TypeError`)
    and slicing returns a tuple;
  - `privileged_hole_cards()` maps exactly the participants' seats (sitting-out seats
    absent, folded players present) to the `Hand`s dealt (checked against a stacked
    deck and the deal-order formula); it is read-only (`MappingProxyType`), the same
    object at every decision of a hand; the cards it reveals for other seats never
    appear in `Decision`, `view()`, `on_hole_cards` or events before showdown;
  - `hole_cards()` with no pending decision → `NoDecisionError`; reading any of the
    three changes no state (`seq`, history length, chips, pending decision).
- `test_step_equivalence` (the interchangeability acceptance test): for several
  hundred seeds, table sizes 2–10, stacks drawn to include short stacks (< 1 BB,
  1–3 BB), all-ins and multi-way side pots, and button positions: play the hand in
  callback mode with an agent that records each `LegalActions` it is offered and picks
  an action from a seeded RNG, then on a fresh environment with the same deck seed
  play it in step mode choosing with the same RNG sequence. Assert identical event
  streams (as received through `on_event`, including `seq` and `hand_no`), identical
  legal actions at every decision, identical `HandResult`s and identical final stacks.
  Also over multi-hand sessions (button moves, busts and rebuys, a leave and a join
  in the middle), and with views requested at every decision vs never (no effect on
  results).
- `test_lazy_views`: counting `Dealer.table_view` calls — zero in step mode without
  `view()` and with agents lacking `on_hole_cards` (also when `history`,
  `hole_cards()` and `privileged_hole_cards()` are read at every decision); exactly
  one per `view()`-requested decision; one per `on_hole_cards` agent per hand; one per
  callback-mode turn of a real agent and none for the default agent.
- `test_benchmark`: `run_benchmark` with a tiny workload (e.g. 4 tables, 50 hands,
  warm-up 10) at 2 and 6 players returns a `BenchResult` with the requested hand count,
  positive decisions and rates, a showdown share in [0, 1]; two runs with the same
  arguments give the same hands, decisions and showdown share; `views=True` and
  `callback=True` run; `format_report` gives one line per result containing its
  hands/s and decisions/s.
- `fuzz/test_fuzz_step` (`engine_fuzz_test`): 100,000 random step-mode hands
  (`ENGINE_FUZZ_HANDS` overrides the count) on tables of random size 2–10 with random
  stacks (0.5–200 BB, players topped up / rebought between hands), random legal
  actions; no exceptions, and for every hand `Σ net` over seats is exactly 0 in chips
  (`round(net · chips_per_bb)`), every final stack is ≥ 0, and the sum of stacks
  before the hand equals the sum after it. A failing seed is printed so it can be
  replayed.
- Regression outside this project: all tests of `//entities`, `//cli_server` and
  `//cli_client`, and all end-to-end cases of `//cli_server`, are run unchanged
  (`buck2 test //...`) — callback mode keeps its behaviour.
