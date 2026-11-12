# Engine — implementation plan

Python library (`engine`) for a **fixed-limit** Omaha Hi-Lo 8-or-better table (the
repo is named "plo8", but the game is limit, not pot-limit). It exports
`Player`, `Dealer`, `Environment` and the classes they depend on (`Action`,
`LegalActions`, views, ...). Card types (`Card`, `Rank`, `Suit`, `Hand`, `Board`,
`Street`, `Deck`) come from `//entities:entities` and are re-exported for convenience. Agents drive players through callbacks; the engine enforces rules,
moves chips, handles all-ins/side pots and decides who wins.

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
    player.py          # Player, agent callback protocol
    pots.py            # Pot, side-pot construction, pot splitting
    dealer.py          # Dealer: dealing, betting rounds, showdown, button movement
    environment.py     # Environment: table setup, joining players, start_round()
    views.py           # read-only snapshots passed to agents (TableView, HandResult)
    remote.py          # adapter for agents running in separate processes
  tests/
    test_evaluator.py
    test_pots.py
    test_betting.py
    test_dealer.py
    test_environment.py
    test_join.py
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
- All public amounts (`rebuy`, `join`, views, `LegalActions`) accept/return BB; conversion is
  done in one place (`to_chips` / `to_bb` helpers) and amounts are rounded to whole chips.
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
- `name` (nick, see below), `seat`, `status` (`WAITING` | `SEATED`), `_chips` (stack),
  `hole_cards` (`Hand | None`, private), `street_bet`, `total_bet` (this hand),
  `folded`, `all_in`, `has_acted`.
- `name` is set when the player is created / joins (`Player(name=None, agent=None)`);
  if omitted the environment assigns `"Player<N>"`. Names must be unique at the table
  (seated and waiting players); `name` is what views show for the player.
- `seat` is `None` while the player is `WAITING` (see Environment → Joining).

Agent-facing API (the "managing agent" calls these during its turn):
- `fold()`, `check()`, `call()`, `raise_()` — each records the pending
  action; validated by the dealer against `LegalActions`. Invalid actions raise
  `IllegalActionError` (the dealer may treat repeated errors / no action as fold).
- `rebuy(amount_bb)` — adds to the stack; only allowed between hands (dealer is not
  in a hand), amount > 0, optional table max buy-in.
- `show_cards() -> Hand` — called by the dealer at showdown; the hole cards
  then become public in the hand result. (Mucking is not modelled: a player
  required to show at showdown always shows.)

Callbacks (set via constructor or `Player.set_agent(agent)`; agent implements
an `Agent` protocol):
- `on_turn(player, view: TableView, legal: LegalActions) -> None` — invoked by the
  dealer when it is this player's turn; the agent must call exactly one of
  `fold/check/call/raise_` on `player` before returning.
- `on_hand_end(player, result: HandResult) -> None` — invoked after stacks are
  updated, whether or not there was a showdown; the agent may call `player.rebuy(...)`.
  Only called for players that were dealt into the hand (not for `WAITING` players).
- Optional `on_seated(player, view: TableView)` — called when a waiting player is
  actually seated (immediately on `join` between hands, or after the current hand).
- Optional `on_event(event)` for observing other players' actions/board cards and
  players joining (`PlayerJoined(name, seat)`).

Default agent (no callbacks set): check if possible, else fold — useful in tests.

## Views (`views.py`)

Snapshots handed to agents so they cannot mutate engine state or see others' cards:
- `TableView`: street (`Street`), board (`Board`), pot total, side pots, current bet, bet unit and
  raises made on this street, button/SB/BB seats, per-seat
  public info (name, stack, street_bet, total_bet, folded, all_in), names of players
  waiting to be seated, action history for the hand, the acting player's own
  `hole_cards`.
- `HandResult`: final `Board`, shown hands (seat → `Hand`) if showdown, winners per pot
  (high / low shares with amounts), net chip change per seat, final stacks,
  `showdown: bool`; per-seat entries also carry the player's name.

## Pots (`pots.py`)

- Contributions are tracked per player via `total_bet`.
- `build_pots(players) -> list[Pot]`: sort distinct contribution levels of all players
  (including folded ones — their chips stay in the pot), create one pot per level;
  each `Pot` has `amount` and `eligible` = non-folded players who contributed at that
  level. Pots with a single eligible player are returned to that player (uncalled bet).
- `split_pot(pot, hands)`:
  - If any eligible player has a qualifying low: half to best high (ties split),
    half to best low (ties split); a player can scoop both halves.
  - Otherwise the whole pot goes to best high.
  - Odd chips: the odd chip from halving goes to the high half; odd chips within a
    tied half go to the first winner(s) left of the button.

## Dealer (`dealer.py`)

Holds the `Deck`, the `Board`, pots, button seat, the list of seated players and an
`in_hand` flag (true from the start of `play_hand()` until end-of-hand processing is
done). Only `SEATED` players are known to the dealer; waiting players are kept by the
`Environment`.

Hand flow (`Dealer.play_hand()`):
1. **Preflop**: post SB and BB (short stacks post all-in), deal 4 hole cards to each
   active player one at a time starting left of the button; betting starts left of BB
   (heads-up: button is SB and acts first preflop, last postflop).
2. **Flop**: deal 3 board cards, betting starts at first active player left of
   the button.
3. **Turn**: deal 1 card, betting round.
4. **River**: deal 1 card, betting round.
5. **Showdown**: if ≥ 2 non-folded players remain.

Betting round (`_betting_round(street)`):
- Reset `street_bet`/`has_acted` (except blinds preflop which count as bets).
- Iterate seats in order, skipping folded/all-in players; for each, compute
  `LegalActions`, invoke the player's `on_turn` callback, read and validate the action,
  apply it (move chips from stack to `street_bet`/`total_bet`), update the current bet
  and the street's raise count, reset `has_acted` for others on a full raise.
- Round ends when every non-folded, non-all-in player has acted and matched the
  current bet (or is all-in).
- If at any point only one non-folded player remains, the hand ends immediately: that
  player wins all pots without showing; remaining streets are skipped.
- If ≤ 1 player can still act (others all-in), skip further betting and run out the
  remaining board cards, then go to showdown.

Showdown:
- Ask every non-folded player to `show_cards()`, evaluate with `best_omaha`, build side
  pots, split each pot as described above, credit stacks.

End of hand:
- Build `HandResult`, call `on_hand_end` for every player dealt into the hand (rebuy window).
- Move the button to the next seated player with chips (SB/BB shift accordingly).
  Players with zero chips who did not rebuy sit out of the next hand.
- Clear `in_hand`; the environment then seats queued joiners (see below).

## Environment (`environment.py`)

- `Environment(players=(), button=0, chips_per_bb=100, seed=None, max_buy_in=None,
  max_seats=10)` — owns the `Dealer`, the seated player list and the join queue.
  Seated players are kept in clockwise order; `Player.seat` is the index in that list.
  Initial `players` are seated directly (each must have a positive stack; unnamed ones
  get a default name).
- `start_round()` — begins a new hand: resets the deck (shuffle), clears hole cards
  and per-hand player state, empties flop/turn/river and pots, has SB/BB players post
  blinds, then lets the dealer run the hand. Returns the `HandResult`. After the hand,
  processes the join queue.
- `run(n_hands)` convenience loop; stops early if fewer than two players have chips.
- Requires 2–`max_seats` seated players with chips to start a hand.

### Joining (`join`)

`join(stack_bb, seat=None, name=None, agent=None) -> Player`:
- `stack_bb` — buy-in in big blinds; must be > 0 and ≤ `max_buy_in` if set.
- `name` — the player's nick; must be unique among seated and waiting players
  (`ValueError` otherwise); default name assigned if omitted.
- `seat` — position **relative to the button** (clockwise offset): `1` = first seat left
  of the button (SB), `2` = BB, …, up to `n` = right of the button (button's seat itself
  is not available; the button stays with its current player). If `None`, the player
  joins **as big blind**: offset 2, or 1 when only one player is seated (heads-up, the
  button posts SB and the other player BB). Offsets outside `1..n` raise `ValueError`.
- Raises `TableFullError` if seated + waiting players would exceed `max_seats`.
- Returns the new `Player` (with `status = WAITING`) so the caller can attach/keep a
  handle to it.

Seating:
- **Between hands** (`dealer.in_hand` is false): the player is seated immediately —
  inserted into the clockwise list at the requested offset from the current button,
  seats are renumbered, `status = SEATED`, `on_seated` / `PlayerJoined` are fired. They
  are dealt in on the next `start_round()`.
- **During a hand**: the request is appended to a FIFO `pending_joins` queue; the
  player does not take part in the current hand (not dealt, not in pots, not asked to
  act, no `on_hand_end`). After the hand's end processing (stack updates, rebuy
  callbacks, button move), the environment seats queued players in order, resolving
  each offset relative to the **new** button position, so a default joiner posts the
  big blind in the hand that follows. Each queued request is re-validated when seated
  (offset clamped to the new `1..n` range).
- Seat renumbering only happens between hands, so seat indices in a `TableView` /
  `HandResult` are stable for the duration of a hand.
- No missed-blind / dead-button rules: a newly seated player is dealt in the next hand
  regardless of position, without posting extra blinds.
- `join` and queue processing are guarded by a `threading.Lock`, so a host thread that
  accepts new agent connections can call `join` while the dealer is running a hand.

## Multi-agent / separate processes (`remote.py`)

The engine itself is single-threaded and runs in one host process that owns the
`Environment` and `Dealer`; agents in other processes play through proxies:
- `RemoteAgent(conn)` implements the `Agent` protocol: `on_turn` sends the
  (picklable) `TableView` + `LegalActions` over a `multiprocessing.Connection`, blocks
  for a reply message (`Action`), and applies it via `player.fold()/call()/...`;
  `on_hand_end` sends the `HandResult` and receives an optional rebuy amount;
  `on_seated` notifies the agent process that it is now at the table.
- Joining from another process: the agent process sends a `Join(name, stack_bb, seat)`
  message on a new connection; a host-side listener thread calls
  `env.join(..., agent=RemoteAgent(conn))` and replies with the result (accepted /
  queued / error such as duplicate name or full table).
- `agent_loop(conn, agent)` helper runs in the agent process, receiving messages and
  dispatching to a local agent object with the same callback interface.
- Optional per-action timeout: no reply in time → check if legal, else fold.
- Each agent only ever receives its own hole cards, so information is isolated per
  process.

## Tests

- Card, hand, board and deck behaviour is tested in `//entities`; engine tests only
  cover how the dealer uses them (no card dealt twice in a hand,
  same seed gives the same hand).
- `test_evaluator`: category ordering, wheel straight, exactly-2-hole-cards rule
  (e.g., four flush cards on board + one in hand is not a flush), low qualification,
  counterfeited lows, A-2 low vs A-3 low.
- `test_pots`: side pots with multiple all-ins of different sizes, folded players'
  contributions, uncalled bet return, hi/lo split, scoop, quartering, odd-chip rules.
- `test_betting`: small bet on preflop/flop and big bet on turn/river, raise cap of
  4 bets and its removal heads-up, all-in for less than half a bet not reopening action
  while half or more does, heads-up blind/action order.
- `test_dealer`: hand ends early when all but one fold, all-in run-out, chip
  conservation (sum of stacks constant across hands apart from rebuys and joins),
  button moves.
- `test_environment`: `start_round` resets deck/hands/board and posts blinds; rebuy
  callback is called after every hand and rebuy increases stack; default and duplicate
  names.
- `test_join`: join between hands with no seat → player posts BB next hand (also with a
  single seated player → heads-up BB); explicit offset places player at that position
  relative to the button; join from inside an `on_turn` callback is queued — player is
  not dealt, gets no `on_hand_end`, and is dealt in the next hand at the offset from the
  moved button; multiple queued joins seated in FIFO order; invalid stack/offset,
  duplicate name and full table are rejected.
- `test_remote`: a hand played with agents in child processes via `RemoteAgent`; an
  agent process joining mid-hand via a `Join` message.
