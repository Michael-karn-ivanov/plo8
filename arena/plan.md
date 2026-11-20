# arena — implementation plan

Python library (`arena`): baseline bots and the duplicate-match evaluator for
fixed-limit Omaha Hi-Lo 8-or-better, on top of `//rl_env`. It gives the later
`//trainer` fixed opponents and a trustworthy "bot A beats bot B by X big bets per
100 hands, ± CI", and gives the later `//bot_client` bots that can sit at a real
`cli_server` table.

Guiding rules:
- **Accounting is the product.** Results are kept in integer chips, keyed by
  (deal, rotation, seat → lineup position), and converted to big bets only when the
  report is built. Zero sum is checked per hand and for the totals; a violation
  raises (`AccountingError`), it is never "fixed up".
- **No rules here.** Betting, pots, showdown and hand evaluation stay in `//engine`;
  table driving stays in `//rl_env`. RuleBot gets hand strength from
  `engine.evaluator.best_omaha` / `rank_low`; arena only interprets the results.
- **Bots see only what a player sees**: `DecisionBatch.views` (own hole cards, board,
  public state), `legal`, `mask`, `obs`. Nothing in `arena/arena/bots.py` or
  `rules.py` imports `rl_env.critic` or calls a privileged accessor (a test scans the
  imports). Only `match.py` reads every seat's hole cards, for hand histories and
  deal-order checks, after the hand is over.

## Alignment with `uber-plan.md` (Phase 2a/2b)

| uber-plan sketch | here |
|---|---|
| `arena/bots.py`: RandomBot, CallingStation, Maniac | `bots.py` (+ `make_bot`, `BOT_NAMES`) |
| `arena/rules.py`: RuleBot | `rules.py` (`RuleBot`, `RULE_PARAMS`, pure helpers) |
| `arena/agent.py`: `BotAgent` | `agent.py` |
| `arena/match.py`: `play_match(controllers, table_size, n_deals, seed)` | `match.py`: `play_match(lineup, n_deals, seed, ...)`, `MatchResult`, `NameStats`, `combine` |
| `history.py` not in sketch | `history.py`: hand history text from `HandResult` + events |
| results table | `bin/match.py` (`//arena:match`) |

## Layout

```
arena/
  spec.md
  plan.md
  BUCK
  arena/
    __init__.py      # public exports
    bots.py          # RandomBot, CallingStation, Maniac, make_bot, BOT_NAMES, Bot base
    rules.py         # RuleBot, RULE_PARAMS, preflop_score, postflop_class, nut-low tables
    agent.py         # BotAgent: any arena bot as an engine Agent
    match.py         # Entrant, play_match, MatchResult, NameStats, HandRecord, combine
    stats.py         # pure accounting: samples -> bb/100, CI (no engine, no tables)
    history.py       # format_hand(...) -> str
  bin/
    match.py         # command-line evaluator (buck2 binary)
  tests/
    helpers.py       # cards("As 2d ..."), StackedDeck, table_at_decision(...)
    test_bots.py
    test_rules.py
    test_agent.py
    test_match.py
    test_stats.py
    test_history.py
    test_imports.py
  slow/
    test_strength.py # RuleBot beats RandomBot / CallingStation (own test target)
```

## Build (buck2)

`arena/BUCK`:
- `python_library(name = "arena", srcs = glob(["arena/**/*.py"]), base_module = "",
  deps = ["//rl_env:rl_env", "//engine:engine", "//entities:entities"],
  visibility = ["PUBLIC"])`. numpy comes from the repo venv (as in rl_env); no other
  third-party import, and no `torch`, `cli_server`, `cli_client` (checked by
  `test_imports`, which scans every import of `arena/arena/` and `arena/bin/`).
- `python_test(name = "arena_test", srcs = glob(["tests/*.py"]), base_module =
  "arena_tests", deps = [":arena"])` — fast tests (seconds).
- `python_test(name = "arena_slow_test", srcs = glob(["slow/*.py"]), base_module =
  "arena_slow", deps = [":arena"])` — the strength acceptance matches (tens of
  seconds); `ARENA_SLOW_DEALS` lowers the deal count for local runs, default = the
  values below.
- `python_binary(name = "match", main = "bin/match.py", base_module = "arena_bin",
  deps = [":arena"])`.

Run: `buck2 test //arena/...`, `buck2 run //arena:match -- [options]`.

## Units

- The engine reports BB; rl_env tables run at `chips_per_bb = 100`. arena converts
  every hand's net to integer chips once: `chips = round(net_bb * 100)`.
- One big bet = 2 BB = 200 chips. All report figures (bb/100, CI, totals in big bets)
  are big bets; the report header says so ("results in big bets (2 BB) per 100
  hands"). The name `bb/100` follows the spec and limit-poker custom: *big bets*.
- Hand histories show amounts in BB (the engine's unit), stated in their header, with
  the final net also in big bets.

## 1. Bots (`bots.py`, `rules.py`)

All bots implement the rl_env `Controller` protocol (`wants_views`,
`act(batch) -> np.ndarray[int64, (B,)]`) and have a class attribute `name` (`random`,
`calling`, `maniac`, `rule`). They are stateless between calls (RandomBot's only
state is its generator); `act` never stores anything from a batch.

Every bot finishes with `_masked(actions, batch.mask)`: an action whose mask bit is
off is replaced by `CHECK_CALL` (always allowed). For the three simple bots this never
triggers by construction; for RuleBot it turns "fold" into "check" when checking is
free and "raise" into "call" when raising is not allowed, so mask safety does not
depend on each rule being written correctly. Tests still assert that the simple bots
never needed it.

| Bot | `wants_views` | `act` |
|---|---|---|
| `RandomBot(seed=0)` | False | wraps `rl_env.RandomController(seed)`: uniform over the masked actions (fold is masked when checking is free) |
| `CallingStation()` | False | `np.full(B, CHECK_CALL)` |
| `Maniac()` | False | `np.where(mask[:, BET_RAISE], BET_RAISE, CHECK_CALL)` |
| `RuleBot()` | True | per row `decide(view, legal)` (below) |

Factory: `BOT_NAMES = ("random", "calling", "maniac", "rule")`,
`make_bot(name: str, seed: int = 0) -> Controller` (`ValueError` listing the names
for an unknown one; `seed` is used by `random` and ignored by the others). The
command-line tools here and in `//bot_client` (`--bot rule`) use it.

### 1.4 RuleBot (`rules.py`)

`RuleBot.act` reads `batch.views[i]` (`hole_cards`, `board`, `street`, `raises` =
bets made on this street, the big blind counting preflop) and `batch.legal[i]`
(`can_check`, `can_raise`). It never reads other seats' cards (there are none in a
view) or `history`. Pure helpers, unit-tested on their own:
- `preflop_score(hand: Hand) -> int`
- `postflop_class(hand: Hand, board: Board) -> PostflopClass` (fields `high_cat`,
  `has_low`, `nut_low`, `nut_low_draw`, `nut_flush_draw`)
- `decide(view, legal) -> int` (masked action)

All numbers live in one frozen dataclass `RULE_PARAMS` at the top of `rules.py`, so
tuning means editing one block:

```python
@dataclass(frozen=True)
class RuleParams:
    # preflop: base low pair, only the best one counts
    a2: int = 30
    a3: int = 20
    two_three: int = 15
    extra_low: int = 6          # per distinct low rank (A..8) beyond 2, only with a base low pair
    suited_ace: int = 10        # per suit holding an ace and another hole card (max 2 suits)
    suited_other: int = 4       # per other suit with >= 2 hole cards
    aces: int = 10              # a pair of aces
    big_two: int = 4            # exactly 2 distinct big ranks (T..A)
    big_three: int = 10         # 3 or more distinct big ranks
    connected: int = 3          # per adjacent pair of distinct ranks (ace both high and low)
    trips: int = -20            # three of one rank
    quads: int = -40            # four of one rank
    nine: int = -6              # per nine
    middle: int = -3            # per T/J/Q when the hand has no ace and no other card >= T
    # preflop thresholds
    raise_at: int = 35
    call_at: int = 18
    # postflop
    strong_high_cat: int = 3    # evaluator category: 3 = three of a kind, 4 = straight, ...
    two_pair_cat: int = 2
    value_raise_cap: int = 3    # one-sided strong hand: bet/raise while bets this street < 3
RULE_PARAMS = RuleParams()
```

Ranks: `Rank` values 2..14; low value via `Rank.low_value` (ace = 1). Categories come
from the first element of `best_omaha`'s high rank tuple (0 high card … 8 straight
flush, the evaluator's own numbering, as in rl_env's hints).

**Preflop** (`view.street` preflop):
- score = best of {A+2, A+3, 2+3} (0 if none) + `extra_low` × max(0, distinct low
  ranks − 2) when a base pair exists + suited terms + `aces` + big-card term +
  `connected` × adjacent pairs + penalties.
- score ≥ `raise_at`: `BET_RAISE` (re-raises while `can_raise`; otherwise call).
- `call_at` ≤ score < `raise_at`: check if free; call when facing a single bet
  (`view.raises == 1`, i.e. only the big blind); fold when `view.raises >= 2`.
- score < `call_at`: check if free, else fold.
- Worked examples (kept in the tests): A♠A♥2♠3♥ = 30 + 6 + 20 + 10 + 6 = 72 → raise;
  9♣9♦9♥K♠ = −20 − 18 = −38 → fold to a bet.

**Postflop** — `best_omaha(view.hole_cards, view.board)` once per row (exactly two hole
cards and three board cards, the engine's rule), giving `high_cat` and `has_low`. Then
from module tables:
- *Nut low* (no card removal): `NUT_LOW_PAIRS[board_low_ranks]` — for every set of
  3..5 distinct board low ranks (182 sets), the set of hole low-rank pairs `{a, b}`
  whose best low on that board equals the best low any two hole cards could make.
  Built once at import by calling `engine.evaluator.rank_low` on every pair × board
  triple (≈ 50k calls, < 0.2 s). The hand has the nut low iff one of its distinct
  hole low-rank pairs is in the set. Card removal by own hole cards is **not**
  considered; a nut low that may be tied (quartered) still counts as nut.
- *Nut-low draw* (flop and turn only): no nut low now, and some low rank `r` not on the
  board such that the board ranks ∪ {r} (≥ 3 low ranks) has one of the hand's hole low
  pairs in `NUT_LOW_PAIRS`.
- *Nut-flush draw* (flop and turn only): a suit with exactly two board cards and at
  least two hole cards, one of which is the highest rank of that suit not on the board
  (the ace unless the ace is on the board, then the king, ...; own cards not removed).
- These are table lookups over ranks/suits; no hand comparison is re-implemented.

Postflop decision (`n = view.raises`, bets made this street):
1. `scoop = strong_high and nut_low` → `BET_RAISE` whenever allowed, else call.
2. `strong_high` (`high_cat >= 3`) or `nut_low` → `BET_RAISE` while `n <
   value_raise_cap`, then check/call.
3. Flop / turn: `nut_low_draw or nut_flush_draw or has_low or high_cat >= 2` →
   check/call (any amount).
4. River: `has_low or high_cat >= 2` → check if free; call when `n == 1`; fold when
   `n >= 2`.
5. Otherwise check if free, else fold.

Caching: none in the first version (each row is a different table, so a per-batch
cache would rarely hit). If the bench shows RuleBot dominating, cache
`postflop_class` per `(hole cards, board)` inside one `act` call only — never across
calls (spec: stateless).

## 2. `BotAgent` (`agent.py`)

```python
class BotAgent:                                     # engine Agent protocol
    def __init__(self, bot, *, rebuy_to_bb: float | None = None): ...
    def on_turn(self, player, view, legal):
        batch = DecisionBatch.single(view, legal)   # rl_env: obs, mask, views=[view]
        out = self.bot.act(batch)
        a = int((out[0] if isinstance(out, tuple) else out)[0])
        t = to_engine(a, legal)                     # raises on a masked action
        {FOLD: player.fold, CHECK: player.check, CALL: player.call, RAISE: player.raise_}[t]()
    def on_hand_end(self, player, result):
        if self.rebuy_to_bb and player.stack < self.rebuy_to_bb:
            player.rebuy(self.rebuy_to_bb - player.stack)
```
- No `on_hole_cards` / `on_event` methods, so the engine builds no views for them.
- Rebuy is off by default (`rebuy_to_bb=None`); a failing rebuy (e.g. max buy-in)
  is logged and ignored.
- Same decision as inside a `TableBatch`: guaranteed because both paths call the same
  `bot.act` on a batch with the same view, legal actions and mask
  (`DecisionBatch.single` equals the batch row, an rl_env test). arena tests it again
  for every bot.

## 3. Duplicate matches (`match.py`, `stats.py`)

### Inputs
```python
@dataclass(frozen=True)
class Entrant:
    name: str                                 # report name; equal names are pooled
    bot: str | Callable[[int], Controller] | Controller
        # bot name for make_bot, a factory taking a seed, or a ready controller

def play_match(lineup: Sequence[Entrant | str], n_deals: int, seed: int, *,
               table_size: int | None = None,      # must equal len(lineup) if given
               stacks: float | Sequence[float] | StackConfig = 100.0,
               n_histories: int = 20,
               n_tables: int = 256,
               record_hands: bool = False) -> MatchResult
```
- A plain `str` entrant means `Entrant(name=s, bot=s)`.
- `2 <= len(lineup) <= 10`, `n_deals >= 1`; otherwise `ValueError`.
- `stacks`: a number → every seat that many BB; a sequence → per seat; a `StackConfig`
  → drawn per deal with `rl_env.draw_stacks` (the CLI's `--stacks training` is
  `StackConfig()`).
- `n_tables` (tables of the `TableBatch`, capped at `n_deals × n`) is part of the
  inputs: RandomBot's generator is consumed in batch order, so a different table
  count gives a different (equally valid) match. Determinism is promised for equal
  inputs including `n_tables`.

### Seeds
`ss = np.random.SeedSequence(seed)`; `deal_ss, bot_ss, hist_ss = ss.spawn(3)`.
- Deals: `rng = np.random.Generator(np.random.PCG64(deal_ss))`; for deal `d = 0..D−1`,
  in this order: stacks (if `StackConfig`), `button = rng.integers(n)`,
  `deck_seed = rng.integers(0, 2**63 − 1)` → `HandSpec(n, stacks, button, deck_seed)`.
  All deals are drawn up front.
- Bots: controllers are built per distinct `(name, bot)` entrant in lineup order;
  named / factory bots get `seed_i = int(bot_ss.spawn(...)[i].generate_state(1)[0])`
  for the i-th distinct entrant. Ready controller objects are used as given (the
  caller seeds them); the same object at several positions is one controller.
- Histories: `hist_rng.choice(D × n, size=min(n_histories, D × n), replace=False)`,
  sorted → the `(deal, rotation)` jobs whose text is kept.

### Rotation
Rotation `r = 0..n−1` of deal `d` plays `spec_d` with lineup position `p` at seat
`(p + r) % n`, i.e. seat `s` is played by position `(s − r) % n`. Same deck seed, same
button, same per-seat stacks in every rotation, so by the engine's deal-order contract
seat `s` gets the same hole cards and the board is the same as far as it is dealt;
over the `n` rotations every position holds every seat's cards (and stack and
position relative to the button) exactly once. No engine change is needed.

### Playing
- `TableBatch(min(n_tables, D·n), observations=needs_obs)`. `needs_obs` is False when
  every controller is an arena bot (they declare `wants_obs = False`) and True
  otherwise (e.g. the trainer's neural policy); see Required rl_env capabilities.
- Jobs `(d, r)` in order `d`-major. Loop: start the next jobs on `idle_tables()`
  (controller keys per seat from the rotation), `finished = batch.step(controllers)`,
  then for each finished hand, **before its table is restarted**: account it, and if
  its job is a sampled one (or `record_hands`), read every seat's hole cards
  (Required rl_env capabilities) and format/record it. Repeat until all jobs are
  done and `not any_in_hand()`.
- Hand ids map back to jobs through a dict filled at `start_hand`.

### Accounting
Per finished hand: `chips[s] = round(net_bb[s] * 100)`; `sum(chips) != 0` →
`AccountingError` (with deal, rotation and nets). Then
`pos_chips[(s − r) % n, d] += chips[s]` (int64 array `(n, D)`) and
`name_chips[name][d] += chips[s]`. Order of finishing does not matter (integer sums).

Per report name with `k` positions (`stats.summarize(chips_per_deal, k, n,
chips_per_big_bet=200)`):
- `hands = k · n · D` — hands played by that name's seats (a pooled name counts one
  hand per seat it held); `deals = D`.
- `total_chips = Σ chips_per_deal`; `total_bb = total_chips / 200` (big bets).
- `samples[d] = chips_per_deal[d] / (k · n) / 200` — the rotation group's result per
  hand, in big bets; all groups have equal weight `k · n`, so
  `bb_per_100 = 100 · mean(samples) = 100 · total_bb / hands` exactly.
- `ci95 = 100 · 1.96 · std(samples, ddof=1) / √D` (in bb/100; `nan` when `D = 1`).
- After all names: `Σ total_chips == 0`, else `AccountingError`.

### Result
```python
@dataclass(frozen=True)
class NameStats:
    name: str; positions: tuple[int, ...]; hands: int; deals: int
    total_chips: int; total_bb: float; bb_per_100: float; ci95: float
    samples: np.ndarray            # float64 (D,), big bets per hand per rotation group

@dataclass(frozen=True)
class HandRecord:                  # only with record_hands=True
    deal: int; rotation: int; spec: HandSpec
    names: tuple[str, ...]         # per seat
    positions: tuple[int, ...]     # per seat
    chips: tuple[int, ...]         # net per seat
    hole_cards: tuple[str, ...]    # per seat, text form
    board: str

@dataclass(frozen=True)
class MatchResult:
    table_size: int; n_deals: int; seed: int; lineup: tuple[str, ...]
    stats: dict[str, NameStats]    # in first-appearance order of the lineup
    position_chips: np.ndarray     # int64 (n, D): per lineup position, per deal
    histories: tuple[str, ...]     # sample hand histories, job order
    hands: tuple[HandRecord, ...] | None
    wall_time: float; hands_per_second: float
```
`combine(*results) -> MatchResult` concatenates deals of matches with the same lineup
names and table size (different seeds or more deals) and recomputes the stats from
the concatenated chips per deal; histories are concatenated. This is how a caller
"keeps playing until the interval is narrower than the difference it cares about".
Automatic stopping at a target width is not implemented (optional in the spec).

### Sample hand histories (`history.py`)
`format_hand(deal, rotation, spec, names, hole_cards, result) -> str`, from the
engine's `HandResult` and its `history` events, cards and boards printed with their
`str` (entities text form):

```
Deal 17, rotation 3 - 6 players, button seat 2 (amounts in BB)
Seat 0  rule      100.00  As 2s Ah 3h
Seat 1  random    100.00  ...
Blinds: seat 3 random SB 0.50; seat 4 random BB 1.00
Preflop: seat 5 random raises to 2.00; seat 0 rule raises to 3.00; seat 1 random folds; ...
Flop 7h 8d Kc: seat 3 random checks; ...
Turn 2s (7h 8d Kc 2s): ...
River ...
Shows: seat 0 rule As 2s Ah 3h; seat 3 random ...
Returned: seat 4 random 1.00
Pot 1 12.00: high seat 0 rule 6.00; low seat 0 rule 6.00
Net: seat 0 rule +8.00 (+4.00 big bets); seat 1 random -1.00 (-0.50); ...
```
Event mapping: `BlindPosted` → blinds line; `PlayerActed` → per street, action word
from `ActionType` (+ "all-in", + "(timeout)/(leave)" reason if any); `BoardDealt` →
street header; `HandShown`, `BetReturned`; pots from `result.pots` (`high` / `low`
shares, `uncontested` → "wins uncontested"); nets from `result.net`. Seat names in
the engine (`s0`…) are replaced by report names. Hole cards of every seat come from
the privileged read (the evaluator is not a player).

## 4. Command-line evaluator (`bin/match.py`, `//arena:match`)

```
buck2 run //arena:match -- --agent rule --opponents random --players 2,6,9 --deals 2000 --seed 0
```
Options: `--agent NAME` (required), `--opponents A,B,...` (required), `--players
2,6,9`, `--deals 2000`, `--seed 0`, `--histories 0`, `--stacks fixed|training`
(`100.0` or `StackConfig()`), `--tables 256`.
- Lineup per size `n`: position 0 = agent, positions 1..n−1 = opponents in order,
  repeated (`opps[i % len(opps)]`). Report names are the bot names; if the agent's
  name also appears among the opponents it is reported as `NAME*` so the measured bot
  is not pooled with its copies.
- Output: header `results in big bets (2 BB) per 100 hands; 95% CI = 1.96·sd/√deals,
  one sample per rotation group`, then one row per report name per table size:

```
agent    opponents            players  hands   bb/100   95% CI
rule     random,random,...    6        12000   +85.31   ±9.12
random   rule                 6        60000   -17.06   ±1.82
```
  (`agent` column = the row's report name, `opponents` = the other names at the
  table), then per match `wall 7.9 s, 1,519 hands/s`, then `--histories N` histories
  (the first N of the match's samples; `n_histories = max(N, 0)`).

## 5. Speed

Target (spec): 2,000 deals 6-handed (12,000 hands), RuleBot vs 5 RandomBots, well
under a minute on one core.

Expected cost: rl_env without observations runs close to engine speed (~140k
decisions/s at 6p); a 6-handed hand has ~25 decisions, ~5 of them RuleBot's, each one
`best_omaha` (tens of µs) plus a few table lookups; RuleBot also makes the engine
build a `TableView` for its decisions. Estimate: a few seconds per 12,000 hands.

Measured with `buck2 run //arena:match -- --agent rule --opponents random --players
2,6,9 --deals 2000 --seed 0` and `--agent random --opponents random`
(development Mac, one core), filled in after the first implementation:

| Lineup | 2p hands/s | 6p hands/s | 9p hands/s | 6p, 2,000 deals wall |
|---|---|---|---|---|
| rule vs random | – | – | – | – |
| random vs random | – | – | – | – |

If the 6p wall time is not well under a minute: profile (`--profile` is not needed,
use `python -m cProfile` on the binary), then in this order: per-`act` cache of
`postflop_class`; skip observations (if not already); ask rl_env for cheaper views
only if `TableView` building dominates.

## 6. Tests

- `test_imports`: arena imports only stdlib, numpy, `engine`, `entities`, `rl_env`;
  `bots.py` / `rules.py` never import `rl_env.critic` nor mention
  `privileged_hole_cards` / `hole_cards_of_finished`.
- `test_bots`:
  - masks: for every bot, `TableBatch` runs of random hands at every size 2..10 with
    `StackConfig()` stacks (short stacks, all-ins); a wrapper asserts every returned
    action is allowed by its row's mask before `submit` (which re-checks and lets the
    engine accept it);
  - CallingStation only `CHECK_CALL`; Maniac `BET_RAISE` exactly when allowed, else
    `CHECK_CALL`, never `FOLD`; RandomBot: over many decisions every allowed action
    occurs with frequency ≈ 1/allowed (loose bounds), never a masked one, same seed →
    same actions, different seed → different;
  - `make_bot` for every name, unknown name raises; simple bots set
    `wants_views = False` (a counting view hook in the batch sees no views for them).
- `test_rules`: `preflop_score` on hand-written hands (A-A-2-3 double suited raises;
  9-9-9-K rainbow below `call_at`; A-2 beats A-3 beats 2-3; trips/quads penalties;
  middle-card penalty only without ace/big cards); `NUT_LOW_PAIRS` spot checks (board
  2-3-7 → A-4 is the only nut pair; board A-2-3: 4-5; board with < 3 low ranks has
  none; counterfeited A-2 on an A-2-x board); nut-low and nut-flush draw detection;
  `decide` on known positions built through a stacked deck (`helpers.table_at_decision`:
  an rl_env `Table` with `deck=StackedDeck(...)` and a scripted action prefix, then
  `view()` / `legal`): raises A-A-2-3ds preflop, folds 9-9-9-K to a raise, bets the
  nut low and a set on the flop, raises a scoop on the river, calls a nut-flush draw,
  calls one river bet with two pair, folds air to a river bet, checks air when free,
  never folds when checking is free.
- `test_agent`: `BotAgent` for every bot at engine callback-mode tables
  (`Environment(players=[Player(..., agent=BotAgent(bot, rebuy_to_bb=100))])`,
  `env.run(...)`) at 2..10 players, ≥ 300 hands per size, no errors, no player sat
  out, no fold when `legal.can_check` (agent wrapper records); and same decisions:
  during `TableBatch` runs with views, every row's action from the batch equals
  `bot.act(DecisionBatch.single(view, legal))`.
- `test_match`:
  - deal order from the arena side (`record_hands=True`): within each rotation group
    every seat has the same hole cards in all rotations and the board strings agree
    on their common prefix, for random lineups at all sizes;
  - rotation: per group, `positions` per seat form a permutation per rotation and each
    position holds each seat exactly once;
  - mirror: CallingStation, Maniac, RuleBot each against copies of itself (same name,
    and again with distinct names) at 2, 6, 9 players: `position_chips == 0`
    everywhere, every `NameStats` exactly 0;
  - RandomBot vs copies (agent named `random-a`, others `random`) at 2 and 6 players,
    500 deals, seed 0: |bb/100| small and the CI includes 0;
  - zero sum: `Σ total_chips == 0` in every match; an injected bad net (patched
    `FinishedHand`) raises `AccountingError`;
  - determinism: two runs with equal inputs give equal stats, `position_chips`,
    samples and identical history strings; a different seed differs;
  - `combine` of two matches equals recomputing from the concatenated per-deal chips;
  - `n_histories` histories are returned, chosen from the seed.
- `test_stats`: `summarize` on hand-made per-deal chips (e.g. `n = 2`, `k = 1`, chips
  per group `[400, −200, 0, 200]`): samples `[1.0, −0.5, 0, 0.5]` big bets per hand,
  bb/100 = 25.0, CI = 100·1.96·sd/√4 computed by hand; sign and units (200 chips = 1
  big bet); pooled `k > 1`; `D = 1` gives `nan` CI. Plus a small real match with
  `record_hands=True` whose stats the test recomputes from the `HandRecord`s.
- `test_history`: a stacked-deck hand with a showdown, a side pot and a split low
  formats with every seat's name, stack and cards, the board by street, every blind
  and action with amounts, pots with high/low winners, nets; a fold-out hand shows
  "uncontested" and the returned bet.
- `slow/test_strength`: RuleBot vs 5 × RandomBot and RuleBot vs 5 × CallingStation,
  6-handed, 1,000 deals each (6,000 hands), seed 0: RuleBot's bb/100 > 0 and its CI
  excludes 0. If a run fails, tune `RULE_PARAMS`, not the seed.
- CLI smoke test (`test_match`): `bin/match.py` `main(["--agent", "rule",
  "--opponents", "random", "--players", "2,6,9", "--deals", "20"])` prints the header
  and the columns agent / opponents / players / hands / bb/100 / 95% CI with a row per
  name per size.
- Regression: `buck2 test //...` — entities, engine, rl_env, cli_server and
  cli_client tests unchanged and green.

## Required rl_env capabilities

Used as they are: `TableBatch` (`start_hand`, `step`, `pending`/`submit`,
`idle_tables`, `any_in_hand`, `collect_finished`), `FinishedHand` (`spec`,
`controllers`, `net_bb`, `result`), `DecisionBatch` (incl. `single`), `Controller`,
`RandomController`, `FOLD` / `CHECK_CALL` / `BET_RAISE`, `to_engine`, `HandSpec`,
`StackConfig`, `draw_stacks`, `Table(deck=...)` for stacked-deck tests.

Requested in `rl_env/spec.md` (section 5):
- **Batches without observation vectors**: a `TableBatch` whose controllers read no
  observations can be run without building them (today the `observations=False`
  flag exists but is documented as bench-only). arena passes it when every controller
  declares `wants_obs = False`. Until it is part of rl_env's contract, arena keeps
  observations on (slower, same results).
- **Every seat's hole cards of a finished hand for evaluation tools**, readable until
  that table's next hand starts (e.g. `TableBatch.finished_hole_cards(table) ->
  Mapping[int, Hand]`), never passed to a controller. Until then `match.py` reads
  `batch.tables[t].privileged_hole_cards()` at the same moment (same data, but
  documented by rl_env as critic-only).

No engine change is needed: deal order, `best_omaha` / `rank_low`, `HandResult`,
events, callback mode and `Player.rebuy` are used as they are.
