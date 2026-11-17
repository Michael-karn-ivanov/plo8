# entities: implementation plan

A pure Python library (standard library only, no third-party dependencies) with
the core value types the Omaha Hi-Lo 8-or-better game engine is built on: `Card`,
`Hand`, `Board` (the shared cards: flop, turn, river) and `Deck`. The deck can be
stacked (chosen cards put on top, in a given order) so recorded hands can be
replayed.

Hand evaluation (hi/lo ranking), betting, pot logic and game flow are out of
scope. They belong to the game engine, which consumes these types.

## Layout

```
entities/
  BUCK
  spec.md
  plan.md
  entities/
    __init__.py      # re-exports Rank, Suit, Card, Hand, Board, Deck
    card.py          # Rank, Suit, Card
    hand.py          # Hand
    board.py         # Board
    deck.py          # Deck
  tests/
    test_card.py
    test_hand.py
    test_board.py
    test_deck.py
```

## Design principles

- **Immutable value types.** `Card`, `Hand` and `Board` are frozen
  (`@dataclass(frozen=True, slots=True)`), hashable, and compare by value. This
  lets them be used as dict keys and set members, and shared safely between
  engine and RL code.
- **Validate at construction.** Invalid input (bad rank or suit, wrong number of
  cards, duplicate cards) raises `ValueError` from the constructor or
  `__post_init__`. A constructed object is always valid.
- **Compact integer encoding.** Every card has a stable index in `0..51` for fast
  lookups and for building RL feature vectors.
- **Text round-trip.** Every type parses from and prints to a short, standard
  string form (`"As"`, `"AsKd7h2c"`, `"Ah7d2c 9s Td"`).

## card.py

### `Rank(IntEnum)`
- Members `TWO = 2` … `NINE = 9`, `TEN = 10`, `JACK = 11`, `QUEEN = 12`,
  `KING = 13`, `ACE = 14`. Integer order is high-hand order.
- `symbol` property: `"2"`…`"9"`, `"T"`, `"J"`, `"Q"`, `"K"`, `"A"`.
- `Rank.from_symbol(s)` accepts the symbols above (case-insensitive, plus `"10"`
  as an alias for `T`). Raises `ValueError` otherwise.
- `low_value` property: the rank's value for low hands. Ace is `1`, every other
  rank is its integer value. Used by the engine for 8-or-better qualification
  (`low_value <= 8`).

### `Suit(Enum)`
- Members `CLUBS`, `DIAMONDS`, `HEARTS`, `SPADES`, with symbols `c`, `d`, `h`,
  `s`. Suits have no poker ordering. They get a fixed index (`0..3` in the order
  above) only to keep the card encoding deterministic.
- `Suit.from_symbol(s)` (case-insensitive). Raises `ValueError` otherwise.

### `Card`
- Fields: `rank: Rank`, `suit: Suit`.
- `Card.parse("As")` / `Card.from_str`: two-character rank+suit (or `"10s"`).
- `__str__` returns `"As"`. `__repr__` returns `Card('As')`.
- `index` property: `(rank - 2) * 4 + suit_index`, in `0..51`.
  `Card.from_index(i)` is the inverse.
- Ordering (`order=True` on the `(rank, suit_index)` key): sorts by rank, then
  suit. This is only for deterministic display and sorting, not a poker
  comparison.
- Module-level `ALL_CARDS: tuple[Card, ...]` holds all 52 cards in index order.
  Cards are interned through it where practical.

## hand.py

### `Hand`
A player's private hole cards: exactly 4 distinct cards.
- Field `cards: tuple[Card, Card, Card, Card]`. The constructor accepts any
  iterable of 4 `Card`s and stores them as a tuple. Input order is preserved
  (`sorted()` is available to callers).
- Raises `ValueError` if the count is not 4 or any card repeats.
- `Hand.parse("AsKd7h2c")` (whitespace between cards is allowed).
- `__str__` returns the concatenated cards. Supports `__iter__`, `__len__` and
  `__contains__`.

## board.py

### `Board`
The shared community cards, revealed in three stages.
- Fields: `flop: tuple[Card, Card, Card] | None`, `turn: Card | None`,
  `river: Card | None`.
- Invariants checked in `__post_init__`:
  - the flop, if present, has exactly 3 cards
  - a turn requires a flop, and a river requires a turn
  - all cards are distinct
- Stage helpers that return new `Board` instances (immutability):
  - `Board.empty()` for preflop
  - `board.with_flop(c1, c2, c3)`
  - `board.with_turn(c)`
  - `board.with_river(c)`

  Each raises `ValueError` if it is called at the wrong stage.
- `cards` property: a flat tuple of revealed cards (0, 3, 4 or 5 of them).
- `street` property: `Street` enum (`PREFLOP`, `FLOP`, `TURN`, `RIVER`), defined
  in `board.py` and exported.
- `Board.parse("Ah7d2c 9s Td")` takes up to 5 cards, assigned to
  flop/turn/river in order. `"Ah7d2c"` and `""` are also valid. `__str__`
  produces the same format.
- Supports `__iter__`, `__len__` and `__contains__` over the revealed cards.

## deck.py

### `Deck`
A mutable 52-card deck. It is the only mutable type in the library.
Internally it is a `list[Card]` whose front (index 0) is the top of the deck:
cards are dealt from the front.
- `Deck(rng: random.Random | None = None)` creates a full deck in `ALL_CARDS`
  order. A `random.Random` can be injected for reproducible shuffles; this is
  needed for RL experiment seeding and for tests.
- `shuffle()` shuffles in place with the deck's RNG.
- `deal(n=1) -> list[Card]` removes and returns `n` cards from the top. Raises
  `ValueError` if fewer than `n` cards remain.
- `deal_one() -> Card`.
- `deal_hand() -> Hand` deals 4 cards.
- `remove(cards)` takes known cards (dead cards, or fixed hands for simulation)
  out of the deck. Accepts a single `Card` or an iterable. Raises `ValueError` if
  a card is not present or is listed twice. All-or-nothing: on error the deck is
  unchanged.
- `put_on_top(cards)` stacks the deck: moves the given cards, which must all
  still be in the deck, to the top in the given order, so the first of them is
  dealt first. See Stacking.
- `__len__`, `__contains__`, `remaining() -> tuple[Card, ...]`. `remaining()`
  lists the cards in deal order (top first).
- `reset()` restores all 52 cards in canonical order. It does not shuffle.
- Pickling: `__getstate__` / `__setstate__` carry the RNG and the card list (the
  class uses `__slots__`), so a pickled deck keeps its order, including a
  stacked top, and shuffles like the original would.

### Stacking
Used to replay recorded hands, e.g. the end-to-end tests of `//cli_server`,
whose `StackedDeck` subclass shuffles and then stacks the recorded cards on top.
- `put_on_top(cards: Card | Iterable[Card]) -> None`. A single `Card` is treated
  as a one-card sequence, as in `remove`.
- Validation, before anything changes (all-or-nothing, like `remove`): every
  item is a `Card` (`TypeError` otherwise), no card is listed twice, and every
  card is still in the deck (`ValueError` otherwise, naming the card). A card
  that was already dealt or removed cannot be stacked.
- Effect: `self._cards = stacked + [c for c in self._cards if c not in
  stacked_set]`. The stacked cards come first, in the given order; the other
  cards keep their relative order. An empty input leaves the deck unchanged.
- Calling it again stacks the new cards above the current top (cards listed
  again are moved, not duplicated), so the last call wins for ordering.
- The stack is just the deck's current order: `deal` consumes it from the top,
  and `shuffle()` or `reset()` discards it. Callers that reshuffle per hand
  (as the engine does: `reset()` and `shuffle()` once per hand) stack again
  after shuffling.
- No new state is added, so `len`, `remaining()`, `__contains__` and pickling
  need no changes.

## Changes requested by engine

The engine (`//engine`) reuses `Card`, `Rank`, `Suit`, `Hand`, `Board`, `Street` and
`Deck` from this library instead of defining its own. It needs no new types or
methods, only one guaranteed property:
- **Pickling** (already works, now a requirement): every type must survive
  `pickle` round-trips with equality and hashing preserved, because the engine sends
  hands, boards and cards to agent processes over `multiprocessing` connections.

## Changes requested by cli_server

`//cli_server`'s end-to-end test mode replays recorded hands with a `Deck`
subclass passed to the engine. It needs `Deck.put_on_top(cards)` (see Stacking),
because the card list is private. Subclasses must be able to override
`shuffle()` and call `super().shuffle()` followed by `self.put_on_top(...)`.

## Build (buck2)

`entities/BUCK`:
- `python_library(name = "entities", srcs = glob(["entities/**/*.py"]),
  base_module = "entities", visibility = ["PUBLIC"])`.
- `python_test(name = "entities_test", srcs = glob(["tests/*.py"]),
  deps = [":entities"])`.

Other projects (the game engine, the RL code, the e2e mode of `cli_server`)
depend on `//entities:entities`.

## Testing

Tests use `unittest` or `pytest`, whichever the repo's buck2 python toolchain
supports, and run with `buck2 test //entities:entities_test`.
- **Card:** parse/str round-trip for all 52 cards, `index`/`from_index`
  bijection, invalid inputs rejected, `low_value` of the ace is 1, cards hash and
  compare as equal.
- **Hand:** exactly 4 cards required, duplicates rejected, parse/str round-trip.
- **Board:** stage transitions and the errors for out-of-order transitions,
  duplicate detection across streets, `street` and `cards` at each stage,
  parse/str round-trip.
- **Deck:** a fresh deck has 52 unique cards; dealing reduces `len` and never
  repeats a card; overdealing raises; a seeded RNG gives the same order every
  time; `remove` and `reset` behave correctly.
- **Stacking:**
  - after `shuffle()` then `put_on_top([c1, c2, c3])`, `deal(3)` returns exactly
    `[c1, c2, c3]`; `len` is unchanged by stacking; the rest of the deck keeps its
    relative order and has no duplicates;
  - a single `Card` argument works; an empty list is a no-op;
  - stacking a card that was dealt or removed, a card listed twice, or a non-`Card`
    raises, and the deck is unchanged afterwards (same `remaining()`);
  - a second `put_on_top` puts its cards above the first stack and moves a card
    listed in both instead of duplicating it;
  - `shuffle()` and `reset()` discard the stack;
  - a stack big enough for a whole hand (e.g. 4 players × 4 hole cards + 5 board
    cards) deals in the given order;
  - a subclass overriding `shuffle()` as `super().shuffle(); self.put_on_top(...)`
    deals the stacked cards first after each `reset()` + `shuffle()`.
- **Pickling:** `Card`, `Hand`, `Board`, `Street` and a partly dealt `Deck` round-trip
  through `pickle` and compare equal (and hash equal) to the originals; a stacked
  deck keeps its order through `pickle`.
