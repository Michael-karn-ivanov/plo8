# entities: implementation plan

A pure Python library (standard library only, no third-party dependencies) with
the core value types the Omaha Hi-Lo 8-or-better game engine is built on: `Card`,
`Hand`, `Board` (the shared cards: flop, turn, river) and `Deck`.

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
- `Deck(rng: random.Random | None = None)` creates a full deck in `ALL_CARDS`
  order. A `random.Random` can be injected for reproducible shuffles; this is
  needed for RL experiment seeding and for tests.
- `shuffle()` shuffles in place with the deck's RNG.
- `deal(n=1) -> list[Card]` removes and returns `n` cards from the top. Raises
  `ValueError` if fewer than `n` cards remain.
- `deal_one() -> Card`.
- `deal_hand() -> Hand` deals 4 cards.
- `remove(cards)` takes known cards (dead cards, or fixed hands for simulation)
  out of the deck. Raises `ValueError` if a card is not present.
- `__len__`, `__contains__`, `remaining() -> tuple[Card, ...]`.
- `reset()` restores all 52 cards in canonical order. It does not shuffle.

## Build (buck2)

`entities/BUCK`:
- `python_library(name = "entities", srcs = glob(["entities/**/*.py"]),
  base_module = "entities", visibility = ["PUBLIC"])`.
- `python_test(name = "entities_test", srcs = glob(["tests/*.py"]),
  deps = [":entities"])`.

Other projects (the game engine, the RL code) depend on `//entities:entities`.

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
