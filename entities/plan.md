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
  lookups and for building RL feature vectors. It is computed once, when the card
  is constructed, and stored on the card, so reading it is a plain slot read.
- **Cheap hot path.** The engine generates training data by self-play, so the
  operations it calls per card are cheap: `card.index` is an attribute read and
  `Deck.deal_one()` is O(1). Everything else (validation, `shuffle`, `remove`,
  `put_on_top`) may stay O(52).
- **Text round-trip.** Every type parses from and prints to a short, standard
  string form (`"As"`, `"AsKd7h2c"`, `"Ah7d2c 9s Td"`).
- **Speed-ups do not change behaviour.** Values, equality, hashing, ordering, error
  types and messages, and pickling are identical to the straightforward
  implementation; the tests below pin this.

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
- `@dataclass(frozen=True, slots=True, order=True)` with fields:
  - `rank: Rank` and `suit: Suit`: the constructor arguments, `compare=False`;
  - `_key: tuple[int, int]`: `(int(rank), suit_index)`, `init=False`,
    `repr=False`, the only field with `compare=True`. Equality, ordering and the
    dataclass-generated `__hash__` (`hash((self._key,))`) use it, exactly as before;
  - `index: int`: `(rank - 2) * 4 + suit_index`, in `0..51`. `init=False`,
    `repr=False`, `compare=False`, so it takes no part in equality, ordering or
    hashing (hash values stay what they were).
- `__post_init__` validates and normalises `rank` (an `int` is converted to `Rank`,
  `bool` and other types are rejected), validates `suit`, then sets `_key` and
  `index` with `object.__setattr__`. Because this runs for every construction, the
  index is stored for any card, not only for the interned `ALL_CARDS` instances:
  `Card(Rank.ACE, Suit.SPADES)` built by a caller (e.g. for a stacked deck) and
  `dataclasses.replace(card, ...)` both get it. There is no `index` property any
  more; the slot replaces it (same name, same value, same type).
- `Card.parse("As")` / `Card.from_str`: two-character rank+suit (or `"10s"`).
  Returns the interned `ALL_CARDS` instance.
- `__str__` returns `"As"`. `__repr__` returns `Card('As')`.
- `Card.from_index(i)` is the inverse of `index` and returns `ALL_CARDS[i]`.
  Raises `ValueError` for anything that is not an `int` in `0..51` (`bool`
  included).
- Ordering (on `_key`): sorts by rank, then suit. This is only for deterministic
  display and sorting, not a poker comparison. (It coincides with `index` order.)
- Pickling: `__reduce__` returns `(_card_from_index, (self.index,))`, so a card
  pickles as its index and unpickles to the interned card, which has `index` set.
  `copy.copy` / `copy.deepcopy` go through the same path.
- Module-level `ALL_CARDS: tuple[Card, ...]` holds all 52 cards in index order.
  Cards are interned through it where practical (`parse`, `from_index`,
  unpickling). Code must not rely on identity (`is`, `id(card)`), since callers
  may build equal, non-interned cards.

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

Internal representation: `__slots__ = ("_rng", "_pile")`. `_pile` is a
`list[Card]` stored **bottom first**: the top of the deck (the next card to be
dealt) is `_pile[-1]`. Dealing one card is then `list.pop()`, which is O(1);
dealing from the front of a list would shift every remaining card. (The attribute
is renamed from the old top-first `_cards` so that any code reaching into the
private list fails loudly instead of silently reading it in the wrong order.)
Module constant `_REVERSED_ALL_CARDS = ALL_CARDS[::-1]` is the bottom-first
canonical order.

All public methods keep their top-first semantics; only the storage is reversed.
- `Deck(rng: random.Random | None = None)` creates a full deck in `ALL_CARDS`
  order (top = `ALL_CARDS[0]`): `_pile = list(_REVERSED_ALL_CARDS)`. A
  `random.Random` can be injected for reproducible shuffles; this is needed for RL
  experiment seeding and for tests.
- `shuffle()` shuffles in place with the deck's RNG and gives **exactly the same
  deal order as before** for the same RNG state: it shuffles the top-first list and
  stores it reversed (`cards = _pile[::-1]; rng.shuffle(cards); cards.reverse();
  _pile = cards`). Same RNG calls, same permutation of the top-first order; the two
  O(52) reversals are negligible next to the shuffle and happen once per hand.
- `deal(n=1) -> list[Card]` removes and returns `n` cards from the top, top card
  first. Validation is unchanged: `ValueError` if `n` is not a non-negative `int`
  (`bool` rejected) or if fewer than `n` cards remain, with the same messages
  (`"cannot deal {n} cards, only {len} remain"`). Implementation: `n == 0` returns
  `[]` (special-cased, because `_pile[-0:]` would be the whole list); otherwise
  `dealt = _pile[-n:]; del _pile[-n:]; dealt.reverse()`.
- `deal_one() -> Card`: `return self._pile.pop()`; on an empty deck the
  `IndexError` is turned into the same `ValueError("cannot deal 1 cards, only 0
  remain")` that `deal(1)` raised. It no longer goes through `deal()`, so it does
  no argument validation or slicing. (Subclasses that want to observe dealing must
  override `deal_one` itself; the engine's recording-deck tests already do.)
- `deal_hand() -> Hand` deals 4 cards (`Hand(self.deal(4))`).
- `remove(cards)` takes known cards (dead cards, or fixed hands for simulation)
  out of the deck. Accepts a single `Card` or an iterable. Raises `ValueError` if
  a card is not present or is listed twice. All-or-nothing: on error the deck is
  unchanged. Filtering `_pile` keeps the other cards' relative order, whatever the
  storage direction.
- `put_on_top(cards)` stacks the deck: moves the given cards, which must all
  still be in the deck, to the top in the given order, so the first of them is
  dealt first. See Stacking.
- `__len__`, `__contains__` (on `_pile`), `remaining() -> tuple[Card, ...]`.
  `remaining()` lists the cards in deal order (top first):
  `tuple(reversed(_pile))`.
- `reset()` restores all 52 cards in canonical order (`_pile =
  list(_REVERSED_ALL_CARDS)`). It does not shuffle.
- Pickling: `__getstate__` returns `(rng, top_first_list)` (`_pile[::-1]`) and
  `__setstate__` reverses it back, so the pickled state has the same format and
  meaning as before (a top-first card list). A pickled deck keeps its order,
  including a stacked top, and shuffles like the original would.

### Stacking
Used to replay recorded hands, e.g. the end-to-end tests of `//cli_server`,
whose `StackedDeck` subclass shuffles and then stacks the recorded cards on top.
- `put_on_top(cards: Card | Iterable[Card]) -> None`. A single `Card` is treated
  as a one-card sequence, as in `remove`.
- Validation, before anything changes (all-or-nothing, like `remove`): every
  item is a `Card` (`TypeError` otherwise), no card is listed twice, and every
  card is still in the deck (`ValueError` otherwise, naming the card). A card
  that was already dealt or removed cannot be stacked. Cards are matched by value,
  so caller-built (non-interned) cards work.
- Effect, in top-first terms: the stacked cards come first, in the given order;
  the other cards keep their relative order. With bottom-first storage:
  `rest = [c for c in _pile if c not in stacked_set]; rest.extend(reversed(stacked));
  _pile = rest`. An empty input leaves the deck unchanged.
- The stacked cards are stored as given (not replaced by the interned instances);
  they already carry `index`, so the engine reads it at the same cost.
- Calling it again stacks the new cards above the current top (cards listed
  again are moved, not duplicated), so the last call wins for ordering.
- The stack is just the deck's current order: `deal` / `deal_one` consume it from
  the top, and `shuffle()` or `reset()` discards it. Callers that reshuffle per
  hand (as the engine does: `reset()` and `shuffle()` once per hand) stack again
  after shuffling.
- No extra state, so `len`, `remaining()`, `__contains__` and pickling need
  nothing special.

## Changes requested by engine

The engine (`//engine`) reuses `Card`, `Rank`, `Suit`, `Hand`, `Board`, `Street` and
`Deck` from this library instead of defining its own. It needs no new types or
methods, only guaranteed properties:
- **Pickling**: every type must survive `pickle` round-trips with equality and
  hashing preserved, because the engine sends hands, boards and cards to agent
  processes over `multiprocessing` connections.
- **Stored card index**: `card.index` is a slot read for every `Card`, including
  caller-built ones (see `Card`). The engine's evaluator tables are indexed by it,
  and it never keys on `id(card)`.
- **O(1) `deal_one()`**: the dealer deals every card with `deal_one()` (see
  `Deck`).
These are measured by the engine's benchmark (`buck2 run //engine:bench`, see
`engine/plan.md`, Speed), not here.

## Changes requested by cli_server

`//cli_server`'s end-to-end test mode replays recorded hands with a `Deck`
subclass passed to the engine. It needs `Deck.put_on_top(cards)` (see Stacking),
because the card list is private. Subclasses must be able to override
`shuffle()` and call `super().shuffle()` followed by `self.put_on_top(...)`.
Subclasses use only the public methods, so the change of internal storage does
not affect them.

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
  bijection, invalid inputs rejected (same exception types and messages), `low_value`
  of the ace is 1, cards hash and compare as equal.
- **Stored index:**
  - for every rank and suit, a freshly built `Card(rank, suit)` (checked to be
    `is not` the interned card) has `index == (rank - 2) * 4 + suit_index`, as a
    plain `int`, equal to the interned card's index; the same for a card built
    with an `int` rank and for `dataclasses.replace(card, suit=...)`;
  - `index` is a slot, not a property (`"index" in Card.__slots__`,
    `not isinstance(inspect.getattr_static(Card, "index"), property)`), and
    assigning it raises `FrozenInstanceError` like the other fields;
  - behaviour unchanged: a caller-built card equals, hashes equal to (`hash(a) ==
    hash(b)`) and sorts like its interned twin; `hash(card) == hash(((int(rank),
    suit_index),))` (the pre-change hash); `sorted(ALL_CARDS)` is index order;
    `repr` is still `Card('As')`; pickling a caller-built card returns the
    interned card with the same index.
- **Hand:** exactly 4 cards required, duplicates rejected, parse/str round-trip.
- **Board:** stage transitions and the errors for out-of-order transitions,
  duplicate detection across streets, `street` and `cards` at each stage,
  parse/str round-trip.
- **Deck:** a fresh deck has 52 unique cards and `remaining() == ALL_CARDS`;
  dealing reduces `len` and never repeats a card; overdealing raises; a seeded RNG
  gives the same order every time; `remove` and `reset` behave correctly.
- **Deal path:**
  - golden order: for `Random(0)` and `Random(12345)`, `shuffle()` then
    `remaining()` equals a card list recorded from the pre-change implementation
    (kept in the test as a string), so seeded experiments and e2e fillers deal
    exactly as before;
  - `deal_one()` × 52 on one deck gives the same cards as `deal(52)` on an equally
    seeded deck, and `remaining()` before dealing; `deal(n)` returns cards top
    first; `deal(0)` returns `[]` and leaves the deck unchanged (also on a full
    deck);
  - `deal_one()` on an empty deck raises `ValueError` with the message `deal(1)`
    gives; `deal(-1)`, `deal(True)`, `deal(1.0)` raise as before;
  - `deal_one()` does not call `deal()` (a subclass overriding `deal` to raise is
    still dealt from by `deal_one`). Constant time is not timed here (a 52-card
    deck gives no measurable difference to assert on); it is covered by the
    implementation being `list.pop()` and by the engine benchmark.
- **Stacking:**
  - after `shuffle()` then `put_on_top([c1, c2, c3])`, `deal(3)` returns exactly
    `[c1, c2, c3]` (and so do three `deal_one()` calls); `len` is unchanged by
    stacking; the rest of the deck keeps its relative order and has no duplicates;
  - stacking caller-built (non-interned) cards works, and the dealt cards have the
    right `index`;
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
  deck keeps its order through `pickle`; a pickled deck and the original shuffle to
  the same order afterwards; `Deck.__getstate__()[1]` is the top-first list
  (`list(remaining())`), and `__setstate__` with such a list restores that deal
  order, so the state format is the one used before the storage change.
