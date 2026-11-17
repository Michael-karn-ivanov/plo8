import copy
import dataclasses
import inspect
import pickle
import unittest

from entities import ALL_CARDS, Card, Rank, Suit
from entities.card import parse_cards


class RankTest(unittest.TestCase):
    def test_order_and_values(self):
        self.assertEqual(int(Rank.TWO), 2)
        self.assertEqual(int(Rank.ACE), 14)
        self.assertLess(Rank.KING, Rank.ACE)
        self.assertEqual(len(Rank), 13)

    def test_symbols_round_trip(self):
        for r in Rank:
            self.assertIs(Rank.from_symbol(r.symbol), r)
            self.assertIs(Rank.from_symbol(r.symbol.lower()), r)
        self.assertEqual(Rank.TEN.symbol, "T")
        self.assertIs(Rank.from_symbol("10"), Rank.TEN)

    def test_invalid_symbol(self):
        for bad in ["", "1", "11", "X", "AA", None, 5]:
            with self.assertRaises(ValueError):
                Rank.from_symbol(bad)

    def test_low_value(self):
        self.assertEqual(Rank.ACE.low_value, 1)
        self.assertEqual(Rank.EIGHT.low_value, 8)
        self.assertEqual(Rank.KING.low_value, 13)
        qualifying = {r for r in Rank if r.low_value <= 8}
        self.assertEqual(
            qualifying,
            {Rank.ACE, Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN, Rank.EIGHT},
        )


class SuitTest(unittest.TestCase):
    def test_symbols_and_index(self):
        self.assertEqual([s.symbol for s in Suit], ["c", "d", "h", "s"])
        self.assertEqual([s.index for s in Suit], [0, 1, 2, 3])
        for s in Suit:
            self.assertIs(Suit.from_symbol(s.symbol), s)
            self.assertIs(Suit.from_symbol(s.symbol.upper()), s)

    def test_invalid_symbol(self):
        for bad in ["", "x", "cc", None]:
            with self.assertRaises(ValueError):
                Suit.from_symbol(bad)


class CardTest(unittest.TestCase):
    def test_parse_str_round_trip_all_cards(self):
        self.assertEqual(len(ALL_CARDS), 52)
        self.assertEqual(len(set(ALL_CARDS)), 52)
        for c in ALL_CARDS:
            self.assertEqual(Card.parse(str(c)), c)
            self.assertIs(Card.parse(str(c)), c)  # interned

    def test_str_repr(self):
        c = Card(Rank.ACE, Suit.SPADES)
        self.assertEqual(str(c), "As")
        self.assertEqual(repr(c), "Card('As')")

    def test_parse_variants(self):
        self.assertEqual(Card.parse("10s"), Card(Rank.TEN, Suit.SPADES))
        self.assertEqual(Card.parse("td"), Card(Rank.TEN, Suit.DIAMONDS))
        self.assertEqual(Card.from_str("Kh"), Card(Rank.KING, Suit.HEARTS))

    def test_parse_invalid(self):
        for bad in ["", "A", "Ax", "1s", "Asd", "XX", "11s", None]:
            with self.assertRaises(ValueError):
                Card.parse(bad)

    def test_invalid_construction(self):
        with self.assertRaises(ValueError):
            Card(1, Suit.CLUBS)
        with self.assertRaises(ValueError):
            Card("A", Suit.CLUBS)
        with self.assertRaises(ValueError):
            Card(Rank.ACE, "s")
        # plain ints are coerced to Rank
        self.assertIs(Card(14, Suit.SPADES).rank, Rank.ACE)

    def test_invalid_construction_messages(self):
        cases = [
            ((1, Suit.CLUBS), "invalid rank: 1"),
            ((True, Suit.CLUBS), "invalid rank: True"),
            (("A", Suit.CLUBS), "invalid rank: 'A'"),
            ((Rank.ACE, "s"), "invalid suit: 's'"),
        ]
        for args, message in cases:
            with self.subTest(args=args):
                with self.assertRaises(ValueError) as ctx:
                    Card(*args)
                self.assertEqual(str(ctx.exception), message)

    def test_index_bijection(self):
        for i in range(52):
            c = Card.from_index(i)
            self.assertEqual(c.index, i)
            self.assertIs(ALL_CARDS[i], c)
        self.assertEqual(Card.parse("2c").index, 0)
        self.assertEqual(Card.parse("As").index, 51)
        for bad in [-1, 52, 1.0, True, "3"]:
            with self.assertRaises(ValueError) as ctx:
                Card.from_index(bad)
            self.assertEqual(
                str(ctx.exception), f"card index must be an int in 0..51, got {bad!r}"
            )

    def test_equality_and_hash(self):
        a = Card(Rank.ACE, Suit.SPADES)
        b = Card.parse("As")
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(len({a, b}), 1)
        self.assertNotEqual(a, Card.parse("Ah"))

    def test_immutable(self):
        c = Card.parse("As")
        with self.assertRaises(Exception):
            c.rank = Rank.TWO

    def test_ordering(self):
        cards = [Card.parse(s) for s in ["As", "2d", "2c", "Kh"]]
        self.assertEqual([str(c) for c in sorted(cards)], ["2c", "2d", "Kh", "As"])
        self.assertEqual(list(ALL_CARDS), sorted(ALL_CARDS))

    def test_parse_cards(self):
        self.assertEqual([str(c) for c in parse_cards("AsKd 7h10c")], ["As", "Kd", "7h", "Tc"])
        self.assertEqual(parse_cards(""), ())
        with self.assertRaises(ValueError):
            parse_cards("AsK")


def expected_index(rank, suit):
    return (int(rank) - 2) * 4 + suit.index


class StoredIndexTest(unittest.TestCase):
    def test_caller_built_cards_have_index(self):
        for rank in Rank:
            for suit in Suit:
                with self.subTest(rank=rank, suit=suit):
                    interned = ALL_CARDS[expected_index(rank, suit)]
                    c = Card(rank, suit)
                    self.assertIsNot(c, interned)
                    self.assertIs(type(c.index), int)
                    self.assertEqual(c.index, expected_index(rank, suit))
                    self.assertEqual(c.index, interned.index)

    def test_int_rank_card_has_index(self):
        for r in range(2, 15):
            for suit in Suit:
                c = Card(r, suit)
                self.assertIs(c.rank, Rank(r))
                self.assertIs(type(c.index), int)
                self.assertEqual(c.index, (r - 2) * 4 + suit.index)

    def test_replace_recomputes_index(self):
        c = Card.parse("As")
        d = dataclasses.replace(c, suit=Suit.CLUBS)
        self.assertEqual(d, Card.parse("Ac"))
        self.assertEqual(d.index, Card.parse("Ac").index)
        e = dataclasses.replace(c, rank=Rank.TWO)
        self.assertEqual(e.index, Card.parse("2s").index)

    def test_index_is_slot_not_property(self):
        self.assertIn("index", Card.__slots__)
        self.assertNotIsInstance(inspect.getattr_static(Card, "index"), property)
        c = Card(Rank.ACE, Suit.SPADES)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            c.index = 3
        with self.assertRaises(dataclasses.FrozenInstanceError):
            c.suit = Suit.CLUBS
        self.assertEqual(c.index, 51)

    def test_index_not_in_compare_or_repr(self):
        fields = {f.name: f for f in dataclasses.fields(Card)}
        self.assertFalse(fields["index"].compare)
        self.assertFalse(fields["index"].repr)
        self.assertFalse(fields["index"].init)
        self.assertEqual([n for n, f in fields.items() if f.compare], ["_key"])

    def test_behaviour_matches_interned_twin(self):
        built = [Card(c.rank, c.suit) for c in ALL_CARDS]
        for b, c in zip(built, ALL_CARDS):
            self.assertEqual(b, c)
            self.assertEqual(hash(b), hash(c))
            # The pre-change hash: dataclass hash over the single compare field.
            self.assertEqual(hash(b), hash(((int(c.rank), c.suit.index),)))
            self.assertEqual(repr(b), repr(c))
        reversed_built = built[::-1]
        self.assertEqual(sorted(reversed_built), list(ALL_CARDS))
        self.assertEqual(sorted(ALL_CARDS), list(ALL_CARDS))
        self.assertEqual(
            [c.index for c in sorted(reversed_built)], list(range(52))
        )
        self.assertEqual(repr(Card(Rank.ACE, Suit.SPADES)), "Card('As')")

    def test_pickle_and_copy_of_caller_built_card(self):
        c = Card(Rank.KING, Suit.DIAMONDS)
        interned = Card.parse("Kd")
        self.assertIsNot(c, interned)
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            back = pickle.loads(pickle.dumps(c, protocol=proto))
            self.assertIs(back, interned)
            self.assertEqual(back.index, c.index)
        self.assertIs(copy.copy(c), interned)
        self.assertIs(copy.deepcopy(c), interned)


if __name__ == "__main__":
    unittest.main()
