import pickle
import random
import unittest

from entities import ALL_CARDS, Board, Card, Deck, Hand, Rank, Street, Suit

PROTOCOLS = range(pickle.HIGHEST_PROTOCOL + 1)


def roundtrip(obj, protocol):
    return pickle.loads(pickle.dumps(obj, protocol=protocol))


class PickleTest(unittest.TestCase):
    def assertRoundTrips(self, obj):
        for proto in PROTOCOLS:
            with self.subTest(obj=obj, protocol=proto):
                copy = roundtrip(obj, proto)
                self.assertIs(type(copy), type(obj))
                self.assertEqual(copy, obj)
                self.assertEqual(hash(copy), hash(obj))
                self.assertEqual(str(copy), str(obj))

    def test_enums(self):
        for r in Rank:
            for proto in PROTOCOLS:
                self.assertIs(roundtrip(r, proto), r)
        for s in Suit:
            for proto in PROTOCOLS:
                self.assertIs(roundtrip(s, proto), s)
        for st in Street:
            for proto in PROTOCOLS:
                self.assertIs(roundtrip(st, proto), st)

    def test_all_cards(self):
        for c in ALL_CARDS:
            self.assertRoundTrips(c)

    def test_card_is_interned(self):
        for proto in PROTOCOLS:
            self.assertIs(roundtrip(Card.parse("As"), proto), Card.parse("As"))

    def test_card_still_orders_and_indexes(self):
        a, k = roundtrip(Card.parse("As"), 2), roundtrip(Card.parse("Kd"), 2)
        self.assertLess(k, a)
        self.assertEqual(a.index, Card.parse("As").index)

    def test_caller_built_card(self):
        c = Card(Rank.SEVEN, Suit.HEARTS)
        for proto in PROTOCOLS:
            copy = roundtrip(c, proto)
            self.assertIs(copy, ALL_CARDS[c.index])
            self.assertEqual(copy, c)
            self.assertEqual(hash(copy), hash(c))
            self.assertEqual(copy.index, c.index)

    def test_hand(self):
        self.assertRoundTrips(Hand.parse("AsKd7h2c"))

    def test_hand_preserves_order(self):
        h = Hand.parse("2c7hKdAs")
        self.assertEqual(roundtrip(h, pickle.HIGHEST_PROTOCOL).cards, h.cards)

    def test_board_every_street(self):
        for text in ("", "Ah7d2c", "Ah7d2c 9s", "Ah7d2c 9s Td"):
            b = Board.parse(text)
            self.assertRoundTrips(b)
            self.assertIs(roundtrip(b, pickle.HIGHEST_PROTOCOL).street, b.street)

    def test_containers_as_keys(self):
        h, b = Hand.parse("AsKd7h2c"), Board.parse("Ah7d2c 9s")
        table = {h: 1, b: 2, Card.parse("Qh"): 3}
        copy = roundtrip(table, pickle.HIGHEST_PROTOCOL)
        self.assertEqual(copy[h], 1)
        self.assertEqual(copy[b], 2)
        self.assertEqual(copy[Card.parse("Qh")], 3)

    def test_partly_dealt_deck(self):
        d = Deck(random.Random(3))
        d.shuffle()
        d.deal(7)
        d.remove([d.remaining()[5]])
        for proto in PROTOCOLS:
            with self.subTest(protocol=proto):
                copy = roundtrip(d, proto)
                self.assertIsInstance(copy, Deck)
                self.assertEqual(len(copy), len(d))
                self.assertEqual(copy.remaining(), d.remaining())
                self.assertEqual(copy.deal_one(), d.remaining()[0])

    def test_stacked_deck_keeps_order(self):
        d = Deck(random.Random(4))
        d.shuffle()
        d.deal(2)
        stack = [c for c in (Card.parse(t) for t in ("As", "Kd", "7h", "2c")) if c in d]
        d.put_on_top(stack)
        for proto in PROTOCOLS:
            with self.subTest(protocol=proto):
                copy = roundtrip(d, proto)
                self.assertEqual(copy.remaining(), d.remaining())
                self.assertEqual(copy.deal(len(stack)), stack)

    def test_deck_copy_is_independent_and_keeps_rng(self):
        d = Deck(random.Random(11))
        d.shuffle()
        d.deal(4)
        copy = roundtrip(d, pickle.HIGHEST_PROTOCOL)
        # Same RNG state: the next shuffle gives the same order on both.
        d.shuffle()
        copy.shuffle()
        self.assertEqual(copy.remaining(), d.remaining())
        # Mutating the copy does not touch the original.
        copy.deal(10)
        self.assertEqual(len(d), 48)
        self.assertEqual(len(copy), 38)

    def test_deck_state_is_top_first(self):
        d = Deck(random.Random(6))
        d.shuffle()
        d.deal(3)
        rng, top_first = d.__getstate__()
        self.assertIsInstance(rng, random.Random)
        self.assertIsInstance(top_first, list)
        self.assertEqual(top_first, list(d.remaining()))
        self.assertEqual(Deck().__getstate__()[1], list(ALL_CARDS))

    def test_deck_setstate_takes_top_first_list(self):
        order = random.Random(9).sample(ALL_CARDS, 20)
        d = Deck.__new__(Deck)
        d.__setstate__((random.Random(1), list(order)))
        self.assertEqual(list(d.remaining()), order)
        self.assertEqual(len(d), 20)
        self.assertEqual(d.deal_one(), order[0])
        self.assertEqual(d.deal(2), order[1:3])
        # The state list is copied, not aliased.
        state_list = list(order)
        e = Deck.__new__(Deck)
        e.__setstate__((random.Random(1), state_list))
        e.deal_one()
        self.assertEqual(state_list, order)


if __name__ == "__main__":
    unittest.main()
