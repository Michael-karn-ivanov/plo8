import random
import unittest

from entities import ALL_CARDS, Card, Deck, Hand


class DeckTest(unittest.TestCase):
    def test_fresh_deck(self):
        d = Deck()
        self.assertEqual(len(d), 52)
        self.assertEqual(len(set(d.remaining())), 52)
        self.assertEqual(d.remaining(), ALL_CARDS)

    def test_dealing(self):
        d = Deck(random.Random(1))
        d.shuffle()
        seen = set()
        first = d.deal(3)
        self.assertEqual(len(first), 3)
        self.assertEqual(len(d), 49)
        seen.update(first)
        c = d.deal_one()
        self.assertIsInstance(c, Card)
        seen.add(c)
        h = d.deal_hand()
        self.assertIsInstance(h, Hand)
        seen.update(h)
        self.assertEqual(len(d), 44)
        seen.update(d.deal(len(d)))
        self.assertEqual(len(seen), 52)
        self.assertEqual(len(d), 0)

    def test_deal_from_top(self):
        d = Deck()
        self.assertEqual(d.deal(2), [Card.parse("2c"), Card.parse("2d")])
        self.assertNotIn(Card.parse("2c"), d)

    def test_overdeal(self):
        d = Deck()
        d.deal(50)
        with self.assertRaises(ValueError):
            d.deal(3)
        self.assertEqual(len(d), 2)  # unchanged on failure
        with self.assertRaises(ValueError):
            d.deal_hand()
        d.deal(2)
        with self.assertRaises(ValueError):
            d.deal_one()
        with self.assertRaises(ValueError):
            Deck().deal(-1)

    def test_seeded_reproducible(self):
        a, b = Deck(random.Random(42)), Deck(random.Random(42))
        a.shuffle()
        b.shuffle()
        self.assertEqual(a.remaining(), b.remaining())
        self.assertNotEqual(a.remaining(), ALL_CARDS)
        self.assertEqual(sorted(a.remaining()), list(ALL_CARDS))

    def test_remove(self):
        d = Deck()
        dead = [Card.parse("As"), Card.parse("Kd")]
        d.remove(dead)
        self.assertEqual(len(d), 50)
        for c in dead:
            self.assertNotIn(c, d)
        d.remove(Hand.parse("2c3d4h5s"))
        self.assertEqual(len(d), 46)
        d.remove(Card.parse("Qh"))
        self.assertEqual(len(d), 45)

    def test_remove_missing_is_atomic(self):
        d = Deck()
        d.remove([Card.parse("As")])
        with self.assertRaises(ValueError):
            d.remove([Card.parse("Kd"), Card.parse("As")])
        self.assertIn(Card.parse("Kd"), d)
        self.assertEqual(len(d), 51)
        with self.assertRaises(ValueError):
            d.remove([Card.parse("Kd"), Card.parse("Kd")])
        self.assertEqual(len(d), 51)

    def test_reset(self):
        d = Deck(random.Random(7))
        d.shuffle()
        d.deal(10)
        d.remove([d.remaining()[0]])
        d.reset()
        self.assertEqual(len(d), 52)
        self.assertEqual(d.remaining(), ALL_CARDS)


if __name__ == "__main__":
    unittest.main()
