import unittest

from entities import Card, Hand


def cards(s):
    return [Card.parse(t) for t in s.split()]


class HandTest(unittest.TestCase):
    def test_construct_from_iterable(self):
        h = Hand(cards("As Kd 7h 2c"))
        self.assertIsInstance(h.cards, tuple)
        self.assertEqual(len(h), 4)
        h2 = Hand(c for c in cards("As Kd 7h 2c"))
        self.assertEqual(h, h2)

    def test_order_preserved(self):
        h = Hand.parse("2cAsKd7h")
        self.assertEqual(str(h), "2cAsKd7h")
        self.assertEqual([str(c) for c in sorted(h)], ["2c", "7h", "Kd", "As"])

    def test_wrong_count(self):
        for s in ["As Kd 7h", "As Kd 7h 2c 3c", ""]:
            with self.assertRaises(ValueError):
                Hand(cards(s))

    def test_duplicates(self):
        with self.assertRaises(ValueError):
            Hand(cards("As Kd As 2c"))
        with self.assertRaises(ValueError):
            Hand.parse("AsKdAs2c")

    def test_non_cards_rejected(self):
        with self.assertRaises(ValueError):
            Hand(["As", "Kd", "7h", "2c"])
        with self.assertRaises(ValueError):
            Hand("AsKd7h2c")

    def test_parse_str_round_trip(self):
        for s in ["AsKd7h2c", "2c3d4h5s", "TcTdThTs"]:
            self.assertEqual(str(Hand.parse(s)), s)
        self.assertEqual(Hand.parse("As Kd 7h 2c"), Hand.parse("AsKd7h2c"))
        self.assertEqual(str(Hand.parse("10sKd7h2c")), "TsKd7h2c")

    def test_container_protocol(self):
        h = Hand.parse("AsKd7h2c")
        self.assertIn(Card.parse("Kd"), h)
        self.assertNotIn(Card.parse("Kh"), h)
        self.assertEqual(list(h), cards("As Kd 7h 2c"))

    def test_hashable_value_type(self):
        a = Hand.parse("AsKd7h2c")
        b = Hand.parse("AsKd7h2c")
        self.assertEqual(a, b)
        self.assertEqual(len({a, b}), 1)
        with self.assertRaises(Exception):
            a.cards = ()


if __name__ == "__main__":
    unittest.main()
