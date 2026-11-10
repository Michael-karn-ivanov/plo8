import unittest

from entities import Board, Card, Street

C = Card.parse


class BoardTest(unittest.TestCase):
    def test_stage_transitions(self):
        b = Board.empty()
        self.assertIs(b.street, Street.PREFLOP)
        self.assertEqual(b.cards, ())
        self.assertEqual(len(b), 0)

        b1 = b.with_flop(C("Ah"), C("7d"), C("2c"))
        self.assertIs(b1.street, Street.FLOP)
        self.assertEqual(b1.cards, (C("Ah"), C("7d"), C("2c")))
        self.assertIs(b.street, Street.PREFLOP)  # original unchanged

        b2 = b1.with_turn(C("9s"))
        self.assertIs(b2.street, Street.TURN)
        self.assertEqual(len(b2), 4)

        b3 = b2.with_river(C("Td"))
        self.assertIs(b3.street, Street.RIVER)
        self.assertEqual(b3.cards, (C("Ah"), C("7d"), C("2c"), C("9s"), C("Td")))

    def test_out_of_order_transitions(self):
        empty = Board.empty()
        with self.assertRaises(ValueError):
            empty.with_turn(C("9s"))
        with self.assertRaises(ValueError):
            empty.with_river(C("9s"))
        flop = empty.with_flop(C("Ah"), C("7d"), C("2c"))
        with self.assertRaises(ValueError):
            flop.with_flop(C("3h"), C("4d"), C("5c"))
        with self.assertRaises(ValueError):
            flop.with_river(C("9s"))
        turn = flop.with_turn(C("9s"))
        with self.assertRaises(ValueError):
            turn.with_turn(C("Td"))
        river = turn.with_river(C("Td"))
        for call in (
            lambda: river.with_flop(C("3h"), C("4d"), C("5c")),
            lambda: river.with_turn(C("3h")),
            lambda: river.with_river(C("3h")),
        ):
            with self.assertRaises(ValueError):
                call()

    def test_constructor_invariants(self):
        with self.assertRaises(ValueError):
            Board(flop=(C("Ah"), C("7d")))
        with self.assertRaises(ValueError):
            Board(turn=C("9s"))
        with self.assertRaises(ValueError):
            Board(flop=(C("Ah"), C("7d"), C("2c")), river=C("9s"))
        with self.assertRaises(ValueError):
            Board(flop=("Ah", "7d", "2c"))
        b = Board(flop=[C("Ah"), C("7d"), C("2c")])
        self.assertIsInstance(b.flop, tuple)

    def test_duplicates_across_streets(self):
        with self.assertRaises(ValueError):
            Board(flop=(C("Ah"), C("Ah"), C("2c")))
        flop = Board.empty().with_flop(C("Ah"), C("7d"), C("2c"))
        with self.assertRaises(ValueError):
            flop.with_turn(C("7d"))
        turn = flop.with_turn(C("9s"))
        with self.assertRaises(ValueError):
            turn.with_river(C("9s"))
        with self.assertRaises(ValueError):
            turn.with_river(C("Ah"))
        with self.assertRaises(ValueError):
            Board.parse("Ah7d2c 9s Ah")

    def test_parse_str_round_trip(self):
        for s in ["", "Ah7d2c", "Ah7d2c 9s", "Ah7d2c 9s Td"]:
            self.assertEqual(str(Board.parse(s)), s)
        self.assertIs(Board.parse("").street, Street.PREFLOP)
        self.assertIs(Board.parse("Ah7d2c 9s Td").street, Street.RIVER)
        self.assertEqual(Board.parse("Ah 7d 2c9sTd"), Board.parse("Ah7d2c 9s Td"))

    def test_parse_invalid_counts(self):
        for s in ["Ah", "Ah7d", "Ah7d2c9sTdJc"]:
            with self.assertRaises(ValueError):
                Board.parse(s)

    def test_container_protocol(self):
        b = Board.parse("Ah7d2c 9s")
        self.assertIn(C("9s"), b)
        self.assertNotIn(C("Td"), b)
        self.assertEqual(list(b), [C("Ah"), C("7d"), C("2c"), C("9s")])

    def test_hashable_value_type(self):
        a = Board.parse("Ah7d2c 9s")
        b = Board.empty().with_flop(C("Ah"), C("7d"), C("2c")).with_turn(C("9s"))
        self.assertEqual(a, b)
        self.assertEqual(len({a, b}), 1)
        with self.assertRaises(Exception):
            a.turn = None


if __name__ == "__main__":
    unittest.main()
