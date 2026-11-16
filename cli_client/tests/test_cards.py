import unittest

from entities import Board, Card, Hand, Suit

from cli_client.cards import (
    back_segments,
    card_segments,
    card_text,
    describe_hand,
    describe_low,
    fmt_amount,
    segments_text,
    suit_color,
    suit_style,
)


class CardTextTest(unittest.TestCase):
    def test_unicode_and_ascii(self):
        self.assertEqual(card_text(Card.parse("Ah")), "A♥")
        self.assertEqual(card_text(Card.parse("Td")), "T♦")
        self.assertEqual(card_text(Card.parse("2s")), "2♠")
        self.assertEqual(card_text(Card.parse("9c")), "9♣")
        self.assertEqual(card_text(Card.parse("Ah"), ascii=True), "Ah")
        self.assertEqual(card_text(Card.parse("Kc"), ascii=True), "Kc")

    def test_four_colour_deck(self):
        self.assertEqual(suit_style(Suit.HEARTS), "hearts")
        self.assertEqual(suit_color(Suit.HEARTS), "red")
        self.assertEqual(suit_color(Suit.DIAMONDS), "blue")
        self.assertEqual(suit_color(Suit.SPADES), "yellow")
        self.assertEqual(suit_color(Suit.CLUBS), "green")

    def test_segments(self):
        segs = card_segments(Hand.parse("AhKd7s2c"))
        self.assertEqual(segments_text(segs), "[A♥][K♦][7♠][2♣]")
        self.assertEqual([s for _, s in segs], ["hearts", "diamonds", "spades", "clubs"])
        self.assertEqual(segments_text(card_segments(Hand.parse("AhKd7s2c"), brackets=False, sep=" ")), "A♥ K♦ 7♠ 2♣")
        self.assertEqual(segments_text(back_segments(2)), "[▒▒][▒▒]")
        self.assertEqual(segments_text(back_segments(1, ascii=True)), "[##]")
        self.assertEqual({s for _, s in back_segments(3)}, {"card_back"})

    def test_amounts(self):
        self.assertEqual(fmt_amount(2.0), "2")
        self.assertEqual(fmt_amount(1.5), "1.5")
        self.assertEqual(fmt_amount(0.25), "0.25")
        self.assertEqual(fmt_amount(100), "100")
        self.assertEqual(fmt_amount(0.1 + 0.2), "0.3")


class DescribeHandTest(unittest.TestCase):
    def test_flush_no_low(self):
        self.assertEqual(
            describe_hand(Hand.parse("AhKh2c3d"), Board.parse("2h7h9hJcQd")),
            "Flush, A high; no low",
        )

    def test_pair_with_low(self):
        self.assertEqual(
            describe_hand(Hand.parse("Ah2dKcQs"), Board.parse("3c5d7hKsJd")),
            "Pair of Ks; 7-5-3-2-A low",
        )

    def test_exactly_two_hole_cards(self):
        # Four hearts on the board but only one in the hand: no flush.
        text = describe_hand(Hand.parse("AhKsQd2c"), Board.parse("2h5h8h9hKc"))
        self.assertNotIn("Flush", text)
        self.assertEqual(text, "Two pair, Ks and 2s; no low")

    def test_full_house_and_straight(self):
        self.assertTrue(
            describe_hand(Hand.parse("KcKd7s2c"), Board.parse("Kh7d7c")).startswith("Full house, Ks full of 7s")
        )
        self.assertTrue(
            describe_hand(Hand.parse("9c8dAsAd"), Board.parse("7h6d5cKs")).startswith("Straight, 9 high")
        )

    def test_low_text_and_preflop(self):
        self.assertEqual(describe_low((7, 5, 4, 2, 1)), "7-5-4-2-A low")
        self.assertEqual(describe_low(None), "no low")
        self.assertEqual(describe_hand(Hand.parse("AhKh2c3d"), Board()), "")


if __name__ == "__main__":
    unittest.main()
