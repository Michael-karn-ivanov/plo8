import random
import unittest

from entities import ALL_CARDS, Card, Deck, Hand


def cards(text):
    return [Card.parse(t) for t in text.split()]


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


class StackingTest(unittest.TestCase):
    def shuffled(self, seed=5):
        d = Deck(random.Random(seed))
        d.shuffle()
        return d

    def test_stacked_cards_dealt_first(self):
        d = self.shuffled()
        before = d.remaining()
        stack = cards("As Kd 7h")
        d.put_on_top(stack)
        self.assertEqual(len(d), 52)
        after = d.remaining()
        self.assertEqual(len(set(after)), 52)
        # The rest keeps its relative order.
        self.assertEqual(after[3:], tuple(c for c in before if c not in stack))
        self.assertEqual(d.deal(3), stack)
        self.assertEqual(len(d), 49)

    def test_single_card(self):
        d = self.shuffled()
        c = Card.parse("Qh")
        d.put_on_top(c)
        self.assertEqual(len(d), 52)
        self.assertEqual(d.deal_one(), c)

    def test_empty_is_noop(self):
        d = self.shuffled()
        before = d.remaining()
        d.put_on_top([])
        self.assertEqual(d.remaining(), before)

    def test_stack_with_generator(self):
        d = self.shuffled()
        stack = cards("2c 3d 4h")
        d.put_on_top(c for c in stack)
        self.assertEqual(d.deal(3), stack)

    def test_dealt_card_rejected_atomically(self):
        d = self.shuffled()
        dealt = d.deal_one()
        before = d.remaining()
        other = before[10]
        with self.assertRaises(ValueError) as ctx:
            d.put_on_top([other, dealt])
        self.assertIn(str(dealt), str(ctx.exception))
        self.assertEqual(d.remaining(), before)

    def test_removed_card_rejected_atomically(self):
        d = self.shuffled()
        removed = Card.parse("As")
        d.remove(removed)
        before = d.remaining()
        with self.assertRaises(ValueError) as ctx:
            d.put_on_top([Card.parse("Kd"), removed])
        self.assertIn("As", str(ctx.exception))
        self.assertEqual(d.remaining(), before)

    def test_duplicate_rejected_atomically(self):
        d = self.shuffled()
        before = d.remaining()
        with self.assertRaises(ValueError):
            d.put_on_top(cards("Kd As Kd"))
        self.assertEqual(d.remaining(), before)

    def test_non_card_rejected_atomically(self):
        d = self.shuffled()
        before = d.remaining()
        with self.assertRaises(TypeError):
            d.put_on_top([Card.parse("Kd"), "As"])
        with self.assertRaises(TypeError):
            d.put_on_top([Card.parse("Kd"), 51])
        self.assertEqual(d.remaining(), before)

    def test_second_stack_goes_above_first(self):
        d = self.shuffled()
        d.put_on_top(cards("As Kd 7h"))
        d.put_on_top(cards("2c Kd"))
        after = d.remaining()
        self.assertEqual(len(after), 52)
        self.assertEqual(len(set(after)), 52)
        self.assertEqual(list(after[:4]), cards("2c Kd As 7h"))

    def test_shuffle_discards_stack(self):
        d = self.shuffled()
        stack = cards("As Kd 7h 2c 3d 4h 5s 6c")
        d.put_on_top(stack)
        d.shuffle()
        self.assertNotEqual(list(d.remaining()[:8]), stack)
        self.assertEqual(len(d), 52)

    def test_reset_discards_stack(self):
        d = self.shuffled()
        d.put_on_top(cards("As Kd 7h"))
        d.reset()
        self.assertEqual(d.remaining(), ALL_CARDS)

    def test_whole_hand_stack(self):
        # 4 players x 4 hole cards + 5 board cards.
        d = self.shuffled()
        stack = list(random.Random(99).sample(ALL_CARDS, 21))
        d.put_on_top(stack)
        hands = [d.deal_hand() for _ in range(4)]
        board = d.deal(5)
        self.assertEqual([c for h in hands for c in h] + board, stack)
        self.assertEqual(len(d), 31)

    def test_subclass_restacks_after_each_shuffle(self):
        hands = [cards("As Kd 7h 2c 9s"), cards("3d 4h 5s 6c Td")]

        class StackedDeck(Deck):
            def __init__(self, stacks, rng=None):
                super().__init__(rng)
                self.stacks = list(stacks)
                self.hand_no = 0

            def shuffle(self):
                super().shuffle()
                self.put_on_top(self.stacks[self.hand_no % len(self.stacks)])
                self.hand_no += 1

        d = StackedDeck(hands, random.Random(1))
        for _ in range(2):
            for stack in hands:
                d.reset()
                d.shuffle()
                self.assertEqual(len(d), 52)
                self.assertEqual(d.deal(len(stack)), stack)


if __name__ == "__main__":
    unittest.main()
