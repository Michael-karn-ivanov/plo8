Python library, defining major types to implement game engine: card (from deuce to ace, 4 suits), hand (4 cards), shared cards (consisting of flop, turn, river), deck.
Deck can also be stacked: chosen cards can be put on top of the deck, in a given order, so that they are dealt first. This is used to replay recorded hands (e.g. end-to-end tests of plo8/cli_server).
