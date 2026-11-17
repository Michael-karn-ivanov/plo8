Python library, defining major types to implement game engine: card (from deuce to ace, 4 suits), hand (4 cards), shared cards (consisting of flop, turn, river), deck.
Deck can also be stacked: chosen cards can be put on top of the deck, in a given order, so that they are dealt first. This is used to replay recorded hands (e.g. end-to-end tests of plo8/cli_server).

The game engine (plo8/engine) generates training data by self-play and needs the hot operations to be cheap:
* a card's compact index (0..51) is stored on the card when it is created, so reading it costs no more than reading an attribute - including for cards created by callers (e.g. for a stacked deck), not only the predefined ones;
* dealing one card from the top of the deck takes constant time.
Behaviour (values, equality, hashing, ordering, errors, pickling) stays exactly as it is.
