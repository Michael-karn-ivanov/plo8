Python library, providing Omaha Hi-Lo game engine.
It exports several classes:
1. Player
    * player has his stack in front of him - amount of big blinds
    * player can call, fold, check, raise during round
    * player can rebuy - add amount of big blinds to his stack 
    * player can show his card to dealer during showdown
2. Dealer
    * dealer has a deck - instance of class Deck. 
    * at the start of the round dealer deals each player 4 cards
    * Then, after rounds of bets, opens flop
    * Then, after round of bets, opens turn
    * Then, after round of bets, opens river
    * Then, after round of bets, showdowns comes
    * at the end of the round, if there is a showdown - it makes players to show their cards and determines winner(s)
    * if at some moment, during betting, only one player left - it is declared as winner
    * it also handles all-ins and side-pots
    * after end of the round, dealer moves button (i.e. shifting Small Blind and Big Blind)
3. Environment
    * collection of players, dealer and starting position of the button
    * environment exposes method to start new round - i.e. deck is reset, player hands are reset, flop/turn/river emptied, corresponding players place big and small blinds.
    * environment can be created with a deck supplied by the caller (e.g. a stacked deck) and with a chip unit (number of chips per big blind), so a recorded hand can be replayed exactly - same cards, same odd-chip splits (used by end-to-end tests of plo8/cli_server). For that the order in which the dealer takes cards from the deck is fixed and documented: hole cards one at a time to each player starting left of the button, four rounds, then flop, turn and river, no burn cards.
    * a starting stack given in big blinds is kept exactly whatever the chip unit, both for players the environment is created with and for players who join: a stack that is a whole number of the table's chips is seated with exactly those chips (plo8/rl_env replays recorded hands with players created directly, not through joins).

Library is designed this way that there will be several agents - like separate processes. 
They will share environment and dealer and will "play" at the table: i.e. each player can see their hand and each player will be responsible for its own actions.
I.e. there should be some callback in Player, that will get called within rounds during betting for an active player and its managing agent will take action through Player (call, fold, etc).
As soon as the cards are dealt (before any betting), each player's managing agent is notified with that player's own hand (and only with it), so it does not have to wait for its first turn to see the cards.
Every public thing that happens at the table (blinds posted, each call / fold / check / raise, board cards opened, hands shown, pots awarded) is reported to every seated player, including the player who did it, together with the stack of the player concerned after it.
After the end of round (no matter whether it was showdown or not) - players stacks are updated - (if they won - if they lost - their bets are deducted from stack during the round) - and there is another callback, so that each managing agent can make decision whether to rebuy or not.
If player has 0 chips left and they did not rebuy at the end of the round - they are not dealt cards and don't take part in the next round.
They still get notification when next and further rounds end so that they can rebuy (and if they do - they return to game).
Players can explicitly leave the game, emptying their seat.

Also, environment has a method for a new player to join the game. They select a seat (index from button, if not passed - they join as big blind) and stack (amount of blinds) they join with. Also let's all the players may have name (nick), which they set when join. 
If player while round is in progress, they don't participate in game but are waiting in the queue for round to complete and get cards dealt starting new round.

Step mode (driving a hand one decision at a time)

Besides the callback mode above, the environment can play a hand in step mode, so that one process can run many tables side by side and decide for all of them in batches (reinforcement learning).
* In step mode the environment does not call the acting player's turn callback. Instead it stops at each decision and hands it to the caller: which player is to act and that player's legal actions. On request (and only then) it also gives that player's table view - the same view the turn callback would have received, showing only that player's own hole cards.
* The caller answers by submitting that player's action. The environment applies it and runs on to the next decision, or to the end of the hand if no decision is left.
* Starting a hand in step mode does everything starting a round does today (deck reset and shuffle, blinds, deal) and stops at the first decision. If no decision is possible at all (for example everyone left in the hand is all-in after the blinds), the hand runs to the end straight away and the caller is told there is nothing to decide.
* When the hand ends, the caller can read the hand result, and all end-of-hand processing has been done exactly as in callback mode: stacks updated, end-of-hand notifications and rebuy window, leave requests and join queue processed, button moved.
* An illegal action submitted in step mode is rejected with an error and leaves the hand exactly as it was, so the caller can submit a legal action instead. The engine never folds or sits out a player for this in step mode, and there is no turn timeout, because the caller is in control.
* Starting a hand while another hand is still in progress, or submitting an action when no decision is pending, is an error.
* Everything else stays as in callback mode: other callbacks a player's agent has (hole cards, table events, end of hand) are still called if present, and the hand's events are recorded the same way.
* Without asking for a view, the step-mode caller can also read (plo8/rl_env builds its observations from these, so it does not pay for a view at every decision):
    * the public events of the current hand so far - the same events the view's history holds - read-only and without a copy being made at each decision, so a caller can process only the events added since its last look;
    * the acting player's own hole cards;
    * every participant's hole cards. This is meant only for training tools that are allowed to see everything (the critic of plo8/rl_env); it is a separate accessor, clearly apart from the decision, the view and anything an agent receives.
* Callback mode and step mode are interchangeable: for the same players, stacks, button, deck and sequence of actions, both produce identical events, identical hand results and identical final stacks, and the legal actions offered at each decision are identical. Callback mode keeps working exactly as today, so plo8/cli_server, plo8/cli_client and their tests and end-to-end cases are unaffected.
* Step mode is deterministic: the same deck (same seed or same stacked cards), seating and actions always give the same events and results.
* Table views are built only when someone needs them - when the step-mode caller asks for one, or when an agent actually has the callback that receives it - never just to be thrown away.

Speed

The engine must be fast enough to generate training data by self-play.
* Target: at least 2,000 hands per second on one CPU core at a 6-player table, playing random legal actions in step mode, engine only (no neural network). Measured baseline before this change on the development Mac: about 4,000 hands/s heads-up, 1,230 at 6 players, 800 at 9 players, about 30,000 decisions/s; hand evaluation and building table views took roughly a third of the time each.
* The project includes a benchmark tool (a buck2 binary, standard library only) that plays random legal actions on many tables (default 512) in step mode in one process and reports hands per second and decisions per second at 2, 6 and 9 players. Its numbers before and after each speed-up are recorded in plan.md.
* The benchmark's workload is also available as a library function (same tables, same random play, same report), so that the benchmark of plo8/rl_env can print the engine-only number from the engine's own code next to its own.
* Hand evaluation (best high hand, best qualifying low, best Omaha hand from 2 hole cards + 3 board cards) may be reimplemented for speed, but its results must stay exactly the same as the current implementation, checked against it on a large number of random hands.
* Speed-ups must not change any rule, event, result or public behaviour described above.
* If the target is not reached with pure Python, the remaining gap and the chosen next step (for example a native evaluator behind the same functions) are written down in plan.md.

Acceptance for step mode and speed
* Callback mode and step mode give identical events, legal actions, results and stacks for many random seeds, table sizes (2-10 players), stacks (including short stacks, all-ins and side pots) and action sequences.
* An illegal step-mode action raises and leaves the state unchanged; a hand with no possible decision completes immediately with a full result.
* In step mode, at every decision, the events read without a view equal the view's history, the acting player's hole cards read without a view equal the view's, and reading them builds no view.
* A starting stack that is a whole number of chips is seated exactly, for players created with the environment and for joiners, at chip units other than the default.
* Fuzz test: 100,000 random step-mode hands at random table sizes and stacks run without errors, and in every hand the players' net results sum to zero (there is no rake).
* All existing tests of plo8/entities, plo8/engine, plo8/cli_server and plo8/cli_client still pass, and so do all end-to-end cases of plo8/cli_server.
* The benchmark meets the target, or the gap and the next step are recorded in plan.md.
