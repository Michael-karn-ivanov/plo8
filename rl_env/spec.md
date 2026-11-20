Python library (`rl_env`), the reinforcement-learning environment for fixed-limit Omaha Hi-Lo 8-or-better.
It sits between plo8/engine and everything that learns or plays with neural networks (the later plo8/arena, plo8/trainer and plo8/bot_client projects). It is the only part of the RL stack that drives the engine: everything above it sees tables, decisions, observation vectors, action masks and results, never engine internals.
Mistakes here do not show up as crashes: a bot trained on a wrong observation or a wrong reward still trains, it just learns the wrong thing. So correctness of what a player is allowed to see, and of how results are counted, matters more than anything else in this project.

Dependencies
* plo8/engine (step mode: start a hand, get each decision, submit the action, read the hand result) and plo8/entities (cards, `Card.index` 0..51).
* numpy, from the repo venv (`requirements-rl.txt`). No other third-party package.
* Tests may also use plo8/cli_server (its protocol module and its recorded end-to-end hands). The library itself must not depend on plo8/cli_server.
* All rules (betting, pots, showdown, hand evaluation) stay in plo8/engine. rl_env never re-implements a rule; if it needs a hand ranking it asks the engine's evaluator.

1. Actions
* A learning player has exactly three actions: fold, check-or-call, bet-or-raise. In limit poker there is no bet size to choose.
* For every decision the environment gives an action mask of these three, derived from the engine's legal actions:
    * check-or-call is always allowed;
    * bet-or-raise is allowed exactly when the engine allows a raise (it is not allowed once the raise cap is reached, when the player cannot put in more than a call, or when nobody else can still act);
    * fold is allowed only when the player faces a bet. Folding when checking is free is never correct, so it is masked out, even though the engine itself would accept it.
* Any action allowed by the mask is turned into an engine action the engine accepts (check-or-call becomes check when checking is free and call otherwise). Submitting an action outside the mask is an error and changes nothing.

2. One table
* A table is one engine environment played in step mode, with 2 to 10 seats. Seats are numbered 0..n-1 clockwise, as in the engine.
* A hand is started with: the table size, each seat's stack in big blinds, the button seat and a deck seed. The same four inputs plus the same sequence of actions always give exactly the same hand (cards, events, result). The deck is a fresh engine deck seeded with the deck seed; because the engine's deal order is a documented contract, the same seed deals the same cards to the same positions relative to the button. This is what later makes duplicate evaluation possible (replaying the same cards with seats rotated).
* Every hand is independent: nothing carries over from the previous hand at the same table (no rebuys, sit-outs, joins, leaves or button movement are needed; stacks, size and button are given anew each hand).
* While a hand is in progress, the table tells: whether a decision is pending, which seat is to act, the action mask, the engine's legal actions and, on request only, the acting seat's table view (the engine's view - only that seat's own hole cards are visible).
* When the hand is over, the table gives each seat's net result for the hand. The reward used for learning is the net result in big bets (one big bet = 2 big blinds; the engine reports amounts in big blinds), with no rake. The net results of all seats sum to zero.
* For the critic only (a value network used during training that may see everything), the table can also give the other seats' hole cards, relative to a given seat. This is the only way hidden information leaves rl_env during a hand, it is clearly separated from the observation, and nothing a playing policy receives may be derived from it.

3. Stacks used for training
* rl_env provides a way to draw starting stacks for a hand from a seeded random source:
    * most hands (by default 80%) start all seats at 100 big blinds (50 big bets), which in limit poker is effectively deep;
    * the rest start each seat with an independent random stack between 5 and 150 big blinds, so that all-ins and side pots appear in training.
* The share and the ranges are parameters; the defaults are the values above.

4. Observation (what a player sees, as a vector)
* For each decision the acting player's situation is turned into a fixed-length vector of 32-bit floats. Its length (`OBS_DIM`) is a constant of the library, and the encoding has a version number (`ENCODER_VERSION`) that changes whenever the meaning or layout of the vector changes, so saved models can tell whether they match.
* The observation is computed only from the engine's table view of the acting player and that player's legal actions - nothing else. This matters twice:
    * it cannot leak hidden information, because the view only contains the acting player's own hole cards;
    * the same function works in training and at a real table: plo8/cli_server already sends the engine's table view and legal actions over its protocol, so a bot connected to the server can build exactly the same vector.
* Seats are described relative to the acting player: slot 0 is the acting player, slot 1 the next seat clockwise, and so on. Seats that do not exist at a smaller table are all zeros. As a result, the same situation seen from different physical seats (same cards, same positions relative to the button, same actions) gives an identical vector.
* Amounts (pot, bets, stacks, amount to call) are expressed in big bets and clipped to a fixed range, so that every value of the vector is finite and between -1 and 2.
* The vector must contain at least this information:
    * the acting player's four hole cards;
    * the board, card by card, with the street each card was dealt on;
    * the current street;
    * the number of players dealt into the hand;
    * the acting player's position relative to the button;
    * the betting state: pot, amount to call, number of bets made this street (with a last bucket meaning "4 or more", since the bet cap is lifted heads-up), the acting player's chips already in the pot, how many players are still in the hand, and the acting player's own stack;
    * for each other seat: whether it is occupied, still in the hand, all-in; its chips in the pot in total and on this street; how many bets it made on each street; whether it was the last aggressor; whether it acts after the acting player on this street; whether it has acted on this street; its stack;
    * the most recent actions of the hand (at least the last 8), each as who acted (relative slot) and what kind of action it was (fold, check-or-call, bet-or-raise);
    * hand-strength hints from the engine's evaluator, from the flop on: the category of the acting player's best high hand, whether they have a qualifying low and how good the best low is; plus, on every street including preflop, how many distinct low ranks (ace to eight) the hole cards hold. Preflop the other hints are zero. The hints are not strictly needed but make early learning much faster.
* The exact layout (order of blocks, one-hot or scalar for each item, clipping ranges) is a design decision for plan.md, documented there block by block with its offset in the vector.
* The critic's extra input (the other seats' hole cards relative to the acting seat, see 2) is a separate, fixed-size array of card indices, with a marker for empty slots or players not dealt in, in the same relative slot order as the observation.

5. Many tables at once
* rl_env runs a batch of tables side by side in one process, so that the decisions of many tables can be sent to a neural network together. Each table in the batch is the table of section 2.
* Each seat of a hand is assigned a controller - the thing that decides for it (for example the current neural policy, an older copy of it, or a hand-written bot). Which controller sits where is chosen by the caller when the hand starts, per seat.
* A controller receives a batch of decisions at once and returns one action (fold, check-or-call, bet-or-raise) per decision. With each batch it receives the observation vectors, the action masks, and also the engine's table views and legal actions, so both neural policies (which read vectors and masks) and rule-based bots (which read views and legal actions) can implement the same interface. Views are built only for controllers that ask for them; a controller that only needs vectors and masks must not pay for views.
* A batch whose controllers read no observation vectors (for example rule-based bots in an evaluation match, plo8/arena) can be run without building them; its controllers then receive masks, legal actions and, on request, views as usual, and everything else (actions, results, determinism) is unchanged.
* The batch reports all pending decisions grouped by controller (at most one pending decision per table, since one player acts at a time), accepts the chosen actions, advances each of those tables to its next decision or to the end of its hand, and reports finished hands with each seat's result. The caller decides when and with what to start the next hand on a finished table.
* For evaluation tools only (plo8/arena's hand histories and deal checks), every seat's hole cards of a finished hand can be read from the batch, until the caller starts that table's next hand. Like the critic input this is hidden information: it is clearly separated from what controllers receive and is never passed to a controller.
* The batch also makes it possible to record, for every decision, everything training needs: table, hand, seat, controller, observation, mask, chosen action, the critic's extra input, and, once the hand is over, that seat's reward. Matching each decision to the final result of its hand must be unambiguous.
* A batch run is deterministic: the same seeds (for decks, stacks, seating and any random choices made by rl_env itself) and the same controller decisions give identical decisions, actions and results.

6. Speed
* The environment must be fast enough to generate training data by self-play on many CPU cores. What matters is the cost a rollout worker pays per decision: the engine plus building the observation and the mask, without any neural network.
* Target: at least 5,000 decisions per second on one CPU core, with the observation and mask built for every decision, running 512 tables in one batch with a controller that picks random legal actions. Measured at 2, 6 and 9 players.
* The project includes a benchmark tool (a buck2 binary) that runs exactly this workload and reports decisions per second and hands per second at 2, 6 and 9 players, next to the engine-only number from the engine's own benchmark, so the share of the observation building is visible. Its numbers are recorded in plan.md, before and after each speed-up.
* If the target is not met, the remaining gap and the chosen next step are recorded in plan.md. Speed-ups first go into rl_env (for example not re-evaluating hand-strength hints that cannot have changed since the player's previous decision on the same street, and not re-reading the whole hand history at every decision); rules or evaluator speed-ups belong in plo8/engine as an engine spec change.

7. Acceptance
* Leakage: for many random situations, changing the other players' hole cards while keeping everything public the same leaves the observation byte-for-byte identical. This is the most important test of the project.
* Rotation: the same situation dealt and played from different physical seats (same button-relative positions) gives identical observations.
* Shape and range: every observation has length `OBS_DIM`, and every value is finite and between -1 and 2, over many random hands at all table sizes and stacks, including short stacks, all-ins and side pots.
* Action mask: matches the engine's legal actions as described in section 1 at every decision; fold is masked exactly when checking is free; every action allowed by the mask is accepted by the engine; an action outside the mask is rejected and changes nothing.
* Hand-strength hints: known hands give the expected high-hand categories and lows (including hands where a low is counterfeited or only completes on the river, and the exactly-two-hole-cards rule).
* Server path: for every decision in a few thousand random hands, the observation built from the engine's view and legal actions equals the observation built from the same view and legal actions after encoding and decoding them with the plo8/cli_server protocol.
* Recorded hands: every recorded end-to-end hand of plo8/cli_server (`cli_server/e2e_test_sets`, 26 hands, including side pots, quartered lows and odd chips) replayed through an rl_env table, with the recorded cards and actions, gives the recorded final stacks and pot splits.
* Determinism: the same seeds and controller decisions give an identical batch run (decisions, actions, results), and the same deck seed deals the same cards relative to the button whatever the stacks.
* Fuzz: 100,000 random hands through the batch at random table sizes (2-10) and random stacks run without errors, and in every hand the seats' net results sum to zero.
* Training records: for random batch runs, every recorded decision is matched to exactly one finished hand, and its reward is that seat's net result in big bets.
* Batches without observations: the same seeds and controller decisions give the same actions and results with and without observation vectors.
* Finished hole cards: for a finished hand, every seat's hole cards read from the batch equal the hands the engine dealt (the shown hands at showdown agree), and controllers never receive them.
* The benchmark meets the target, or the gap and the next step are recorded in plan.md.
* All existing tests of plo8/entities, plo8/engine, plo8/cli_server and plo8/cli_client still pass.
