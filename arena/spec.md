Python library (`arena`): baseline bots and the duplicate-match evaluator for fixed-limit Omaha Hi-Lo 8-or-better.
It sits on top of plo8/rl_env. Its purpose is to provide fixed opponents and a trustworthy way to measure "bot A beats bot B by X per 100 hands", with a confidence interval. This is needed before any training (the later plo8/trainer project): without it, learning cannot be told apart from noise. The later plo8/trainer runs its periodic evaluations with this evaluator, and the later plo8/bot_client plays these bots at the real table (plo8/cli_server).
Mistakes here do not crash anything: a wrong seat assignment, a wrong sign or a wrong unit in the results gives plausible-looking numbers that are simply wrong. So the correctness of the accounting matters more than speed or features.

Dependencies
* plo8/rl_env (tables, `TableBatch`, the `Controller` protocol, `DecisionBatch` including `DecisionBatch.single`, the action constants, mask and `to_engine`, `RandomController`, `HandSpec`), plo8/engine (table views, legal actions, hand results and events, the evaluator `best_omaha`, the `Agent` protocol) and plo8/entities (cards, boards, their text form).
* numpy, from the repo venv. No other third-party package. arena must not depend on torch, cli_server or cli_client.
* All poker rules stay in plo8/engine and all table driving stays in plo8/rl_env. arena never re-implements a rule (betting, pots, showdown, hand evaluation); bots that need hand strength ask the engine's evaluator.
* Dependency direction: entities <- engine <- rl_env <- arena. Nothing in engine or rl_env may depend on arena.

1. Bots
* arena provides four bots: RandomBot, CallingStation, Maniac and RuleBot.
* Every bot implements the rl_env `Controller` protocol: it receives a batch of decisions and returns one action (fold, check-or-call, bet-or-raise) per decision, and every returned action is allowed by that decision's action mask. The same bot object can therefore sit at any number of tables of a `TableBatch` at once.
* A bot decides only from what the acting player may see: its table view (own hole cards, board, public actions, stacks, pot) and its legal actions, or the observation vector and mask. A bot never uses the critic input or any other hidden information.
* Only bots that need the table view ask the batch for views (RuleBot does); RandomBot, CallingStation and Maniac decide from the mask and legal actions alone, so they do not pay for building views.
* Bots are stateless between decisions: a decision depends only on that decision's view, legal actions and (for RandomBot) the bot's own random generator. In particular a bot does not remember earlier hands.
* CallingStation, Maniac and RuleBot are deterministic: the same view and legal actions always give the same action. RandomBot is reproducible from its seed.
* Every bot has a short name used in reports and on the command line: `random`, `calling`, `maniac`, `rule`. arena provides a way to create a bot from its name (and a seed, for bots that use randomness), so command-line tools (here and in the later plo8/bot_client, `--bot rule`) can select a bot by name.

1.1 RandomBot
* Picks uniformly at random among the actions allowed by the mask (so it never folds when checking is free). It may wrap rl_env's `RandomController`. Its random generator is seeded when the bot is created.
* Purpose: the weakest possible opponent; any learning bot must beat it by a large margin.

1.2 CallingStation
* Always checks or calls; never folds, never bets or raises.
* Purpose: catches bots that only win by bluffing.

1.3 Maniac
* Bets or raises whenever that is allowed; otherwise checks or calls. Never folds.
* Purpose: catches bots that fold too much under pressure.

1.4 RuleBot (tight-aggressive, a human-like baseline)
* A simple, readable rule-based player, about 100 lines. It is a yardstick, not a product: it should be clearly better than RandomBot and CallingStation, and its rules should be easy to read and adjust.
* It never folds when checking is free (the mask already forbids it).
* Preflop it scores its four hole cards and acts on the score:
    * the score rewards: holding an ace together with a 2 (best), an ace with a 3, or a 2 with a 3; further distinct low cards (ace to eight) beyond those two; an ace suited with another hole card (nut-flush potential); other suited pairs of hole cards (less); a pair of aces; several big cards (ten to ace) that work together for high hands; connected cards;
    * the score penalises: three or four cards of the same rank; middle cards (nine and, to a lesser degree, ten to queen without an ace or other big cards) that make neither good lows nor good highs;
    * with a high score it raises (and re-raises while raising is allowed); with a medium score it calls a single bet but folds to a raise or more; with a low score it folds when facing a bet and checks when free.
* After the flop it evaluates its hand with the engine's evaluator (`best_omaha`, exactly two hole cards and three board cards):
    * it bets or raises with a strong high hand (a straight or better, or three of a kind) or with the nut low (a qualifying low that no other two hole cards can beat on this board), and always with both at once (a scooping hand);
    * it checks or calls with a good draw on the flop and the turn: a draw to the nut low (two low hole cards that make the nut low if one more suitable low card comes), a draw to the nut flush (an ace or the highest missing card of a suit with two board cards of that suit), or a qualifying low that is not the nut low; and with two pair or better on any street;
    * on the river it calls a single bet with any qualifying low or two pair or better, and folds otherwise;
    * with nothing of the above it checks when free and folds to a bet.
* The exact point values, thresholds and the definition of "nut low" and "nut-flush draw" (e.g. whether card removal by its own hole cards is considered) are design decisions for plan.md; plan.md lists them in one place so they can be tuned. The behaviour above is the requirement.

2. Bots at the real table (engine agent adapter)
* arena provides `BotAgent`, which wraps any arena bot as an engine agent (the engine's `Agent` protocol), so the same bot can play in engine callback mode and, later, at plo8/cli_server through plo8/bot_client.
* On each turn it builds a one-row decision batch from the engine's table view and legal actions (rl_env's `DecisionBatch.single`), asks the bot, turns the chosen action into an engine action (rl_env's `to_engine`) and submits it to the player.
* At the end of each hand it may rebuy up to a configured starting stack (off by default); everything else (hole-card notifications, events) is ignored.
* A bot behind `BotAgent` must make the same decision as the same bot inside a `TableBatch` for the same view and legal actions.

3. Duplicate matches (the evaluator)
* arena provides a match function, a plain library function callable from unit tests and from the later plo8/trainer, that plays a duplicate match:
    * inputs: a lineup (one bot per seat, each with a report name; several seats may hold copies of the same bot under the same name), the table size (2 to 10, equal to the lineup length), the number of deals, and a seed; optionally the starting stacks (default: every seat 100 big blinds);
    * a deal is one deck seed with one button seat and one set of starting stacks per seat. For each deal the same deck seed, button and stacks are played `table_size` times (a rotation group), rotating the lineup by one seat each time, so that every lineup position plays every seat's cards exactly once. Because the engine's deal order is a documented contract, the same deck seed deals the same cards to the same seats whatever the actions are, so this needs no engine change;
    * the deck seeds, buttons and any random stacks of the deals, and the seeds of random bots, all come from the match seed, so the same inputs give exactly the same match (actions, results, report);
    * the hands are played through an rl_env `TableBatch` with many tables at once, so a match runs at rl_env speed and bots receive batched decisions.
* Results are accounted per report name (copies of one bot under the same name are pooled), exactly, in chips, and converted to big bets only for the report:
    * **bb/100** = 100 × the mean result per hand in big bets (one big bet = 2 big blinds, as in rl_env's rewards; in limit poker results are customarily quoted in big bets, and the report says so in its header);
    * **95% confidence interval** = ±1.96 × the standard deviation / √(number of deals), where one sample is the result of one rotation group (all `table_size` hands of a deal), expressed per hand, so the correlation between the rotations of one deal is accounted for;
    * the number of hands and deals played, per report name.
* The net results of all seats of every hand sum to zero in chips; the match checks this and fails loudly if it ever does not hold.
* The match returns a result object with, per report name: hands, deals, total result in chips and in big bets, bb/100, the 95% interval, and the per-deal samples (so callers can combine matches or compute other statistics); and also the sample hand histories below.
* Sample hand histories: about 20 per match (the number is a parameter), chosen reproducibly from the match seed, as readable text built from the engine's hand result and events, using the text form of cards and boards. Each shows: deal and rotation number, table size, button, each seat's report name, starting stack and hole cards (the evaluator is not a player and may show all hole cards), the board by street, every blind and action in order with amounts, the pots with their high and low winners and amounts, and each seat's net result.
* "Keep playing until the interval is narrower than the difference you care about": the match function can be called again with more deals, or with another seed and the results combined; the result object makes this easy. Stopping automatically at a target width is optional.

4. Command-line evaluator
* arena includes a buck2 binary that plays matches and prints a results table, for example: `buck2 run //arena:match -- --agent rule --opponents random --players 2,6,9 --deals 2000 --seed 0`.
    * `--agent` is the bot being measured, sitting in one lineup position; `--opponents` is a comma-separated list of bot names filling the other seats in order, repeated as needed; `--players` lists the table sizes, one match per size; `--deals` and `--seed` as in section 3; optional `--histories N` prints N sample hand histories per match; optional `--stacks` chooses fixed 100 BB stacks (default) or rl_env's training stack distribution.
* The results table has one row per bot name per table size, with columns: agent, opponents, players, hands, bb/100, 95% CI. The header states that results are in big bets per 100 hands.
* It also reports wall time and hands per second per match.

5. Speed
* The evaluator must be fast enough to be run often during training (the later trainer evaluates every few iterations). A match of 2,000 deals 6-handed (12,000 hands) between RuleBot and five RandomBots should finish in well under a minute on one core of the development Mac; the measured numbers (hands per second at 2, 6 and 9 players, with and without RuleBot) are recorded in plan.md.
* RuleBot's per-decision cost is dominated by the evaluator; it may cache its hand evaluation per hand and street within one decision batch, but it must stay stateless across decisions as required in section 1.

6. Acceptance
* Masks: over many random hands at all table sizes and stacks, every action returned by every bot is allowed by the mask, and is accepted by the engine.
* Bot behaviour: CallingStation only checks or calls; Maniac bets or raises whenever allowed and never folds; RandomBot's choices are spread over all allowed actions and are reproducible from its seed; RuleBot raises its best preflop hands (e.g. A-A-2-3 double suited), folds its worst to a bet (e.g. 9-9-9-K rainbow), bets or raises the nut low and strong highs after the flop, and folds hands with neither a made hand nor a draw to a bet on the river; known positions (board, hole cards, betting) give the expected actions.
* Agent adapter: bots wrapped in `BotAgent` play complete hands in engine callback mode (2 to 10 players, at least a few hundred hands) without errors and without ever folding when checking is free, and make the same decisions as the same bots inside a `TableBatch` for the same views and legal actions.
* Deal-order contract from the arena side: within a rotation group every seat gets the same hole cards and the board is the same in every rotation in which it is dealt, whatever the actions.
* Rotation: within a rotation group every lineup position holds every seat's cards exactly once.
* Mirror matches: a deterministic bot (CallingStation, Maniac, RuleBot) against copies of itself gives **exactly 0** for every rotation group of every lineup position (this catches seat-assignment and accounting bugs).
* RandomBot against copies of itself gives a result near 0 with a 95% interval that includes 0.
* RuleBot beats RandomBot and CallingStation 6-handed (one RuleBot against five copies of the opponent) with a 95% interval that excludes 0, at a fixed seed and a number of deals chosen in plan.md.
* Zero sum: in every hand of every match the seats' nets sum to zero in chips, and the reported totals of all report names sum to zero.
* Determinism: the same lineup, table size, deals, stacks and seed give an identical match result and identical sample histories.
* Statistics: bb/100 and the interval are checked against a hand computation on a small match with known per-hand results (sign, units in big bets, one sample per rotation group).
* The command-line evaluator prints the results table (agent, opponents, players, hands, bb/100, 95% CI) for 2, 6 and 9 players.
* The speed numbers of section 5 are recorded in plan.md.
* All existing tests of plo8/entities, plo8/engine, plo8/rl_env, plo8/cli_server and plo8/cli_client still pass.
