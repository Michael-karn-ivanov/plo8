This is a python binary, that implements local server, hosting single Omaha Hi-Lo game.
It should start the game once launched, but no active actions happen until 2 or more clients (players) are connected.
After that it starts serving game round by round, using classes, defined in //engine library.
If the game is in progress, newly-connected client joins as a player waiting (see details in engine/spec.md and engine/plan.md).
If the game is not started and is about to start (after 2nd client joined) - server should wait graceful period (15 seconds) after last-joined client, allowing other clients to join.

Interaction between this server and (future) clients happen over TCP in both directions.
We assume server and all clients processes run on the same host (localhost).
When server starts, it takes following arguments:
    * optional argument "port number" on which it listens for incoming clients - default is 8008
    * optional graceful period at the start of the game for all clients to join - default is 15 seconds
    * optional time to act for client - default is 180 seconds
    * optional time between hands - for players to see showdown - default - 3 seconds
Each client connecting informs server with information about player (name, stack etc - see engine/plan.md and code) and about port it (client) is listening for callbacks.
Server creates player object, corresponding to connected client and join it to the game.
Each callback to player is later passed as a call to client on the port it (client) is listening (the one that was provided).
Every call from Client to Server is passed through the same original port, server was listening, and server differentiates between clients who called it by clients (players) name.
Each client is informed about everything on the table - shown hands, bets made, stack sizes.

So example of calls may look like:
client A == (join as player A, 40 BB) ==> server 
(gracefull period starts ticking)
client B == (join as player B, 40 BB) ==> server 
(gracefull period starts ticking)
client C == (join as player C, 40 BB) ==> server 
(gracefull period starts ticking)
(graceful period ended)
Server initiates environment and starts first round
Server == informs player A on A's updated stack (after putting small blind) ==> client A
Server == informs player B on A's updated stack (after putting small blind) ==> client B
Server == informs player C on A's updated stack (after putting small blind) ==> client C
Server == informs player A on B's updated stack (after putting big blind) ==> client A
Server == informs player B on B's updated stack (after putting big blind) ==> client B
Server == informs player C on B's updated stack (after putting big blind) ==> client C
Server == hand for player A ==> client A
Server == hand for player B ==> client B
Server == hand for player C ==> client C
client D == (join as player D, 40 BB) ==> server (put as waiting player)
Server == Call for action from player C ==> client C
client C == fold ==> server 
Server == inform A on C's fold ==> client A
Server == inform B on C's fold ==> client B
Server == Call for action from player A ==> client A
client A == raise ==> server 
Server == informs player A on its updated stack (after putting raise) ==> client A
Server == informs player B on A's raise and updated stack (after putting raise) ==> client B
Server == informs player C on A's raise and updated stack (after putting raise) ==> client C
Server == Call for action from player B ==> client B
client B == call ==> server 
Server == informs player A on B's call and updated stack (after call) ==> client B
Server == informs player B on its updated stack (after call) ==> client B
Server == informs player B on B's call and updated stack (after call) ==> client B
Dealer shows flop
Server == send Flop to player A ==> client A
Server == send Flop to player B ==> client B
Server == send Flop to player C ==> client C
Server == Call for action from player A ==> client A
client A == check ==> server 
Server ==> informs B on A's check ==> client B
Server ==> informs C on A's check ==> client C
Server == Call for action from player B ==> client B
client B == raise ==> server 
Server == informs player A on B's raise and updated stack (after raise) ==> client A
Server == informs player B on its updated stack (after raise) ==> client B
Server == informs player C on B's raise and updated stack (after raise) ==> client C
Server == Call for action from player A ==> client A
client A == fold ==> server 
Server == informs player B on A's fold ==> client A
Server == informs player C on A's fold ==> client C
Server == send round summary - winner, pot won, winning hand (if shown), other hands (if showns) and updated stacks of each player ==> client A
Server == send round summary - winner, pot won, winning hand (if shown), other hands (if showns) and updated stacks of each player ==> client B
Server == send round summary - winner, pot won, winning hand (if shown), other hands (if showns) and updated stacks of each player ==> client C
(Client / player D now joined from waiting)
Server starts 2nd route.

Clients (e.g. plo8/cli_client) are expected to render the table the way online poker rooms do. To support that, the server also:
- answers a successful join with the table settings: blinds and bet sizes, maximum number of seats, maximum buy-in (if any), time given to act, and the grace period;
- whenever a player is asked to act, informs every other seated client whose turn it is (seat, name) and how much time that player has, so clients can highlight the acting player and show a countdown;
- while no hand is running, keeps connected clients informed about the lobby: how many active players are seated, how many are needed to start, and - while the graceful period is ticking - in how many seconds the game starts (re-sent whenever this changes);
- pauses for a short, configurable time between hands, so players can see the showdown and the pot being awarded before the next hand starts.
All these additions are optional for clients: a client that ignores them still plays correctly.

Server can also support end-to-end testing mode. In this mode it gets 2 extra arguments into cli input:
--e2e_test - flag, false when not passed
--case - name of the file in ./e2e_test_sets - e.g. 12_heads_up_three_raises_preflop_bb_scoops. 
If --case not passed when e2e_test passed - run e2e tests against all files in e2e_test_sets.
In this mode server spins up server itself, but also spins up several very simple "client" processes (not like fully-functional cli_client-s). And it replays hand described in the file: i.e. emulating client decisions and also controlling which hands dealt to clients and which table is opened.
What is asserted during each test replay:
- all clients receive correct updates on information
- stacks of the clients are correctly updated through the hand and at the end of it
- winners correctly determined and pot correctly splitted
So this mode is actual e2e test engine both for server (its api etc) and for game engine (in ../engine) itself.

The recorded hands are reusable by tests of other projects: the files in e2e_test_sets, together with the code that loads and validates a case and builds its seating (button first, clockwise) and its stacked deck, are available to them as a library, without starting a server. plo8/rl_env replays every recorded hand through its own tables this way. The wire protocol module stays usable on its own as well (plo8/rl_env tests encode and decode table views and legal actions with it).
