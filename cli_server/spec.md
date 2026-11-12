This is a python binary, that implements local server, hosting single Omaha Hi-Lo game.
It should start the game once launched, but no active actions happen until 2 or more clients (players) are connected.
After that it starts serving game round by round, using classes, defined in //engine library.
If the game is in progress, newly-connected client joins as a player waiting (see details in engine/spec.md and engine/plan.md).
If the game is not started and is about to start (after 2nd client joined) - server should wait graceful period (15 seconds) after last-joined client, allowing other clients to join.

Interaction between this server and (future) clients happen over TCP in both directions.
We assume server and all clients processes run on the same host (localhost).
When server starts, it takes optional argument "port number" on which it listens for incoming clients.
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
