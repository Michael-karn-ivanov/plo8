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

Library is designed this way that there will be several agents - like separate processes. 
They will share environment and dealer and will "play" at the table: i.e. each player can see their hand and each player will be responsible for its own actions.
I.e. there should be some callback in Player, that will get called within rounds during betting for an active player and its managing agent will take action through Player (call, fold, etc).
After the end of round (no matter whether it was showdown or not) - players stacks are updated - (if they won - if they lost - their bets are deducted from stack during the round) - and there is another callback, so that each managing agent can make decision whether to rebuy or not.
If player has 0 chips left and they did not rebuy at the end of the round - they are not dealt cards and don't take part in the next round.
They still get notification when next and further rounds end so that they can rebuy (and if they do - they return to game).
Players can explicitly leave the game, emptying their seat.

Also, environment has a method for a new player to join the game. They select a seat (index from button, if not passed - they join as big blind) and stack (amount of blinds) they join with. Also let's all the players may have name (nick), which they set when join. 
If player while round is in progress, they don't participate in game but are waiting in the queue for round to complete and get cards dealt starting new round.