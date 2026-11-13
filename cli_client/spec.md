This is a python binary, that implements local cli client to play Omaha Hi-Lo through server, defined in plo8/cli_server.

You can see details & protocol to communicate with server in plo8/cli_server - see spec.md and plan.md.

On startup it takes arguments: 
    - port of the running server - to connect to 
    - its own port - to open and listen for server events
    - player name (nick)
    - player's stack (default 100B)

It should provide cli interface where player can
* see table, players, their names, their stacks, their actions, their betsm total bank, opened flop/turn/river, small and big blinds
* interface should be colourfull, in particular, 4 colours used to render suits - red hearts, blue diamonds, yellow spades, green clubs
* it should highlight actions that happened on the table with the text on the right side of the screen, e.g.:
        player B raise
        player C fold
        played D check
        flop opened: ...
* when its client turn to take an action, it should provide convinient menu select, e.g.:
    [1] Fold
    [2] Call
    [3] Raise
or
    [1] Rebuy
    [2] Sit-out

Despite it should be regular cli, it is very much encouraged to make interface as realistic (comparable to actual online poker rooms) as possible.
If you're missing some capabilities from server for this problem - you are eligible to propose changes into plo8/cli_server spec.
