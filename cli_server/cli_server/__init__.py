"""cli_server: a localhost TCP server hosting one fixed-limit Omaha Hi-Lo 8-or-better table.

`cli_server.protocol` is the language-neutral JSON-lines wire format (shared with
clients; library `//cli_server:protocol`); `cli_server.session`,
`cli_server.server` and `cli_server.main` are the server itself.
`cli_server.e2e` is the end-to-end test mode (`--e2e_test`), which replays
recorded hands through a real server and scripted client processes; its case
loader, seating and stacked deck (`case`, `cases`, `deck`) are the library
`//cli_server:e2e`, usable by other projects' tests without a server.

This package file imports nothing, so importing a library module does not load
the server.
"""
