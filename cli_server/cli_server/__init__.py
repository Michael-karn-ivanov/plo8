"""cli_server: a localhost TCP server hosting one fixed-limit Omaha Hi-Lo 8-or-better table.

`cli_server.protocol` is the language-neutral JSON-lines wire format (shared with
clients); `cli_server.session`, `cli_server.server` and `cli_server.main` are the
server itself.
"""
