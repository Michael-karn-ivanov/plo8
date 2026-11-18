"""End-to-end test mode of cli_server (`cli_server --e2e_test [--case NAME]`).

Recorded hands (`e2e_test_sets/*.json`) are replayed through a real `GameServer`
and real client processes, with a stacked deck; what every client receives is
checked against the recording.

Library `//cli_server:e2e` (public, no server, no process; other projects replay
the recorded hands through their own tables with it):
- `case`: load and validate a test-set file.
- `cases`: where the case files are; list / find / load them.
- `deck`: the engine seating of the case players and `StackedDeck`.

Runner `//cli_server:e2e_runner` (private to cli_server):
- `expect`: what every client is expected to receive.
- `client`: the scripted client process (`python -m cli_server.e2e.client`).
- `driver`: runs one case (server + clients) and all cases.
- `check`: compares client transcripts with the expectations.

This package file imports nothing, so importing a library module does not load
the runner.
"""
