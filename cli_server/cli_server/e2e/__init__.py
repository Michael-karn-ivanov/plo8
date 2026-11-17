"""End-to-end test mode of cli_server (`cli_server --e2e_test [--case NAME]`).

Recorded hands (`e2e_test_sets/*.json`) are replayed through a real `GameServer`
and real client processes, with a stacked deck; what every client receives is
checked against the recording.

- `case`: load and validate a test-set file.
- `deck`: the engine seating of the case players and `StackedDeck`.
- `expect`: what every client is expected to receive.
- `client`: the scripted client process (`python -m cli_server.e2e.client`).
- `driver`: runs one case (server + clients) and all cases.
- `check`: compares client transcripts with the expectations.
"""
