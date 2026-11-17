---
name: e2e-testing
description: Run e2e tests on server and game engine
---

Use this skill when you need to perform e2e tests on plo8/engine or plo8/cli_server code changes.

The e2e mode of `cli_server` replays the recorded hands in `cli_server/e2e_test_sets/*.json`
through a real server and scripted client processes, checking every client update, stacks,
winners and pot splits (it exercises both `cli_server` and `engine`).

- run all test cases if you need to assess broader set of changes
`buck2 run //cli_server:cli_server -- --e2e_test`

- if you need to re-run previously failing test case 
`buck2 run //cli_server:cli_server -- --e2e_test --case 12_heads_up_three_raises_preflop_bb_scoops`

- The run prints `PASS` / `FAIL` / `ERROR` per case and a `N passed, N failed, N errors`
  summary, and exits 0 only if every case passed. Server logs go to stderr; add
  `2>/dev/null` to see only the report, or `--log-level WARNING` to quiet them.
- `buck2 test //cli_server/...` also runs every case (`tests/test_e2e_cases.py`), and
  `tests/test_e2e_case.py` checks that every case file is valid.