---
name: commit-loop
description: Take a change request from the user, apply it, then repeatedly commit (via commit-author) and wait for the post-commit hooks (planner / coder sub-agents) until a commit leaves the working tree clean.
argument-hint: <description of what needs to be done>
---

The user's request is: $ARGUMENTS

If the request above is empty, ask the user what needs to be done and stop until they answer.

## Background

`.githooks/post-commit` runs synchronously inside `git commit`:
- a commit touching `spec.md` runs the `planner` sub-agent, which edits `plan.md`;
- a commit touching `plan.md` runs the `coder` sub-agent, which edits code and tests.

The agents leave their edits uncommitted. So every commit can produce new changes, and
those need committing too. This workflow keeps going until a commit produces nothing new.

## Step 1: apply the request

1. Read the `spec.md` (and `plan.md` if it helps) of the project folder(s) the request is about.
2. Make the change the request describes. A behaviour or feature change goes into
   `spec.md`, and the hooks carry it into `plan.md` and code. Edit `plan.md` or code
   directly only when the user explicitly asks for that.
3. Briefly tell the user what you changed. Then start the loop without waiting.

## Step 2: the loop

Repeat:

1. Run `git status --porcelain`. If the output is empty, go to Step 3.
2. Invoke the `commit-author` skill and follow it exactly. It picks what goes into this
   commit (one spec.md, or one plan.md, or a series of small code commits for one
   sub-directory). Leave everything else in the working tree for later iterations.
3. Run each `git commit` with the Bash tool's `run_in_background: true`, because the hook
   may run a sub-agent for longer than the foreground timeout allows. Wait for the
   completion notification. Don't poll and don't start the next commit before it arrives.
   Then read the output and check that the commit exists (`git log -1 --stat`).
   - If the hook printed `sub-agent failed`, or the commit failed, stop and report it to
     the user with the relevant output. Don't retry blindly.
   - Never use `--no-verify` or `PLO8_SKIP_CLAUDE_HOOK=1`. Running the hooks is the point.
4. Tell the user in one line what was committed and what the hook did (for example
   "committed engine/spec.md, planner updated engine/plan.md").
5. Go back to 1.

Safety cap: stop after 20 commits in one run and ask the user whether to continue. This
prevents a planner/coder ping-pong that never settles.

Don't run `git push`, `git reset`, `git checkout -- <files>` or anything else that
discards or publishes work.

## Step 3: finish

When `git status --porcelain` is empty right after a commit and its hook have finished:
1. Run `buck2 test` on the affected project folders. If `engine` or `cli_server` changed,
   use the `e2e-testing` skill as well. Report the results honestly. Don't fix failures
   unless the user asks you to.
2. Summarise for the user: the commits made (`git log --oneline <start>..HEAD`), which
   hooks ran, and the test results.
