---
name: commit-author
description: Create local git commit
---

Use git add if necessary and git commit with a meaningfull subject.

When asked to commit the changes and there are multiple files changed, you use following prioritization list:
1. If there are multiple spec.md changed (along with other changes) - you select and commit only one spec.md, following this priority list:
arena > rl_env > cli_client > cli_server > engine > entities
You do not commit anything else
2. If multiple files were changed and among them single spec.md - you always commit only spec.md.
3. If there are multiple plan.md changed (along with other changes, but no spec.md - otherwise it would be rule 1 or rule 2) - you select and commit only one plan.md, following this priority list:
entities > engine > cli_server > cli_client > rl_env > arena
4. If multiple files were changed and among them single plan.md (along with other changes, but no spec.md - otherwise it would be rule 1 or rule 2) - you commit only plan.md.
5. If multiple files were changed and amond them no spec.md and plan.md - you commit changes working through sub-directories in this priority order:
entities > engine > cli_server > cli_client > rl_env
I.e. each single commit there will be changes only from one sub-directory.
You create sequence of commits following this workflow:
1. Identify minimal meaningful change, that does not have dependancies on yet-uncommited code. It could be:
a) newly added function to existing class
b) newly added class
c) change in implementation of existing function or block of code
d) change of some enum or constant
2. If this change is still big - more than +- 100 lines of code - especially and likely if it is (a) or (b) - you can introduce temporary intermediate steps in this change - those could be:
a) class with stubbed functions without actual functions implementations - and functions implementations go each into its own commit on top until reach needed state
b) function without exception handling or other harness - and those are added later on top as its own commits
3. If there is unit test - added or changed - that covers changes in this small commit - include it into this commit also.
4. Find meaningful name for each commit, describing this small change. Prefix it with sub-directory name, e.g. [rl_env] xyz
5. Proceed with step 1 for next portion of changes. Try to keep relevant commits close - i.e. if in one commit you added function, in next commit it will make sense to include its callsite. If in one commit you expanded enum - in next commit bring code that uses new value of the enum etc

So that at the end instead of one big commit there was a sequence of small meaningfull changes.

If you commit spec.md - put message of a format "[SPEC]: <meaningfull 1-liner describing changes>"
If you commit plan.md - in the body provide explanation - how different changes in spec.md influenced changes in plan.md
If you commit code - in the body provide explanation - how different changes in plan.md influenced code changes - and separately - how they influenced test changes. If it is first or last commit in commit sequence created following workflow abouve - into summary of the first and last xommit in this sequence put brief description how this sequence achieves desired outcome.
