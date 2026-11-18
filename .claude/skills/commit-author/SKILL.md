---
name: commit-author
description: Create local git commit
---

Use git add if necessary and git commit with a meaningfull subject.
If multiple files were changed and among them spec.md - you always commit only spec.md.
If multiple files were changed and among them plan.md and no plan.md - you commit only plan.md.
If multiple files were changed and amond them no spec.md and plan.md - you create sequence of commits following this workflow:
1. Identify minimal meaningful change, that does not have dependancies on yet-uncommited code. It could be:
a) newly added function to existing class
b) newly added class
c) change in implementation of existing function or block of code
d) change of some enum or constant
2. If this change is still big - more than +- 100 lines of code - especially and likely if it is (a) or (b) - you can introduce temporary intermediate steps in this change - those could be:
a) class with stubbed functions without actual functions implementations - and functions implementations go each into its own commit on top until reach needed state
b) function without exception handling or other harness - and those are added later on top as its own commits
3. If there is unit test - added or changed - that covers changes in this small commit - include it into this commit also.
4. Find meaningful name for each commit, describing this small change
5. Proceed with step 1 for next portion of changes. Try to keep relevant commits close - i.e. if in one commit you added function, in next commit it will make sense to include its callsite. If in one commit you expanded enum - in next commit bring code that uses new value of the enum etc

So that at the end instead of one big commit there was a sequence of small meaningfull changes.

If spec.d was changed - put message of a format "[SPEC]: <meaningfull 1-liner describing changes>"
If plan.md was changed - in the body provide explanation - how different changes in spec.md influenced changes in plan.md
If code was changed - in the body provide explanation - how different changes in plan.md influenced code changes - and separately - how they influenced test changes. If it is first or last commit in commit sequence created following workflow abouve - into summary of the first and last xommit in this sequence put brief description how this sequence achieves desired outcome.
