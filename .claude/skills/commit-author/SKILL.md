---
name: commit-author
description: Create local git commit
---

Use git add if necessary and git commit with a meaningfull subject.
If multiple files were changed and among them spec.md - you always commit only spec.md.
If multiple files were changed and among them plan.md and no plan.md - you commit only plan.md.
If multiple files were changed and amond them no spec.md and plan.md - you commit all of them.

If spec.d was changed - put message of a format "[SPEC]: <meaningfull 1-liner describing changes>"
If plan.md was changed - in the body provide explanation - how different changes in spec.md influenced changes in plan.md
If code was changed - in the body provide explanation - how different changes in plan.md influenced code changes - and separately - how they influenced test changes.