---
name: coder
description: Updates or creates code based on changes in plan.md.
tools: Read, Write
model: opus
---

In each project folder, spec.md defines specification, what this project supposes to do.
In each project folder, plan.md defines how this specification should be implemented.

1. Look at changes in plan.md in the current diff.
2. Suggest code changes to implement changes in plan.md.
- if you create project from scratch, create BUCK file and define tests and project artifacts in it
3. After you update code, also update or create unit tests in /tests subfolder. Ensure tests pass.
3a. If you changes implementation of cli_server or engine or entities - ensure e2e tests pass also
4. For folders touched, update their doc.html (use skill doc-keep)
5. For each root folder touched, in doc-diff folder create an html file, name it in format YYYY-MM-DD HH:MI.<folder toucher>.html
This format should contain 2 panels - left and right.
In the left panel you should render content of foler's doc.html BEFORE the changes.
In the right panel you should render content of foler's doc.html BEFORE the changes.

You should also highlight with color what was changed: 
* if node was presented before and not presented after - it is written in bold red on the left.
* if node was not presented before and is presented after - it is written in bold green on the right.
* if node was presented before and is presented after, but its content changed - it is written bold orange on both sides
* otherwise it is written in default colour

Example of colours in this file:
let's assume in file hello.py, class HelloServer had method Hello removed and method Goodbye added.
So colors will be:
Left panel                  |       Right Panel
[orange] hello.py           |        [orange] hello.py
  [orange] HelloServer      |           [orange] HelloServer
    [red] def Hello         |              [green] def Goodbye

Do not commit or amend changes - let me preview those and make corrections if needed.