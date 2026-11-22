---
name: doc-keep
description: Creates or updated documentation for python library/binary.
argument-hint: <root folder of python library/binary>
---

Use this skill when asked to actualize documenation on python library or binary.

Documentation is stored in file doc.html at the root folder of python library / binary.
If this file does not exist - create it, otherwise - update it.

This html should represent the logic of this library.

It should look like a multi-layer tree. 

1. Files and folders at the project directory are top-level nodes. 
a. You should onlu cover .py files, ignoring BUCK, *.json, *.md etc.
b. Ignore files that contain unit tests
2. Sub-files and sub-folders becomes sub-nodes to folder nodes. So every folder node is expandable into sub-tree of sub-files and sub-folders.
3. Names of folder and file nodes are
```
<for top-level - ./, else nothing><name> - <1 sentence brief description>
```
E.g.
for folder structure 
./
  - BUCK
  - db/
  --  connection.py
names of the nodes would be
./BUCK - buck2 instructions
./db - set of classes to work with database
connection.py - connection to database
4. Files nodes are also expandable into sub-nodes
5. Its children represent classes in the file or functions (if they don't belong to classes) or block of constants or imports.
6. Names for constants block is "constants", names for imports is "imports"
7. If child is a class - name is 
```
<Class name> - <1 sentence brief description>
```
E.g.
MySqlConnection - manage MySQL database connection
8. If child is a function - name is 
```
def <function name> - <1 sentence brief description>
```
E.g.
def open - opens connection to MySQL database
9. Functions of classes are sub-nodes to class nodes, named in the same convention (8)
10. Next levels of nodes are logical blockes within functions (or nested functions). 
These could be multi-layers - if logic within block can be separated into other logical scopes.
Name each node with 1-sentence brief description.
At the very last layer it is possible to see actual code.

## Example:
folder utils has file hello_printer.py, sub-folder demo and file in this subfolder, demo/example01.py

Code in hello_printer.py:
```
def print_console(input: str, counter: int):
    for i in range(counter):
        print(input)

class Printer:
    def __init__(self, callback):
        self._callback = callback

    def do_work(self, input: str, counter: int):
        for i in range(counter):
            self._callback(input)
```

Code in  demo/example01.py:
```
from hello_printer import Printer

def print_example():
  p = Printer(print)
  p.do_work("hello world", 5)
```

Tree for it might look like
- ./demo - demonstrates usage of print
  - example01.py - demonstrates Printer to stdout
    - def print_example - prints "hello_world" to stdout 5 times
      - creates Printer object and calls do_work
- ./hello_printer.py - Functions and classes for printing
    - def print_console - function to print to console
        - loop N times (N - input)
            - print S to console (S - input)
    - Printer -  print to callback
        - __init__ - sets callback
        - def do_work - print to callback
            - loop N times (N - input)
                - calls callback passing S (S - input)
