"""Command-line evaluator: buck2 run //arena:match -- [options]

The code lives in `arena.cli` (part of the library, so tests can call it).
"""

import sys

from arena.cli import main

if __name__ == "__main__":
    sys.exit(main())
