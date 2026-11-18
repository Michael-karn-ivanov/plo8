"""Where the recorded hands are: list, find and load the case files.

The JSON files of `e2e_test_sets` are resources of the `//cli_server:e2e`
library, so they sit next to the `cli_server` package in the source tree and in
any built binary or test that includes the library. This module imports only
the standard library and `case` (no server, no engine, no process).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Union

from .case import Case, CaseError, load_case

__all__ = [
    "E2E_DIR",
    "UnknownCaseError",
    "available_cases",
    "case_name",
    "case_path",
    "load_cases",
]

E2E_DIR = Path(__file__).resolve().parents[2] / "e2e_test_sets"


class UnknownCaseError(KeyError):
    """No file in `e2e_test_sets` has that name."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else "unknown case"


def _dir(directory: Optional[Union[str, Path]]) -> Path:
    return E2E_DIR if directory is None else Path(directory)


def available_cases(directory: Optional[Union[str, Path]] = None) -> list[str]:
    """Names (file stems) of every case, sorted."""
    d = _dir(directory)
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.json"))


def case_name(name: str) -> str:
    """`NAME` or `NAME.json` -> `NAME`."""
    return name[:-5] if name.endswith(".json") else name


def case_path(name: str, directory: Optional[Union[str, Path]] = None) -> Path:
    """The file of a case (with or without `.json`). Raises UnknownCaseError,
    listing the available cases."""
    d = _dir(directory)
    stem = case_name(name)
    path = d / f"{stem}.json"
    if not stem or "/" in stem or os.sep in stem or not path.is_file():
        available = ", ".join(available_cases(d)) or "none"
        raise UnknownCaseError(f"no e2e case {name!r} in {d} (available: {available})")
    return path


def load_cases(
    names: Optional[Iterable[str]] = None, directory: Optional[Union[str, Path]] = None
) -> list[Case]:
    """Load the named cases (all of them, sorted by name, when `names` is None).

    An invalid file raises CaseError naming the file, so a caller that iterates
    the cases fails loudly instead of skipping one.
    """
    d = _dir(directory)
    selected = available_cases(d) if names is None else list(names)
    out = []
    for name in selected:
        path = case_path(name, d)
        try:
            out.append(load_case(path))
        except CaseError as e:
            raise CaseError(f"{path.name}: {e}") from None
    return out
