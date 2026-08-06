"""Import source files as a lightweight packaging smoke check."""

from __future__ import annotations

import sys
import traceback
from importlib import import_module
from pathlib import Path


def _module_name(file: str) -> str:
    """Convert a package-relative Python path to its importable module name."""
    parts = list(Path(file).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def main(files: list[str]) -> int:
    """Import each supplied file and return a failing status for any exception."""
    has_failure = False
    for file in files:
        try:
            import_module(_module_name(file))
        except Exception:
            has_failure = True
            print(file)  # noqa: T201
            traceback.print_exc()
            print()  # noqa: T201
    return int(has_failure)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
