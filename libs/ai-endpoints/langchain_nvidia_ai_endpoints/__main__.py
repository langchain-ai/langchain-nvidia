# This CLI writes usage to stderr by design.
# ruff: noqa: T201
"""Run the connector's read-only endpoint preflight."""

import sys

from langchain_nvidia_ai_endpoints._doctor import main

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "doctor":
        print(
            "Usage: python -m langchain_nvidia_ai_endpoints doctor [options]",
            file=sys.stderr,
        )
        sys.exit(2)
    sys.exit(main(sys.argv[2:]))
