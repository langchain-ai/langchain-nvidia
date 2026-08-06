"""Compile-only integration-test marker used by repository CI."""

import pytest


@pytest.mark.compile
def test_placeholder() -> None:
    """Allow CI to collect the integration-test package without paid calls."""
