"""Validate values before passing them across typed API boundaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast


def require_mapping(value: object, path: str) -> Mapping[str, object]:
    """Return a mapping or raise an error naming its source path."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return cast(Mapping[str, object], value)


def require_sequence(value: object, path: str) -> Sequence[object]:
    """Return a sequence while rejecting strings and other byte sequences."""
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{path} must be a sequence")
    return cast(Sequence[object], value)


def require_string_list(value: object, path: str) -> list[str]:
    """Return a list of strings without silently accepting other sequences."""
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{path} must be a list of strings")
    return cast(list[str], value)
