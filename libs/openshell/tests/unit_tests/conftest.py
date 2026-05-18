"""Shared fixtures for ``OpenShellSandbox`` unit tests.

Provides a ``FakeSandbox`` that mirrors the slice of the OpenShell SDK we
depend on (``id`` property + ``exec`` method returning an ``ExecResult``-shaped
record). Lets us exercise the wrapper end-to-end without installing the
``openshell`` package or talking to a real gateway.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

import pytest


@dataclass
class FakeExecResult:
    """Mirror of ``openshell.ExecResult``."""

    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""


@dataclass
class FakeExecCall:
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    stdin: bytes | None = None
    workdir: str | None = None
    timeout_seconds: int | None = None


class FakeSandbox:
    """Minimal stand-in for ``openshell.Sandbox``.

    Behaviour is driven by either:

    * a queue of pre-canned :class:`FakeExecResult`s (``queue(*results)``),
      consumed in FIFO order; or
    * a ``handler(call) -> FakeExecResult`` callable that decides each
      response dynamically (used for the upload/download bootstraps which
      need to react to the env vars + stdin payload).
    """

    def __init__(
        self,
        *,
        sandbox_id: str = "openshell-fake-1",
    ) -> None:
        self.id = sandbox_id
        self.calls: list[FakeExecCall] = []
        self._queue: list[FakeExecResult] = []
        self._handler: Callable[[FakeExecCall], FakeExecResult] | None = None
        self._raise_on_next: BaseException | None = None

    # -- programming helpers -------------------------------------------------

    def queue(self, *results: FakeExecResult) -> None:
        self._queue.extend(results)

    def set_handler(self, handler: Callable[[FakeExecCall], FakeExecResult]) -> None:
        self._handler = handler

    def raise_next(self, exc: BaseException) -> None:
        self._raise_on_next = exc

    # -- SDK surface ---------------------------------------------------------

    def exec(  # noqa: A003 - matches OpenShell SDK method name
        self,
        command: Sequence[str],
        *,
        stream_output: bool = False,
        workdir: str | None = None,
        env: Mapping[str, str] | None = None,
        stdin: bytes | None = None,
        timeout_seconds: int | None = None,
    ) -> Any:
        call = FakeExecCall(
            command=list(command),
            env=dict(env or {}),
            stdin=stdin,
            workdir=workdir,
            timeout_seconds=timeout_seconds,
        )
        self.calls.append(call)

        if self._raise_on_next is not None:
            exc = self._raise_on_next
            self._raise_on_next = None
            raise exc

        if self._handler is not None:
            return self._handler(call)
        if not self._queue:
            return FakeExecResult(exit_code=0, stdout="", stderr="")
        return self._queue.pop(0)


@pytest.fixture
def fake_sandbox() -> FakeSandbox:
    return FakeSandbox()


# ---------------------------------------------------------------------------
# Filesystem-aware fake handler (drives the upload/download bootstraps).
# ---------------------------------------------------------------------------


def make_filesystem_handler(
    initial: dict[str, bytes] | None = None,
) -> tuple[dict[str, bytes], Callable[[FakeExecCall], FakeExecResult]]:
    """Return a (storage, handler) pair that simulates a tiny filesystem.

    The handler interprets the upload/download bootstraps the wrapper sends:

    * ``OPENSHELL_UPLOAD_PATH`` + base64 stdin → write into ``storage[path]``.
    * ``OPENSHELL_DOWNLOAD_PATH`` → emit ``base64(storage[path])`` on stdout.

    Any other command is reported as a no-op success (so callers can mix in
    arbitrary ``execute`` checks against the same fake).
    """
    storage: dict[str, bytes] = dict(initial or {})

    def handler(call: FakeExecCall) -> FakeExecResult:
        path = call.env.get("OPENSHELL_UPLOAD_PATH")
        if path is not None:
            mode = call.env.get("OPENSHELL_UPLOAD_MODE", "wb")
            payload = base64.b64decode(call.stdin or b"")
            if mode == "ab":
                storage[path] = storage.get(path, b"") + payload
            else:
                storage[path] = payload
            return FakeExecResult(exit_code=0)

        path = call.env.get("OPENSHELL_DOWNLOAD_PATH")
        if path is not None:
            if path not in storage:
                return FakeExecResult(
                    exit_code=1,
                    stderr=f"python3: {path}: No such file or directory\n",
                )
            return FakeExecResult(
                exit_code=0,
                stdout=base64.b64encode(storage[path]).decode("ascii"),
            )

        return FakeExecResult(exit_code=0)

    return storage, handler


@pytest.fixture
def fs_sandbox() -> tuple[FakeSandbox, dict[str, bytes]]:
    sandbox = FakeSandbox()
    storage, handler = make_filesystem_handler()
    sandbox.set_handler(handler)
    return sandbox, storage


# ---------------------------------------------------------------------------
# Quality-of-life assertion helpers
# ---------------------------------------------------------------------------


def find_call_with_env_key(sandbox: FakeSandbox, key: str) -> FakeExecCall | None:
    for call in sandbox.calls:
        if key in call.env:
            return call
    return None


# Re-export so tests can ``from conftest import find_call_with_env_key``-style.
__all__ = [
    "FakeExecCall",
    "FakeExecResult",
    "FakeSandbox",
    "find_call_with_env_key",
    "fake_sandbox",
    "fs_sandbox",
    "make_filesystem_handler",
]
