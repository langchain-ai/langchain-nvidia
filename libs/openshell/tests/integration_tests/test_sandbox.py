"""Integration tests for :class:`OpenShellSandbox`.

Two tracks:

1.  **Standard conformance** — runs LangChain's
    ``langchain_tests.integration_tests.SandboxIntegrationTests`` suite against
    a real OpenShell sandbox. This is the canonical interop check used by all
    sandbox partner integrations (Daytona, Modal, Runloop).

2.  **Smoke** — a pair of end-to-end tests covering ``execute``, file
    upload/download round-trip, and a small policy-friendly Python invocation,
    so the suite stays useful even if the standard tests are unavailable.

Both tracks require a configured OpenShell gateway. Tests are auto-skipped
when ``openshell`` is not installed or no gateway is reachable, so the file is
safe to ship in CI.

Run locally with::

    pip install langchain-tests openshell
    openshell sandbox create --keep --no-tty -- bash   # one-time per host
    poetry run pytest tests/integration_tests
"""

from __future__ import annotations

import os
import pathlib
from typing import TYPE_CHECKING

import pytest

# Skip the entire module unless the OpenShell SDK is present.
openshell = pytest.importorskip("openshell")

from langchain_nvidia_openshell import OpenShellSandbox  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents.backends.protocol import SandboxBackendProtocol


def _gateway_configured() -> bool:
    """Return True if an OpenShell gateway is locally reachable.

    We check both the ``OPENSHELL_GATEWAY`` env override and the CLI's
    ``~/.config/openshell/active_gateway`` pointer (the same lookup the SDK
    performs in :meth:`SandboxClient.from_active_cluster`).
    """
    if os.environ.get("OPENSHELL_GATEWAY"):
        return True
    xdg = os.environ.get("XDG_CONFIG_HOME")
    config_home = pathlib.Path(xdg) if xdg else (pathlib.Path.home() / ".config")
    return (config_home / "openshell" / "active_gateway").exists()


pytestmark = pytest.mark.skipif(
    not _gateway_configured(),
    reason=(
        "no OpenShell gateway configured — set OPENSHELL_GATEWAY or run "
        "`openshell sandbox create` once to populate ~/.config/openshell/."
    ),
)


# ---------------------------------------------------------------------------
# Compile-only marker so the file is collectable without a live gateway.
# Mirrors the convention used in libs/ai-endpoints/tests/integration_tests.
# ---------------------------------------------------------------------------


@pytest.mark.compile
def test_compile_marker() -> None:
    """Placeholder so `pytest --collect-only` picks something up."""


# ---------------------------------------------------------------------------
# Smoke: end-to-end against a real sandbox
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_backend() -> "Iterator[OpenShellSandbox]":
    sandbox = openshell.Sandbox()
    sandbox.__enter__()
    try:
        yield OpenShellSandbox(sandbox=sandbox)
    finally:
        sandbox.__exit__(None, None, None)


def test_execute_smoke(real_backend: OpenShellSandbox) -> None:
    result = real_backend.execute("echo hello-openshell")
    assert result.exit_code == 0
    assert "hello-openshell" in result.output


def test_upload_download_round_trip(real_backend: OpenShellSandbox) -> None:
    payload = b"\x00\x01\x02hello-openshell\n"
    [up] = real_backend.upload_files([("/sandbox/round-trip.bin", payload)])
    assert up.error is None
    [down] = real_backend.download_files(["/sandbox/round-trip.bin"])
    assert down.error is None
    assert down.content == payload


def test_python_invocation(real_backend: OpenShellSandbox) -> None:
    """Run a small Python script inside the sandbox via execute()."""
    result = real_backend.execute("python3 -c 'print(2 + 2)'")
    assert result.exit_code == 0
    assert "4" in result.output


def test_notebook_zen_tool_pattern(real_backend: OpenShellSandbox) -> None:
    """Validate the demo notebook's @tool-around-execute pattern end-to-end.

    This mirrors `make_zen_tool` in
    ``docs/nvidia_openshell_sandbox.ipynb``. We don't assert on whether the
    sandbox's network policy allows ``api.github.com`` (that's the
    operator's choice and is what the notebook explicitly demos both ways).
    We only assert the wrapper composes cleanly with a
    ``langchain_core.tools.@tool``: success returns a non-empty string,
    denial returns a ``Tool failed`` string. Either is a valid outcome.
    """
    from langchain_core.tools import tool

    @tool
    def github_zen() -> str:
        """Fetch a Zen of GitHub quote (a short proverb)."""
        result = real_backend.execute(
            "curl -sSf --max-time 5 https://api.github.com/zen",
        )
        if result.exit_code != 0:
            return f"Tool failed (exit {result.exit_code}): {result.output[:200]}"
        return result.output.strip()

    out = github_zen.invoke({})

    assert isinstance(out, str)
    assert out  # non-empty either way
    # Cross-check: the output is either a quote or a structured failure.
    assert out.startswith("Tool failed") or len(out.splitlines()) <= 3


# ---------------------------------------------------------------------------
# Standard conformance suite
# ---------------------------------------------------------------------------

# Skip cleanly if `langchain-tests` isn't installed.
SandboxIntegrationTests = None
try:
    from langchain_tests.integration_tests import (  # type: ignore[import-not-found]
        SandboxIntegrationTests as _SandboxIntegrationTests,
    )

    SandboxIntegrationTests = _SandboxIntegrationTests
except ImportError:  # pragma: no cover - environment-dependent
    pass


if SandboxIntegrationTests is not None:

    class TestOpenShellSandboxStandard(SandboxIntegrationTests):  # type: ignore[misc, valid-type]
        """Run LangChain's standard sandbox conformance suite."""

        @pytest.fixture(scope="class")
        def sandbox(self) -> "Iterator[SandboxBackendProtocol]":
            sandbox = openshell.Sandbox()
            sandbox.__enter__()
            try:
                yield OpenShellSandbox(sandbox=sandbox)
            finally:
                sandbox.__exit__(None, None, None)
