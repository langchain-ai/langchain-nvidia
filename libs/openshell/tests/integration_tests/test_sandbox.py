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
from typing import TYPE_CHECKING, Any

import pytest

openshell: Any | None
_OPENSHELL_IMPORT_ERROR: Exception | None
try:
    import openshell as _openshell
except Exception as exc:  # pragma: no cover - environment-dependent
    openshell = None
    _OPENSHELL_IMPORT_ERROR = exc
else:
    openshell = _openshell
    _OPENSHELL_IMPORT_ERROR = None

_OPENSHELL_GATEWAY_ERROR: str | None = None

from langchain_nvidia_openshell import OpenShellSandbox  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Iterator

    from deepagents.backends.protocol import SandboxBackendProtocol


def _gateway_configured() -> bool:
    """Return True if an OpenShell gateway is locally reachable.

    We check both the ``OPENSHELL_GATEWAY`` env override and the CLI's
    ``~/.config/openshell/active_gateway`` pointer (the same lookup the SDK
    performs in :meth:`SandboxClient.from_active_cluster`). A stale
    ``active_gateway`` file is not enough: the SDK must be able to perform a
    cheap list request without creating a sandbox.
    """
    global _OPENSHELL_GATEWAY_ERROR
    _OPENSHELL_GATEWAY_ERROR = None

    has_gateway_pointer = bool(os.environ.get("OPENSHELL_GATEWAY"))
    xdg = os.environ.get("XDG_CONFIG_HOME")
    config_home = pathlib.Path(xdg) if xdg else (pathlib.Path.home() / ".config")
    has_gateway_pointer = (
        has_gateway_pointer or (config_home / "openshell" / "active_gateway").exists()
    )
    if not has_gateway_pointer or openshell is None:
        return False

    try:
        client = openshell.SandboxClient.from_active_cluster(timeout=2.0)
        try:
            client.list(limit=1)
        finally:
            client.close()
    except Exception as exc:  # noqa: BLE001 - SDK/grpc readiness failures vary
        _OPENSHELL_GATEWAY_ERROR = _brief_error(exc)
        return False
    return True


def _brief_error(exc: Exception) -> str:
    details = getattr(exc, "details", None)
    if callable(details):
        try:
            message = details()
        except Exception:  # pragma: no cover - defensive around grpc internals
            message = ""
        if message:
            return str(message)
    message = str(exc).strip()
    return message.splitlines()[0] if message else type(exc).__name__


# ---------------------------------------------------------------------------
# Compile-only marker so the file is collectable without a live gateway.
# Mirrors the convention used in libs/ai-endpoints/tests/integration_tests.
# ---------------------------------------------------------------------------


@pytest.mark.compile
def test_compile_marker() -> None:
    """Placeholder so `pytest --collect-only` picks something up."""


def _require_openshell_gateway() -> None:
    """Skip runtime tests unless the SDK imports and a gateway is configured."""
    if _OPENSHELL_IMPORT_ERROR is not None:
        pytest.skip(f"OpenShell SDK import failed: {_OPENSHELL_IMPORT_ERROR}")
    if not _gateway_configured():
        detail = f" ({_OPENSHELL_GATEWAY_ERROR})" if _OPENSHELL_GATEWAY_ERROR else ""
        pytest.skip(
            "no OpenShell gateway configured — set OPENSHELL_GATEWAY or run "
            "`openshell sandbox create` once to populate ~/.config/openshell/."
            f"{detail}"
        )


# ---------------------------------------------------------------------------
# Smoke: end-to-end against a real sandbox
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_backend() -> "Iterator[OpenShellSandbox]":
    _require_openshell_gateway()
    assert openshell is not None
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
    ``docs/sandboxes/nvidia_openshell_sandbox.ipynb``. We don't assert on whether the
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


def test_deep_agent_execute_tool_uses_openshell_backend(
    real_backend: OpenShellSandbox,
) -> None:
    """Run Deep Agents' built-in execute tool against OpenShell without API keys."""
    from deepagents import create_deep_agent
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel,
    )
    from langchain_core.messages import AIMessage, ToolMessage

    class ToolCallingFakeModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, *, tool_choice=None, **kwargs):  # type: ignore[no-untyped-def]
            return self

    model = ToolCallingFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute",
                        "args": {"command": "printf deepagents-openshell-smoke"},
                        "id": "call_execute_1",
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    agent = create_deep_agent(model=model, backend=real_backend)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Run the smoke command."}]}
    )

    tool_messages = [msg for msg in result["messages"] if isinstance(msg, ToolMessage)]
    assert tool_messages
    assert tool_messages[0].name == "execute"
    assert "deepagents-openshell-smoke" in tool_messages[0].content
    assert "[Command succeeded with exit code 0]" in tool_messages[0].content


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
            _require_openshell_gateway()
            assert openshell is not None
            sandbox = openshell.Sandbox()
            sandbox.__enter__()
            try:
                yield OpenShellSandbox(sandbox=sandbox)
            finally:
                sandbox.__exit__(None, None, None)
