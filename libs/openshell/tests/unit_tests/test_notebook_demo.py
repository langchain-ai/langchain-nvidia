"""Validate the deny->allow tool pattern shown in the demo notebook.

The notebook (`libs/openshell/docs/nvidia_openshell_sandbox.ipynb`)
demonstrates a `langchain_core.tools.@tool`-decorated function that calls
`OpenShellSandbox.execute("curl ...")` and returns the result to a Deep
Agent. We assert that the wrapper composes correctly with that pattern in
both the **denied** and **allowed** policy states, using the same
`FakeSandbox` fixture the rest of the unit suite uses.

We also smoke-validate the notebook file itself: it parses as JSON,
contains both `%%writefile` policy cells, contains a `create_deep_agent`
call, and ends with a cleanup cell.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.tools import tool

from langchain_nvidia_openshell import OpenShellSandbox

from .conftest import FakeExecResult, FakeSandbox

# ---------------------------------------------------------------------------
# In-notebook tool pattern: a @tool that wraps backend.execute(...)
# ---------------------------------------------------------------------------


def _make_zen_tool(backend: OpenShellSandbox):
    """Mirror of the notebook's `make_zen_tool` factory, verbatim shape."""

    @tool
    def github_zen() -> str:
        """Fetch a Zen of GitHub quote (a short proverb) from api.github.com.

        Use whenever the user asks for a GitHub Zen quote or a programming
        proverb. Returns the quote on success or an error message on failure.
        """
        result = backend.execute("curl -sSf --max-time 5 https://api.github.com/zen")
        if result.exit_code != 0:
            return f"Tool failed (exit {result.exit_code}): {result.output[:200]}"
        return result.output.strip()

    return github_zen


def test_zen_tool_returns_quote_under_allow_policy(fake_sandbox: FakeSandbox) -> None:
    """Sandbox returns a successful curl: tool emits the cleaned quote."""
    fake_sandbox.queue(
        FakeExecResult(exit_code=0, stdout="Speak like a human.\n"),
    )
    backend = OpenShellSandbox(sandbox=fake_sandbox)

    zen = _make_zen_tool(backend)
    out = zen.invoke({})

    assert out == "Speak like a human."
    # The wrapper sent a single bash -c <curl> argv to the sandbox
    [call] = fake_sandbox.calls
    assert call.command[:2] == ["bash", "-c"]
    assert "curl" in call.command[2]


def test_zen_tool_reports_failure_under_deny_policy(
    fake_sandbox: FakeSandbox,
) -> None:
    """Sandbox simulates the egress proxy denying the request."""
    fake_sandbox.queue(
        FakeExecResult(
            exit_code=7,
            stderr=(
                "curl: (7) Failed to connect to api.github.com port 443: "
                "Connection refused\n"
            ),
        ),
    )
    backend = OpenShellSandbox(sandbox=fake_sandbox)

    zen = _make_zen_tool(backend)
    out = zen.invoke({})

    assert out.startswith("Tool failed (exit 7):")
    assert "api.github.com" in out


def test_deny_then_allow_flow_with_one_backend(fake_sandbox: FakeSandbox) -> None:
    """One backend, two consecutive invocations: simulates the notebook flow.

    First call returns curl exit 7 (proxy denied); second call returns
    exit 0 with a quote (policy was swapped). The same tool invocation
    succeeds on the second try, demonstrating the ``OpenShellSandbox``
    wrapper is stateless w.r.t. the underlying sandbox's policy.
    """
    fake_sandbox.queue(
        FakeExecResult(
            exit_code=7,
            stderr="curl: (7) Failed to connect to api.github.com port 443\n",
        ),
        FakeExecResult(exit_code=0, stdout="Approachable is better than simple.\n"),
    )
    backend = OpenShellSandbox(sandbox=fake_sandbox)
    zen = _make_zen_tool(backend)

    denied = zen.invoke({})
    allowed = zen.invoke({})

    assert "Tool failed (exit 7)" in denied
    assert allowed == "Approachable is better than simple."


# ---------------------------------------------------------------------------
# Notebook structural smoke
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def notebook() -> dict:
    nb_path = (
        Path(__file__).resolve().parents[2] / "docs" / "nvidia_openshell_sandbox.ipynb"
    )
    return json.loads(nb_path.read_text())


def test_notebook_is_valid_json_and_nbformat4(notebook: dict) -> None:
    assert notebook["nbformat"] == 4
    assert isinstance(notebook["cells"], list)
    assert notebook["cells"]


def _cell_sources(notebook: dict) -> list[str]:
    return ["".join(c["source"]) for c in notebook["cells"]]


def test_notebook_has_both_policy_files(notebook: dict) -> None:
    sources = _cell_sources(notebook)
    assert any("%%writefile policy-deny.yaml" in s for s in sources), (
        "expected a deny-by-default policy cell"
    )
    assert any("%%writefile policy-allow.yaml" in s for s in sources), (
        "expected an allow-list policy cell"
    )


def test_notebook_creates_a_deep_agent_with_tools(notebook: dict) -> None:
    sources = _cell_sources(notebook)
    assert any("create_deep_agent" in s for s in sources)
    assert any("ChatNVIDIA" in s for s in sources)
    assert any("OpenShellSandbox" in s for s in sources)
    assert any("@tool" in s for s in sources)


def test_notebook_cleans_up_at_the_end(notebook: dict) -> None:
    sources = _cell_sources(notebook)
    cleanup = "\n".join(sources[-3:])  # last few cells
    assert "openshell sandbox delete openshell-demo" in cleanup
    assert "rm -f policy-deny.yaml policy-allow.yaml" in cleanup
    assert "openshell sandbox list" in cleanup


def test_notebook_cites_canonical_sources(notebook: dict) -> None:
    sources = _cell_sources(notebook)
    last_md = sources[-1]
    assert "github.com/NVIDIA/OpenShell" in last_md
    assert "docs.nvidia.com/openshell" in last_md
    assert "docs.langchain.com" in last_md
