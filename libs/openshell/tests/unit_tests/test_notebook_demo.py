"""Validate the real Deep Agent pattern shown in the demo notebook.

The notebook (`libs/openshell/docs/sandboxes/nvidia_openshell_sandbox.ipynb`)
demonstrates a live `deepagents.create_deep_agent(...)` agent with
`backend=OpenShellSandbox(...)`. The Deep Agent uses its built-in `execute`
tool to run commands inside OpenShell and infer which requests were allowed or
blocked by policy.

Unit tests do not call the live LLM or OpenShell gateway. They keep the
notebook structure honest so the tutorial cannot drift back into a deterministic
host-side runner.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def notebook() -> dict:
    nb_path = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "sandboxes"
        / "nvidia_openshell_sandbox.ipynb"
    )
    return json.loads(nb_path.read_text())


def test_notebook_is_valid_json_and_nbformat4(notebook: dict) -> None:
    assert notebook["nbformat"] == 4
    assert isinstance(notebook["cells"], list)
    assert notebook["cells"]


def _cell_sources(notebook: dict) -> list[str]:
    return ["".join(c["source"]) for c in notebook["cells"]]


def test_notebook_code_cells_compile(notebook: dict) -> None:
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if any(
            line.lstrip().startswith(("%%", "!", "%")) for line in source.splitlines()
        ):
            continue
        compile(source, f"notebook-cell-{index}", "exec")


def test_notebook_has_both_policy_files(notebook: dict) -> None:
    sources = _cell_sources(notebook)
    assert any(
        "%%writefile policy-github.yaml" in s for s in sources
    ), "expected a GitHub-only policy cell"
    assert any(
        "%%writefile policy-expanded.yaml" in s for s in sources
    ), "expected an expanded GitHub + PyPI policy cell"


def test_notebook_uses_real_deep_agent_with_openshell_backend(notebook: dict) -> None:
    sources = _cell_sources(notebook)
    joined = "\n".join(sources)

    assert "from deepagents import create_deep_agent" in joined
    assert "from langchain_nvidia_ai_endpoints import ChatNVIDIA" in joined
    assert "OpenShellSandbox(sandbox=client.get_session" in joined
    assert "agent = create_deep_agent(" in joined
    assert "backend=backend" in joined
    assert "agent.invoke(" in joined

    assert "make_policy_scout_tools" not in joined
    assert "run_policy_scout_agent" not in joined
    assert "from langchain_core.tools import tool" not in joined


def test_notebook_prompts_agent_to_call_execute_for_each_policy_check(
    notebook: dict,
) -> None:
    joined = "\n".join(_cell_sources(notebook))
    assert "Use the `execute` tool for every command" in joined
    assert "Do not combine commands; call `execute` once per listed command" in joined
    assert "github_zen" in joined
    assert "github_repo_summary" in joined
    assert "pypi_openshell_version" in joined
    assert "external_ip_probe" in joined


def test_notebook_displays_execute_trace_and_agent_audit(notebook: dict) -> None:
    joined = "\n".join(_cell_sources(notebook))
    assert "_execute_trace_markdown(result)" in joined
    assert "_final_message(result)" in joined
    assert 'display(Markdown("### Execute Tool Trace\\n"' in joined
    assert 'display(Markdown("### Agent Audit\\n"' in joined


def test_notebook_requires_nvidia_api_key(notebook: dict) -> None:
    joined = "\n".join(_cell_sources(notebook))
    assert "NVIDIA_API_KEY" in joined
    assert "getpass.getpass" in joined
    assert "ready to run the Deep Agent" in joined


def test_notebook_uses_current_openshell_pin(notebook: dict) -> None:
    joined = "\n".join(_cell_sources(notebook))
    assert "./setup_openshell.sh --openshell-version 0.0.72" in joined
    assert "langchain-nvidia-openshell-demo:0.0.72" in joined
    assert "grpcio" in joined
    assert "0.0.57" not in joined
    assert "0.0.40" not in joined
    assert "0.0.39" not in joined


def test_notebook_uses_nemotron_default_model(notebook: dict) -> None:
    joined = "\n".join(_cell_sources(notebook))
    assert 'NVIDIA_DEEP_AGENT_MODEL", "nvidia/nemotron-3-nano-30b-a3b"' in joined
    assert "openai/gpt-oss-120b" not in joined


def test_notebook_explains_named_sandbox_lifecycle(notebook: dict) -> None:
    joined = "\n".join(_cell_sources(notebook))
    assert "Create a named OpenShell sandbox with a policy" in joined
    assert 'SandboxClient.get_session("openshell-demo")' in joined
    assert "OpenShellSandbox` does not create or delete this sandbox" in joined


def test_notebook_setup_uses_bash_script(notebook: dict) -> None:
    joined = "\n".join(_cell_sources(notebook))
    assert "setup_openshell.sh" in joined
    assert "Poetry environment" in joined
    assert "Python SDK wheel compatibility" in joined


def test_notebook_cleans_up_at_the_end(notebook: dict) -> None:
    sources = _cell_sources(notebook)
    cleanup = "\n".join(sources[-3:])  # last few cells
    assert "openshell sandbox delete openshell-demo" in cleanup
    assert "rm -f policy-github.yaml policy-expanded.yaml" in cleanup
    assert "openshell sandbox list" in cleanup


def test_notebook_cites_canonical_sources(notebook: dict) -> None:
    sources = _cell_sources(notebook)
    last_md = sources[-1]
    assert "github.com/NVIDIA/OpenShell" in last_md
    assert "docs.nvidia.com/openshell" in last_md
    assert "docs.langchain.com" in last_md
