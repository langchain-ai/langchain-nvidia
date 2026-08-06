"""Test the installable Switchyard LangChain package contract."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import cast

from conftest import PACKAGE_ROOT, REPOSITORY_ROOT


def _poetry() -> dict[str, object]:
    config = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text())
    return cast(dict[str, object], config["tool"]["poetry"])


def test_package_metadata_declares_installable_public_contract() -> None:
    poetry = _poetry()

    assert poetry["name"] == "langchain-nvidia-switchyard"
    assert poetry["license"] == "MIT"
    assert poetry["readme"] == "README.md"
    assert poetry["packages"] == [{"include": "langchain_nvidia_switchyard"}]
    assert poetry["dependencies"] == {
        "python": ">=3.12,<4.0",
        "nemo-switchyard": ">=0.2.0",
        "langchain": ">=1.3.14,<2",
        "deepagents": {"version": ">=0.7.4,<0.8", "optional": True},
        "langchain-openrouter": {"version": ">=0.2.7,<0.3", "optional": True},
        "python-dotenv": {"version": ">=1,<2", "optional": True},
    }
    assert poetry["extras"] == {
        "deepagents": ["deepagents"],
        "openrouter": ["deepagents", "langchain-openrouter", "python-dotenv"],
    }


def test_package_carries_the_repository_mit_license() -> None:
    assert (PACKAGE_ROOT / "LICENSE").read_text() == (REPOSITORY_ROOT / "LICENSE").read_text()


def test_pytest_defaults_never_select_paid_e2e() -> None:
    config = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text())

    assert config["tool"]["pytest"]["ini_options"]["addopts"] == (
        '--strict-markers --strict-config --durations=5 -m "not e2e"'
    )
    assert config["tool"]["pytest"]["ini_options"]["markers"] == [
        "e2e: makes paid calls to real OpenRouter models",
        "compile: marks compile-only integration tests",
    ]


def test_env_example_contains_names_but_no_secret() -> None:
    assert (PACKAGE_ROOT / ".env.example").read_text() == (
        "OPENROUTER_API_KEY=\n"
        "OPENROUTER_EFFICIENT_MODEL=nvidia/nemotron-3-ultra-550b-a55b\n"
        "OPENROUTER_CAPABLE_MODEL=anthropic/claude-sonnet-4.6\n"
    )


def test_public_package_exports_only_the_two_adapters() -> None:
    import langchain_nvidia_switchyard

    assert langchain_nvidia_switchyard.__all__ == [
        "LangChainLlmClient",
        "SwitchyardRoutingMiddleware",
    ]


def test_package_root_is_the_expected_directory() -> None:
    assert Path(PACKAGE_ROOT).name == "switchyard"


def test_readme_installs_switchyard_from_the_canonical_source_repository() -> None:
    readme = (PACKAGE_ROOT / "README.md").read_text()

    assert "https://github.com/NVIDIA-NeMo/Switchyard" in readme
    assert "python -m pip install -e ./Switchyard" in readme


def test_poetry_lock_contains_switchyard_without_a_local_path_source() -> None:
    lock = (PACKAGE_ROOT / "poetry.lock").read_text()

    assert 'name = "nemo-switchyard"' in lock
    assert "../../../" not in lock
