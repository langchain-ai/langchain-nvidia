"""Test safety checks and paid OpenRouter E2E for the runnable example."""

from __future__ import annotations

import os

import pytest
from conftest import REPOSITORY_ROOT
from langchain_core.messages import AIMessage

import example


def test_repository_env_path_targets_the_checkout_root() -> None:
    assert example.repository_env_path() == REPOSITORY_ROOT / ".env"


def test_create_models_uses_nemotron_as_the_default_efficient_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("OPENROUTER_EFFICIENT_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_CAPABLE_MODEL", raising=False)

    efficient, capable = example._create_models()

    assert efficient.model_name == "nvidia/nemotron-3-ultra-550b-a55b"
    assert capable.model_name == "anthropic/claude-sonnet-4.6"


async def test_demo_requires_explicit_spend_opt_in_before_model_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(example, "load_repository_environment", lambda: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("SWITCHYARD_LANGCHAIN_E2E", raising=False)

    def fail_if_models_are_constructed() -> object:
        pytest.fail("provider models must not be constructed without explicit spend opt-in")

    monkeypatch.setattr(example, "_create_models", fail_if_models_are_constructed)

    with pytest.raises(RuntimeError, match="SWITCHYARD_LANGCHAIN_E2E=1"):
        await example.run_demo()


async def test_demo_requires_key_before_model_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(example, "load_repository_environment", lambda: None)
    monkeypatch.setenv("SWITCHYARD_LANGCHAIN_E2E", "1")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def fail_if_models_are_constructed() -> object:
        pytest.fail("provider models must not be constructed without OPENROUTER_API_KEY")

    monkeypatch.setattr(example, "_create_models", fail_if_models_are_constructed)

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        await example.run_demo()


def test_demo_result_preserves_validated_structured_response() -> None:
    structured = example.RoutingCheck(
        status="ok",
        message="Structured routing confirmed.",
    )
    result = {
        "messages": [
            AIMessage(
                '{"status":"ok","message":"Structured routing confirmed."}',
                response_metadata={"switchyard": {"selected_model": "efficient"}},
            )
        ],
        "structured_response": structured,
    }

    demo = example._demo_result("structured", result)

    assert demo.structured_response == structured
    assert demo.selected_model == "efficient"


@pytest.mark.e2e
async def test_paid_deep_agent_routes_both_openrouter_models() -> None:
    example.load_repository_environment()
    if os.environ.get("SWITCHYARD_LANGCHAIN_E2E") != "1":
        pytest.skip("SWITCHYARD_LANGCHAIN_E2E=1 is required for paid E2E")
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY is required for paid E2E")

    results = await example.run_demo()

    assert [result.case for result in results] == ["simple", "failed-tool"]
    assert [result.selected_model for result in results] == ["efficient", "capable"]
    assert all(result.text.strip() for result in results)
    assert results[0].structured_response is not None
    assert results[0].structured_response.status == "ok"
    assert results[1].structured_response is None
