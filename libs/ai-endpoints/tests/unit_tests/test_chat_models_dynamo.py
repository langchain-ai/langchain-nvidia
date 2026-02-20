"""Tests for ChatNVIDIADynamo."""

from typing import Any

import pytest
from requests_mock import Mocker

from langchain_nvidia_ai_endpoints.chat_models_dynamo import ChatNVIDIADynamo


@pytest.fixture(autouse=True)
def mock_v1_models(requests_mock: Mocker) -> None:
    requests_mock.get(
        "https://integrate.api.nvidia.com/v1/models",
        json={
            "data": [
                {
                    "id": "mock-model",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "OWNER",
                }
            ]
        },
    )


def _make_llm(**kwargs: Any) -> ChatNVIDIADynamo:
    kwargs.setdefault("model", "mock-model")
    kwargs.setdefault("nvidia_api_key", "nvapi-test")
    return ChatNVIDIADynamo(**kwargs)


def _get_payload(llm: ChatNVIDIADynamo, **kwargs: Any) -> dict:
    kwargs.setdefault("stop", None)
    return llm._get_payload(
        inputs=[{"role": "user", "content": "hello"}],
        **kwargs,
    )


# --- Construction ---


def test_default_field_values() -> None:
    llm = _make_llm()
    assert llm.osl == 512
    assert llm.iat == 250
    assert llm.latency_sensitivity == 1.0
    assert llm.priority == 1


def test_custom_field_values() -> None:
    llm = _make_llm(osl=1024, iat=100, latency_sensitivity=0.5, priority=3)
    assert llm.osl == 1024
    assert llm.iat == 100
    assert llm.latency_sensitivity == 0.5
    assert llm.priority == 3


def test_llm_type() -> None:
    llm = _make_llm()
    assert llm._llm_type == "chat-nvidia-ai-playground-dynamo"


# --- Payload ---


def test_payload_has_nvext_agent_hints() -> None:
    llm = _make_llm()
    payload = _get_payload(llm)

    assert "nvext" in payload
    hints = payload["nvext"]["agent_hints"]
    assert hints["prefix_id"].startswith("langchain-dynamo-")
    assert hints["osl"] == 512
    assert hints["iat"] == 250
    assert hints["latency_sensitivity"] == 1.0
    assert hints["priority"] == 1


def test_prefix_id_auto_generated_per_request() -> None:
    llm = _make_llm()
    p1 = _get_payload(llm)
    p2 = _get_payload(llm)
    id1 = p1["nvext"]["agent_hints"]["prefix_id"]
    id2 = p2["nvext"]["agent_hints"]["prefix_id"]
    assert id1.startswith("langchain-dynamo-")
    assert id2.startswith("langchain-dynamo-")
    assert id1 != id2


def test_payload_all_fields() -> None:
    llm = _make_llm(osl=256, iat=100, latency_sensitivity=0.8, priority=5)
    payload = _get_payload(llm)
    hints = payload["nvext"]["agent_hints"]
    assert hints["osl"] == 256
    assert hints["iat"] == 100
    assert hints["latency_sensitivity"] == 0.8
    assert hints["priority"] == 5


def test_per_invocation_overrides() -> None:
    llm = _make_llm(osl=512, iat=250)
    payload = _get_payload(llm, osl=2048, iat=50)
    hints = payload["nvext"]["agent_hints"]
    assert hints["osl"] == 2048
    assert hints["iat"] == 50
    assert hints["latency_sensitivity"] == 1.0  # unchanged default
    assert hints["priority"] == 1  # unchanged default


def test_per_invocation_override_latency_and_priority() -> None:
    llm = _make_llm()
    payload = _get_payload(llm, latency_sensitivity=0.1, priority=10)
    hints = payload["nvext"]["agent_hints"]
    assert hints["latency_sensitivity"] == 0.1
    assert hints["priority"] == 10


def test_existing_nvext_keys_preserved() -> None:
    """Existing nvext keys like guided_json should not be clobbered."""
    llm = _make_llm()
    payload = _get_payload(llm, nvext={"guided_json": {"type": "object"}})
    assert payload["nvext"]["guided_json"] == {"type": "object"}
    assert "agent_hints" in payload["nvext"]


def test_dynamo_keys_not_in_base_payload() -> None:
    """Dynamo-specific keys should not leak into the base payload."""
    llm = _make_llm()
    payload = _get_payload(llm, osl=2048)
    for key in ("osl", "iat", "latency_sensitivity", "priority"):
        assert key not in payload


def test_messages_still_present() -> None:
    llm = _make_llm()
    payload = _get_payload(llm)
    assert "messages" in payload
    assert payload["messages"][0]["content"] == "hello"


# --- Streaming payload ---


def test_streaming_payload_has_agent_hints() -> None:
    llm = _make_llm()
    payload = llm._get_payload(
        inputs=[{"role": "user", "content": "hi"}],
        stop=None,
        stream=True,
        stream_options={"include_usage": True},
    )
    hints = payload["nvext"]["agent_hints"]
    assert hints["prefix_id"].startswith("langchain-dynamo-")
    assert hints["osl"] == 512
    assert hints["iat"] == 250
    assert hints["latency_sensitivity"] == 1.0
    assert hints["priority"] == 1
