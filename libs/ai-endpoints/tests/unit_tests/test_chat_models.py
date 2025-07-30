"""Test chat model integration."""

import warnings
from typing import Any

import pytest
from requests_mock import Mocker

from langchain_nvidia_ai_endpoints.chat_models import ChatNVIDIA


@pytest.fixture
def mock_local_models(requests_mock: Mocker) -> None:
    requests_mock.get(
        "http://localhost:8888/v1/models",
        json={
            "data": [
                {
                    "id": "unknown_model",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "OWNER",
                    "root": "unknown_model",
                },
            ]
        },
    )


def test_base_url_unknown_model(mock_local_models: None) -> None:
    llm = ChatNVIDIA(model="unknown_model", base_url="http://localhost:8888/v1")
    assert llm.model == "unknown_model"


def test_integration_initialization() -> None:
    """Test chat model initialization."""
    ChatNVIDIA(
        model="meta/llama2-70b",
        nvidia_api_key="nvapi-...",
        temperature=0.5,
        top_p=0.9,
        max_tokens=50,
    )
    ChatNVIDIA(model="meta/llama2-70b", nvidia_api_key="nvapi-...")


def test_unavailable(empty_v1_models: None) -> None:
    with pytest.warns(UserWarning, match="Model not-a-real-model is unknown"):
        ChatNVIDIA(api_key="BOGUS", model="not-a-real-model")


def test_max_tokens_deprecation_warning() -> None:
    """Test that using max_tokens raises a deprecation warning."""
    with pytest.warns(
        DeprecationWarning,
        match=(
            "The 'max_tokens' parameter is deprecated and will be removed "
            "in a future version"
        ),
    ):
        ChatNVIDIA(model="meta/llama2-70b", max_tokens=50)


def test_max_completion_tokens() -> None:
    """Test that max_completion_tokens works without warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        llm = ChatNVIDIA(
            model="meta/llama2-70b",
            max_completion_tokens=50,
            nvidia_api_key="nvapi-...",
        )
        assert len(w) == 0
        assert llm.max_tokens == 50
        payload = llm._get_payload(
            inputs=[{"role": "user", "content": "test"}],
            stop=None,
        )
        assert payload["max_tokens"] == 50


def test_max_tokens_value() -> None:
    """Test that max_tokens value is correctly set and reflected in payload."""
    llm = ChatNVIDIA(
        model="meta/llama2-70b",
        max_tokens=50,
        nvidia_api_key="nvapi-...",
    )
    assert llm.max_tokens == 50
    payload = llm._get_payload(
        inputs=[{"role": "user", "content": "test"}],
        stop=None,
    )
    assert payload["max_tokens"] == 50


@pytest.mark.parametrize(
    "thinking_mode",
    [False, True],
    ids=["thinking_off", "thinking_on"],
)
def test_payload_for_thinking_mode(requests_mock: Mocker, thinking_mode: bool) -> None:
    """Test that thinking mode correctly modifies the payload."""
    captured_requests = []

    def capture_request(request: Any, context: Any) -> dict:
        captured_requests.append(request.json())
        return {"choices": [{"message": {"role": "assistant", "content": "test"}}]}

    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        json=capture_request,
    )
    llm = ChatNVIDIA(
        model="nvidia/llama-3.1-nemotron-nano-8b-v1", api_key="BOGUS"
    ).with_thinking_mode(enabled=thinking_mode)
    llm.invoke("test message")

    messages = captured_requests[0]["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    if thinking_mode:
        assert messages[0]["content"] == "detailed thinking on"
    else:
        assert messages[0]["content"] == "detailed thinking off"


@pytest.mark.parametrize(
    "thinking_mode",
    [False, True],
    ids=["thinking_off", "thinking_on"],
)
def test_warning_for_thinking_mode_unsupported_model(thinking_mode: bool) -> None:
    """Test warning for thinking mode with a model that does not support it"""
    with pytest.warns(
        UserWarning,
        match="does not support thinking mode",
    ):
        ChatNVIDIA(
            model="meta/llama2-70b",
            nvidia_api_key="nvapi-...",
        ).with_thinking_mode(enabled=thinking_mode)


@pytest.mark.parametrize(
    "thinking_mode",
    [False, True],
    ids=["thinking_off", "thinking_on"],
)
def test_no_warning_for_thinking_mode_supported_model(thinking_mode: bool) -> None:
    """Test that no warning is raised for thinking mode with a supported model"""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ChatNVIDIA(
            model="nvidia/llama-3.1-nemotron-nano-8b-v1",
            nvidia_api_key="nvapi-...",
        ).with_thinking_mode(enabled=thinking_mode)
        assert len(w) == 0


@pytest.mark.parametrize(
    "verify_ssl,expected_verify_ssl",
    [
        (None, True),  # Default behavior
        (True, True),  # Explicit True
        (False, False),  # Explicit False
    ],
    ids=["default", "true", "false"],
)
def test_verify_ssl_behavior(
    verify_ssl: bool | None, expected_verify_ssl: bool
) -> None:
    """Test verify_ssl parameter behavior with different values."""
    kwargs = {
        "model": "meta/llama2-70b",
        "nvidia_api_key": "nvapi-...",
        "base_url": "https://example.com/v1",
    }
    if verify_ssl is not None:
        kwargs["verify_ssl"] = verify_ssl

    llm = ChatNVIDIA(**kwargs)

    # Test that session factory creates sessions with correct verify setting
    assert llm._client.get_session_fn().verify is expected_verify_ssl
