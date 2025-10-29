"""Test chat model integration."""

import warnings
from typing import Any, Optional, Union

import pytest
from requests_mock import Mocker

from langchain_nvidia_ai_endpoints.chat_models import ChatNVIDIA


@pytest.fixture(autouse=True)
def mock_v1_models(requests_mock: Mocker) -> None:
    requests_mock.get(
        "https://integrate.api.nvidia.com/v1/models",
        json={
            "data": [
                {
                    "id": "meta/llama3-8b-instruct",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "OWNER",
                }
            ]
        },
    )


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
        model="meta/llama3-70b-instruct",
        nvidia_api_key="nvapi-...",
        temperature=0.5,
        top_p=0.9,
        max_tokens=50,
    )
    ChatNVIDIA(model="meta/llama3-70b-instruct", nvidia_api_key="nvapi-...")


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
        ChatNVIDIA(model="meta/llama3-70b-instruct", max_tokens=50)


def test_max_completion_tokens() -> None:
    """Test that max_completion_tokens works without warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        llm = ChatNVIDIA(
            model="meta/llama3-70b-instruct",
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
        model="meta/llama3-70b-instruct",
        max_tokens=50,
        nvidia_api_key="nvapi-...",
    )
    assert llm.max_tokens == 50
    payload = llm._get_payload(
        inputs=[{"role": "user", "content": "test"}],
        stop=None,
    )
    assert payload["max_tokens"] == 50


def test_min_tokens_parameter() -> None:
    """Test that min_tokens parameter works correctly."""
    llm = ChatNVIDIA(
        model="meta/llama2-70b",
        min_tokens=10,
        nvidia_api_key="nvapi-...",
    )
    assert llm.min_tokens == 10
    payload = llm._get_payload(
        inputs=[{"role": "user", "content": "test"}],
        stop=None,
    )
    assert payload["min_tokens"] == 10


def test_ignore_eos_parameter() -> None:
    """Test that ignore_eos parameter works correctly."""
    llm = ChatNVIDIA(
        model="meta/llama2-70b",
        ignore_eos=True,
        nvidia_api_key="nvapi-...",
    )
    assert llm.ignore_eos is True
    payload = llm._get_payload(
        inputs=[{"role": "user", "content": "test"}],
        stop=None,
    )
    assert payload["ignore_eos"] is True


def test_optional_parameters_default_values() -> None:
    """Test that optional parameters have correct default values
    and payload behavior."""
    llm = ChatNVIDIA(
        model="meta/llama2-70b",
        nvidia_api_key="nvapi-...",
    )
    # Parameters that default to None
    assert llm.temperature is None
    assert llm.top_p is None
    assert llm.seed is None
    assert llm.stop is None
    assert llm.min_tokens is None
    assert llm.ignore_eos is None

    # Parameters that have non-None default values
    assert llm.max_tokens == 1024
    assert llm.stream_options == {"include_usage": True}

    payload = llm._get_payload(
        inputs=[{"role": "user", "content": "test"}],
        stop=None,
    )
    assert "temperature" not in payload
    assert "top_p" not in payload
    assert "seed" not in payload
    assert "stop" not in payload
    assert "min_tokens" not in payload
    assert "ignore_eos" not in payload

    assert "max_tokens" in payload
    assert payload["max_tokens"] == 1024


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
            model="meta/llama3-70b-instruct",
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
        ("/path/to/ca.pem", "/path/to/ca.pem"),  # CA certificate path
    ],
    ids=["default", "true", "false", "ca_path"],
)
def test_verify_ssl_behavior(
    verify_ssl: Optional[Union[bool, str]], expected_verify_ssl: Union[bool, str]
) -> None:
    """Test verify_ssl parameter behavior with different values."""
    kwargs: dict[str, Any] = {
        "model": "meta/llama3-70b-instruct",
        "nvidia_api_key": "nvapi-...",
        "base_url": "https://example.com/v1",
    }
    if verify_ssl is not None:
        kwargs["verify_ssl"] = verify_ssl

    llm = ChatNVIDIA(**kwargs)

    # Test that session factory creates sessions with correct verify setting
    assert llm._client.get_session_fn().verify is expected_verify_ssl


def test_default_headers(requests_mock: Mocker) -> None:
    """Test that default_headers are passed to requests."""
    model = "meta/llama3-8b-instruct"
    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )

    llm = ChatNVIDIA(
        model=model,
        nvidia_api_key="a-bogus-key",
        default_headers={"X-Test": "test"},
    )
    assert llm.default_headers == {"X-Test": "test"}

    _ = llm.invoke("Hello")
    assert requests_mock.last_request is not None
    assert requests_mock.last_request.headers["X-Test"] == "test"
