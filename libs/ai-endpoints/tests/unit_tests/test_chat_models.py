"""Test chat model integration."""

import warnings
from typing import Any, Optional, Union

import pytest
from requests_mock import Mocker

from langchain_nvidia_ai_endpoints.chat_models import ChatNVIDIA

from .conftest import MockHTTP


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
async def test_async_payload_for_thinking_mode(
    thinking_mode: bool, mock_http: MockHTTP
) -> None:
    """Async parity of thinking mode payload injection for ainvoke."""
    mock_http.set_post(
        json_body={"choices": [{"message": {"role": "assistant", "content": "test"}}]}
    )

    llm = ChatNVIDIA(
        model="nvidia/llama-3.1-nemotron-nano-8b-v1", api_key="BOGUS"
    ).with_thinking_mode(enabled=thinking_mode)
    await llm.ainvoke("test message")

    request_payload = mock_http.aio.post.call_args.kwargs.get("json", {})
    messages = request_payload["messages"]
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
    "thinking_mode",
    [False, True],
    ids=["thinking_off", "thinking_on"],
)
def test_thinking_prefix_appends_to_existing_system_message(
    requests_mock: Mocker, thinking_mode: bool
) -> None:
    """Test thinking prefix appends to existing system message."""
    captured_requests = []

    def capture_request(request: Any, context: Any) -> dict:
        captured_requests.append(request.json())
        return {"choices": [{"message": {"role": "assistant", "content": "response"}}]}

    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        json=capture_request,
    )

    llm = ChatNVIDIA(
        model="nvidia/llama-3.1-nemotron-nano-8b-v1", api_key="BOGUS"
    ).with_thinking_mode(enabled=thinking_mode)

    # Invoke with a system message
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hello"),
    ]
    llm.invoke(messages)

    # Check the captured request
    request_messages = captured_requests[0]["messages"]

    # Should still have 2 messages (system and user)
    assert len(request_messages) == 2
    assert request_messages[0]["role"] == "system"
    assert request_messages[1]["role"] == "user"

    # System message should have the original content + thinking prefix
    expected_prefix = (
        "detailed thinking on" if thinking_mode else "detailed thinking off"
    )
    expected_content = f"You are a helpful assistant.\n{expected_prefix}"
    assert request_messages[0]["content"] == expected_content


@pytest.mark.parametrize(
    "model_name,expected_on,expected_off",
    [
        # Detailed thinking models (use "detailed thinking on/off")
        (
            "nvidia/llama-3.1-nemotron-nano-8b-v1",
            "detailed thinking on",
            "detailed thinking off",
        ),
        (
            "nvidia/llama-3.1-nemotron-nano-4b-v1.1",
            "detailed thinking on",
            "detailed thinking off",
        ),
        (
            "nvidia/llama-3.1-nemotron-ultra-253b-v1",
            "detailed thinking on",
            "detailed thinking off",
        ),
        (
            "nvidia/llama-3.3-nemotron-super-49b-v1",
            "detailed thinking on",
            "detailed thinking off",
        ),
        # /think models (use "/think" and "/no_think")
        ("nvidia/llama-3.3-nemotron-super-49b-v1.5", "/think", "/no_think"),
        ("nvidia/nvidia-nemotron-nano-9b-v2", "/think", "/no_think"),
    ],
)
@pytest.mark.parametrize(
    "thinking_mode",
    [False, True],
    ids=["thinking_off", "thinking_on"],
)
def test_different_thinking_prefixes_for_different_models(
    requests_mock: Mocker,
    model_name: str,
    expected_on: str,
    expected_off: str,
    thinking_mode: bool,
) -> None:
    """Test different models use correct thinking prefixes."""
    captured_requests = []

    def capture_request(request: Any, context: Any) -> dict:
        captured_requests.append(request.json())
        return {"choices": [{"message": {"role": "assistant", "content": "response"}}]}

    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        json=capture_request,
    )

    llm = ChatNVIDIA(model=model_name, api_key="BOGUS").with_thinking_mode(
        enabled=thinking_mode
    )
    llm.invoke("test message")

    messages = captured_requests[0]["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "system"

    expected_content = expected_on if thinking_mode else expected_off
    assert messages[0]["content"] == expected_content


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


def test_model_kwargs_extra_parameters() -> None:
    """Test that extra parameters are captured in model_kwargs and included
    in payload."""
    # Test 1: Extra parameters via constructor kwargs
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        llm = ChatNVIDIA(
            model="meta/llama3-70b-instruct",
            nvidia_api_key="nvapi-...",
            custom_param="custom_value",
        )
        # Should get warning about non-default parameter
        assert len(w) == 1
        assert "custom_param" in str(w[0].message)

    # Check that extra param is stored in model_kwargs
    assert "custom_param" in llm.model_kwargs
    assert llm.model_kwargs["custom_param"] == "custom_value"

    # Check that custom param appears in payload
    payload = llm._get_payload(
        inputs=[{"role": "user", "content": "test"}],
        stop=None,
    )
    assert payload["custom_param"] == "custom_value"

    # Test 2: Explicit model_kwargs parameter alongside explicit params
    llm2 = ChatNVIDIA(
        model="meta/llama3-70b-instruct",
        nvidia_api_key="nvapi-...",
        temperature=0.5,
        model_kwargs={"top_k": 10, "custom_param": "from_model_kwargs"},
    )

    # Explicit temperature parameter is set
    assert llm2.temperature == 0.5
    assert "top_k" in llm2.model_kwargs
    assert llm2.model_kwargs["top_k"] == 10

    # Both explicit and model_kwargs parameters appear in payload
    payload2 = llm2._get_payload(
        inputs=[{"role": "user", "content": "test"}],
        stop=None,
    )
    assert payload2["temperature"] == 0.5
    assert payload2["top_k"] == 10
    assert payload2["custom_param"] == "from_model_kwargs"

    # Test 3: Invoke-time kwargs override explicit parameters and model_kwargs
    payload3 = llm2._get_payload(
        inputs=[{"role": "user", "content": "test"}],
        stop=None,
        temperature=0.3,
        top_k=20,
        custom_param="from_invoke",
    )
    assert payload3["temperature"] == 0.3
    assert payload3["top_k"] == 20
    assert payload3["custom_param"] == "from_invoke"
