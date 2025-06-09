"""Test chat model integration."""

import warnings

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
