"""Test chat model integration."""


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
    with pytest.raises(ValueError):
        ChatNVIDIA(model="not-a-real-model")
