"""Test chat model integration."""


import pytest

from langchain_nvidia_ai_endpoints.chat_models import ChatNVIDIA


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


def test_unavailable() -> None:
    with pytest.raises(ValueError):
        ChatNVIDIA(model="not-a-real-model")


@pytest.mark.parametrize(
    "base_url",
    [
        "bogus",
        "http:/",
        "http://",
        "http:/oops",
    ],
)
def test_param_base_url_negative(base_url: str) -> None:
    with pytest.raises(ValueError):
        ChatNVIDIA(base_url=base_url)
