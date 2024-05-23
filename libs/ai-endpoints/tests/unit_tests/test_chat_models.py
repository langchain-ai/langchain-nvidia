"""Test chat model integration."""


import warnings

import pytest

from langchain_nvidia_ai_endpoints._statics import MODEL_TABLE
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


@pytest.mark.parametrize(
    "model",
    [
        name
        for ls in [
            [model.id] + (model.aliases or [])  # alises can be None
            for model in MODEL_TABLE.values()
            if model.deprecated and model.model_name
        ]
        for name in ls
    ],
)
def test_deprecated(model: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ChatNVIDIA()
    with pytest.warns(UserWarning):
        ChatNVIDIA(model=model)


@pytest.mark.parametrize(
    "model",
    [
        name
        for ls in [
            [model.id] + (model.aliases or [])  # alises can be None
            for model in MODEL_TABLE.values()
            if model.deprecated and not model.model_name
        ]
        for name in ls
    ],
)
def test_unavailable(model: str) -> None:
    with pytest.raises(ValueError):
        ChatNVIDIA(model=model)


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
