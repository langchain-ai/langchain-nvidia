"""Test chat model integration."""


import warnings

import pytest

from langchain_nvidia_ai_endpoints._statics import MODEL_SPECS
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
    ChatNVIDIA(model="mistral", nvidia_api_key="nvapi-...")


@pytest.mark.parametrize(
    "model",
    [
        name
        for pair in [
            (model, model.replace("playground_", ""))
            for model, config in MODEL_SPECS.items()
            if "api_type" in config and config["api_type"] == "aifm"
        ]
        for name in pair
    ],
)
def test_aifm_deprecated(model: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ChatNVIDIA()
    with pytest.deprecated_call():
        ChatNVIDIA(model=model)


def test_param_bad_deprecated() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ChatNVIDIA()
    with pytest.deprecated_call():
        ChatNVIDIA(bad=["bad"])


def test_param_labels_deprecated() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ChatNVIDIA()
    with pytest.deprecated_call():
        ChatNVIDIA(labels={"label": 1.0})
