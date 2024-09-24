import warnings
from typing import Any, Generator

import pytest

from langchain_nvidia_ai_endpoints import Model, register_model
from langchain_nvidia_ai_endpoints._statics import MODEL_TABLE


def test_model_listing(public_class: Any, mock_model: str) -> None:
    warnings.filterwarnings("ignore", message=f"Default model is set as: {mock_model}")
    # we set base_url to avoid having results filtered by the public_class name
    models = public_class.get_available_models(base_url="https://mock/v1")
    assert any(model.id == mock_model for model in models)


@pytest.fixture
def model_table() -> Generator[None, None, None]:
    """
    Reset MODEL_TABLE between tests.
    """
    original = MODEL_TABLE.copy()
    yield
    MODEL_TABLE.clear()
    MODEL_TABLE.update(original)


def test_model_listing_hosted(
    public_class: Any, mock_model: str, model_table: None
) -> None:
    model = Model(
        id=mock_model,
        model_type={
            "ChatNVIDIA": "chat",
            "NVIDIAEmbeddings": "embedding",
            "NVIDIARerank": "ranking",
            "NVIDIA": "completions",
        }[public_class.__name__],
        client=public_class.__name__,
        endpoint="BOGUS",
    )
    register_model(model)
    models = public_class.get_available_models()
    assert any(model.id == mock_model for model in models)
