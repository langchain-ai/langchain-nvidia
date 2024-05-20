from typing import Any

import pytest

from langchain_nvidia_ai_endpoints._statics import MODEL_TABLE, determine_model


@pytest.fixture(params=MODEL_TABLE.keys())
def entry(request: Any) -> str:
    return request.param


def test_model_table_integrity_name_id(entry: str) -> None:
    model = MODEL_TABLE[entry]
    assert model.id == entry


def test_model_table_integrity_deprecated_alternative(entry: str) -> None:
    model = MODEL_TABLE[entry]
    if model.deprecated:
        # model_name is an optional alternative
        if model.model_name:
            assert model.model_name in MODEL_TABLE


def test_model_table_integrity_playground_aliases(entry: str) -> None:
    model = MODEL_TABLE[entry]
    if "playground_" in model.id:
        assert model.aliases
        assert model.id.replace("playground_", "") in model.aliases


def test_determine_model_deprecated_alternative_warns(entry: str) -> None:
    model = MODEL_TABLE[entry]
    if model.deprecated and model.model_name:
        with pytest.warns(UserWarning):
            determine_model(model.id)
