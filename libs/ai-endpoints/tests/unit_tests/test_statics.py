import warnings
from typing import Any, List

import pytest

from langchain_nvidia_ai_endpoints._statics import MODEL_TABLE, determine_model


@pytest.fixture(params=MODEL_TABLE.keys())
def entry(request: Any) -> str:
    return request.param


@pytest.fixture(
    params=[
        alias
        for ls in [model.aliases for model in MODEL_TABLE.values() if model.aliases]
        for alias in ls
    ]
)
def alias(request: Any) -> str:
    return request.param


def test_model_table_integrity_name_id(entry: str) -> None:
    model = MODEL_TABLE[entry]
    assert model.id == entry


def test_determine_model_deprecated_alternative_warns(alias: str) -> None:
    with pytest.warns(UserWarning) as record:
        determine_model(alias)
    record_list: List[warnings.WarningMessage] = list(record)
    assert len(record_list) == 1
    assert f"Model {alias} is deprecated" in str(record_list[0].message)
