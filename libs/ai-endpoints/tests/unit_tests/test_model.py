import warnings
from itertools import chain
from typing import Any, List

import pytest
from requests_mock import Mocker

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
from langchain_nvidia_ai_endpoints._statics import (
    CHAT_MODEL_TABLE,
    EMBEDDING_MODEL_TABLE,
    MODEL_TABLE,
    QA_MODEL_TABLE,
    RANKING_MODEL_TABLE,
    VLM_MODEL_TABLE,
)


@pytest.fixture
def known_unknown() -> str:
    return "mock-model"


@pytest.fixture(autouse=True)
def mock_v1_models(requests_mock: Mocker, known_unknown: str) -> None:
    requests_mock.get(
        "https://integrate.api.nvidia.com/v1/models",
        json={
            "data": [
                {
                    "id": known_unknown,
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "OWNER",
                },
            ]
        },
    )


@pytest.fixture(autouse=True)
def mock_v1_local_models(requests_mock: Mocker, known_unknown: str) -> None:
    requests_mock.get(
        "http://localhost:8000/v1/models",
        json={
            "data": [
                {
                    "id": known_unknown,
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "OWNER",
                    "root": known_unknown,
                },
                {
                    "id": "lora1",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "OWNER",
                    "root": known_unknown,
                },
            ]
        },
    )


@pytest.mark.parametrize(
    "alias, client",
    [
        (alias, ChatNVIDIA)
        for model in list(
            chain(
                CHAT_MODEL_TABLE.values(),
                VLM_MODEL_TABLE.values(),
                QA_MODEL_TABLE.values(),
            )
        )
        if model.aliases is not None
        for alias in model.aliases
    ]
    + [
        (alias, NVIDIAEmbeddings)
        for model in EMBEDDING_MODEL_TABLE.values()
        if model.aliases is not None
        for alias in model.aliases
    ]
    + [
        (alias, NVIDIARerank)
        for model in RANKING_MODEL_TABLE.values()
        if model.aliases is not None
        for alias in model.aliases
    ],
)
def test_aliases(alias: str, client: Any) -> None:
    """
    Test that the aliases for each model in the model table are accepted
    with a warning about deprecation of the alias.
    """
    with pytest.warns(UserWarning) as record:
        x = client(model=alias, nvidia_api_key="a-bogus-key")
        assert x.model == x._client.mdl_name
    record_list: List[warnings.WarningMessage] = list(record)
    assert isinstance(record_list[0].message, Warning)
    assert "deprecated" in record_list[0].message.args[0]


def test_known(public_class: type) -> None:
    """
    Test that a model in the model table will be accepted.
    """
    # find a model that matches the public_class under test
    known = None
    for model in MODEL_TABLE.values():
        if model.client == public_class.__name__:
            known = model.id
            break
    assert known is not None, f"Model not found for client {public_class.__name__}"
    x = public_class(model=known, nvidia_api_key="a-bogus-key")
    assert x.model == known


def test_known_unknown(public_class: type, known_unknown: str) -> None:
    """
    Test that a model in /v1/models but not in the model table will be accepted
    with a warning.
    """
    with pytest.warns(UserWarning) as record:
        x = public_class(model=known_unknown, nvidia_api_key="a-bogus-key")
        assert x.model == known_unknown
    record_list: List[warnings.WarningMessage] = list(record)
    assert isinstance(record_list[0].message, Warning)
    assert "Found" in record_list[0].message.args[0]
    assert "unknown" in record_list[0].message.args[0]


def test_unknown_unknown(public_class: type, empty_v1_models: None) -> None:
    """
    Test that a model not in /v1/models and not in known model table will be
    rejected.
    """
    # todo: make this work for local NIM
    with pytest.raises(ValueError) as e:
        public_class(model="test/unknown-unknown", nvidia_api_key="a-bogus-key")
    assert "unknown" in str(e.value)


def test_default_known(public_class: type, known_unknown: str) -> None:
    """
    Test that a model in the model table will be accepted.
    """
    # check if default model is getting set
    with pytest.warns(UserWarning) as record:
        x = public_class(base_url="http://localhost:8000/v1")
        assert x.model == known_unknown
    record_list: List[warnings.WarningMessage] = list(record)
    assert "Default model is set as: mock-model" in str(record_list[0].message)


def test_default_lora(public_class: type) -> None:
    """
    Test that a model in the model table will be accepted.
    """
    # find a model that matches the public_class under test
    x = public_class(base_url="http://localhost:8000/v1", model="lora1")
    assert x.model == "lora1"


def test_default(public_class: type) -> None:
    x = public_class(api_key="BOGUS")
    assert x.model is not None


@pytest.mark.parametrize(
    "model, client",
    [(model.id, model.client) for model in MODEL_TABLE.values()],
)
def test_all_incompatible(public_class: type, model: str, client: str) -> None:
    if client == public_class.__name__:
        pytest.skip("Compatibility expected.")

    with pytest.raises(ValueError) as err_msg:
        public_class(model=model, nvidia_api_key="a-bogus-key")
    assert f"Model {model} is incompatible with client {public_class.__name__}" in str(
        err_msg.value
    )
