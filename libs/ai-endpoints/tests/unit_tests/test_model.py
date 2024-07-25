from itertools import chain
from typing import Any

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
        assert x.model == x._client.model
    assert isinstance(record[0].message, Warning)
    assert "deprecated" in record[0].message.args[0]


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
    assert isinstance(record[0].message, Warning)
    assert "Found" in record[0].message.args[0]
    assert "unknown" in record[0].message.args[0]


def test_unknown_unknown(public_class: type) -> None:
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
    with pytest.warns(UserWarning):
        x = public_class(base_url="http://localhost:8000/v1")
        assert x.model == known_unknown


def test_default_lora(public_class: type) -> None:
    """
    Test that a model in the model table will be accepted.
    """
    # find a model that matches the public_class under test
    x = public_class(base_url="http://localhost:8000/v1", model="lora1")
    assert x.model == "lora1"
