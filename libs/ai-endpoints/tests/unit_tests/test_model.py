import pytest
from requests_mock import Mocker

from langchain_nvidia_ai_endpoints._statics import MODEL_TABLE


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


@pytest.mark.parametrize(
    "alias",
    [
        alias
        for model in MODEL_TABLE.values()
        if model.aliases is not None
        for alias in model.aliases
    ],
)
def test_aliases(public_class: type, alias: str) -> None:
    """
    Test that the aliases for each model in the model table are accepted
    with a warning about deprecation of the alias.
    """
    with pytest.warns(UserWarning) as record:
        x = public_class(model=alias, nvidia_api_key="a-bogus-key")
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


