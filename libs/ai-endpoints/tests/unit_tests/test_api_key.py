import os
import warnings
from contextlib import contextmanager
from typing import Any, Generator, List

import pytest
from pydantic import SecretStr
from requests_mock import Mocker


@contextmanager
def no_env_var(var: str) -> Generator[None, None, None]:
    try:
        if val := os.environ.get(var, None):
            del os.environ[var]
        yield
    finally:
        if val:
            os.environ[var] = val
        else:
            if var in os.environ:
                del os.environ[var]


@pytest.fixture(autouse=True)
def mock_endpoint_models(requests_mock: Mocker) -> None:
    requests_mock.get(
        "https://integrate.api.nvidia.com/v1/models",
        json={
            "data": [
                {
                    "id": "meta/llama3-8b-instruct",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "OWNER",
                    "root": "model1",
                },
            ]
        },
    )


@pytest.fixture(autouse=True)
def mock_v1_local_models(requests_mock: Mocker) -> None:
    requests_mock.get(
        "https://test_url/v1/models",
        json={
            "data": [
                {
                    "id": "model",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "OWNER",
                    "root": "model",
                },
            ]
        },
    )


def test_create_without_api_key(public_class: type) -> None:
    with no_env_var("NVIDIA_API_KEY"):
        with pytest.warns(UserWarning) as record:
            public_class()
    record_list: List[warnings.WarningMessage] = list(record)
    assert len(record_list) == 1
    assert "API key is required for the hosted" in str(record_list[0].message)


def test_create_unknown_url_no_api_key(public_class: type) -> None:
    with no_env_var("NVIDIA_API_KEY"):
        with pytest.warns(UserWarning) as record:
            public_class(base_url="https://test_url/v1")
    record_list: List[warnings.WarningMessage] = list(record)
    assert len(record_list) == 1
    assert "Default model is set as" in str(record_list[0].message)


@pytest.mark.parametrize("param", ["nvidia_api_key", "api_key"])
def test_create_with_api_key(public_class: type, param: str) -> None:
    with no_env_var("NVIDIA_API_KEY"):
        public_class(**{param: "just testing no failure"})


def test_api_key_priority(public_class: type) -> None:
    def get_api_key(instance: Any) -> str:
        return instance._client.api_key.get_secret_value()

    with no_env_var("NVIDIA_API_KEY"):
        os.environ["NVIDIA_API_KEY"] = "ENV"
        assert get_api_key(public_class()) == "ENV"
        assert get_api_key(public_class(nvidia_api_key="PARAM")) == "PARAM"
        assert get_api_key(public_class(api_key="PARAM")) == "PARAM"
        assert get_api_key(public_class(api_key="LOW", nvidia_api_key="HIGH")) == "HIGH"


def test_api_key_type(public_class: type) -> None:
    # Test case to make sure the api_key is SecretStr and not str
    def get_api_key(instance: Any) -> str:
        return instance._client.api_key

    with no_env_var("NVIDIA_API_KEY"):
        os.environ["NVIDIA_API_KEY"] = "ENV"
        assert type(get_api_key(public_class())) == SecretStr
