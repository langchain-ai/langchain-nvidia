import os
from contextlib import contextmanager
from typing import Any, Generator

import pytest
from langchain_core.pydantic_v1 import SecretStr
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
def mock_v1_local_models(requests_mock: Mocker) -> None:
    requests_mock.get(
        "https://test_url/v1/models",
        json={
            "data": [
                {
                    "id": "model1",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "OWNER",
                    "root": "model1",
                },
            ]
        },
    )


def test_create_without_api_key(public_class: type) -> None:
    with no_env_var("NVIDIA_API_KEY"):
        with pytest.warns(UserWarning):
            public_class()


def test_create_unknown_url_no_api_key(public_class: type) -> None:
    with no_env_var("NVIDIA_API_KEY") and pytest.warns(UserWarning):
        public_class(base_url="https://test_url/v1")


@pytest.mark.parametrize("param", ["nvidia_api_key", "api_key"])
def test_create_with_api_key(public_class: type, param: str) -> None:
    with no_env_var("NVIDIA_API_KEY"):
        public_class(**{param: "just testing no failure"})


def test_api_key_priority(public_class: type) -> None:
    def get_api_key(instance: Any) -> str:
        return instance._client.client.api_key.get_secret_value()

    with no_env_var("NVIDIA_API_KEY"):
        os.environ["NVIDIA_API_KEY"] = "ENV"
        assert get_api_key(public_class()) == "ENV"
        assert get_api_key(public_class(nvidia_api_key="PARAM")) == "PARAM"
        assert get_api_key(public_class(api_key="PARAM")) == "PARAM"
        assert get_api_key(public_class(api_key="LOW", nvidia_api_key="HIGH")) == "HIGH"


def test_api_key_type(public_class: type) -> None:
    # Test case to make sure the api_key is SecretStr and not str
    def get_api_key(instance: Any) -> str:
        return instance._client.client.api_key

    with no_env_var("NVIDIA_API_KEY"):
        os.environ["NVIDIA_API_KEY"] = "ENV"
        assert type(get_api_key(public_class())) == SecretStr
