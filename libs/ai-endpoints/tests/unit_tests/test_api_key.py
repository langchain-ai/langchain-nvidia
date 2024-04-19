import inspect
import os
from contextlib import contextmanager
from typing import Any, Generator

import pytest

import langchain_nvidia_ai_endpoints

public_classes = [
    member[1]
    for member in inspect.getmembers(langchain_nvidia_ai_endpoints, inspect.isclass)
]


@contextmanager
def no_env_var(var: str) -> Generator[None, None, None]:
    try:
        if val := os.environ.get(var, None):
            del os.environ[var]
        yield
    finally:
        if val:
            os.environ[var] = val


@pytest.mark.parametrize("cls", public_classes)
def test_create_without_api_key(cls: type) -> None:
    with no_env_var("NVIDIA_API_KEY"):
        cls()


@pytest.mark.parametrize("cls", public_classes)
@pytest.mark.parametrize("param", ["nvidia_api_key", "api_key"])
def test_create_with_api_key(cls: type, param: str) -> None:
    with no_env_var("NVIDIA_API_KEY"):
        cls(**{param: "just testing no failure"})


@pytest.mark.parametrize("cls", public_classes)
def test_api_key_priority(cls: type) -> None:
    # ChatNVIDIA and NVIDIAEmbeddings currently expose a client attribute
    def get_api_key(instance: Any) -> str:
        if isinstance(instance, langchain_nvidia_ai_endpoints.ChatNVIDIA) or isinstance(
            instance, langchain_nvidia_ai_endpoints.NVIDIAEmbeddings
        ):
            return instance.client.api_key.get_secret_value()
        return instance._client.client.api_key.get_secret_value()

    with no_env_var("NVIDIA_API_KEY"):
        os.environ["NVIDIA_API_KEY"] = "ENV"
        assert get_api_key(cls()) == "ENV"
        assert get_api_key(cls(nvidia_api_key="PARAM")) == "PARAM"
        assert get_api_key(cls(api_key="PARAM")) == "PARAM"
        assert get_api_key(cls(api_key="LOW", nvidia_api_key="HIGH")) == "HIGH"
