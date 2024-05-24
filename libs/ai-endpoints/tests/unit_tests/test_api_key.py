import os
from contextlib import contextmanager
from typing import Any, Generator

import pytest


@contextmanager
def no_env_var(var: str) -> Generator[None, None, None]:
    try:
        if val := os.environ.get(var, None):
            del os.environ[var]
        yield
    finally:
        if val:
            os.environ[var] = val


def test_create_without_api_key(public_class: type) -> None:
    with no_env_var("NVIDIA_API_KEY"):
        public_class()


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
