import inspect
import os

import pytest

import langchain_nvidia_ai_endpoints

from ..unit_tests.test_api_key import no_env_var

public_classes = [
    member[1]
    for member in inspect.getmembers(langchain_nvidia_ai_endpoints, inspect.isclass)
]


@pytest.mark.parametrize("cls", public_classes)
def test_missing_api_key_error(cls: type) -> None:
    with no_env_var("NVIDIA_API_KEY"):
        client = cls()
        with pytest.raises(ValueError) as exc_info:
            client.available_models
        message = str(exc_info.value)
        assert "401" in message
        assert "Unauthorized" in message
        assert "API key" in message


@pytest.mark.parametrize("cls", public_classes)
def test_bogus_api_key_error(cls: type) -> None:
    with no_env_var("NVIDIA_API_KEY"):
        client = cls(nvidia_api_key="BOGUS")
        with pytest.raises(ValueError) as exc_info:
            client.available_models
        message = str(exc_info.value)
        assert "401" in message
        assert "Unauthorized" in message
        assert "API key" in message


@pytest.mark.parametrize("cls", public_classes)
@pytest.mark.parametrize("param", ["nvidia_api_key", "api_key"])
def test_api_key(cls: type, param: str) -> None:
    api_key = os.environ.get("NVIDIA_API_KEY")
    with no_env_var("NVIDIA_API_KEY"):
        client = cls(**{param: api_key})
        assert client.available_models
