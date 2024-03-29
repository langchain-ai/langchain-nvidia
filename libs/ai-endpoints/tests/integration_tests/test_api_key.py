import pytest

from langchain_nvidia_ai_endpoints import ChatNVIDIA

from ..unit_tests.test_api_key import no_env_var


def test_missing_api_key_error() -> None:
    with no_env_var("NVIDIA_API_KEY"):
        chat = ChatNVIDIA()
        with pytest.raises(ValueError) as exc_info:
            chat.invoke("Hello, world!")
        message = str(exc_info.value)
        assert "401" in message
        assert "Unauthorized" in message
        assert "API key" in message


def test_bogus_api_key_error() -> None:
    with no_env_var("NVIDIA_API_KEY"):
        chat = ChatNVIDIA(nvidia_api_key="BOGUS")
        with pytest.raises(ValueError) as exc_info:
            chat.invoke("Hello, world!")
        message = str(exc_info.value)
        assert "401" in message
        assert "Unauthorized" in message
        assert "API key" in message
