import os
from typing import Any

import pytest
from langchain_core.messages import HumanMessage

from langchain_nvidia_ai_endpoints import ChatNVIDIA

from ..unit_tests.test_api_key import no_env_var


def test_missing_api_key_error(public_class: type, contact_service: Any) -> None:
    with no_env_var("NVIDIA_API_KEY"):
        with pytest.warns(UserWarning) as record:
            client = public_class()
        assert len(record) == 1
        assert "API key is required for the hosted" in str(record[0].message)
        with pytest.raises(Exception) as exc_info:
            contact_service(client)
        message = str(exc_info.value)
        assert "401" in message
        assert "Unauthorized" in message
        assert "API key" in message


def test_bogus_api_key_error(public_class: type, contact_service: Any) -> None:
    with no_env_var("NVIDIA_API_KEY"):
        client = public_class(nvidia_api_key="BOGUS")
        with pytest.raises(Exception) as exc_info:
            contact_service(client)
        message = str(exc_info.value)
        assert "401" in message
        assert "Unauthorized" in message
        assert "API key" in message


@pytest.mark.parametrize("param", ["nvidia_api_key", "api_key"])
def test_api_key(public_class: type, param: str, contact_service: Any) -> None:
    api_key = os.environ.get("NVIDIA_API_KEY")
    with no_env_var("NVIDIA_API_KEY"):
        client = public_class(**{param: api_key})
        contact_service(client)


def test_api_key_leakage(chat_model: str, mode: dict) -> None:
    """Test ChatNVIDIA wrapper."""
    chat = ChatNVIDIA(model=chat_model, temperature=0.7, **mode)
    message = HumanMessage(content="Hello")
    chat.invoke([message])

    # check last_input post request
    last_inputs = chat._client.last_inputs
    assert last_inputs

    authorization_header = last_inputs.get("headers", {}).get("Authorization")

    if authorization_header:
        key = authorization_header.split("Bearer ")[1]

        assert not key.startswith("nvapi-")
        assert key == "**********"
