import os
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank

from ..unit_tests.test_api_key import no_env_var


def contact_service(instance: Any) -> None:
    if isinstance(instance, ChatNVIDIA):
        instance.invoke("Hello")
    elif isinstance(instance, NVIDIAEmbeddings):
        instance.embed_documents(["Hello"])
    elif isinstance(instance, NVIDIARerank):
        instance.compress_documents(
            documents=[Document(page_content="World")], query="Hello"
        )


def test_missing_api_key_error(public_class: type) -> None:
    with no_env_var("NVIDIA_API_KEY"):
        with pytest.warns(UserWarning):
            client = public_class()
        with pytest.raises(Exception) as exc_info:
            contact_service(client)
        message = str(exc_info.value)
        assert "401" in message
        assert "Unauthorized" in message
        assert "API key" in message


def test_bogus_api_key_error(public_class: type) -> None:
    with no_env_var("NVIDIA_API_KEY"):
        client = public_class(nvidia_api_key="BOGUS")
        with pytest.raises(Exception) as exc_info:
            contact_service(client)
        message = str(exc_info.value)
        assert "401" in message
        assert "Unauthorized" in message
        assert "API key" in message


@pytest.mark.parametrize("param", ["nvidia_api_key", "api_key"])
def test_api_key(public_class: type, param: str) -> None:
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
    last_inputs = chat._client.client.last_inputs
    assert last_inputs

    authorization_header = last_inputs.get("headers", {}).get("Authorization")

    if authorization_header:
        key = authorization_header.split("Bearer ")[1]

        assert not key.startswith("nvapi-")
        assert key == "**********"
