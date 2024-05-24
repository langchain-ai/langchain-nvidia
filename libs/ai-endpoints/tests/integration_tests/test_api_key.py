import os
from typing import Any

import pytest
from langchain_core.documents import Document

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
