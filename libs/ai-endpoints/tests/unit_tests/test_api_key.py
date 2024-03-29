import os
from contextlib import contextmanager
from typing import Generator

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings


@contextmanager
def no_env_var(var: str) -> Generator[None, None, None]:
    try:
        if key := os.environ.get(var, None):
            del os.environ[var]
            yield
    finally:
        if key:
            os.environ[var] = key


def test_create_chat_without_api_key() -> None:
    with no_env_var("NVIDIA_API_KEY"):
        ChatNVIDIA()


def test_create_embeddings_without_api_key() -> None:
    with no_env_var("NVIDIA_API_KEY"):
        NVIDIAEmbeddings()
