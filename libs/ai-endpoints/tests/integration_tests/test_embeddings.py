"""Test NVIDIA AI Foundation Model Embeddings.

Note: These tests are designed to validate the functionality of NVIDIAEmbeddings.
"""

import pytest
import requests_mock

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings


def test_nvai_play_embedding_documents(embedding_model: str, mode: dict) -> None:
    """Test NVIDIA embeddings for documents."""
    documents = ["foo bar"]
    embedding = NVIDIAEmbeddings(model=embedding_model).mode(**mode)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 1024  # Assuming embedding size is 2048


def test_nvai_play_embedding_documents_multiple(
    embedding_model: str, mode: dict
) -> None:
    """Test NVIDIA embeddings for multiple documents."""
    documents = ["foo bar", "bar foo", "foo"]
    embedding = NVIDIAEmbeddings(model=embedding_model).mode(**mode)
    output = embedding.embed_documents(documents)
    assert len(output) == 3
    assert all(len(doc) == 1024 for doc in output)


def test_nvai_play_embedding_query(embedding_model: str, mode: dict) -> None:
    """Test NVIDIA embeddings for a single query."""
    query = "foo bar"
    embedding = NVIDIAEmbeddings(model=embedding_model).mode(**mode)
    output = embedding.embed_query(query)
    assert len(output) == 1024


async def test_nvai_play_embedding_async_query(
    embedding_model: str, mode: dict
) -> None:
    """Test NVIDIA async embeddings for a single query."""
    query = "foo bar"
    embedding = NVIDIAEmbeddings(model=embedding_model).mode(**mode)
    output = await embedding.aembed_query(query)
    assert len(output) == 1024


async def test_nvai_play_embedding_async_documents(
    embedding_model: str, mode: dict
) -> None:
    """Test NVIDIA async embeddings for multiple documents."""
    documents = ["foo bar", "bar foo", "foo"]
    embedding = NVIDIAEmbeddings(model=embedding_model).mode(**mode)
    output = await embedding.aembed_documents(documents)
    assert len(output) == 3
    assert all(len(doc) == 1024 for doc in output)


def test_embed_available_models(mode: dict) -> None:
    if mode:
        pytest.skip(f"available_models test only valid against API Catalog, not {mode}")
    embedding = NVIDIAEmbeddings()
    models = embedding.available_models
    assert len(models) >= 2  # nvolveqa_40k and ai-embed-qa-4
    assert "nvolveqa_40k" in [model.id for model in models]
    assert "ai-embed-qa-4" in [model.id for model in models]


def test_embed_available_models_cached() -> None:
    """Test NVIDIA embeddings for available models."""
    pytest.skip("There's a bug that needs to be fixed")
    with requests_mock.Mocker(real_http=True) as mock:
        embedding = NVIDIAEmbeddings()
        assert not mock.called
        embedding.available_models
        assert mock.called
        embedding.available_models
        embedding.available_models
        assert mock.call_count == 1


def test_embed_long_query_text(embedding_model: str, mode: dict) -> None:
    embedding = NVIDIAEmbeddings(model=embedding_model).mode(**mode)
    text = "nvidia " * 2048
    with pytest.raises(Exception):
        embedding.embed_query(text)


def test_embed_many_texts(embedding_model: str, mode: dict) -> None:
    embedding = NVIDIAEmbeddings(model=embedding_model).mode(**mode)
    texts = ["nvidia " * 32] * 1000
    output = embedding.embed_documents(texts)
    assert len(output) == 1000
    assert all(len(embedding) == 1024 for embedding in output)


def test_embed_mixed_long_texts(embedding_model: str, mode: dict) -> None:
    if embedding_model == "nvolveqa_40k":
        pytest.skip("AI Foundation Model trucates by default")
    embedding = NVIDIAEmbeddings(model=embedding_model).mode(**mode)
    texts = ["nvidia " * 32] * 50
    texts[42] = "nvidia " * 2048
    with pytest.raises(Exception):
        embedding.embed_documents(texts)
