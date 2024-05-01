"""Test NVIDIA AI Foundation Model Embeddings.

Note: These tests are designed to validate the functionality of NVIDIAEmbeddings.
"""

import pytest
import requests_mock

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings


def test_embed_query(embedding_model: str, mode: dict) -> None:
    """Test NVIDIA embeddings for a single query."""
    query = "foo bar"
    embedding = NVIDIAEmbeddings(model=embedding_model).mode(**mode)
    output = embedding.embed_query(query)
    assert len(output) == 1024


async def test_embed_query_async(embedding_model: str, mode: dict) -> None:
    """Test NVIDIA async embeddings for a single query."""
    query = "foo bar"
    embedding = NVIDIAEmbeddings(model=embedding_model).mode(**mode)
    output = await embedding.aembed_query(query)
    assert len(output) == 1024


def test_embed_documents_single(embedding_model: str, mode: dict) -> None:
    """Test NVIDIA embeddings for documents."""
    documents = ["foo bar"]
    embedding = NVIDIAEmbeddings(model=embedding_model).mode(**mode)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) == 1024  # Assuming embedding size is 2048


def test_embed_documents_multiple(embedding_model: str, mode: dict) -> None:
    """Test NVIDIA embeddings for multiple documents."""
    documents = ["foo bar", "bar foo", "foo"]
    embedding = NVIDIAEmbeddings(model=embedding_model).mode(**mode)
    output = embedding.embed_documents(documents)
    assert len(output) == 3
    assert all(len(doc) == 1024 for doc in output)


async def test_embed_documents_multiple_async(embedding_model: str, mode: dict) -> None:
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


def test_embed_query_long_text(embedding_model: str, mode: dict) -> None:
    embedding = NVIDIAEmbeddings(model=embedding_model).mode(**mode)
    text = "nvidia " * 2048
    with pytest.raises(Exception):
        embedding.embed_query(text)


def test_embed_documents_batched_texts(embedding_model: str, mode: dict) -> None:
    embedding = NVIDIAEmbeddings(model=embedding_model).mode(**mode)
    count = NVIDIAEmbeddings._default_max_batch_size * 2 + 1
    texts = ["nvidia " * 32] * count
    output = embedding.embed_documents(texts)
    assert len(output) == count
    assert all(len(embedding) == 1024 for embedding in output)


def test_embed_documents_mixed_long_texts(embedding_model: str, mode: dict) -> None:
    if embedding_model == "nvolveqa_40k":
        pytest.skip("AI Foundation Model trucates by default")
    embedding = NVIDIAEmbeddings(model=embedding_model).mode(**mode)
    count = NVIDIAEmbeddings._default_max_batch_size * 2 - 1
    texts = ["nvidia " * 32] * count
    texts[len(texts) // 2] = "nvidia " * 2048
    with pytest.raises(Exception):
        embedding.embed_documents(texts)


@pytest.mark.parametrize("truncate", ["START", "END"])
def test_embed_query_truncate(embedding_model: str, mode: dict, truncate: str) -> None:
    if embedding_model == "nvolveqa_40k":
        pytest.skip("AI Foundation Model does not support truncate option")
    embedding = NVIDIAEmbeddings(model=embedding_model, truncate=truncate).mode(**mode)
    text = "nvidia " * 2048
    output = embedding.embed_query(text)
    assert len(output) == 1024


@pytest.mark.parametrize("truncate", ["START", "END"])
def test_embed_documents_truncate(
    embedding_model: str, mode: dict, truncate: str
) -> None:
    embedding = NVIDIAEmbeddings(model=embedding_model, truncate=truncate).mode(**mode)
    count = 10
    texts = ["nvidia " * 32] * count
    texts[len(texts) // 2] = "nvidia " * 2048
    output = embedding.embed_documents(texts)
    assert len(output) == count


# todo: test model_type ("passage" and embed_query,
#                        "query" and embed_documents; compare results)
# todo: test max_length > max length accepted by the model
# todo: test max_batch_size > max batch size accepted by the model
