"""Test NVIDIA AI Foundation Model Embeddings.

Note: These tests are designed to validate the functionality of NVIDIAEmbeddings.
"""

import pytest

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_nvidia_ai_endpoints.embeddings import _DEFAULT_BATCH_SIZE


def test_embed_query(embedding_model: str, mode: dict) -> None:
    """Test NVIDIA embeddings for a single query."""
    query = "foo bar"
    embedding = NVIDIAEmbeddings(model=embedding_model, **mode)
    output = embedding.embed_query(query)
    assert len(output) > 3


async def test_embed_query_async(embedding_model: str, mode: dict) -> None:
    """Test NVIDIA async embeddings for a single query."""
    query = "foo bar"
    embedding = NVIDIAEmbeddings(model=embedding_model, **mode)
    output = await embedding.aembed_query(query)
    assert len(output) > 3


def test_embed_documents_single(embedding_model: str, mode: dict) -> None:
    """Test NVIDIA embeddings for documents."""
    documents = ["foo bar"]
    embedding = NVIDIAEmbeddings(model=embedding_model, **mode)
    output = embedding.embed_documents(documents)
    assert len(output) == 1
    assert len(output[0]) > 3


def test_embed_documents_multiple(embedding_model: str, mode: dict) -> None:
    """Test NVIDIA embeddings for multiple documents."""
    documents = ["foo bar", "bar foo", "foo"]
    embedding = NVIDIAEmbeddings(model=embedding_model, **mode)
    output = embedding.embed_documents(documents)
    assert len(output) == 3
    assert all(len(doc) > 4 for doc in output)


async def test_embed_documents_multiple_async(embedding_model: str, mode: dict) -> None:
    """Test NVIDIA async embeddings for multiple documents."""
    documents = ["foo bar", "bar foo", "foo"]
    embedding = NVIDIAEmbeddings(model=embedding_model, **mode)
    output = await embedding.aembed_documents(documents)
    assert len(output) == 3
    assert all(len(doc) > 4 for doc in output)


def test_embed_query_long_text(embedding_model: str, mode: dict) -> None:
    embedding = NVIDIAEmbeddings(model=embedding_model, **mode)
    text = "nvidia " * 10240
    with pytest.raises(Exception):
        embedding.embed_query(text)


def test_embed_documents_batched_texts(embedding_model: str, mode: dict) -> None:
    embedding = NVIDIAEmbeddings(model=embedding_model, **mode)
    count = _DEFAULT_BATCH_SIZE * 2 + 1
    texts = ["nvidia " * 32] * count
    output = embedding.embed_documents(texts)
    assert len(output) == count
    assert all(len(embedding) > 3 for embedding in output)


def test_embed_documents_mixed_long_texts(embedding_model: str, mode: dict) -> None:
    embedding = NVIDIAEmbeddings(model=embedding_model, **mode)
    count = _DEFAULT_BATCH_SIZE * 2 - 1
    texts = ["nvidia " * 32] * count
    texts[len(texts) // 2] = "nvidia " * 10240
    with pytest.raises(Exception):
        embedding.embed_documents(texts)


@pytest.mark.parametrize("truncate", ["START", "END"])
def test_embed_query_truncate(embedding_model: str, mode: dict, truncate: str) -> None:
    embedding = NVIDIAEmbeddings(model=embedding_model, truncate=truncate, **mode)
    text = "nvidia " * 2048
    output = embedding.embed_query(text)
    assert len(output) > 3


@pytest.mark.parametrize("truncate", ["START", "END"])
def test_embed_documents_truncate(
    embedding_model: str, mode: dict, truncate: str
) -> None:
    embedding = NVIDIAEmbeddings(model=embedding_model, truncate=truncate, **mode)
    count = 10
    texts = ["nvidia " * 32] * count
    texts[len(texts) // 2] = "nvidia " * 10240
    output = embedding.embed_documents(texts)
    assert len(output) == count


@pytest.mark.parametrize("dimensions", [32, 64, 128, 2048])
def test_embed_query_with_dimensions(embedding_model: str, mode: dict, dimensions: int) -> None:
    if embedding_model != "nvidia/llama-3.2-nv-embedqa-1b-v2":
        pytest.skip("Model does not support custom dimensions.")
    query = "foo bar"
    embedding = NVIDIAEmbeddings(model=embedding_model, dimensions=dimensions, **mode)
    assert len(embedding.embed_query(query)) == dimensions


@pytest.mark.parametrize("dimensions", [32, 64, 128, 2048])
def test_embed_documents_with_dimensions(embedding_model: str, mode: dict, dimensions: int) -> None:
    if embedding_model != "nvidia/llama-3.2-nv-embedqa-1b-v2":
        pytest.skip("Model does not support custom dimensions.")
    documents = ["foo bar", "bar foo"]
    embedding = NVIDIAEmbeddings(model=embedding_model, dimensions=dimensions, **mode)
    output = embedding.embed_documents(documents)
    assert len(output) == len(documents)
    assert all(len(doc) == dimensions for doc in output)


@pytest.mark.parametrize("dimensions", [102400])
def test_embed_query_with_large_dimensions(embedding_model: str, mode: dict, dimensions: int) -> None:
    if embedding_model != "nvidia/llama-3.2-nv-embedqa-1b-v2":
        pytest.skip("Model does not support custom dimensions.")
    query = "foo bar"
    embedding = NVIDIAEmbeddings(model=embedding_model, dimensions=dimensions, **mode)
    assert 2048 <= len(embedding.embed_query(query)) < dimensions


@pytest.mark.parametrize("dimensions", [102400])
def test_embed_documents_with_large_dimensions(embedding_model: str, mode: dict, dimensions: int) -> None:
    if embedding_model != "nvidia/llama-3.2-nv-embedqa-1b-v2":
        pytest.skip("Model does not support custom dimensions.")
    documents = ["foo bar", "bar foo"]
    embedding = NVIDIAEmbeddings(model=embedding_model, dimensions=dimensions, **mode)
    output = embedding.embed_documents(documents)
    assert len(output) == len(documents)
    assert all(2048 <= len(doc) < dimensions for doc in output)


@pytest.mark.parametrize("dimensions", [-1])
def test_embed_query_invalid_dimensions(embedding_model: str, mode: dict, dimensions: int) -> None:
    if embedding_model != "nvidia/llama-3.2-nv-embedqa-1b-v2":
        pytest.skip("Model does not support custom dimensions.")
    query = "foo bar"
    with pytest.raises(Exception) as exc:
        NVIDIAEmbeddings(model=embedding_model, dimensions=dimensions, **mode).embed_query(query)
    assert "400" in str(exc.value)


@pytest.mark.parametrize("dimensions", [-1])
def test_embed_documents_invalid_dimensions(embedding_model: str, mode: dict, dimensions: int) -> None:
    if embedding_model != "nvidia/llama-3.2-nv-embedqa-1b-v2":
        pytest.skip("Model does not support custom dimensions.")
    documents = ["foo bar", "bar foo"]
    with pytest.raises(Exception) as exc:
        NVIDIAEmbeddings(model=embedding_model, dimensions=dimensions, **mode).embed_documents(documents)
    assert "400" in str(exc.value)


# todo: test max_length > max length accepted by the model
# todo: test max_batch_size > max batch size accepted by the model
