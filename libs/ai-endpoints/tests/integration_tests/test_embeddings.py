"""Test NVIDIA AI Foundation Model Embeddings.

Note: These tests are designed to validate the functionality of NVIDIAEmbeddings.
"""

import pytest

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings


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
    if embedding_model in ["playground_nvolveqa_40k", "nvolveqa_40k"]:
        pytest.skip("Skip test for nvolveqa-40k due to compat override of truncate")
    embedding = NVIDIAEmbeddings(model=embedding_model, **mode)
    text = "nvidia " * 10240
    with pytest.raises(Exception):
        embedding.embed_query(text)


def test_embed_documents_batched_texts(embedding_model: str, mode: dict) -> None:
    embedding = NVIDIAEmbeddings(model=embedding_model, **mode)
    count = NVIDIAEmbeddings._default_max_batch_size * 2 + 1
    texts = ["nvidia " * 32] * count
    output = embedding.embed_documents(texts)
    assert len(output) == count
    assert all(len(embedding) > 3 for embedding in output)


def test_embed_documents_mixed_long_texts(embedding_model: str, mode: dict) -> None:
    if embedding_model in ["playground_nvolveqa_40k", "nvolveqa_40k"]:
        pytest.skip("Skip test for nvolveqa-40k due to compat override of truncate")
    embedding = NVIDIAEmbeddings(model=embedding_model, **mode)
    count = NVIDIAEmbeddings._default_max_batch_size * 2 - 1
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


@pytest.mark.parametrize("nvolveqa_40k", ["playground_nvolveqa_40k", "nvolveqa_40k"])
def test_embed_nvolveqa_40k_compat(nvolveqa_40k: str, mode: dict) -> None:
    if mode:
        pytest.skip("Test only relevant for API Catalog")
    with pytest.warns(UserWarning):
        embedding = NVIDIAEmbeddings(model=nvolveqa_40k, truncate="NONE", **mode)
    text = "nvidia " * 2048
    output = embedding.embed_query(text)
    assert len(output) > 3


# todo: test model_type ("passage" and embed_query,
#                        "query" and embed_documents; compare results)
# todo: test max_length > max length accepted by the model
# todo: test max_batch_size > max batch size accepted by the model
