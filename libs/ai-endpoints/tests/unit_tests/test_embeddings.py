from typing import Any, Generator

import pytest
from requests_mock import Mocker

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings


@pytest.fixture(autouse=True)
def mock_v1_models(requests_mock: Mocker) -> None:
    requests_mock.get(
        "https://integrate.api.nvidia.com/v1/models",
        json={
            "data": [
                {
                    "id": "mock-model",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "OWNER",
                }
            ]
        },
    )


@pytest.fixture
def embedding(requests_mock: Mocker) -> Generator[NVIDIAEmbeddings, None, None]:
    model = "mock-model"
    requests_mock.post(
        "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/ID",
        json={
            "data": [
                {
                    "embedding": [
                        0.1,
                        0.2,
                        0.3,
                    ],
                    "index": 0,
                }
            ],
            "usage": {"prompt_tokens": 8, "total_tokens": 8},
        },
    )
    with pytest.warns(UserWarning) as record:
        yield NVIDIAEmbeddings(model=model, nvidia_api_key="a-bogus-key")
    assert len(record) == 1
    assert "type is unknown and inference may fail" in str(record[0].message)


def test_embed_documents_negative_input_int(embedding: NVIDIAEmbeddings) -> None:
    documents = 1
    with pytest.raises(ValueError):
        embedding.embed_documents(documents)  # type: ignore


def test_embed_documents_negative_input_float(embedding: NVIDIAEmbeddings) -> None:
    documents = 1.0
    with pytest.raises(ValueError):
        embedding.embed_documents(documents)  # type: ignore


def test_embed_documents_negative_input_str(embedding: NVIDIAEmbeddings) -> None:
    documents = "subscriptable string, not a list"
    with pytest.raises(ValueError):
        embedding.embed_documents(documents)  # type: ignore


def test_embed_documents_negative_input_list_int(embedding: NVIDIAEmbeddings) -> None:
    documents = [1, 2, 3]
    with pytest.raises(ValueError):
        embedding.embed_documents(documents)  # type: ignore


def test_embed_documents_negative_input_list_float(embedding: NVIDIAEmbeddings) -> None:
    documents = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        embedding.embed_documents(documents)  # type: ignore


def test_embed_documents_negative_input_list_mixed(embedding: NVIDIAEmbeddings) -> None:
    documents = ["1", 2.0, 3]
    with pytest.raises(ValueError):
        embedding.embed_documents(documents)  # type: ignore


@pytest.mark.parametrize("truncate", [True, False, 1, 0, 1.0, "BOGUS"])
def test_embed_query_truncate_invalid(truncate: Any) -> None:
    with pytest.raises(ValueError):
        NVIDIAEmbeddings(truncate=truncate)


def test_default_headers(requests_mock: Mocker) -> None:
    """Test that default_headers are passed to requests."""
    import warnings

    model = "mock-model"
    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/embeddings",
        json={
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            "usage": {"prompt_tokens": 8, "total_tokens": 8},
        },
    )

    warnings.filterwarnings("ignore", ".*type is unknown and inference may fail.*")
    embedder = NVIDIAEmbeddings(
        model=model,
        nvidia_api_key="a-bogus-key",
        default_headers={"X-Test": "test"},
    )
    assert embedder.default_headers == {"X-Test": "test"}

    _ = embedder.embed_documents(["test document"])
    assert requests_mock.last_request is not None
    assert requests_mock.last_request.headers["X-Test"] == "test"


# todo: test max_batch_size (-50, 0, 1, 50)
