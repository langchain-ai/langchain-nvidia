import warnings
from typing import Any, Generator

import pytest
from requests_mock import Mocker

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings


@pytest.fixture
def embedding(requests_mock: Mocker) -> Generator[NVIDIAEmbeddings, None, None]:
    model = "mock-model"
    requests_mock.get(
        "https://api.nvcf.nvidia.com/v2/nvcf/functions",
        json={
            "functions": [
                {
                    "id": "ID",
                    "ncaId": "NCA-ID",
                    "versionId": "VERSION-ID",
                    "name": model,
                    "status": "ACTIVE",
                    "ownedByDifferentAccount": True,
                    "apiBodyFormat": "CUSTOM",
                    "healthUri": "/v2/health/ready",
                    "createdAt": "0000-00-00T00:00:00.000Z",
                }
            ]
        },
    )
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
    yield NVIDIAEmbeddings(model=model, nvidia_api_key="a-bogus-key")


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


def test_embed_deprecated_nvolvqa_40k() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        NVIDIAEmbeddings()
    with pytest.deprecated_call():
        NVIDIAEmbeddings(model="nvolveqa_40k")
    with pytest.deprecated_call():
        NVIDIAEmbeddings(model="playground_nvolveqa_40k")


def test_embed_max_length_deprecated() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        NVIDIAEmbeddings()
    with pytest.deprecated_call():
        NVIDIAEmbeddings(max_length=43)


@pytest.mark.parametrize("truncate", [True, False, 1, 0, 1.0, "BOGUS"])
def test_embed_query_truncate_invalid(truncate: Any) -> None:
    with pytest.raises(ValueError):
        NVIDIAEmbeddings(truncate=truncate)


# todo: test max_batch_size (-50, 0, 1, 50)
