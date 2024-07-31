from typing import Any, Generator, Literal

import pytest
from requests_mock import Mocker

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings


@pytest.fixture
def embedding(requests_mock: Mocker) -> Generator[NVIDIAEmbeddings, None, None]:
    model = "mock-model"
    requests_mock.get(
        "https://integrate.api.nvidia.com/v1/models",
        json={
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "OWNER",
                },
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
    with pytest.warns(UserWarning):
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


@pytest.mark.parametrize("truncate", [True, False, 1, 0, 1.0, "BOGUS"])
def test_embed_query_truncate_invalid(truncate: Any) -> None:
    with pytest.raises(ValueError):
        NVIDIAEmbeddings(truncate=truncate)


@pytest.mark.parametrize("model_type", ["query", "passage"])
def test_embed_model_type_deprecated(model_type: Literal["query", "passage"]) -> None:
    with pytest.warns(UserWarning) as record:
        NVIDIAEmbeddings(api_key="BOGUS", model_type=model_type)
    assert len(record) == 1
    assert "`model_type` is deprecated" in str(record[0].message)
    x = NVIDIAEmbeddings(api_key="BOGUS")
    with pytest.warns(UserWarning) as record:
        x.model_type = model_type
    assert len(record) == 1
    assert "`model_type` is deprecated" in str(record[0].message)


# todo: test max_batch_size (-50, 0, 1, 50)
