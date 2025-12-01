from typing import Any, Generator

import pytest
from requests_mock import Mocker

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from .conftest import MockHTTP


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


mock_embedding_response = {
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
}


@pytest.fixture
def embedding(requests_mock: Mocker) -> Generator[NVIDIAEmbeddings, None, None]:
    model = "mock-model"
    requests_mock.post(
        "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/ID",
        json=mock_embedding_response,
    )
    with pytest.warns(UserWarning) as record:
        yield NVIDIAEmbeddings(model=model, nvidia_api_key="a-bogus-key")
    assert len(record) == 1
    assert "type is unknown and inference may fail" in str(record[0].message)


@pytest.fixture
def aembedding(mock_http: MockHTTP) -> Generator[NVIDIAEmbeddings, None, None]:
    model = "mock-model"
    mock_http.set_post(json_body=mock_embedding_response)
    with pytest.warns(UserWarning) as record:
        yield NVIDIAEmbeddings(model=model, nvidia_api_key="a-bogus-key")
    assert len(record) == 1
    assert "type is unknown and inference may fail" in str(record[0].message)


@pytest.mark.parametrize(
    "func, emb_fixture",
    [("embed_documents", "embedding"), ("aembed_documents", "aembedding")],
)
@pytest.mark.asyncio
async def test_embed_documents_negative_input_int(
    func: str,
    emb_fixture: str,
    request: pytest.FixtureRequest,
    mock_http: MockHTTP,
) -> None:
    emb: NVIDIAEmbeddings = request.getfixturevalue(emb_fixture)
    documents = 1
    with pytest.raises(ValueError):
        if func == "aembed_documents":
            await emb.aembed_documents(documents)  # type: ignore
        else:
            emb.embed_documents(documents)  # type: ignore


@pytest.mark.parametrize(
    "func, emb_fixture",
    [("embed_documents", "embedding"), ("aembed_documents", "aembedding")],
)
@pytest.mark.asyncio
async def test_embed_documents_negative_input_float(
    func: str,
    emb_fixture: str,
    request: pytest.FixtureRequest,
    mock_http: MockHTTP,
) -> None:
    emb: NVIDIAEmbeddings = request.getfixturevalue(emb_fixture)
    documents = 1.0
    with pytest.raises(ValueError):
        if func == "aembed_documents":
            await emb.aembed_documents(documents)  # type: ignore
        else:
            emb.embed_documents(documents)  # type: ignore


@pytest.mark.parametrize(
    "func, emb_fixture",
    [("embed_documents", "embedding"), ("aembed_documents", "aembedding")],
)
@pytest.mark.asyncio
async def test_embed_documents_negative_input_str(
    func: str,
    emb_fixture: str,
    request: pytest.FixtureRequest,
    mock_http: MockHTTP,
) -> None:
    emb: NVIDIAEmbeddings = request.getfixturevalue(emb_fixture)
    documents = "subscriptable string, not a list"
    with pytest.raises(ValueError):
        if func == "aembed_documents":
            await emb.aembed_documents(documents)  # type: ignore
        else:
            emb.embed_documents(documents)  # type: ignore


@pytest.mark.parametrize(
    "func, emb_fixture",
    [("embed_documents", "embedding"), ("aembed_documents", "aembedding")],
)
@pytest.mark.asyncio
async def test_embed_documents_negative_input_list_int(
    func: str,
    emb_fixture: str,
    request: pytest.FixtureRequest,
    mock_http: MockHTTP,
) -> None:
    emb: NVIDIAEmbeddings = request.getfixturevalue(emb_fixture)
    documents = [1, 2, 3]
    with pytest.raises(ValueError):
        if func == "aembed_documents":
            await emb.aembed_documents(documents)  # type: ignore
        else:
            emb.embed_documents(documents)  # type: ignore


@pytest.mark.parametrize(
    "func, emb_fixture",
    [("embed_documents", "embedding"), ("aembed_documents", "aembedding")],
)
@pytest.mark.asyncio
async def test_embed_documents_negative_input_list_float(
    func: str,
    emb_fixture: str,
    request: pytest.FixtureRequest,
    mock_http: MockHTTP,
) -> None:
    emb: NVIDIAEmbeddings = request.getfixturevalue(emb_fixture)
    documents = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        if func == "aembed_documents":
            await emb.aembed_documents(documents)  # type: ignore
        else:
            emb.embed_documents(documents)  # type: ignore


@pytest.mark.parametrize(
    "func, emb_fixture",
    [("embed_documents", "embedding"), ("aembed_documents", "aembedding")],
)
@pytest.mark.asyncio
async def test_embed_documents_negative_input_list_mixed(
    func: str,
    emb_fixture: str,
    request: pytest.FixtureRequest,
    mock_http: MockHTTP,
) -> None:
    emb: NVIDIAEmbeddings = request.getfixturevalue(emb_fixture)
    documents = ["1", 2.0, 3]
    with pytest.raises(ValueError):
        if func == "aembed_documents":
            await emb.aembed_documents(documents)  # type: ignore
        else:
            emb.embed_documents(documents)  # type: ignore


@pytest.mark.parametrize("truncate", [True, False, 1, 0, 1.0, "BOGUS"])
def test_embed_query_truncate_invalid(truncate: Any) -> None:
    with pytest.raises(ValueError):
        NVIDIAEmbeddings(truncate=truncate)


@pytest.mark.parametrize(
    "func",
    ["embed_documents", "aembed_documents"],
)
@pytest.mark.asyncio
async def test_default_headers(
    func: str, requests_mock: Mocker, mock_http: MockHTTP
) -> None:
    """Test that default_headers are passed to requests."""
    import warnings

    model = "mock-model"

    if func == "aembed_documents":
        mock_http.set_post(
            json_body={
                "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
                "usage": {"prompt_tokens": 8, "total_tokens": 8},
            }
        )
    else:
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

    if func == "aembed_documents":
        _ = await embedder.aembed_documents(["test document"])
        assert len(mock_http.history) > 0
        last_request = mock_http.history[-1]
        assert last_request is not None
        assert last_request.kwargs.get("headers", {})["X-Test"] == "test"
        # Verify the correct endpoint was called
        assert "embeddings" in last_request.url or "v2/nvcf/pexec" in last_request.url
    else:
        _ = embedder.embed_documents(["test document"])
        assert requests_mock.last_request is not None
        assert requests_mock.last_request.headers["X-Test"] == "test"


# todo: test max_batch_size (-50, 0, 1, 50)
