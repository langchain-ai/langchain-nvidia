"""Unit tests for NVIDIARetriever."""

import asyncio
import json
from unittest.mock import patch

import aiohttp
import pytest
import requests
import requests_mock

from langchain_nvidia_ai_endpoints.retrievers import (
    NVIDIARAGConnectionError,
    NVIDIARAGError,
    NVIDIARAGServerError,
    NVIDIARAGValidationError,
    NVIDIARetriever,
)


def test_nvidia_rag_exception_inheritance() -> None:
    """Test that all RAG exceptions inherit from NVIDIARAGError."""
    assert issubclass(NVIDIARAGConnectionError, NVIDIARAGError)
    assert issubclass(NVIDIARAGServerError, NVIDIARAGError)
    assert issubclass(NVIDIARAGValidationError, NVIDIARAGError)


def test_nvidia_retriever_init() -> None:
    """Test NVIDIARetriever initialization."""
    retriever = NVIDIARetriever(
        base_url="http://localhost:8081",
        collection_names=["test_multimodal_query"],
        k=4,
    )
    assert retriever.base_url == "http://localhost:8081"
    assert retriever.collection_names == ["test_multimodal_query"]
    assert retriever.k == 4


def test_nvidia_retriever_base_url_rejects_v1_search() -> None:
    """Test that base_url must not include /v1/search."""
    with pytest.raises(ValueError) as exc_info:
        NVIDIARetriever(
            base_url="http://localhost:8081/v1/search",
            collection_names=["test"],
        )
    assert "/v1/search" in str(exc_info.value)
    assert "must not include" in str(exc_info.value)

    with pytest.raises(ValueError):
        NVIDIARetriever(
            base_url="http://localhost:8081/v1/search/",
            collection_names=["test"],
        )


def test_nvidia_retriever_build_payload() -> None:
    """Test payload construction."""
    retriever = NVIDIARetriever(
        base_url="http://localhost:8081",
        collection_names=["col1"],
        k=5,
        vdb_top_k=50,
        enable_reranker=True,
        filter_expr='content_metadata["x"] == "y"',
    )
    payload = retriever._build_payload("test query")
    assert payload["query"] == "test query"
    assert payload["reranker_top_k"] == 5
    assert payload["vdb_top_k"] == 50
    assert payload["collection_names"] == ["col1"]
    assert payload["enable_reranker"] is True
    assert payload["filter_expr"] == 'content_metadata["x"] == "y"'


def test_nvidia_retriever_results_to_documents() -> None:
    """Test conversion of API results to Documents."""
    retriever = NVIDIARetriever(base_url="http://localhost:8081")
    results = [
        {
            "document_id": "doc1",
            "content": "Hello world",
            "document_name": "file.pdf",
            "document_type": "text",
            "score": 0.95,
            "metadata": {"page": 1},
        },
    ]
    docs = retriever._results_to_documents(results)
    assert len(docs) == 1
    assert docs[0].page_content == "Hello world"
    assert docs[0].metadata["document_id"] == "doc1"
    assert docs[0].metadata["document_name"] == "file.pdf"
    assert docs[0].metadata["score"] == 0.95
    assert docs[0].metadata["page"] == 1


def test_nvidia_retriever_invoke_success() -> None:
    """Test successful invoke."""
    retriever = NVIDIARetriever(
        base_url="http://localhost:8081",
        collection_names=["test_multimodal_query"],
        k=2,
    )
    mock_response = {
        "total_results": 2,
        "results": [
            {
                "document_id": "id1",
                "content": "Content 1",
                "document_name": "doc1.pdf",
                "document_type": "text",
                "score": 0.9,
                "metadata": {},
            },
            {
                "document_id": "id2",
                "content": "Content 2",
                "document_name": "doc2.pdf",
                "document_type": "text",
                "score": 0.8,
                "metadata": {},
            },
        ],
    }

    with requests_mock.Mocker() as m:
        m.post(
            "http://localhost:8081/v1/search",
            json=mock_response,
        )
        docs = retriever.invoke("What is AI?")
        assert len(docs) == 2
        assert docs[0].page_content == "Content 1"
        assert docs[1].page_content == "Content 2"
        assert m.call_count == 1
        req_body = json.loads(m.request_history[0].body)
        assert req_body["query"] == "What is AI?"
        assert req_body["collection_names"] == ["test_multimodal_query"]
        assert req_body["reranker_top_k"] == 2


def test_nvidia_retriever_connection_error() -> None:
    """Test NVIDIARAGConnectionError when server is unreachable."""
    retriever = NVIDIARetriever(base_url="http://localhost:8081")

    with requests_mock.Mocker() as m:
        m.post(
            "http://localhost:8081/v1/search",
            exc=requests.exceptions.ConnectionError("Connection refused"),
        )

        with pytest.raises(NVIDIARAGConnectionError) as exc_info:
            retriever.invoke("query")
        assert "Cannot connect to RAG server" in str(exc_info.value)
        assert "rag-server" in str(exc_info.value)


def test_nvidia_retriever_timeout_error() -> None:
    """Test NVIDIARAGConnectionError when request times out."""
    retriever = NVIDIARetriever(base_url="http://localhost:8081")

    with requests_mock.Mocker() as m:
        m.post(
            "http://localhost:8081/v1/search",
            exc=requests.exceptions.Timeout("Request timed out"),
        )

        with pytest.raises(NVIDIARAGConnectionError) as exc_info:
            retriever.invoke("query")
        assert "timed out" in str(exc_info.value)
        assert "8081" in str(exc_info.value)


def test_nvidia_retriever_request_exception() -> None:
    """Test NVIDIARAGConnectionError for generic RequestException."""
    retriever = NVIDIARetriever(base_url="http://localhost:8081")

    with requests_mock.Mocker() as m:
        m.post(
            "http://localhost:8081/v1/search",
            exc=requests.exceptions.RequestException("Generic request failed"),
        )

        with pytest.raises(NVIDIARAGConnectionError) as exc_info:
            retriever.invoke("query")
        assert "failed" in str(exc_info.value).lower()


def test_nvidia_retriever_invalid_json() -> None:
    """Test NVIDIARAGServerError when server returns 200 with invalid JSON."""
    retriever = NVIDIARetriever(base_url="http://localhost:8081")

    with requests_mock.Mocker() as m:
        m.post(
            "http://localhost:8081/v1/search",
            status_code=200,
            text="<html>Internal Error Page</html>",
        )

        with pytest.raises(NVIDIARAGServerError) as exc_info:
            retriever.invoke("query")
        assert exc_info.value.status_code == 200
        assert "invalid JSON" in str(exc_info.value)
        assert exc_info.value.body == "<html>Internal Error Page</html>"


def test_nvidia_retriever_response_not_dict() -> None:
    """Test NVIDIARAGServerError when 200 response is JSON but not a dict (e.g. [])."""
    retriever = NVIDIARetriever(base_url="http://localhost:8081")

    with requests_mock.Mocker() as m:
        m.post(
            "http://localhost:8081/v1/search",
            status_code=200,
            text="[]",
        )

        with pytest.raises(NVIDIARAGServerError) as exc_info:
            retriever.invoke("query")
        assert exc_info.value.status_code == 200
        assert "expected dict" in str(exc_info.value)


def test_nvidia_retriever_results_not_list() -> None:
    """Test NVIDIARAGServerError when results field is not a list."""
    retriever = NVIDIARetriever(base_url="http://localhost:8081")

    with requests_mock.Mocker() as m:
        m.post(
            "http://localhost:8081/v1/search",
            status_code=200,
            json={"results": "not a list"},
        )

        with pytest.raises(NVIDIARAGServerError) as exc_info:
            retriever.invoke("query")
        assert exc_info.value.status_code == 200
        assert "must be a list" in str(exc_info.value)


def test_nvidia_retriever_server_error() -> None:
    """Test NVIDIARAGServerError when server returns 500."""
    retriever = NVIDIARetriever(base_url="http://localhost:8081")

    with requests_mock.Mocker() as m:
        m.post(
            "http://localhost:8081/v1/search",
            status_code=500,
            text="Internal Server Error",
        )

        with pytest.raises(NVIDIARAGServerError) as exc_info:
            retriever.invoke("query")
        assert exc_info.value.status_code == 500
        assert "500" in str(exc_info.value)


def test_nvidia_retriever_validation_error() -> None:
    """Test NVIDIARAGValidationError when server returns 422."""
    retriever = NVIDIARetriever(base_url="http://localhost:8081")

    with requests_mock.Mocker() as m:
        m.post(
            "http://localhost:8081/v1/search",
            status_code=422,
            text='{"detail": "Invalid query"}',
        )

        with pytest.raises(NVIDIARAGValidationError) as exc_info:
            retriever.invoke("")
        assert "422" in str(exc_info.value)


# -----------------------------------------------------------------------------
# Async exception tests
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_nvidia_retriever_async_connection_error() -> None:
    """Test NVIDIARAGConnectionError when async request fails to connect."""
    retriever = NVIDIARetriever(base_url="http://localhost:8081")

    def mock_post(*args, **kwargs):
        raise aiohttp.ClientError("Connection refused")

    with patch.object(aiohttp.ClientSession, "post", side_effect=mock_post):
        with pytest.raises(NVIDIARAGConnectionError) as exc_info:
            await retriever.ainvoke("query")
        assert "Cannot connect to RAG server" in str(exc_info.value)


@pytest.mark.asyncio
async def test_nvidia_retriever_async_timeout_error() -> None:
    """Test NVIDIARAGConnectionError when async request times out."""
    retriever = NVIDIARetriever(base_url="http://localhost:8081")

    def mock_post(*args, **kwargs):
        raise asyncio.TimeoutError("Request timed out")

    with patch.object(aiohttp.ClientSession, "post", side_effect=mock_post):
        with pytest.raises(NVIDIARAGConnectionError) as exc_info:
            await retriever.ainvoke("query")
        assert "timed out" in str(exc_info.value)


@pytest.mark.asyncio
async def test_nvidia_retriever_async_invalid_json() -> None:
    """Test NVIDIARAGServerError when async response is 200 with invalid JSON."""

    class MockResponse:
        status = 200

        async def text(self):
            return "not valid json {{{"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    retriever = NVIDIARetriever(base_url="http://localhost:8081")

    def mock_post(*args, **kwargs):
        return MockResponse()

    with patch.object(aiohttp.ClientSession, "post", side_effect=mock_post):
        with pytest.raises(NVIDIARAGServerError) as exc_info:
            await retriever.ainvoke("query")
        assert exc_info.value.status_code == 200
        assert "invalid JSON" in str(exc_info.value)


@pytest.mark.asyncio
async def test_nvidia_retriever_async_server_error() -> None:
    """Test NVIDIARAGServerError when async server returns 500."""

    class MockResponse:
        status = 500

        async def text(self):
            return "Internal Server Error"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    retriever = NVIDIARetriever(base_url="http://localhost:8081")

    with patch.object(aiohttp.ClientSession, "post", return_value=MockResponse()):
        with pytest.raises(NVIDIARAGServerError) as exc_info:
            await retriever.ainvoke("query")
        assert exc_info.value.status_code == 500
        assert "500" in str(exc_info.value)


@pytest.mark.asyncio
async def test_nvidia_retriever_async_validation_error() -> None:
    """Test NVIDIARAGValidationError when async server returns 422."""

    class MockResponse:
        status = 422

        async def text(self):
            return '{"detail": "Invalid query"}'

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    retriever = NVIDIARetriever(base_url="http://localhost:8081")

    with patch.object(aiohttp.ClientSession, "post", return_value=MockResponse()):
        with pytest.raises(NVIDIARAGValidationError) as exc_info:
            await retriever.ainvoke("")
        assert "422" in str(exc_info.value)


@pytest.mark.asyncio
async def test_nvidia_retriever_ainvoke_success() -> None:
    """Test successful async invoke."""
    from langchain_core.documents import Document

    retriever = NVIDIARetriever(
        base_url="http://localhost:8081",
        collection_names=["test_multimodal_query"],
        k=1,
    )
    expected_docs = [Document(page_content="Async content", metadata={"score": 0.85})]

    with patch.object(retriever, "_search_async", return_value=expected_docs):
        docs = await retriever.ainvoke("async query")
        assert len(docs) == 1
        assert docs[0].page_content == "Async content"
        retriever._search_async.assert_called_once_with("async query")
