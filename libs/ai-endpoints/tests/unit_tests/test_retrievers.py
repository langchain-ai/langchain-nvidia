"""Unit tests for NVIDIARetriever."""

import json
from unittest.mock import patch

import pytest
import requests
import requests_mock

from langchain_nvidia_ai_endpoints.retrievers import (
    NVIDIARAGConnectionError,
    NVIDIARAGServerError,
    NVIDIARAGValidationError,
    NVIDIARetriever,
)


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
        assert "rag-server container" in str(exc_info.value)


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
