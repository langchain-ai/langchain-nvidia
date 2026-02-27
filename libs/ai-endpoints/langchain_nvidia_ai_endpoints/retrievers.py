"""NVIDIARetriever for NVIDIA RAG Blueprint /search endpoint.

Connects to a containerized NVIDIA RAG server that exposes the /v1/search endpoint.
Supports all DocumentSearch API parameters.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, List, Optional, Union

import aiohttp
import requests
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field

# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------


class NVIDIARAGError(Exception):
    """Base exception for NVIDIA RAG retriever errors."""

    pass


class NVIDIARAGConnectionError(NVIDIARAGError):
    """Raised when the RAG server endpoint is unreachable.

    Common causes:
    - RAG server container is not running
    - Wrong base_url or port (default: 8081)
    - Network/firewall blocking the connection
    """

    pass


class NVIDIARAGServerError(NVIDIARAGError):
    """Raised when the RAG server returns an error response.

    Includes the HTTP status code and response body for debugging.
    """

    def __init__(self, message: str, status_code: Optional[int] = None, body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class NVIDIARAGValidationError(NVIDIARAGError):
    """Raised when the request payload is invalid (e.g. 422 Unprocessable Entity)."""

    def __init__(self, message: str, body: str = ""):
        super().__init__(message)
        self.body = body


# -----------------------------------------------------------------------------
# NVIDIARetriever
# -----------------------------------------------------------------------------

_SEARCH_PATH = "/v1/search"
_DEFAULT_TIMEOUT = 60


class NVIDIARetriever(BaseRetriever):
    """LangChain retriever that queries the NVIDIA RAG Blueprint /v1/search endpoint.

    Targets containerized RAG deployments where the rag-server exposes the search API.
    Supports all DocumentSearch parameters from the RAG server schema.

    Example:
        .. code-block:: python

            from langchain_nvidia_ai_endpoints import NVIDIARetriever

            retriever = NVIDIARetriever(
                base_url="http://localhost:8081",
                collection_names=["test_multimodal_query"],
                k=4,
            )
            docs = retriever.invoke("What is machine learning?")
    """

    model_config = ConfigDict(validate_assignment=True)

    base_url: str = Field(
        ...,
        description="Base URL of the RAG server (e.g. http://localhost:8081). Must not include /v1/search.",
    )

    k: int = Field(
        default=10,
        ge=0,
        le=25,
        description="Number of document chunks to return (reranker_top_k). Maps to reranker_top_k in the API.",
    )

    collection_names: List[str] = Field(
        default_factory=lambda: ["multimodal_data"],
        description="Names of collections to search in the vector database.",
    )

    vdb_top_k: int = Field(
        default=100,
        ge=0,
        le=400,
        description="Number of top results to retrieve from the vector database before reranking.",
    )

    vdb_endpoint: str = Field(
        default="http://milvus:19530",
        description="Endpoint URL of the vector database server.",
    )

    enable_reranker: bool = Field(
        default=True,
        description="Enable or disable reranking by the ranker model.",
    )

    enable_query_rewriting: bool = Field(
        default=False,
        description="Enable or disable query rewriting.",
    )

    enable_filter_generator: bool = Field(
        default=False,
        description="Enable or disable automatic filter expression generation from natural language.",
    )

    enable_citations: bool = Field(
        default=True,
        description="Enable or disable image/table/chart citations as part of response.",
    )

    filter_expr: Union[str, List[dict], None] = Field(
        default=None,
        description="Filter expression to filter retrieved documents (Milvus filter syntax).",
    )

    confidence_threshold: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score threshold for filtering chunks. Requires enable_reranker=True.",
    )

    embedding_model: Optional[str] = Field(
        default=None,
        description="Name of the embedding model. Uses server default if not set.",
    )

    embedding_endpoint: Optional[str] = Field(
        default=None,
        description="Endpoint URL for the embedding model server.",
    )

    reranker_model: Optional[str] = Field(
        default=None,
        description="Name of the reranker model. Uses server default if not set.",
    )

    reranker_endpoint: Optional[str] = Field(
        default=None,
        description="Endpoint URL for the reranker model server.",
    )

    messages: List[dict] = Field(
        default_factory=list,
        description="Conversation history for context-aware retrieval. Last message should have role 'user'.",
    )

    timeout: float = Field(
        default=_DEFAULT_TIMEOUT,
        ge=1.0,
        description="HTTP request timeout in seconds.",
    )

    def _build_payload(self, query: str) -> dict[str, Any]:
        """Build the DocumentSearch request payload."""
        payload: dict[str, Any] = {
            "query": query,
            "reranker_top_k": self.k,
            "vdb_top_k": self.vdb_top_k,
            "vdb_endpoint": self.vdb_endpoint,
            "collection_names": self.collection_names,
            "messages": self.messages,
            "enable_query_rewriting": self.enable_query_rewriting,
            "enable_reranker": self.enable_reranker,
            "enable_filter_generator": self.enable_filter_generator,
            "enable_citations": self.enable_citations,
            "confidence_threshold": self.confidence_threshold,
        }
        if self.filter_expr is not None:
            payload["filter_expr"] = self.filter_expr
        if self.embedding_model is not None:
            payload["embedding_model"] = self.embedding_model
        if self.embedding_endpoint is not None:
            payload["embedding_endpoint"] = self.embedding_endpoint
        if self.reranker_model is not None:
            payload["reranker_model"] = self.reranker_model
        if self.reranker_endpoint is not None:
            payload["reranker_endpoint"] = self.reranker_endpoint
        return payload

    def _results_to_documents(self, results: List[dict]) -> List[Document]:
        """Convert API results (SourceResult) to LangChain Documents."""
        docs: List[Document] = []
        for r in results:
            content = r.get("content", "")
            metadata: dict[str, Any] = {
                "document_id": r.get("document_id", ""),
                "document_name": r.get("document_name", ""),
                "document_type": r.get("document_type", "text"),
                "score": r.get("score", 0.0),
            }
            if "metadata" in r and isinstance(r["metadata"], dict):
                metadata.update(r["metadata"])
            docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def _search_sync(self, query: str) -> List[Document]:
        """Synchronous implementation of search using requests."""
        url = f"{self.base_url.rstrip('/')}{_SEARCH_PATH}"
        payload = self._build_payload(query)

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
                headers={"Content-Type": "application/json"},
            )
        except requests.exceptions.ConnectionError as e:
            raise NVIDIARAGConnectionError(
                f"Cannot connect to RAG server at {url}. "
                f"Ensure the rag-server container is running and base_url is correct (default port: 8081). Error: {e}"
            ) from e
        except requests.exceptions.Timeout as e:
            raise NVIDIARAGConnectionError(
                f"Request to RAG server at {url} timed out after {self.timeout}s. "
                f"Server may be overloaded or unreachable."
            ) from e
        except requests.exceptions.RequestException as e:
            raise NVIDIARAGConnectionError(
                f"Request to RAG server at {url} failed. Error: {e}"
            ) from e

        body = response.text
        if response.status_code == 200:
            try:
                data = json.loads(body)
            except json.JSONDecodeError as e:
                raise NVIDIARAGServerError(
                    f"RAG server returned invalid JSON. Response: {body[:500]}",
                    status_code=response.status_code,
                    body=body,
                ) from e
            results = data.get("results", [])
            return self._results_to_documents(results)
        elif response.status_code == 422:
            raise NVIDIARAGValidationError(
                f"RAG server validation error (422). Check query and parameters. Response: {body[:500]}",
                body=body,
            )
        else:
            raise NVIDIARAGServerError(
                f"RAG server returned HTTP {response.status_code}. Response: {body[:500]}",
                status_code=response.status_code,
                body=body,
            )

    async def _search_async(self, query: str) -> List[Document]:
        """Async implementation of search."""
        url = f"{self.base_url.rstrip('/')}{_SEARCH_PATH}"
        payload = self._build_payload(query)
        timeout = aiohttp.ClientTimeout(total=self.timeout)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    body = await response.text()
                    if response.status == 200:
                        try:
                            data = json.loads(body)
                        except json.JSONDecodeError as e:
                            raise NVIDIARAGServerError(
                                f"RAG server returned invalid JSON. Response: {body[:500]}",
                                status_code=response.status,
                                body=body,
                            ) from e
                        results = data.get("results", [])
                        return self._results_to_documents(results)
                    elif response.status == 422:
                        raise NVIDIARAGValidationError(
                            f"RAG server validation error (422). Check query and parameters. Response: {body[:500]}",
                            body=body,
                        )
                    else:
                        raise NVIDIARAGServerError(
                            f"RAG server returned HTTP {response.status}. Response: {body[:500]}",
                            status_code=response.status,
                            body=body,
                        )
        except aiohttp.ClientError as e:
            raise NVIDIARAGConnectionError(
                f"Cannot connect to RAG server at {url}. "
                f"Ensure the rag-server container is running and base_url is correct. Error: {e}"
            ) from e
        except asyncio.TimeoutError as e:
            raise NVIDIARAGConnectionError(
                f"Request to RAG server at {url} timed out after {self.timeout}s. "
                f"Server may be overloaded or unreachable."
            ) from e

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve documents from the NVIDIA RAG /v1/search endpoint."""
        return self._search_sync(query)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Async retrieve documents from the NVIDIA RAG /v1/search endpoint."""
        return await self._search_async(query)
