from __future__ import annotations

import os
from typing import Optional, Sequence

import requests
from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.documents import Document
from pydantic import BaseModel


class Ranking(BaseModel):
    index: int
    score: float
    logit: Optional[float] = None


class Reranker(BaseDocumentCompressor):
    """
    LangChain Document Compressor that uses the NVIDIA NeMo Retriever Reranking API.
    """

    top_n: int = 5
    """The max number of documents to return."""
    model: str = "ignored"
    """The model to use for reranking."""
    endpoint: str = os.environ.get("NVIDIA_NEMO_RERANKING_ENDPOINT", "http://localhost:1976")
    """The endpoint to use for reranking."""

    def __init__(self, top_k: Optional[int] = None) -> None:
        super().__init__()

    # todo: batching when len(documents) > endpoint's max batch size
    def _rank(self, documents: list[str], query: str) -> list[Ranking]:
        request = {
            "model": self.model,
            "query": {"text": query},
            "passages": [{"text": passage} for passage in documents],
        }

        url = f"{self.endpoint}/v1/ranking"
        response = requests.post(url, json=request)
        # todo: handle errors
        rankings = response.json()["rankings"]
        return [Ranking(**ranking) for ranking in rankings[: self.top_n]]

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using the NVIDIA NeMo Retriever Reranking microservice API.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        if len(documents) == 0 or self.top_n < 1:
            return []
        # todo: consider optimization for len(documents) == 1
        doc_list = list(documents)
        _docs = [d.page_content for d in doc_list]
        rankings = self._rank(query=query, documents=_docs)
        final_results = []
        for ranking in rankings:
            doc = doc_list[ranking.index]
            doc.metadata["relevance_score"] = ranking.score
            final_results.append(doc)
        return final_results
