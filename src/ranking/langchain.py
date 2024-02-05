from __future__ import annotations

import os
from typing import Optional, Sequence

import requests
from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.documents import Document

endpoint = os.environ["NVIDIA_NEMO_RERANKING_ENDPOINT"]


class Reranker(BaseDocumentCompressor):
    """Document compressor that uses `NV Rerank API`."""

    def __init__(self) -> None:
        super().__init__()

    def fetch_reranking(
        self, query: str, documents: list[str], topN: int
    ) -> tuple[list[int], list[float]]:
        request = {
            "model": "ignored",
            "query": {"text": query},
            "passages": [{"text": passage} for passage in documents],
        }

        url = f"{endpoint}/v1/ranking"
        response = requests.post(url, json=request)
        rankings = response.json()[
            "rankings"
        ]  # list of {"index": int, "score": float} with length equal to passages
        idx = [rankings[i]["index"] for i in range(topN)]
        score = [rankings[i]["score"] for i in range(topN)]
        return idx, score

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """
        Compress documents using NVIDIA's rerank Microservice API.

        Args:
            documents: A sequence of documents to compress.
            query: The query to use for compressing the documents.
            callbacks: Callbacks to run during the compression process.

        Returns:
            A sequence of compressed documents.
        """
        if len(documents) == 0:  # to avoid empty api call
            return []
        doc_list = list(documents)
        _docs = [d.page_content for d in doc_list]
        idx_ls, scores = self.fetch_reranking(query=query, documents=_docs, topN=3)
        final_results = []
        for idx, score in zip(idx_ls, scores):
            doc = doc_list[idx]
            doc.metadata["relevance_score"] = score
            final_results.append(doc)
        return final_results

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Asynchronously run the LLM on the given prompt and input."""
        # This is just a placeholder implementation. Replace with your actual implementation.
        raise NotImplementedError("Asynchronous document compression is not yet implemented.")
