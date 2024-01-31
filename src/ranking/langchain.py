from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.pydantic_v1 import Extra, root_validator

from langchain.callbacks.manager import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain.utils import get_from_dict_or_env
import os
import requests


nim_url = os.environ["NVReranker_URL"] 


class Reranker(BaseDocumentCompressor):
    """Document compressor that uses `NV Rerank API`."""
    def __init__(self):        
        super().__init__()        
    
    def fetch_reranking(self, query,documents, topN):
        request = {
          "model": "ignored",
          "query": {"text": query},
          "passages": [{"text": passage} for passage in documents]
        }
        
        response = requests.post(nim_url, json=request)
        #print(response.json())
        rankings = response.json()["rankings"] # list of {"index": int, "score": float} with length equal to passages
        idx=[rankings[i]['index'] for i in range(topN)]
        score=[rankings[i]['score'] for i in range(topN)]
        #print(f"high scoring passage: {passages[rankings[0]['index']]}")
        #print(f"low scoring passage: {passages[rankings[-1]['index']]}")
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
        _docs = [d.page_content.strip().replace('\n','') for d in doc_list]
        idx_ls,scores = self.fetch_reranking(
            query=query,documents=_docs, topN=3
        )
        final_results = []
        for idx,score in zip(idx_ls,scores):
            doc = doc_list[idx]
            doc.metadata["relevance_score"] = score
            final_results.append(doc)
        return final_results
    def acompress_documents(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
        ) -> LLMResult:            
            """Asynchronously run the LLM on the given prompt and input."""
            # This is just a placeholder implementation. Replace with your actual implementation.
            pass