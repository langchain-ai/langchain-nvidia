from __future__ import annotations

import os
from typing import Any, List, Optional, Sequence

from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.pydantic_v1 import BaseModel, Field, PrivateAttr

from ._common import _MODE_TYPE, _NVIDIAClient
from ._statics import Model


class Ranking(BaseModel):
    index: int
    logit: float


class NVIDIARerank(BaseDocumentCompressor):
    """
    LangChain Document Compressor that uses the NVIDIA NeMo Retriever Reranking API.
    """

    _client: _NVIDIAClient = PrivateAttr(_NVIDIAClient)

    top_n: int = 5
    """The max number of documents to return."""
    model: str = Field(
        "ai-rerank-qa-mistral-4b", description="The model to use for reranking."
    )
    """The model to use for reranking."""
    _base_url: str = os.environ.get("NIM_ENDPOINT", "http://localhost:1976/v1")
    """The endpoint to use for reranking."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._client = _NVIDIAClient(model=self.model)

    @property
    def available_models(self) -> list[Model]:
        """
        Get a list of available models that work with ChatNVIDIA.
        """
        return self._client.get_available_models(
            client=self._client,
            filter=self.__class__.__name__,
        )

    @classmethod
    def get_available_models(
        cls,
        mode: Optional[_MODE_TYPE] = None,
        list_all: bool = False,
        **kwargs: Any,
    ) -> List[Model]:
        """
        Get a list of available models. These models will work with the ChatNVIDIA
        interface.

        Use the mode parameter to specify the mode to use. See the docs for mode()
        to understand additional keyword arguments required when setting mode.

        It is possible to get a list of all models, including those that are not
        chat models, by setting the list_all parameter to True.
        """
        self = cls(**kwargs).mode(mode=mode, **kwargs)
        return self._client.get_available_models(
            mode=mode,
            list_all=list_all,
            client=self._client,
            filter=cls.__name__,
            **kwargs,
        )

    def mode(
        self,
        mode: Optional[_MODE_TYPE] = "nvidia",
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> NVIDIARerank:
        """
        Change the mode.

        There are two modes, "nvidia" and "nim". The "nvidia" mode is the default mode
        and is used to interact with hosted NVIDIA AI endpoints. The "nim" mode is
        used to interact with NVIDIA NIM endpoints, which are typically hosted
        on-premises.

        For the "nvidia" mode, the "api_key" parameter is available to specify your
        API key. If not specified, the NVIDIA_API_KEY environment variable will be used.

        For the "nim" mode, the "base_url" and "model" parameters are required. Set
        base_url to the url of your NVIDIA NIM endpoint. For instance,
        "https://localhost:9999/v1". Additionally, the "model" parameter must be set
        to the name of the model inside the NIM.
        """
        self._client = self._client.mode(
            mode=mode,
            base_url=base_url,
            model=model,
            api_key=api_key,
            infer_path="{base_url}/ranking",
            **kwargs,
        )
        self.model = self._client.model
        return self

    # todo: batching when len(documents) > endpoint's max batch size
    def _rank(self, documents: list[str], query: str) -> List[Ranking]:
        response = self._client.client.get_req(
            model_name=self.model,
            payload={
                "model": "nv-rerank-qa-mistral-4b:1",
                "query": {"text": query},
                "passages": [{"text": passage} for passage in documents],
            },
            endpoint="infer",
        )
        if response.status_code != 200:
            response.raise_for_status()
        # todo: handle errors
        rankings = response.json()["rankings"]
        # todo: callback support
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
            doc.metadata["relevance_score"] = ranking.logit
            final_results.append(doc)
        return final_results
