from __future__ import annotations

from typing import Any, Generator, List, Literal, Optional, Sequence

from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.pydantic_v1 import BaseModel, Field, PrivateAttr

from langchain_nvidia_ai_endpoints._common import _NVIDIAClient
from langchain_nvidia_ai_endpoints._statics import Model


class Ranking(BaseModel):
    index: int
    logit: float


class NVIDIARerank(BaseDocumentCompressor):
    """
    LangChain Document Compressor that uses the NVIDIA NeMo Retriever Reranking API.
    """

    class Config:
        validate_assignment = True

    _client: _NVIDIAClient = PrivateAttr(_NVIDIAClient)

    _default_batch_size: int = 32
    _default_model_name: str = "nv-rerank-qa-mistral-4b:1"

    base_url: str = Field(
        "https://integrate.api.nvidia.com/v1",
        description="Base url for model listing an invocation",
    )
    top_n: int = Field(5, ge=0, description="The number of documents to return.")
    model: Optional[str] = Field(description="The model to use for reranking.")
    truncate: Optional[Literal["NONE", "END"]] = Field(
        description=(
            "Truncate input text if it exceeds the model's maximum token length. "
            "Default is model dependent and is likely to raise error if an "
            "input is too long."
        ),
    )
    max_batch_size: int = Field(
        _default_batch_size, ge=1, description="The maximum batch size."
    )

    def __init__(self, **kwargs: Any):
        """
        Create a new NVIDIARerank document compressor.

        This class provides access to a NVIDIA NIM for reranking. By default, it
        connects to a hosted NIM, but can be configured to connect to a local NIM
        using the `base_url` parameter. An API key is required to connect to the
        hosted NIM.

        Args:
            model (str): The model to use for reranking.
            nvidia_api_key (str): The API key to use for connecting to the hosted NIM.
            api_key (str): Alternative to nvidia_api_key.
            base_url (str): The base URL of the NIM to connect to.
            truncate (str): "NONE", "END", truncate input text if it exceeds
                            the model's context length. Default is model dependent and
                            is likely to raise an error if an input is too long.

        API Key:
        - The recommended way to provide the API key is through the `NVIDIA_API_KEY`
            environment variable.
        """
        super().__init__(**kwargs)
        self._client = _NVIDIAClient(
            base_url=self.base_url,
            model=self.model,
            default_model=self._default_model_name,
            api_key=kwargs.get("nvidia_api_key", kwargs.get("api_key", None)),
            infer_path="{base_url}/ranking",
        )
        # todo: only store the model in one place
        # the model may be updated to a newer name during initialization
        self.model = self._client.model

    @property
    def available_models(self) -> List[Model]:
        """
        Get a list of available models that work with NVIDIARerank.
        """
        return self._client.get_available_models(self.__class__.__name__)

    @classmethod
    def get_available_models(
        cls,
        **kwargs: Any,
    ) -> List[Model]:
        """
        Get a list of available models that work with NVIDIARerank.
        """
        return cls(**kwargs).available_models

    # todo: batching when len(documents) > endpoint's max batch size
    def _rank(self, documents: List[str], query: str) -> List[Ranking]:
        payload = {
            "model": self.model,
            "query": {"text": query},
            "passages": [{"text": passage} for passage in documents],
        }
        if self.truncate:
            payload["truncate"] = self.truncate
        response = self._client.client.get_req(payload=payload)
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

        def batch(ls: list, size: int) -> Generator[List[Document], None, None]:
            for i in range(0, len(ls), size):
                yield ls[i : i + size]

        doc_list = list(documents)
        results = []
        for doc_batch in batch(doc_list, self.max_batch_size):
            rankings = self._rank(
                query=query, documents=[d.page_content for d in doc_batch]
            )
            for ranking in rankings:
                assert (
                    0 <= ranking.index < len(doc_batch)
                ), "invalid response from server: index out of range"
                doc = doc_batch[ranking.index]
                doc.metadata["relevance_score"] = ranking.logit
                results.append(doc)

        # if we batched, we need to sort the results
        if len(doc_list) > self.max_batch_size:
            results.sort(key=lambda x: x.metadata["relevance_score"], reverse=True)

        return results[: self.top_n]
