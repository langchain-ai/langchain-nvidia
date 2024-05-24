from __future__ import annotations

from typing import Any, Generator, List, Optional, Sequence

from langchain_core._api import deprecated, warn_deprecated
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

    class Config:
        validate_assignment = True

    _client: _NVIDIAClient = PrivateAttr(_NVIDIAClient)

    _default_batch_size: int = 32
    _deprecated_model: str = "ai-rerank-qa-mistral-4b"
    _default_model_name: str = "nv-rerank-qa-mistral-4b:1"

    top_n: int = Field(5, ge=0, description="The number of documents to return.")
    model: str = Field(
        _default_model_name, description="The model to use for reranking."
    )
    max_batch_size: int = Field(
        _default_batch_size, ge=1, description="The maximum batch size."
    )
    _is_hosted: bool = PrivateAttr(True)

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

        API Key:
        - The recommended way to provide the API key is through the `NVIDIA_API_KEY`
            environment variable.
        """
        super().__init__(**kwargs)
        self._client = _NVIDIAClient(
            model=self.model,
            api_key=kwargs.get("nvidia_api_key", kwargs.get("api_key", None)),
        )
        if base_url := kwargs.get("base_url", None):
            # todo: detect if the base_url points to hosted NIM, this depends on
            #       moving from NVCF inference to API Catalog inference
            self._is_hosted = False
            self._client.client.base_url = base_url
            self._client.client.endpoints["infer"] = "{base_url}/ranking"
            self._client.client.endpoints = {
                "infer": "{base_url}/ranking",
                "status": None,
                "models": None,
            }

    @property
    def available_models(self) -> List[Model]:
        """
        Get a list of available models that work with NVIDIARerank.
        """
        if self._client.curr_mode == "nim" or not self._is_hosted:
            # local NIM supports a single model and no /models endpoint
            models = [
                Model(
                    id=NVIDIARerank._default_model_name,
                    model_name=NVIDIARerank._default_model_name,
                    model_type="ranking",
                    client="NVIDIARerank",
                    path="magic",
                ),
                Model(
                    id=NVIDIARerank._deprecated_model,
                    model_name=NVIDIARerank._default_model_name,
                    model_type="ranking",
                    client="NVIDIARerank",
                    path="magic",
                ),
            ]
        else:
            models = self._client.get_available_models(
                client=self._client,
                filter=self.__class__.__name__,
            )
        return models

    @classmethod
    def get_available_models(
        cls,
        mode: Optional[_MODE_TYPE] = None,
        list_all: bool = False,
        **kwargs: Any,
    ) -> List[Model]:
        """
        Get a list of available models. These models will work with the NVIDIARerank
        interface.

        Use the mode parameter to specify the mode to use. See the docs for mode()
        to understand additional keyword arguments required when setting mode.

        It is possible to get a list of all models, including those that are not
        chat models, by setting the list_all parameter to True.
        """
        if mode is not None:
            warn_deprecated(since="0.0.17", removal="0.1.0", alternative="`base_url`")
        self = cls(**kwargs).mode(mode=mode, **kwargs)
        if mode == "nim" or not self._is_hosted:
            # ignoring list_all because there is one
            models = self.available_models
        else:
            models = self._client.get_available_models(
                mode=mode,
                list_all=list_all,
                client=self._client,
                filter=cls.__name__,
                **kwargs,
            )
        return models

    @deprecated(
        since="0.0.17",
        removal="0.1.0",
        alternative="`base_url` to constructor",
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
        Deprecated: use NVIDIARerank(base_url=...) instead.
        """
        # set a default base_url for nim mode
        if not base_url and mode == "nim":
            base_url = "http://localhost:1976/v1"
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
    def _rank(self, documents: List[str], query: str) -> List[Ranking]:
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
                doc = doc_batch[ranking.index]
                doc.metadata["relevance_score"] = ranking.logit
                results.append(doc)

        # if we batched, we need to sort the results
        if len(doc_list) > self.max_batch_size:
            results.sort(key=lambda x: x.metadata["relevance_score"], reverse=True)

        return results[: self.top_n]
