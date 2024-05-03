from __future__ import annotations

from typing import Any, Generator, List, Optional, Sequence

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

    def __init__(self, **kwargs: Any):
        """
        Create a new NVIDIARerank document compressor.

        Unless you plan to use the "nim" mode, you need to provide an API key. Your
        options are -
         0. Pass the key as the nvidia_api_key parameter.
         1. Pass the key as the api_key parameter.
         2. Set the NVIDIA_API_KEY environment variable, recommended.
        Precedence is in the order listed above.
        """
        super().__init__(**kwargs)
        self._client = _NVIDIAClient(
            model=self.model,
            api_key=kwargs.get("nvidia_api_key", kwargs.get("api_key", None)),
        )

    @property
    def available_models(self) -> List[Model]:
        """
        Get a list of available models that work with NVIDIARerank.
        """
        if self._client.curr_mode in ["nim", "open"]:
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
        self = cls(**kwargs).mode(mode=mode, **kwargs)
        if mode in ["nim", "open"]:
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
