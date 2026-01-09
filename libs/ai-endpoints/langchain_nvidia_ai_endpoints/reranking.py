from __future__ import annotations

import json
import warnings
from typing import Any, Dict, Generator, List, Literal, Optional, Sequence

from langchain_core.callbacks.manager import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
)

from langchain_nvidia_ai_endpoints._common import _NVIDIAClient
from langchain_nvidia_ai_endpoints._statics import Model


class Ranking(BaseModel):
    index: int
    logit: float


_DEFAULT_MODEL_NAME: str = "nvidia/llama-3.2-nv-rerankqa-1b-v2"
_DEFAULT_BATCH_SIZE: int = 32


class NVIDIARerank(BaseDocumentCompressor):
    """
    LangChain Document Compressor that uses the NVIDIA NeMo Retriever Reranking API.
    """

    model_config = ConfigDict(
        validate_assignment=True,
    )

    _client: _NVIDIAClient = PrivateAttr()

    base_url: Optional[str] = Field(
        default=None,
        description="Base url for model listing an invocation",
    )

    top_n: int = Field(5, ge=0, description="The number of documents to return.")

    model: Optional[str] = Field(None, description="The model to use for reranking.")

    truncate: Optional[Literal["NONE", "END"]] = Field(
        default=None,
        description=(
            "Truncate input text if it exceeds the model's maximum token length. "
            "Default is model dependent and is likely to raise error if an "
            "input is too long."
        ),
    )

    max_batch_size: int = Field(
        _DEFAULT_BATCH_SIZE, ge=1, description="The maximum batch size."
    )

    default_headers: dict = Field(
        default_factory=dict,
        description="Default headers merged into all requests.",
    )

    extra_headers: dict = Field(
        default_factory=dict,
        description="Deprecated: Use 'default_headers' instead. "
        "Extra headers to include in the request.",
    )

    def __init__(
        self,
        *,
        nvidia_api_key: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        """Create a new `NVIDIARerank` document compressor.

        This class provides access to a NVIDIA NIM for reranking. By default, it
        connects to a hosted NIM, but can be configured to connect to a local NIM
        using the `base_url` parameter.

        An API key is required to connect to the hosted NIM.

        Args:
            nvidia_api_key: The API key to use for connecting to the hosted NIM.
            api_key: Alternative to `nvidia_api_key`.
            **kwargs: Additional parameters passed to the underlying client.

        The recommended way to provide the API key is through the `NVIDIA_API_KEY`
        environment variable.

        Base URL:

        - Connect to a self-hosted model with NVIDIA NIM using the `base_url` arg to
            link to the local host at `localhost:8000`:

            ```python
            ranker = NVIDIARerank(base_url="http://localhost:8000/v1")
            ```

        Example:
            ```python
            from langchain_nvidia_ai_endpoints import NVIDIARerank
            from langchain_core.documents import Document

            query = "What is the GPU memory bandwidth of H100 SXM?"
            passages = [
                "The Hopper GPU is paired with the Grace CPU using NVIDIA's ultra-fast
                chip-to-chip interconnect, delivering 900GB/s of bandwidth, 7X faster
                than PCIe Gen5. This innovative design will deliver up to 30X higher
                aggregate system memory bandwidth to the GPU compared to today's fastest
                servers and up to 10X higher performance for applications running
                terabytes of data.",

                "A100 provides up to 20X higher performance over the prior generation
                and can be partitioned into seven GPU instances to dynamically adjust to
                shifting demands. The A100 80GB debuts the world's fastest memory
                bandwidth at over 2 terabytes per second (TB/s) to run the largest
                models and datasets.",

                "Accelerated servers with H100 deliver the compute power—along with 3
                terabytes per second (TB/s) of memory bandwidth per GPU and scalability
                with NVLink and NVSwitch™.",
            ]

            client = NVIDIARerank(
                model="nvidia/llama-3.2-nv-rerankqa-1b-v2",
                api_key="$API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC"
            )

            response = client.compress_documents(
                query=query,
                documents=[Document(page_content=passage) for passage in passages]
            )

            print(f"Most relevant: {response[0].page_content}"
                f"Least relevant: {response[-1].page_content}"
            )

            # Most relevant: Accelerated servers with H100 deliver the compute
            # power—along with 3 terabytes per second (TB/s) of memory bandwidth per GPU
            # and scalability with NVLink and NVSwitch™.
            # Least relevant: A100 provides up to 20X higher performance over the prior
            # generation and can be partitioned into seven GPU instances to dynamically
            # adjust to shifting demands. The A100 80GB debuts the world's fastest
            # memory bandwidth at over 2 terabytes per second (TB/s) to run the largest
            # models and datasets.
            ```
        """

        super().__init__(**kwargs)

        # Handle backwards compatibility: if extra_headers is set but default_headers
        # is not, use extra_headers
        if self.extra_headers and not self.default_headers:
            warnings.warn(
                "The 'extra_headers' parameter is deprecated and will be removed "
                "in a future version. Please use 'default_headers' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.default_headers = self.extra_headers

        # allow nvidia_base_url as an alternative for base_url
        base_url = kwargs.pop("nvidia_base_url", self.base_url)

        # allow nvidia_api_key as an alternative for api_key
        api_key = nvidia_api_key or api_key

        # Extract verify_ssl from kwargs, default to True
        verify_ssl = kwargs.pop("verify_ssl", True)

        self._client = _NVIDIAClient(
            **({"base_url": base_url} if base_url else {}),  # only pass if set
            mdl_name=self.model,
            default_hosted_model_name=_DEFAULT_MODEL_NAME,
            **({"api_key": api_key} if api_key else {}),  # only pass if set
            infer_path="{base_url}/ranking",
            cls=self.__class__.__name__,
            verify_ssl=verify_ssl,
        )

        # todo: only store the model in one place
        # the model may be updated to a newer name during initialization
        self.model = self._client.mdl_name

        # same for base_url
        self.base_url = self._client.base_url

    @property
    def available_models(self) -> List[Model]:
        """Get a list of available models that work with `NVIDIARerank`."""
        return self._client.get_available_models(self.__class__.__name__)

    @classmethod
    def get_available_models(
        cls,
        **kwargs: Any,
    ) -> List[Model]:
        """Get a list of available models that work with `NVIDIARerank`."""
        return cls(**kwargs).available_models

    def _prepare_payload(self, documents: List[str], query: str) -> Dict[str, Any]:
        """Prepare payload for both sync and async methods.

        Args:
            documents: List of document texts to rank
            query: Query text

        Returns:
            Payload dictionary
        """
        payload = {
            "model": self.model,
            "query": {"text": query},
            "passages": [{"text": passage} for passage in documents],
        }
        if self.truncate:
            payload["truncate"] = self.truncate
        return payload

    def _process_response(self, result: Dict[str, Any]) -> List[Ranking]:
        """Process response for both sync and async methods.

        Args:
            result: Parsed JSON response from the API

        Returns:
            List of rankings limited to top_n
        """
        rankings = result["rankings"]
        # todo: callback support
        return [Ranking(**ranking) for ranking in rankings[: self.top_n]]

    @staticmethod
    def _batch(ls: list, size: int) -> Generator[List[Document], None, None]:
        """Batch documents into chunks of specified size.

        Args:
            ls: List to batch
            size: Batch size

        Yields:
            Batches of documents
        """
        for i in range(0, len(ls), size):
            yield ls[i : i + size]

    def _process_batch_rankings(
        self,
        doc_batch: List[Document],
        rankings: List[Ranking],
        results: List[Document],
    ) -> None:
        """Process rankings for a batch of documents.

        Args:
            doc_batch: Batch of documents
            rankings: Rankings for the batch
            results: List to append processed documents to
        """
        for ranking in rankings:
            assert (
                0 <= ranking.index < len(doc_batch)
            ), "invalid response from server: index out of range"
            doc = doc_batch[ranking.index]
            doc.metadata["relevance_score"] = ranking.logit
            results.append(doc)

    @staticmethod
    def _sort_by_relevance(results: List[Document]) -> None:
        """Sort results by relevance score in descending order.

        Args:
            results: List of documents to sort in-place
        """
        results.sort(key=lambda x: x.metadata["relevance_score"], reverse=True)

    # todo: batching when len(documents) > endpoint's max batch size
    def _rank(self, documents: List[str], query: str) -> List[Ranking]:
        payload = self._prepare_payload(documents, query)
        response = self._client.get_req(
            payload=payload, extra_headers=self.default_headers
        )
        if response.status_code != 200:
            response.raise_for_status()
        # todo: handle errors
        return self._process_response(response.json())

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

        doc_list = list(documents)
        results: List[Document] = []
        for doc_batch in self._batch(doc_list, self.max_batch_size):
            rankings = self._rank(
                query=query, documents=[d.page_content for d in doc_batch]
            )
            self._process_batch_rankings(doc_batch, rankings, results)

        # if we batched, we need to sort the results
        if len(doc_list) > self.max_batch_size:
            results.sort(key=lambda x: x.metadata["relevance_score"], reverse=True)

        return results[: self.top_n]

    async def _arank(self, documents: List[str], query: str) -> List[Ranking]:
        """Async version of _rank."""
        payload = self._prepare_payload(documents, query)
        response_text = await self._client.aget_req(
            payload=payload, extra_headers=self.default_headers
        )
        result = json.loads(response_text)
        return self._process_response(result)

    async def acompress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Async version of compress_documents."""
        if len(documents) == 0 or self.top_n < 1:
            return []

        doc_list = list(documents)
        results: List[Document] = []
        for doc_batch in self._batch(doc_list, self.max_batch_size):
            rankings = await self._arank(
                query=query, documents=[d.page_content for d in doc_batch]
            )
            self._process_batch_rankings(doc_batch, rankings, results)

        # if we batched, we need to sort the results
        if len(doc_list) > self.max_batch_size:
            results.sort(key=lambda x: x.metadata["relevance_score"], reverse=True)

        return results[: self.top_n]
