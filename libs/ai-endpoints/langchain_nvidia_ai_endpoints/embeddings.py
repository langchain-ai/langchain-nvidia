import json
from typing import Any, Dict, List, Literal, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.outputs.llm_result import LLMResult
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
)

from langchain_nvidia_ai_endpoints._common import _NVIDIAClient
from langchain_nvidia_ai_endpoints._statics import Model
from langchain_nvidia_ai_endpoints.callbacks import usage_callback_var

_DEFAULT_MODEL_NAME: str = "nvidia/nv-embedqa-e5-v5"
_DEFAULT_BATCH_SIZE: int = 50


class NVIDIAEmbeddings(BaseModel, Embeddings):
    """
    Client to NVIDIA embeddings models.

    Attributes:
        model: The name of the model to use
        truncate: `'NONE'`, `'START'`, `'END'`, truncate input text if it exceeds the
            model's maximum token length. Default is `'NONE'`, which raises an error if
            an input is too long.
        dimensions: The number of dimensions for the embeddings. This parameter is
            not supported by all models.
    """

    model_config = ConfigDict(
        validate_assignment=True,
    )

    _client: _NVIDIAClient = PrivateAttr()

    base_url: Optional[str] = Field(
        default=None,
        description="Base url for model listing an invocation",
    )

    model: Optional[str] = Field(None, description="Name of the model to invoke")

    truncate: Literal["NONE", "START", "END"] = Field(
        default="NONE",
        description=(
            "Truncate input text if it exceeds the model's maximum token length. "
            "Default is 'NONE', which raises an error if an input is too long."
        ),
    )

    dimensions: Optional[int] = Field(
        default=None,
        description=(
            "The number of dimensions for the embeddings. This parameter is not "
            "supported by all models."
        ),
    )

    max_batch_size: int = Field(default=_DEFAULT_BATCH_SIZE)

    default_headers: dict = Field(
        default_factory=dict,
        description="Default headers merged into all requests.",
    )

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        nvidia_api_key: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        trucate: Optional[Literal["NONE", "START", "END"]] = None,
        dimensions: Optional[int] = None,
        **kwargs: Any,
    ):
        """
        Create a new `NVIDIAEmbeddings` embedder.

        This class provides access to a NVIDIA NIM for embedding. By default, it
        connects to a hosted NIM, but can be configured to connect to a local NIM
        using the `base_url` parameter.

        An API key is required to connect to the hosted NIM.

        Args:
            model: The model to use for embedding.
            nvidia_api_key: The API key to use for connecting to the hosted NIM.
            api_key: Alternative to `nvidia_api_key`.
            base_url: The base URL of the NIM to connect to.
                Format for base URL is http://host:port
            trucate: `'NONE'`, `'START'`, `'END'`, truncate input text if it
                exceeds the model's context length. Default is `'NONE'`, which raises
                an error if an input is too long.
            dimensions: The number of dimensions for the embeddings. This
                parameter is not supported by all models.
            **kwargs: Additional parameters passed to the underlying client.

        The recommended way to provide the API key is through the `NVIDIA_API_KEY`
        environment variable.

        Base URL:

        - Connect to a self-hosted model with NVIDIA NIM using the `base_url` arg to
            link to the local host at `localhost:8000`:

            ```python
            embedder = NVIDIAEmbeddings(base_url="http://localhost:8080/v1")
            ```
        """
        init_kwargs: Dict[str, Any] = {}
        if model is not None:
            init_kwargs["model"] = model
        if base_url is not None:
            init_kwargs["base_url"] = base_url
        if trucate is not None:
            init_kwargs["truncate"] = trucate
        if dimensions is not None:
            init_kwargs["dimensions"] = dimensions

        init_kwargs.update(kwargs)

        super().__init__(**init_kwargs)

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
            infer_path="{base_url}/embeddings",
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
        """Get a list of available models that work with `NVIDIAEmbeddings`."""
        return self._client.get_available_models(self.__class__.__name__)

    @classmethod
    def get_available_models(
        cls,
        **kwargs: Any,
    ) -> List[Model]:
        """Get a list of available models that work with `NVIDIAEmbeddings`."""
        return cls(**kwargs).available_models

    def _prepare_payload(
        self, texts: List[str], model_type: Literal["passage", "query"]
    ) -> Dict[str, Any]:
        """Prepare payload for both sync and async methods.

        Args:
            texts: List of texts to embed
            model_type: Type of embedding ("passage" or "query")

        Returns:
            Payload dictionary
        """
        payload: Dict[str, Any] = {
            "input": texts,
            "model": self.model,
            "encoding_format": "float",
            "input_type": model_type,
        }
        if self.truncate:
            payload["truncate"] = self.truncate
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        return payload

    def _process_response(self, result: Dict[str, Any]) -> List[List[float]]:
        """Process response for both sync and async methods.

        Args:
            result: Parsed JSON response from the API

        Returns:
            List of embeddings sorted by index
        """
        data = result.get("data", result)
        if not isinstance(data, list):
            raise ValueError(f"Expected data with a list of embeddings. Got: {data}")
        embedding_list = [(res["embedding"], res["index"]) for res in data]
        self._invoke_callback_vars(result)
        return [x[0] for x in sorted(embedding_list, key=lambda x: x[1])]

    def _embed(
        self, texts: List[str], model_type: Literal["passage", "query"]
    ) -> List[List[float]]:
        """Embed a single text entry to either passage or query type"""
        # API Catalog API -
        #  input: str | list[str]              -- char limit depends on model
        #  model: str                          -- model name, e.g. NV-Embed-QA
        #  encoding_format: "float" | "base64"
        #  input_type: "query" | "passage"
        #  user: str                           -- ignored
        #  truncate: "NONE" | "START" | "END"  -- default "NONE", error raised if
        #                                         an input is too long
        #  dimensions: int                     -- not supported by all models
        payload = self._prepare_payload(texts, model_type)
        response = self._client.get_req(
            payload=payload,
            extra_headers=self.default_headers,
        )
        response.raise_for_status()
        result = response.json()
        return self._process_response(result)

    def _validate_texts(self, texts: List[str]) -> None:
        """Validate that texts is a list of strings.

        Args:
            texts: List to validate

        Raises:
            ValueError: If texts is not a list of strings
        """
        if not isinstance(texts, list) or not all(
            isinstance(text, str) for text in texts
        ):
            raise ValueError(f"`texts` must be a list of strings, given: {repr(texts)}")

    def embed_query(self, text: str) -> List[float]:
        """Input pathway for query embeddings."""
        return self._embed([text], model_type="query")[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Input pathway for document embeddings."""
        self._validate_texts(texts)

        all_embeddings = []
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i : i + self.max_batch_size]
            all_embeddings.extend(self._embed(batch, model_type="passage"))
        return all_embeddings

    async def _aembed(
        self, texts: List[str], model_type: Literal["passage", "query"]
    ) -> List[List[float]]:
        """Async version of _embed."""
        payload = self._prepare_payload(texts, model_type)
        response_text = await self._client.aget_req(
            payload=payload,
            extra_headers=self.default_headers,
        )
        result = json.loads(response_text)
        return self._process_response(result)

    async def aembed_query(self, text: str) -> List[float]:
        """Async input pathway for query embeddings."""
        return (await self._aembed([text], model_type="query"))[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async input pathway for document embeddings."""
        self._validate_texts(texts)

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i : i + self.max_batch_size]
            all_embeddings.extend(await self._aembed(batch, model_type="passage"))
        return all_embeddings

    def _invoke_callback_vars(self, response: dict) -> None:
        """Invoke the callback context variables if there are any."""
        callback_vars = [
            usage_callback_var.get(),
        ]
        llm_output = {**response, "model_name": self.model}
        result = LLMResult(generations=[[]], llm_output=llm_output)
        for cb_var in callback_vars:
            if cb_var:
                cb_var.on_llm_end(result)
