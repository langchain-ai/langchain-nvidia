"""Embeddings Components Derived from NVEModel/Embeddings"""

import warnings
from typing import Any, List, Literal, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.outputs.llm_result import LLMResult
from langchain_core.pydantic_v1 import BaseModel, Field, PrivateAttr, validator

from langchain_nvidia_ai_endpoints._common import _NVIDIAClient
from langchain_nvidia_ai_endpoints._statics import Model
from langchain_nvidia_ai_endpoints.callbacks import usage_callback_var


class NVIDIAEmbeddings(BaseModel, Embeddings):
    """
    Client to NVIDIA embeddings models.

    Fields:
    - model: str, the name of the model to use
    - truncate: "NONE", "START", "END", truncate input text if it exceeds the model's
        maximum token length. Default is "NONE", which raises an error if an input is
        too long.
    """

    class Config:
        validate_assignment = True

    _client: _NVIDIAClient = PrivateAttr(_NVIDIAClient)
    _default_model: str = "NV-Embed-QA"
    _default_max_batch_size: int = 50
    base_url: str = Field(
        "https://integrate.api.nvidia.com/v1",
        description="Base url for model listing an invocation",
    )
    model: Optional[str] = Field(description="Name of the model to invoke")
    truncate: Literal["NONE", "START", "END"] = Field(
        default="NONE",
        description=(
            "Truncate input text if it exceeds the model's maximum token length. "
            "Default is 'NONE', which raises an error if an input is too long."
        ),
    )
    max_batch_size: int = Field(default=_default_max_batch_size)
    model_type: Optional[Literal["passage", "query"]] = Field(
        None, description="(DEPRECATED) The type of text to be embedded."
    )

    def __init__(self, **kwargs: Any):
        """
        Create a new NVIDIAEmbeddings embedder.

        This class provides access to a NVIDIA NIM for embedding. By default, it
        connects to a hosted NIM, but can be configured to connect to a local NIM
        using the `base_url` parameter. An API key is required to connect to the
        hosted NIM.

        Args:
            model (str): The model to use for embedding.
            nvidia_api_key (str): The API key to use for connecting to the hosted NIM.
            api_key (str): Alternative to nvidia_api_key.
            base_url (str): The base URL of the NIM to connect to.
            trucate (str): "NONE", "START", "END", truncate input text if it exceeds
                            the model's context length. Default is "NONE", which raises
                            an error if an input is too long.

        API Key:
        - The recommended way to provide the API key is through the `NVIDIA_API_KEY`
            environment variable.
        """
        super().__init__(**kwargs)
        self._client = _NVIDIAClient(
            base_url=self.base_url,
            model=self.model,
            default_model=self._default_model,
            api_key=kwargs.get("nvidia_api_key", kwargs.get("api_key", None)),
            infer_path="{base_url}/embeddings",
        )
        # todo: only store the model in one place
        # the model may be updated to a newer name during initialization
        self.model = self._client.model

        # todo: remove when nvolveqa_40k is removed from MODEL_TABLE
        if "model" in kwargs and kwargs["model"] in [
            "playground_nvolveqa_40k",
            "nvolveqa_40k",
        ]:
            warnings.warn(
                'Setting truncate="END" for nvolveqa_40k backward compatibility'
            )
            self.truncate = "END"

    @validator("model_type")
    def _validate_model_type(
        cls, v: Optional[Literal["passage", "query"]]
    ) -> Optional[Literal["passage", "query"]]:
        if v:
            warnings.warn(
                "Warning: `model_type` is deprecated and will be removed "
                "in a future release. Please use `embed_query` or "
                "`embed_documents` appropriately."
            )
        return v

    @property
    def available_models(self) -> List[Model]:
        """
        Get a list of available models that work with NVIDIAEmbeddings.
        """
        return self._client.get_available_models(self.__class__.__name__)

    @classmethod
    def get_available_models(
        cls,
        **kwargs: Any,
    ) -> List[Model]:
        """
        Get a list of available models that work with NVIDIAEmbeddings.
        """
        return cls(**kwargs).available_models

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
        payload = {
            "input": texts,
            "model": self.model,
            "encoding_format": "float",
            "input_type": model_type,
        }
        if self.truncate:
            payload["truncate"] = self.truncate

        response = self._client.client.get_req(
            payload=payload,
        )
        response.raise_for_status()
        result = response.json()
        data = result.get("data", result)
        if not isinstance(data, list):
            raise ValueError(f"Expected data with a list of embeddings. Got: {data}")
        embedding_list = [(res["embedding"], res["index"]) for res in data]
        self._invoke_callback_vars(result)
        return [x[0] for x in sorted(embedding_list, key=lambda x: x[1])]

    def embed_query(self, text: str) -> List[float]:
        """Input pathway for query embeddings."""
        return self._embed([text], model_type=self.model_type or "query")[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Input pathway for document embeddings."""
        if not isinstance(texts, list) or not all(
            isinstance(text, str) for text in texts
        ):
            raise ValueError(f"`texts` must be a list of strings, given: {repr(texts)}")

        all_embeddings = []
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i : i + self.max_batch_size]
            all_embeddings.extend(
                self._embed(batch, model_type=self.model_type or "passage")
            )
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
