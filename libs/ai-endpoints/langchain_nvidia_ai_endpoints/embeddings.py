"""Embeddings Components Derived from NVEModel/Embeddings"""

from typing import List, Literal, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.outputs.llm_result import LLMResult
from langchain_core.pydantic_v1 import Field

from langchain_nvidia_ai_endpoints._common import _NVIDIAClient
from langchain_nvidia_ai_endpoints.callbacks import usage_callback_var

from ._statics import MODEL_SPECS


class NVIDIAEmbeddings(_NVIDIAClient, Embeddings):
    """NVIDIA's AI Foundation Retriever Question-Answering Asymmetric Model."""

    _default_model: str = "nvolveqa_40k"
    infer_endpoint: str = Field("{base_url}/embeddings")
    model: str = Field(_default_model, description="Name of the model to invoke")
    max_length: int = Field(2048, ge=1, le=2048)
    max_batch_size: int = Field(default=50)
    model_type: Optional[Literal["passage", "query"]] = Field(
        None, description="The type of text to be embedded."
    )

    def _embed(
        self, texts: List[str], model_type: Literal["passage", "query"]
    ) -> List[List[float]]:
        """Embed a single text entry to either passage or query type"""
        # AI Foundation Model API -
        #  input: str | list[str]              -- <= 2048 characters, <= 50 inputs
        #  model: "query" | "passage"          -- type of input text to be embedded
        #  encoding_format: "float" | "base64"
        # API Catalog API -
        #  input: str | list[str]              -- char limit depends on model
        #  model: str                          -- model name, e.g. NV-Embed-QA
        #  encoding_format: "float" | "base64"
        #  input_type: "query" | "passage"
        #  user: str                           -- ignored
        #  truncate: "NONE" | "START" | "END"  -- default "NONE", error raised if
        #                                         an input is too long
        # todo: remove the playground aliases
        model_name = self.model
        if model_name not in MODEL_SPECS:
            if f"playground_{model_name}" in MODEL_SPECS:
                model_name = f"playground_{model_name}"
        if MODEL_SPECS.get(model_name, {}).get("api_type", None) == "aifm":
            payload = {
                "input": texts,
                "model": model_type,
                "encoding_format": "float",
            }
        else:  # default to the API Catalog API
            payload = {
                "input": texts,
                "model": self.get_binding_model() or self.model,
                "encoding_format": "float",
                "input_type": model_type,
            }

        response = self.client.get_req(
            model_name=self.model,
            payload=payload,
            endpoint="infer",
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

        # From https://catalog.ngc.nvidia.com/orgs/nvidia/teams/ai-foundation/models/nvolve-40k/documentation
        # The input must not exceed the 2048 max input characters and inputs above 512
        # model tokens will be truncated. The input array must not exceed 50 input
        #  strings.
        all_embeddings = []
        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i : i + self.max_batch_size]
            truncated = [
                text[: self.max_length] if len(text) > self.max_length else text
                for text in batch
            ]
            all_embeddings.extend(
                self._embed(truncated, model_type=self.model_type or "passage")
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
