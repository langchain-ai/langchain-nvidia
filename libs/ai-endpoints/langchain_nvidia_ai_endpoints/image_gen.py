"""Embeddings Components Derived from NVEModel/Embeddings"""
import base64
from io import BytesIO
from typing import Any, List, Optional

import requests
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import LLM
from langchain_core.pydantic_v1 import Field
from langchain_core.runnables import Runnable, RunnableLambda
from PIL import Image

from langchain_nvidia_ai_endpoints._common import _NVIDIAClient


def _get_pil_from_response(data: str) -> Image.Image:
    if data.startswith("url: "):
        body = requests.get(data[4:], stream=True).raw
    elif data.startswith("b64_json: "):
        body = BytesIO(base64.decodebytes(bytes(data[10:], "utf-8")))
    else:
        raise ValueError(f"Invalid response format: {str(data)[:100]}")
    return Image.open(body)


def ImageParser() -> RunnableLambda[str, Image.Image]:
    return RunnableLambda(_get_pil_from_response)


class ImageGenNVIDIA(_NVIDIAClient, LLM):
    """NVIDIA's AI Foundation Retriever Question-Answering Asymmetric Model."""

    _default_model: str = "sdxl"
    infer_endpoint: str = Field("{base_url}/images/generations")
    model: str = Field(_default_model, description="Name of the model to invoke")
    negative_prompt: Optional[str] = Field(description="Sampling temperature in [0, 1]")
    sampler: Optional[str] = Field(description="Sampling strategy for process")
    guidance_scale: Optional[float] = Field(description="The scale of guidance")
    seed: Optional[int] = Field(description="The seed for deterministic results")

    @property
    def _llm_type(self) -> str:
        """Return type of NVIDIA AI Foundation Model Interface."""
        return "nvidia-image-gen-model"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the Image Gen Model on the given prompt and input."""
        payload = {
            "prompt": prompt,
            "negative_prompt": kwargs.get("negative_prompt", self.negative_prompt),
            "sampler": kwargs.get("sampler", self.sampler),
            "guidance_scale": kwargs.get("guidance_scale", self.guidance_scale),
            "seed": kwargs.get("seed", self.seed),
        }
        if self.get_binding_model():
            payload["model"] = self.get_binding_model()
        response = self.client.get_req(
            model_name=self.model, payload=payload, endpoint="infer"
        )
        response.raise_for_status()
        out_dict = response.json()
        if "data" in out_dict:
            out_dict = out_dict.get("data")[0]
        if "url" in out_dict:
            output = "url: {}".format(out_dict.get("url"))
        elif "b64_json" in out_dict:
            output = "b64_json: {}".format(out_dict.get("b64_json"))
        else:
            output = str(out_dict)
        return output

    def as_pil(self, **kwargs: Any) -> Runnable:
        """Returns a model that outputs a PIL image by default"""
        return self | ImageParser(**kwargs)
