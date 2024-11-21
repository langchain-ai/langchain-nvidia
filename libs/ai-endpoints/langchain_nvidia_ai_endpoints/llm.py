from __future__ import annotations

import warnings
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from pydantic import ConfigDict, Field, PrivateAttr

from langchain_nvidia_ai_endpoints._common import _NVIDIAClient
from langchain_nvidia_ai_endpoints._statics import Model

_DEFAULT_MODEL_NAME: str = "nvidia/mistral-nemo-minitron-8b-base"


class NVIDIA(LLM):
    """
    LangChain LLM that uses the Completions API with NVIDIA NIMs.
    """

    model_config = ConfigDict(
        validate_assignment=True,
    )

    _client: _NVIDIAClient = PrivateAttr()
    _default_model_name: str = "nvidia/mistral-nemo-minitron-8b-base"
    base_url: Optional[str] = Field(
        default=None,
        description="Base url for model listing and invocation",
    )
    model: Optional[str] = Field(None, description="The model to use for completions.")

    _init_args: Dict[str, Any] = PrivateAttr()
    """Stashed arguments given to the constructor that can be passed to
    the Completions API endpoint."""

    def __check_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check kwargs, warn for unknown keys, and return a copy recognized keys.
        """
        completions_arguments = {
            "frequency_penalty",
            "max_tokens",
            "presence_penalty",
            "seed",
            "stop",
            "temperature",
            "top_p",
            "best_of",
            "echo",
            "logit_bias",
            "logprobs",
            "n",
            "suffix",
            "user",
            "stream",
        }

        recognized_kwargs = {
            k: v for k, v in kwargs.items() if k in completions_arguments
        }
        unrecognized_kwargs = set(kwargs) - completions_arguments
        if len(unrecognized_kwargs) > 0:
            warnings.warn(f"Unrecognized, ignored arguments: {unrecognized_kwargs}")

        return recognized_kwargs

    def __init__(self, **kwargs: Any):
        """
        Create a new NVIDIA LLM for Completions APIs.

        This class provides access to a NVIDIA NIM for completions. By default, it
        connects to a hosted NIM, but can be configured to connect to a local NIM
        using the `base_url` parameter. An API key is required to connect to the
        hosted NIM.

        Args:
            model (str): The model to use for completions.
            nvidia_api_key (str): The API key to use for connecting to the hosted NIM.
            api_key (str): Alternative to nvidia_api_key.
            base_url (str): The base URL of the NIM to connect to.

        API Key:
        - The recommended way to provide the API key is through the `NVIDIA_API_KEY`
            environment variable.

        Additional arguments that can be passed to the Completions API:
        - max_tokens (int): The maximum number of tokens to generate.
        - stop (str or List[str]): The stop sequence to use for generating completions.
        - temperature (float): The temperature to use for generating completions.
        - top_p (float): The top-p value to use for generating completions.
        - frequency_penalty (float): The frequency penalty to apply to the completion.
        - presence_penalty (float): The presence penalty to apply to the completion.
        - seed (int): The seed to use for generating completions.

        These additional arguments can also be passed with `bind()`, e.g.
        `NVIDIA().bind(max_tokens=512)`, or pass directly to `invoke()` or `stream()`,
        e.g. `NVIDIA().invoke("prompt", max_tokens=512)`.
        """
        super().__init__(**kwargs)
        # allow nvidia_base_url as an alternative for base_url
        base_url = kwargs.pop("nvidia_base_url", self.base_url)
        # allow nvidia_api_key as an alternative for api_key
        api_key = kwargs.pop("nvidia_api_key", kwargs.pop("api_key", None))
        self._client = _NVIDIAClient(
            **({"base_url": base_url} if base_url else {}),  # only pass if set
            mdl_name=self.model,
            default_hosted_model_name=_DEFAULT_MODEL_NAME,
            **({"api_key": api_key} if api_key else {}),  # only pass if set
            infer_path="{base_url}/completions",
            cls=self.__class__.__name__,
        )
        # todo: only store the model in one place
        # the model may be updated to a newer name during initialization
        self.model = self._client.mdl_name
        # same for base_url
        self.base_url = self._client.base_url

        # stash all additional args that can be passed to the Completions API,
        # but first make sure we pull out any args that are processed elsewhere.
        for key in [
            "model",
            "nvidia_base_url",
            "base_url",
        ]:
            if key in kwargs:
                del kwargs[key]
        self._init_args = self.__check_kwargs(kwargs)

    @property
    def available_models(self) -> List[Model]:
        """
        Get a list of available models that work with NVIDIA.
        """
        return self._client.get_available_models(self.__class__.__name__)

    @classmethod
    def get_available_models(
        cls,
        **kwargs: Any,
    ) -> List[Model]:
        """
        Get a list of available models that work with the Completions API.
        """
        return cls(**kwargs).available_models

    @property
    def _llm_type(self) -> str:
        """
        Get the type of language model used by this chat model.
        Used for logging purposes only.
        """
        return "NVIDIA"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """
        Get parameters used to help identify the LLM.
        """
        return {
            "model": self.model,
            "base_url": self.base_url,
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            **self._init_args,
            **self.__check_kwargs(kwargs),
        }
        if stop:
            payload["stop"] = stop

        if payload.get("stream", False):
            warnings.warn("stream set to true for non-streaming call, ignoring")
            del payload["stream"]

        response = self._client.get_req(payload=payload)
        response.raise_for_status()

        # todo: handle response's usage and system_fingerprint

        choices = response.json()["choices"]
        # todo: write a test for this by setting n > 1 on the request
        #       aug 2024: n > 1 is not supported by endpoints
        if len(choices) > 1:
            warnings.warn(
                f"Multiple choices in response, returning only the first: {choices}"
            )

        return choices[0]["text"]

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            **self._init_args,
            **self.__check_kwargs(kwargs),
        }
        if stop:
            payload["stop"] = stop

        # we construct payload w/ **kwargs positioned to override stream=True,
        # this lets us know if a user passed stream=False
        if not payload.get("stream", True):
            warnings.warn("stream set to false for streaming call, ignoring")
            payload["stream"] = True

        for chunk in self._client.get_req_stream(payload=payload):
            content = chunk["content"]
            generation = GenerationChunk(text=content)
            if run_manager:  # todo: add tests for run_manager
                run_manager.on_llm_new_token(content, chunk=generation)
            yield generation
