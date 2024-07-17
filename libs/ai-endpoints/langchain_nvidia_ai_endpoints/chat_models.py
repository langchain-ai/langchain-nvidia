"""Chat Model Components Derived from ChatModel/NVIDIA"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
import urllib.parse
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Type,
    Union,
)

import requests
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    BaseMessage,
    ChatMessage,
    ChatMessageChunk,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)
from langchain_core.pydantic_v1 import BaseModel, Field, PrivateAttr
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from langchain_nvidia_ai_endpoints._common import _NVIDIAClient
from langchain_nvidia_ai_endpoints._statics import Model

_CallbackManager = Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
_DictOrPydanticClass = Union[Dict[str, Any], Type[BaseModel]]
_DictOrPydantic = Union[Dict, BaseModel]

try:
    import PIL.Image

    has_pillow = True
except ImportError:
    has_pillow = False

logger = logging.getLogger(__name__)


def _is_url(s: str) -> bool:
    try:
        result = urllib.parse.urlparse(s)
        return all([result.scheme, result.netloc])
    except Exception as e:
        logger.debug(f"Unable to parse URL: {e}")
        return False


def _resize_image(img_data: bytes, max_dim: int = 1024) -> str:
    if not has_pillow:
        print(  # noqa: T201
            "Pillow is required to resize images down to reasonable scale."
            " Please install it using `pip install pillow`."
            " For now, not resizing; may cause NVIDIA API to fail."
        )
        return base64.b64encode(img_data).decode("utf-8")
    image = PIL.Image.open(io.BytesIO(img_data))
    max_dim_size = max(image.size)
    aspect_ratio = max_dim / max_dim_size
    new_h = int(image.size[1] * aspect_ratio)
    new_w = int(image.size[0] * aspect_ratio)
    resized_image = image.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
    output_buffer = io.BytesIO()
    resized_image.save(output_buffer, format="JPEG")
    output_buffer.seek(0)
    resized_b64_string = base64.b64encode(output_buffer.read()).decode("utf-8")
    return resized_b64_string


def _url_to_b64_string(image_source: str) -> str:
    b64_template = "data:image/png;base64,{b64_string}"
    try:
        if _is_url(image_source):
            response = requests.get(
                image_source, headers={"User-Agent": "langchain-nvidia-ai-endpoints"}
            )
            response.raise_for_status()
            encoded = base64.b64encode(response.content).decode("utf-8")
            if sys.getsizeof(encoded) > 200000:
                ## (VK) Temporary fix. NVIDIA API has a limit of 250KB for the input.
                encoded = _resize_image(response.content)
            return b64_template.format(b64_string=encoded)
        elif image_source.startswith("data:image"):
            return image_source
        elif os.path.exists(image_source):
            with open(image_source, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                return b64_template.format(b64_string=encoded)
        else:
            raise ValueError(
                "The provided string is not a valid URL, base64, or file path."
            )
    except Exception as e:
        raise ValueError(f"Unable to process the provided image source: {e}")


class ChatNVIDIA(BaseChatModel):
    """NVIDIA chat model.

    Example:
        .. code-block:: python

            from langchain_nvidia_ai_endpoints import ChatNVIDIA


            model = ChatNVIDIA(model="meta/llama2-70b")
            response = model.invoke("Hello")
    """

    _client: _NVIDIAClient = PrivateAttr(_NVIDIAClient)
    _default_model: str = "meta/llama3-8b-instruct"
    base_url: str = Field(
        "https://integrate.api.nvidia.com/v1",
        description="Base url for model listing an invocation",
    )
    model: Optional[str] = Field(description="Name of the model to invoke")
    temperature: Optional[float] = Field(description="Sampling temperature in [0, 1]")
    max_tokens: Optional[int] = Field(
        1024, description="Maximum # of tokens to generate"
    )
    top_p: Optional[float] = Field(description="Top-p for distribution sampling")
    seed: Optional[int] = Field(description="The seed for deterministic results")
    stop: Optional[Sequence[str]] = Field(description="Stop words (cased)")

    def __init__(self, **kwargs: Any):
        """
        Create a new NVIDIAChat chat model.

        This class provides access to a NVIDIA NIM for chat. By default, it
        connects to a hosted NIM, but can be configured to connect to a local NIM
        using the `base_url` parameter. An API key is required to connect to the
        hosted NIM.

        Args:
            model (str): The model to use for chat.
            nvidia_api_key (str): The API key to use for connecting to the hosted NIM.
            api_key (str): Alternative to nvidia_api_key.
            base_url (str): The base URL of the NIM to connect to.
            temperature (float): Sampling temperature in [0, 1].
            max_tokens (int): Maximum number of tokens to generate.
            top_p (float): Top-p for distribution sampling.
            seed (int): A seed for deterministic results.
            stop (list[str]): A list of cased stop words.

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
            infer_path="{base_url}/chat/completions",
        )
        # todo: only store the model in one place
        # the model may be updated to a newer name during initialization
        self.model = self._client.model

    @property
    def available_models(self) -> List[Model]:
        """
        Get a list of available models that work with ChatNVIDIA.
        """
        return self._client.get_available_models(self.__class__.__name__)

    @classmethod
    def get_available_models(
        cls,
        **kwargs: Any,
    ) -> List[Model]:
        """
        Get a list of available models that work with ChatNVIDIA.
        """
        return cls(**kwargs).available_models

    @property
    def _llm_type(self) -> str:
        """Return type of NVIDIA AI Foundation Model Interface."""
        return "chat-nvidia-ai-playground"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        inputs = self._custom_preprocess(messages)
        payload = self._get_payload(inputs=inputs, stop=stop, stream=False, **kwargs)
        response = self._client.client.get_req(payload=payload)
        responses, _ = self._client.client.postprocess(response)
        self._set_callback_out(responses, run_manager)
        message = ChatMessage(**self._custom_postprocess(responses))
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation], llm_output=responses)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[Sequence[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Allows streaming to model!"""
        inputs = self._custom_preprocess(messages)
        payload = self._get_payload(inputs=inputs, stop=stop, stream=True, **kwargs)
        for response in self._client.client.get_req_stream(payload=payload):
            self._set_callback_out(response, run_manager)
            chunk = ChatGenerationChunk(
                message=ChatMessageChunk(**self._custom_postprocess(response))
            )
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    def _set_callback_out(
        self,
        result: dict,
        run_manager: Optional[_CallbackManager],
    ) -> None:
        result.update({"model_name": self.model})
        if run_manager:
            for cb in run_manager.handlers:
                if hasattr(cb, "llm_output"):
                    cb.llm_output = result

    def _custom_preprocess(  # todo: remove
        self, msg_list: Sequence[BaseMessage]
    ) -> List[Dict[str, str]]:
        def _preprocess_msg(msg: BaseMessage) -> Dict[str, str]:
            if isinstance(msg, BaseMessage):
                role_convert = {"ai": "assistant", "human": "user"}
                if isinstance(msg, ChatMessage):
                    role = msg.role
                else:
                    role = msg.type
                role = role_convert.get(role, role)
                content = self._process_content(msg.content)
                return {"role": role, "content": content}
            raise ValueError(f"Invalid message: {repr(msg)} of type {type(msg)}")

        return [_preprocess_msg(m) for m in msg_list]

    def _process_content(self, content: Union[str, List[Union[dict, str]]]) -> str:
        if isinstance(content, str):
            return content
        string_array: list = []

        for part in content:
            if isinstance(part, str):
                string_array.append(part)
            elif isinstance(part, Mapping):
                # OpenAI Format
                if "type" in part:
                    if part["type"] == "text":
                        string_array.append(str(part["text"]))
                    elif part["type"] == "image_url":
                        img_url = part["image_url"]
                        if isinstance(img_url, dict):
                            if "url" not in img_url:
                                raise ValueError(
                                    f"Unrecognized message image format: {img_url}"
                                )
                            img_url = img_url["url"]
                        b64_string = _url_to_b64_string(img_url)
                        string_array.append(f'<img src="{b64_string}" />')
                    else:
                        raise ValueError(
                            f"Unrecognized message part type: {part['type']}"
                        )
                else:
                    raise ValueError(f"Unrecognized message part format: {part}")
        return "".join(string_array)

    def _custom_postprocess(self, msg: dict) -> dict:  # todo: remove
        kw_left = msg.copy()
        out_dict = {
            "role": kw_left.pop("role", "assistant") or "assistant",
            "name": kw_left.pop("name", None),
            "id": kw_left.pop("id", None),
            "content": kw_left.pop("content", "") or "",
            "additional_kwargs": {},
            "response_metadata": {},
        }
        for k in list(kw_left.keys()):
            if "tool" in k:
                out_dict["additional_kwargs"][k] = kw_left.pop(k)
        out_dict["response_metadata"] = kw_left
        return out_dict

    ######################################################################################
    ## Core client-side interfaces

    def _get_payload(
        self, inputs: Sequence[Dict], **kwargs: Any
    ) -> dict:  # todo: remove
        """Generates payload for the _NVIDIAClient API to send to service."""
        messages: List[Dict[str, Any]] = []
        for msg in inputs:
            if isinstance(msg, str):
                # (WFH) this shouldn't ever be reached but leaving this here bcs
                # it's a Chesterton's fence I'm unwilling to touch
                messages.append(dict(role="user", content=msg))
            elif isinstance(msg, dict):
                if msg.get("content", None) is None:
                    raise ValueError(f"Message {msg} has no content")
                messages.append(msg)
            else:
                raise ValueError(f"Unknown message received: {msg} of type {type(msg)}")

        # special handling for "stop" because it always comes in kwargs.
        # if user provided "stop" to invoke/stream, it will be non-None
        # in kwargs.
        # note: we cannot tell if the user specified stop=None to invoke/stream because
        #       the default value of stop is None.
        # todo: remove self.stop
        assert "stop" in kwargs, '"stop" param is expected in kwargs'
        if kwargs["stop"] is None:
            kwargs.pop("stop")

        # setup default payload values
        payload: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "seed": self.seed,
            "stop": self.stop,
        }

        # merge incoming kwargs with attr_kwargs giving preference to
        # the incoming kwargs
        payload.update(kwargs)

        # remove keys with None values from payload
        payload = {k: v for k, v in payload.items() if v is not None}

        return {"messages": messages, **payload}

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[dict, str, Literal["auto", "none"], bool]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        raise NotImplementedError(
            "Not implemented, awaiting server-side function-recieving API"
            " Consider following open-source LLM agent spec techniques:"
            " https://huggingface.co/blog/open-source-llms-as-agents"
        )

    def bind_functions(
        self,
        functions: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable]],
        function_call: Optional[str] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        raise NotImplementedError(
            "Not implemented, awaiting server-side function-recieving API"
            " Consider following open-source LLM agent spec techniques:"
            " https://huggingface.co/blog/open-source-llms-as-agents"
        )

    def with_structured_output(
        self,
        schema: _DictOrPydanticClass,
        *,
        method: Literal["function_calling", "json_mode"] = "function_calling",
        return_type: Literal["parsed", "all"] = "parsed",
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        raise NotImplementedError(
            "Not implemented, awaiting server-side function-recieving API"
            " Consider following open-source LLM agent spec techniques:"
            " https://huggingface.co/blog/open-source-llms-as-agents"
        )
