"""Chat Model Components Derived from ChatModel/NVIDIA"""

from __future__ import annotations

import base64
import enum
import logging
import os
import re
import urllib.parse
import warnings
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.language_models.chat_models import LangSmithParams
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
)
from langchain_core.output_parsers import (
    BaseOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    Generation,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils import get_pydantic_field_names
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.pydantic import is_basemodel_subclass
from langchain_core.utils.utils import _build_model_kwargs
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from langchain_nvidia_ai_endpoints._common import _NVIDIAClient
from langchain_nvidia_ai_endpoints._statics import Model
from langchain_nvidia_ai_endpoints._utils import convert_message_to_dict

# Type variable for generic parser types
T_Parser = TypeVar("T_Parser", bound="BaseOutputParser")

_CallbackManager = Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]

logger = logging.getLogger(__name__)


def _is_url(s: str) -> bool:
    try:
        result = urllib.parse.urlparse(s)
        return all([result.scheme, result.netloc])
    except Exception as e:
        logger.debug(f"Unable to parse URL: {e}")
        return False


def _url_to_b64_string(image_source: str) -> str:
    try:
        if _is_url(image_source):
            return image_source
            # import sys
            # import io
            # try:
            #     import PIL.Image
            #     has_pillow = True
            # except ImportError:
            #     has_pillow = False
            # def _resize_image(img_data: bytes, max_dim: int = 1024) -> str:
            #     if not has_pillow:
            #         print(  # noqa: T201
            #             "Pillow is required to resize images down to reasonable scale."  # noqa: E501
            #             " Please install it using `pip install pillow`."
            #             " For now, not resizing; may cause NVIDIA API to fail."
            #         )
            #         return base64.b64encode(img_data).decode("utf-8")
            #     image = PIL.Image.open(io.BytesIO(img_data))
            #     max_dim_size = max(image.size)
            #     aspect_ratio = max_dim / max_dim_size
            #     new_h = int(image.size[1] * aspect_ratio)
            #     new_w = int(image.size[0] * aspect_ratio)
            #     resized_image = image.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)  # noqa: E501
            #     output_buffer = io.BytesIO()
            #     resized_image.save(output_buffer, format="JPEG")
            #     output_buffer.seek(0)
            #     resized_b64_string = base64.b64encode(output_buffer.read()).decode("utf-8")  # noqa: E501
            #     return resized_b64_string
            # b64_template = "data:image/png;base64,{b64_string}"
            # response = requests.get(
            #     image_source, headers={"User-Agent": "langchain-nvidia-ai-endpoints"}
            # )
            # response.raise_for_status()
            # encoded = base64.b64encode(response.content).decode("utf-8")
            # if sys.getsizeof(encoded) > 200000:
            #     ## (VK) Temporary fix. NVIDIA API has a limit of 250KB for the input.
            #     encoded = _resize_image(response.content)
            # return b64_template.format(b64_string=encoded)
        elif image_source.startswith("data:image"):
            return image_source
        elif os.path.exists(image_source):
            with open(image_source, "rb") as f:
                image_data = f.read()
                import filetype  # type: ignore

                kind = filetype.guess(image_data)
                image_type = kind.extension if kind else "unknown"
                encoded = base64.b64encode(image_data).decode("utf-8")
                return f"data:image/{image_type};base64,{encoded}"
        else:
            raise ValueError(
                "The provided string is not a valid URL, base64, or file path."
            )
    except Exception as e:
        raise ValueError(f"Unable to process the provided image source: {e}")


def _deep_merge(base: dict, update: dict) -> dict:
    """Deep merge update dict into base dict."""
    result = base.copy()
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _nv_vlm_adjust_input(
    message_dict: Dict[str, Any], model_type: str
) -> Dict[str, Any]:
    """The NVIDIA VLM API input `message.content`:
        {
            "role": "user",
            "content": [
                ...,
                {
                    "type": "image_url",
                    "image_url": "{data}"
                },
                ...
            ]
        }

    where OpenAI VLM API input `message.content`:
        {
            "role": "user",
            "content": [
                ...,
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "{url | data}"
                    }
                },
                ...
            ]
        }

    This function converts the OpenAI VLM API input message to NVIDIA VLM API input
    message, in place.

    In the process, it accepts a url or file and converts them to data urls.
    """
    if content := message_dict.get("content"):
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and "image_url" in part:
                    if (
                        isinstance(part["image_url"], dict)
                        and "url" in part["image_url"]
                    ):
                        if "detail" in part["image_url"]:
                            detail = part["image_url"]["detail"]
                            if detail not in ["auto", "low", "high"]:
                                raise ValueError(
                                    f"Invalid detail value: {detail!r}. "
                                    "Must be one of 'auto', 'low', or 'high'. "
                                )
                        url = _url_to_b64_string(part["image_url"]["url"])
                        if model_type == "nv-vlm":
                            part["image_url"] = url
                        else:
                            part["image_url"]["url"] = url
    return message_dict


def parse_thinking_content(
    content: str, *, remove_tags: bool = True
) -> tuple[str, str, str]:
    """Parse thinking content from text.

    This function handles multiple formats by trying to find the reasoning content
    1. Content with single </think> tag delimiter
    2. Content with <think></think> paired tags
    3. Plain content without reasoning

    Args:
        content: The full content including potential thinking tags
        remove_tags: If True (default), removes tags.
            If False, keeps for backward compat.

    Returns:
        tuple: (reasoning_content, content_with_tags, content_without_tags)
    """
    if not content:
        return "", "", ""

    # Check for single </think> tag (everything before is reasoning)
    if "</think>" in content and "<think>" not in content:
        think_end_idx = content.find("</think>")
        reasoning_part = content[:think_end_idx]
        actual_content = content[think_end_idx + len("</think>") :]

        reasoning = reasoning_part.strip("\n").strip()
        actual = actual_content.strip("\n").strip()

        if remove_tags:
            return reasoning, actual, actual
        else:
            return reasoning, content, actual

    # Check for paired <think></think> tags
    if "<think>" in content and "</think>" in content:
        think_start_idx = content.find("<think>")
        think_end_idx = content.find("</think>")

        # Make sure both tags are in the right order
        if (
            think_start_idx != -1
            and think_end_idx != -1
            and think_start_idx < think_end_idx
        ):
            reasoning_part = content[think_start_idx + len("<think>") : think_end_idx]
            actual_content = content[think_end_idx + len("</think>") :]

            reasoning = reasoning_part.strip("\n").strip()
            actual = actual_content.strip("\n").strip()

            if remove_tags:
                return reasoning, actual, actual
            else:
                return reasoning, content, actual

    # No reasoning found, return plain content
    return "", content, content


def _is_structured_output(payload: dict) -> bool:
    """Check if the payload indicates structured output mode.

    Structured output is enabled when nvext contains guided_json or guided_choice.

    Args:
        payload: The request payload dictionary

    Returns:
        True if structured output mode is enabled, False otherwise
    """
    nvext = payload.get("nvext", {})
    return bool(nvext and ("guided_json" in nvext or "guided_choice" in nvext))


def _nv_vlm_get_asset_ids(
    content: Union[str, List[Union[str, Dict[str, Any]]]],
) -> List[str]:
    """VLM APIs accept asset IDs as input in two forms:

    - `content = [{"image_url": {"url": "data:image/{type};asset_id,{asset_id}"}}*]`
    - `content = .*<img src="data:image/{type};asset_id,{asset_id}"/>.*`

    This function extracts asset IDs from the message content.
    """

    def extract_asset_id(data: str) -> List[str]:
        pattern = re.compile(r'data:image/[^;]+;asset_id,([^"\'\s]+)')
        return pattern.findall(data)

    asset_ids = []
    if isinstance(content, str):
        asset_ids.extend(extract_asset_id(content))
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, str):
                asset_ids.extend(extract_asset_id(part))
            elif isinstance(part, dict) and "image_url" in part:
                image_url = part["image_url"]
                if isinstance(image_url, dict) and "url" in image_url:
                    asset_ids.extend(extract_asset_id(image_url["url"]))

    return asset_ids


def _process_for_vlm(
    inputs: List[Dict[str, Any]],
    model: Optional[Model],  # not optional, Optional for type alignment
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Process inputs for NVIDIA VLM models.

    This function processes the input messages for NVIDIA VLM models.

    Extracts asset IDs from the input messages and adds them to the headers for the
    NVIDIA VLM API.
    """
    if not model or not model.model_type:
        return inputs, {}

    extra_headers = {}
    if "vlm" in model.model_type:
        asset_ids = []
        for input in inputs:
            if "content" in input:
                asset_ids.extend(_nv_vlm_get_asset_ids(input["content"]))
        if asset_ids:
            extra_headers["NVCF-INPUT-ASSET-REFERENCES"] = ",".join(asset_ids)
        inputs = [_nv_vlm_adjust_input(message, model.model_type) for message in inputs]
    return inputs, extra_headers


_DEFAULT_MODEL_NAME: str = "meta/llama3-8b-instruct"


class ChatNVIDIA(BaseChatModel):
    """NVIDIA chat model.

    Example:
        ```python
        from langchain_nvidia_ai_endpoints import ChatNVIDIA


        model = ChatNVIDIA(model="meta/llama2-70b")
        response = model.invoke("Hello")
        ```
    """

    model_config = ConfigDict(populate_by_name=True)

    _client: _NVIDIAClient = PrivateAttr()

    base_url: Optional[str] = Field(
        default=None,
        description="Base url for model listing an invocation",
    )

    model: Optional[str] = Field(None, description="Name of the model to invoke")

    temperature: Optional[float] = Field(
        None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature in [0, 2]",
    )

    max_tokens: Optional[int] = Field(
        1024,
        gt=0,
        description="Maximum # of tokens to generate",
        alias="max_completion_tokens",
    )

    top_p: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Top-p for nucleus sampling in [0, 1]",
    )

    seed: Optional[int] = Field(
        None,
        ge=0,
        description="The seed for deterministic results",
    )

    stop: Optional[Union[str, List[str]]] = Field(
        None, description="Stop words (cased)"
    )

    stream_options: Optional[Dict[str, Any]] = Field(
        {"include_usage": True},
        description="Stream options for the model. Set to None to disable",
    )

    default_headers: dict = Field(
        default_factory=dict,
        description="Default headers merged into all requests.",
    )

    model_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional model parameters that are not explicitly defined "
            "to be added during invocation."
        ),
    )

    # Reference: https://github.com/langchain-ai/langchain/blob/master/libs/partners/openai/langchain_openai/llms/base.py#L295
    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        return _build_model_kwargs(values, all_required_field_names)

    def __init__(
        self,
        *,
        model: Optional[str] = None,
        nvidia_api_key: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_completion_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        default_headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ):
        """
        Create a new `NVIDIAChat` chat model.

        This class provides access to a NVIDIA NIM for chat. By default, it
        connects to a hosted NIM, but can be configured to connect to a local NIM
        using the `base_url` parameter.

        An API key is required to connect to the hosted NIM.

        Args:
            model: The model to use for chat.
            nvidia_api_key: The API key to use for connecting to the hosted NIM.
            api_key: Alternative to `nvidia_api_key`.
            base_url: The base URL of the NIM to connect to.

                Format for base URL is `http://host:port`
            temperature: Sampling temperature in `[0, 2]`.
            max_completion_tokens: Maximum number of tokens to generate.
            top_p: Top-p for distribution sampling in `[0, 1]`.
            seed: A seed for deterministic results.
            stop: A string or list of strings specifying stop sequences.
            default_headers: Default headers merged into all requests.
            **kwargs: Additional parameters passed to the underlying client.

        The recommended way to provide the API key is through the `NVIDIA_API_KEY`
        environment variable.

        **Base URL:**

        - Connect to a self-hosted model with NVIDIA NIM using the `base_url` arg to
            link to the local host at `localhost:8000`:

            ```python
            llm = ChatNVIDIA(
                base_url="http://localhost:8000/v1",
                model="meta-llama3-8b-instruct"
            )
            ```
        """
        # Show deprecation warning if max_tokens was used
        if "max_tokens" in kwargs:
            warnings.warn(
                "The 'max_tokens' parameter is deprecated and will be removed "
                "in a future version. "
                "Please use 'max_completion_tokens' instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        init_kwargs: Dict[str, Any] = {}
        if model is not None:
            init_kwargs["model"] = model
        if base_url is not None:
            init_kwargs["base_url"] = base_url
        if temperature is not None:
            init_kwargs["temperature"] = temperature
        if max_completion_tokens is not None:
            init_kwargs["max_completion_tokens"] = max_completion_tokens
        if top_p is not None:
            init_kwargs["top_p"] = top_p
        if seed is not None:
            init_kwargs["seed"] = seed
        if stop is not None:
            init_kwargs["stop"] = stop
        if default_headers is not None:
            init_kwargs["default_headers"] = default_headers

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
            infer_path="{base_url}/chat/completions",
            # instead of self.__class__.__name__ to assist in subclassing ChatNVIDIA
            cls="ChatNVIDIA",
            verify_ssl=verify_ssl,
        )
        # todo: only store the model in one place
        # the model may be updated to a newer name during initialization
        self.model = self._client.mdl_name
        # same for base_url
        self.base_url = self._client.base_url

    @property
    def available_models(self) -> List[Model]:
        """Get a list of available models that work with `ChatNVIDIA`."""
        return self._client.get_available_models(self.__class__.__name__)

    @classmethod
    def get_available_models(
        cls,
        **kwargs: Any,
    ) -> List[Model]:
        """Get a list of available models that work with `ChatNVIDIA`."""
        return cls(**kwargs).available_models

    @property
    def _llm_type(self) -> str:
        """Return type of NVIDIA AI Foundation Model Interface."""
        return "chat-nvidia-ai-playground"

    def _get_ls_params(
        self,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        """Get standard LangSmith parameters for tracing."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        return LangSmithParams(
            ls_provider="NVIDIA",
            # error: Incompatible types (expression has type "Optional[str]",
            #  TypedDict item "ls_model_name" has type "str")  [typeddict-item]
            ls_model_name=self.model or "UNKNOWN",
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
            # TODO: remove max_tokens once all models support max_completion_tokens
            ls_max_tokens=(
                params.get("max_completion_tokens", self.max_tokens)
                or params.get("max_tokens", self.max_tokens)
            ),
            # mypy error: Extra keys ("ls_top_p", "ls_seed")
            #  for TypedDict "LangSmithParams"  [typeddict-item]
            # ls_top_p=params.get("top_p", self.top_p),
            # ls_seed=params.get("seed", self.seed),
            ls_stop=params.get("stop", self.stop),
        )

    def _prepare_inputs_and_payload(
        self,
        messages: List[BaseMessage],
        stop: Optional[Union[List[str], Sequence[str]]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, str]]:
        """Prepare inputs and payload for both sync and async methods.

        Args:
            messages: List of messages
            stop: Stop words
            stream: Whether to stream the response
            kwargs: Additional keyword arguments

        Returns:
            Tuple of `(inputs, payload, extra_headers)`
        """
        inputs = [
            message
            for message in [convert_message_to_dict(message) for message in messages]
        ]
        inputs, extra_headers = _process_for_vlm(inputs, self._client.model)
        # Merge default_headers with extra_headers from VLM processing.
        # VLM headers (auto-generated) take precedence in case of conflicts.
        if self.default_headers:
            conflicts = set(self.default_headers.keys()) & set(extra_headers.keys())
            if conflicts:
                warnings.warn(
                    f"default_headers keys {conflicts} conflict with "
                    f"auto-generated VLM headers and will be overridden. "
                    f"Remove them from default_headers.",
                    UserWarning,
                    stacklevel=2,
                )
            extra_headers = {**self.default_headers, **extra_headers}

        if stream:
            payload = self._get_payload(
                inputs=inputs,
                stop=stop,
                stream=True,
                stream_options=self.stream_options,
                **kwargs,
            )
            # remove stream_options if user set it to None or if model
            # doesn't support it
            # todo: get vlm endpoints fixed and remove this
            #       vlm endpoints do not accept standard stream_options parameter
            if self.stream_options is None or (
                self._client.model
                and self._client.model.model_type
                and self._client.model.model_type in ["nv-vlm", "qa"]
            ):
                payload.pop("stream_options", None)
        else:
            payload = self._get_payload(
                inputs=inputs, stop=stop, stream=False, **kwargs
            )

        return inputs, payload, extra_headers

    def _process_generate_response(
        self,
        response: Any,
        run_manager: Optional[
            Union[CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun]
        ] = None,
        structured_output: bool = False,
    ) -> ChatResult:
        """Process response for both sync and async generate methods.

        Args:
            response: Raw response from the API
            run_manager: Callback manager
            structured_output: Whether this is structured output mode

        Returns:
            ChatResult with generated message
        """
        responses, _ = self._client.postprocess(response)
        self._set_callback_out(responses, run_manager)
        parsed_response = self._custom_postprocess(
            responses, streaming=False, structured_output=structured_output
        )
        # for pre 0.2 compatibility w/ ChatMessage
        # ChatMessage had a role property that was not present in AIMessage
        parsed_response.update({"role": "assistant"})
        generation = ChatGeneration(message=AIMessage(**parsed_response))
        return ChatResult(generations=[generation], llm_output=responses)

    def _process_stream_chunk(
        self,
        response: Any,
        run_manager: Optional[
            Union[CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun]
        ] = None,
        structured_output: bool = False,
    ) -> ChatGenerationChunk:
        """Process a single stream chunk for both sync and async stream methods.

        Args:
            response: Raw response chunk from the API
            run_manager: Callback manager
            structured_output: Whether this is structured output mode

        Returns:
            ChatGenerationChunk with the parsed message
        """
        self._set_callback_out(response, run_manager)
        parsed_response = self._custom_postprocess(
            response, streaming=True, structured_output=structured_output
        )
        # for pre 0.2 compatibility w/ ChatMessageChunk
        # ChatMessageChunk had a role property that was not
        # present in AIMessageChunk
        # unfortunately, AIMessageChunk does not have extensible propery
        # parsed_response.update({"role": "assistant"})
        message = AIMessageChunk(**parsed_response)
        chunk = ChatGenerationChunk(message=message)
        return chunk

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        _, payload, extra_headers = self._prepare_inputs_and_payload(
            messages, stop, stream=False, **kwargs
        )
        response = self._client.get_req(payload=payload, extra_headers=extra_headers)
        return self._process_generate_response(
            response, run_manager, _is_structured_output(payload)
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[Sequence[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Allows streaming to model!"""
        _, payload, extra_headers = self._prepare_inputs_and_payload(
            messages, stop, stream=True, **kwargs
        )
        structured_output = _is_structured_output(payload)
        for response in self._client.get_req_stream(
            payload=payload, extra_headers=extra_headers
        ):
            chunk = self._process_stream_chunk(response, run_manager, structured_output)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async version of `_generate`."""
        _, payload, extra_headers = self._prepare_inputs_and_payload(
            messages, stop, stream=False, **kwargs
        )
        response = await self._client.aget_req(
            payload=payload, extra_headers=extra_headers
        )
        return self._process_generate_response(
            response, run_manager, _is_structured_output(payload)
        )

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[Sequence[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Async version of `_stream`."""
        _, payload, extra_headers = self._prepare_inputs_and_payload(
            messages, stop, stream=True, **kwargs
        )
        structured_output = _is_structured_output(payload)
        async for response in self._client.aget_req_stream(
            payload=payload, extra_headers=extra_headers
        ):
            chunk = self._process_stream_chunk(response, run_manager, structured_output)
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
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

    def _custom_postprocess(
        self, msg: dict, streaming: bool = False, structured_output: bool = False
    ) -> dict:  # todo: remove
        kw_left = msg.copy()
        content = kw_left.pop("content", "") or ""

        # Extract reasoning: check reasoning_content field first,
        # then parse <think> tags if needed
        reasoning_from_reasoning_content = kw_left.pop("reasoning_content", None)

        # Parse thinking content
        # For structured output: remove tags
        # For regular output: keep tags for backward compatibility
        (
            reasoning_from_tags,
            content_with_tags,
            content_without_tags,
        ) = parse_thinking_content(content, remove_tags=structured_output)

        # Warn user if reasoning was parsed from tags
        # (only if not provided via reasoning_content field)
        if reasoning_from_tags and not reasoning_from_reasoning_content:
            if structured_output:
                warnings.warn(
                    "Reasoning content with <think> tags was detected in "
                    "structured output mode. The tags have been removed from "
                    "the content to ensure valid structured output. "
                    "Note: The reasoning will be removed after the output "
                    "parser extracts the structured object. To preserve "
                    "reasoning, include a 'reasoning' field in your output "
                    "schema and ask the model to populate it in the JSON.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                warnings.warn(
                    "Reasoning content was parsed from <think> tags in model "
                    "output. The tags are currently preserved in the content "
                    "but will be removed in a future version. The reasoning "
                    "is available in additional_kwargs['reasoning_content'] "
                    "and in the reasoning content block in content_blocks. "
                    "Use reasoning_content instead of parsing tags manually.",
                    UserWarning,
                    stacklevel=2,
                )

        # Prioritize reasoning from reasoning_content field
        reasoning = reasoning_from_reasoning_content or reasoning_from_tags
        final_content = content_without_tags if structured_output else content_with_tags

        out_dict = {
            "role": kw_left.pop("role", "assistant") or "assistant",
            "name": kw_left.pop("name", None),
            "id": kw_left.pop("id", None),
            "content": final_content,
            "additional_kwargs": {},
            "response_metadata": {},
        }

        if reasoning:
            out_dict["additional_kwargs"]["reasoning_content"] = reasoning

        if token_usage := kw_left.pop("token_usage", None):
            out_dict["usage_metadata"] = {
                "input_tokens": token_usage.get("prompt_tokens", 0),
                "output_tokens": token_usage.get("completion_tokens", 0),
                "total_tokens": token_usage.get("total_tokens", 0),
            }
        # "tool_calls" is set for invoke and stream responses
        if tool_calls := kw_left.pop("tool_calls", None):
            assert isinstance(
                tool_calls, list
            ), "invalid response from server: tool_calls must be a list"
            # todo: break this into post-processing for invoke and stream
            if not streaming:
                out_dict["additional_kwargs"]["tool_calls"] = tool_calls
            elif streaming:
                out_dict["tool_call_chunks"] = []
                for tool_call in tool_calls:
                    # todo: the nim api does not return the function index
                    #       for tool calls in stream responses. this is
                    #       an issue that needs to be resolved server-side.
                    #       the only reason we can skip this for now
                    #       is because the nim endpoint returns only full
                    #       tool calls, no deltas.
                    # assert "index" in tool_call, (
                    #     "invalid response from server: "
                    #     "tool_call must have an 'index' key"
                    # )
                    assert "function" in tool_call, (
                        "invalid response from server: "
                        "tool_call must have a 'function' key"
                    )
                    out_dict["tool_call_chunks"].append(
                        {
                            "index": tool_call.get("index", None),
                            "id": tool_call.get("id", None),
                            "name": tool_call["function"].get("name", None),
                            "args": tool_call["function"].get("arguments", None),
                        }
                    )
        # we only create the response_metadata from the last message in a stream.
        # if we do it for all messages, we'll end up with things like
        # "model_name" = "mode-xyz" * # messages.
        if "finish_reason" in kw_left:
            out_dict["response_metadata"] = kw_left
        return out_dict

    ######################################################################################
    ## Core client-side interfaces

    def _get_payload(
        self, inputs: Sequence[Dict], **kwargs: Any
    ) -> dict:  # todo: remove
        """Generates payload for the `_NVIDIAClient` API to send to service."""
        messages: List[Dict[str, Any]] = []

        for msg in inputs:
            if isinstance(msg, str):
                # (WFH) this shouldn't ever be reached but leaving this here bcs
                # it's a Chesterton's fence I'm unwilling to touch
                messages.append(dict(role="user", content=msg))
            elif isinstance(msg, dict):
                if msg.get("content", None) is None:
                    # content=None is valid for assistant messages (tool calling)
                    if not msg.get("role") == "assistant":
                        raise ValueError(f"Message {msg} has no content.")
                messages.append(msg)
            else:
                raise ValueError(f"Unknown message received: {msg} of type {type(msg)}")

        # Handle thinking mode via parameters or prefix to system message
        thinking_mode = kwargs.pop("thinking_mode", None)
        if thinking_mode is not None and self._client.model:
            # Check if model uses param-based thinking
            thinking_params = (
                self._client.model.thinking_param_enable
                if thinking_mode
                else self._client.model.thinking_param_disable
            )

            if thinking_params:
                # Param-based thinking: merge parameters into kwargs
                kwargs = _deep_merge(kwargs, thinking_params)
            else:
                # Tag-based thinking: use system message prefix
                prefix = (
                    (self._client.model.thinking_prefix or "")
                    if thinking_mode
                    else (self._client.model.no_thinking_prefix or "")
                )

                if prefix:
                    # Find existing system message and append prefix
                    system_msg_found = False
                    for msg in messages:
                        if msg.get("role") == "system":
                            system_msg_found = True
                            existing_content = msg.get("content", "")
                            # Append prefix at the end of existing system message
                            if existing_content:
                                msg["content"] = f"{existing_content}\n{prefix}"
                            else:
                                msg["content"] = prefix
                            break

                    # If no system message exists, create one with the prefix
                    if not system_msg_found:
                        messages.insert(0, {"role": "system", "content": prefix})

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

        # merge model_kwargs first
        payload.update(self.model_kwargs)

        # merge incoming kwargs with attr_kwargs giving preference to
        # the incoming kwargs
        payload.update(kwargs)

        # remove keys with None values from payload
        payload = {k: v for k, v in payload.items() if v is not None}

        return {"messages": messages, **payload}

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type, Callable, BaseTool]],
        *,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "any", "required"], bool]
        ] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """
        Bind tools to the model.

        !!! note

            The `strict` mode is always in effect, if you need it disabled, please file
            an issue.

        Args:
            tools (list): A list of tools to bind to the model.
            tool_choice: Control tool choice.

                Options:

                - `'any'` or `'required'` – Force a tool call.
                - `'auto'` – Let the model decide.
                - `'none'` – Force no tool call.
                - `str` or `dict` – Force a specific tool call.
                - `bool` – If `True`, force a tool call; if `False`, force no tool call.

                Defaults to passing no value.
            **kwargs: Additional keyword arguments.
        """
        # check if the model supports tools, warn if it does not
        if self._client.model and not self._client.model.supports_tools:
            warnings.warn(
                f"Model '{self.model}' is not known to support tools. "
                "Your tool binding may fail at inference time."
            )

        if kwargs.get("strict", True) is not True:
            warnings.warn("The `strict` parameter is not necessary and is ignored.")

        tool_name = None
        if isinstance(tool_choice, bool):
            tool_choice = "required" if tool_choice else "none"
        elif isinstance(tool_choice, str):
            # LangChain documents "any" as an option, server API uses "required"
            if tool_choice == "any":
                tool_choice = "required"
            # if a string that's not "auto", "none", or "required"
            # then it must be a tool name
            if tool_choice not in ["auto", "none", "required"]:
                tool_name = tool_choice
                tool_choice = {
                    "type": "function",
                    "function": {"name": tool_choice},
                }
        elif isinstance(tool_choice, dict):
            # if a dict, it must be a tool choice dict, e.g.
            #  {"type": "function", "function": {"name": "my_tool"}}
            if "type" not in tool_choice:
                tool_choice["type"] = "function"
            if "function" not in tool_choice:
                raise ValueError("Tool choice dict must have a 'function' key")
            if "name" not in tool_choice["function"]:
                raise ValueError("Tool choice function dict must have a 'name' key")
            tool_name = tool_choice["function"]["name"]

        # check that the specified tool is in the tools list
        tool_dicts = [convert_to_openai_tool(tool) for tool in tools]
        if tool_name:
            if not any(tool["function"]["name"] == tool_name for tool in tool_dicts):
                raise ValueError(
                    f"Tool choice '{tool_name}' not found in the tools list"
                )

        return super().bind(
            tools=tool_dicts,
            tool_choice=tool_choice,
            **kwargs,
        )

    def bind_functions(
        self,
        functions: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable]],
        function_call: Optional[str] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        raise NotImplementedError("Not implemented, use `bind_tools` instead.")

    # we have an Enum extension to BaseChatModel.with_structured_output and
    # as a result need to type ignore for the schema parameter and return type.
    def with_structured_output(  # type: ignore
        self,
        schema: Union[Dict, Type],
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """
        Bind a structured output schema to the model.

        Args:
            schema (Union[Dict, Type]): The schema to bind to the model.
            include_raw (bool): Always `False`. Passing `True` raises an error.
            **kwargs: Additional keyword arguments.

        !!! note

            - The `strict` mode is always in effect, if you need it disabled, please file
                an issue.
            - If you need `include_raw=True` consider using an unstructured model and
                output formatter, or file an issue.

        The schema can be:

        1. A dictionary representing a JSON schema
        2. A Pydantic object
        3. An `Enum`

        If a dictionary is provided, the model will return a dictionary.

        !!! example "Dictionary schema"
            ```python
            json_schema = {
                "title": "joke",
                "description": "Joke to tell user.",
                "type": "object",
                "properties": {
                    "setup": {
                        "type": "string",
                        "description": "The setup of the joke",
                    },
                    "punchline": {
                        "type": "string",
                        "description": "The punchline to the joke",
                    },
                },
                "required": ["setup", "punchline"],
            }

            structured_llm = llm.with_structured_output(json_schema)
            structured_llm.invoke("Tell me a joke about NVIDIA")
            # Output: {'setup': 'Why did NVIDIA go broke? The hardware ate all the software.',
            #          'punchline': 'It took a big bite out of their main board.'}
            ```

        If a Pydantic schema is provided, the model will return a Pydantic object.

        !!! example "Pydantic schema"

            ```python
            from pydantic import BaseModel, Field
            class Joke(BaseModel):
                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke")

            structured_llm = llm.with_structured_output(Joke)
            structured_llm.invoke("Tell me a joke about NVIDIA")
            # Output: Joke(setup='Why did NVIDIA go broke? The hardware ate all the software.',
            #              punchline='It took a big bite out of their main board.')
            ```

        If an `Enum` is provided, all values must be strings, and the model will return
        an `Enum` object.

        !!! example "Enum schema"

            ```python
            import enum
            class Choices(enum.Enum):
                A = "A"
                B = "B"
                C = "C"

            structured_llm = llm.with_structured_output(Choices)
            structured_llm.invoke("What is the first letter in this list? [X, Y, Z, C]")
            # Output: <Choices.C: 'C'>
            ```

        ???+ note "Streaming"

            Unlike other streaming responses, the streamed chunks will be increasingly
            complete. They will not be deltas. The last chunk will contain the complete
            response.

            For instance with a dictionary schema, the chunks will be:

            ```python
            structured_llm = llm.with_structured_output(json_schema)
            for chunk in structured_llm.stream("Tell me a joke about NVIDIA"):
                print(chunk)

            # Output:
            # {}
            # {'setup': ''}
            # {'setup': 'Why'}
            # {'setup': 'Why did'}
            # {'setup': 'Why did N'}
            # {'setup': 'Why did NVID'}
            # ...
            # {'setup': 'Why did NVIDIA go broke? The hardware ate all the software.', 'punchline': 'It took a big bite out of their main board'}
            # {'setup': 'Why did NVIDIA go broke? The hardware ate all the software.', 'punchline': 'It took a big bite out of their main board.'}
            ```

            For instance with a Pydantic schema, the chunks will be:

            ```python
            structured_llm = llm.with_structured_output(Joke)
            for chunk in structured_llm.stream("Tell me a joke about NVIDIA"):
                print(chunk)

            # Output:
            # setup='Why did NVIDIA go broke? The hardware ate all the software.' punchline=''
            # setup='Why did NVIDIA go broke? The hardware ate all the software.' punchline='It'
            # setup='Why did NVIDIA go broke? The hardware ate all the software.' punchline='It took'
            # ...
            # setup='Why did NVIDIA go broke? The hardware ate all the software.' punchline='It took a big bite out of their main board'
            # setup='Why did NVIDIA go broke? The hardware ate all the software.' punchline='It took a big bite out of their main board.'
            ```

            For Pydantic schema and `Enum`, the output will be `None` if the response is
            insufficient to construct the object or otherwise invalid.

            ```python
            llm = ChatNVIDIA(max_completion_tokens=1)
            structured_llm = llm.with_structured_output(Joke)
            print(structured_llm.invoke("Tell me a joke about NVIDIA"))

            # Output: None
            ```

            For more, see docs on [structured output](https://docs.langchain.com/oss/python/langchain/structured-output).
        """  # noqa: E501

        if "method" in kwargs:
            warnings.warn(
                "The 'method' parameter is unnecessary and is ignored. "
                "The appropriate method will be chosen automatically depending "
                "on the type of schema provided."
            )

        if kwargs.get("strict", True) is not True:
            warnings.warn(
                "Structured output always follows strict validation. "
                "`strict` is ignored. Please file an issue if you "
                "need strict validation disabled."
            )

        if include_raw:
            raise NotImplementedError(
                "include_raw=True is not implemented, consider "
                "https://python.langchain.com/docs/how_to/"
                "structured_output/#prompting-and-parsing-model"
                "-outputs-directly or rely on the structured response "
                "being None when the LLM produces an incomplete response."
            )

        # check if the model supports structured output, warn if it does not
        known_good = False
        guided_schema: Union[Dict[str, Any], Any] = schema
        # todo: we need to store model: Model in this class
        #       instead of model: str (= Model.id)
        #  this should be: if not self.model.supports_tools: warnings.warn...
        candidates = [
            model for model in self.available_models if model.id == self.model
        ]
        if not candidates:  # user must have specified the model themselves
            known_good = False
        else:
            assert len(candidates) == 1, "Multiple models with the same id"
            known_good = candidates[0].supports_structured_output is True
        if not known_good:
            warnings.warn(
                f"Model '{self.model}' is not known to support structured output. "
                "Your output may fail at inference time."
            )

        if isinstance(schema, dict):
            output_parser: BaseOutputParser = JsonOutputParser()
            nvext_param: Dict[str, Any] = {"guided_json": schema}
        elif issubclass(schema, enum.Enum):
            # langchain's EnumOutputParser is not in langchain_core
            # and doesn't support streaming. this is a simple implementation
            # that supports streaming with our semantics of returning None
            # if no complete object can be constructed.
            class EnumOutputParser(BaseOutputParser):
                enum: Type[enum.Enum]

                def parse(self, response: str) -> Any:
                    try:
                        return self.enum(response.strip())
                    except ValueError:
                        pass
                    return None

            # guided_choice only supports string choices
            choices = [choice.value for choice in schema]
            if not all(isinstance(choice, str) for choice in choices):
                # instead of erroring out we could coerce the enum values to
                # strings, but would then need to coerce them back to their
                # original type for Enum construction.
                raise ValueError(
                    "Enum schema must only contain string choices. "
                    "Use StrEnum or ensure all member values are strings."
                )
            output_parser = EnumOutputParser(enum=schema)
            nvext_param = {"guided_choice": choices}
            guided_schema = choices

        elif is_basemodel_subclass(schema):
            # PydanticOutputParser does not support streaming. what we do
            # instead is ignore all inputs that are incomplete wrt the
            # underlying Pydantic schema. if the entire input is invalid,
            # we return None.
            class ForgivingPydanticOutputParser(PydanticOutputParser):
                def parse_result(
                    self, result: List[Generation], *, partial: bool = False
                ) -> Any:
                    try:
                        return super().parse_result(result, partial=partial)
                    except OutputParserException:
                        pass
                    return None

            output_parser = ForgivingPydanticOutputParser(pydantic_object=schema)
            if hasattr(schema, "model_json_schema"):
                json_schema = schema.model_json_schema()
            else:
                json_schema = schema.schema()
            nvext_param = {"guided_json": json_schema}
            guided_schema = json_schema

        else:
            raise ValueError(
                "Schema must be a Pydantic object, a dictionary "
                "representing a JSON schema, or an Enum."
            )

        ls_structured_output_format = {
            "schema": guided_schema,
        }

        return (
            super().bind(
                nvext=nvext_param,
                ls_structured_output_format=ls_structured_output_format,
            )
            | output_parser
        )

    def with_thinking_mode(
        self,
        enabled: bool = True,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """
        Configure the model to use thinking mode.

        Args:
            enabled (bool): Whether to enable thinking mode.
            **kwargs: Additional keyword arguments.

        Returns:
            A runnable that will use thinking mode when enabled.

        Example:
            ```python
            from langchain_nvidia_ai_endpoints import ChatNVIDIA

            model = ChatNVIDIA(model="nvidia/nvidia-nemotron-nano-9b-v2")

            # Enable thinking mode
            thinking_model = model.with_thinking_mode(enabled=True)
            response = thinking_model.invoke("Hello")

            # Disable thinking mode
            no_thinking_model = model.with_thinking_mode(enabled=False)
            response = no_thinking_model.invoke("Hello")
            ```
        """
        # check if the model supports thinking mode, warn if it does not
        if self._client.model and not self._client.model.supports_thinking:
            warnings.warn(
                f"Model '{self.model}' does not support thinking mode. "
                "The thinking mode configuration will be ignored."
            )

        return super().bind(
            thinking_mode=enabled,
            **kwargs,
        )
