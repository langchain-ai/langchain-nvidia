from __future__ import annotations

import logging
import urllib.parse
from typing import (
    Any,
    Dict,
)

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

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
        elif image_source.startswith("data:image"):
            return image_source
        else:
            raise ValueError(
                "The provided string is not a valid URL or data:image URI. "
                "Local file paths are no longer supported for image inputs. "
                "To use a trusted local file, read it explicitly and pass a "
                "data:image/...;base64 URI."
            )
    except Exception as e:
        raise ValueError(f"Unable to process the provided image source: {e}")


def _standard_image_to_openai(block: dict[str, Any]) -> dict[str, Any]:
    """Convert a LangChain image content block to chat-completions format."""
    if "base64" in block:
        base64_data = block["base64"]
        mime_type = block.get("mime_type")
        if not isinstance(base64_data, str) or not base64_data:
            raise ValueError("Image base64 content must be a nonempty string")
        if not isinstance(mime_type, str) or not mime_type.startswith("image/"):
            raise ValueError("Image base64 content requires an image MIME type")
        url = f"data:{mime_type};base64,{base64_data}"
    elif "url" in block:
        url = block["url"]
        if not isinstance(url, str) or not url:
            raise ValueError("Image URL must be a nonempty string")
    else:
        raise ValueError("ChatNVIDIA image content requires a URL or base64 data")
    return {"type": "image_url", "image_url": {"url": url}}


def _normalize_content(content: Any) -> Any:
    """Normalize message content to handle LangChain 1.0 content blocks.

    In LangChain 1.0, `message.content` can be:

    - A string (traditional)
    - A list of content blocks (new in v1.0)
    - `None`

    This function converts list content to string or `None` as needed.

    For multimodal content, converts standard image blocks to `image_url`
    blocks and preserves the other blocks.
    """
    if content is None or isinstance(content, str):
        return content

    if not isinstance(content, list):
        return str(content)

    # Process list of content blocks
    text_parts = []

    for block in content:
        if isinstance(block, str):
            text_parts.append(block)
        elif isinstance(block, dict):
            block_type = block.get("type")

            # Preserve multimodal content (images and videos) as-is for VLM models
            if block_type in ("image_url", "image", "video_url", "video"):
                return [
                    _standard_image_to_openai(part)
                    if isinstance(part, dict)
                    and part.get("type") == "image"
                    and ("base64" in part or "url" in part)
                    else part
                    for part in content
                ]

            # Extract text from text blocks
            if block_type == "text" and "text" in block:
                text_parts.append(block["text"])
            # Ignore other block types (tool_call, etc.) - they're handled elsewhere

    # Join text blocks, return None if empty
    result = "".join(text_parts)
    return result if result else None


def convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_dict: Dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {
            "role": message.role,
            "content": _normalize_content(message.content),
        }
    elif isinstance(message, HumanMessage):
        message_dict = {
            "role": "user",
            "content": _normalize_content(message.content),
        }
    elif isinstance(message, AIMessage):
        message_dict = {
            "role": "assistant",
            "content": _normalize_content(message.content),
        }
        # Forward reasoning fields only when they originated from separate API
        # response fields, not from <think> tags.
        api_fields = message.additional_kwargs.get("_reasoning_api_fields", [])
        if "reasoning_content" in api_fields:
            message_dict["reasoning_content"] = message.additional_kwargs[
                "reasoning_content"
            ]
        if "reasoning" in api_fields:
            message_dict["reasoning"] = message.additional_kwargs["reasoning"]
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
            # If function call only, content is None not empty string
            if message_dict["content"] == "":
                message_dict["content"] = None
        if "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
            # If tool calls only, content is None not empty string
            if message_dict["content"] == "":
                message_dict["content"] = None
    elif isinstance(message, SystemMessage):
        message_dict = {
            "role": "system",
            "content": _normalize_content(message.content),
        }
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "content": _normalize_content(message.content),
            "name": message.name,
        }
    elif isinstance(message, ToolMessage):
        # An empty tool result is valid and must remain an empty string. A list
        # with no text blocks normalizes to None, which ChatNVIDIA rejects for
        # tool messages when constructing the next request.
        content = _normalize_content(message.content)
        message_dict = {
            "role": "tool",
            "content": "" if content is None else content,
            "tool_call_id": message.tool_call_id,
        }
    else:
        raise TypeError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict
