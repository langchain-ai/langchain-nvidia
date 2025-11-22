from __future__ import annotations

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


def _normalize_content(content: Any) -> Any:
    """Normalize message content to handle LangChain 1.0 content blocks.

    In LangChain 1.0, `message.content` can be:

    - A string (traditional)
    - A list of content blocks (new in v1.0)
    - `None`

    This function converts list content to string or `None` as needed.

    For multimodal content (images), returns the list as-is.
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

            # Preserve multimodal content (images) as-is for VLM models
            if block_type in ("image_url", "image"):
                return content

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
        message_dict = {
            "role": "tool",
            "content": _normalize_content(message.content),
            "tool_call_id": message.tool_call_id,
        }
    else:
        raise TypeError(f"Got unknown type {message}")
    if "name" in message.additional_kwargs:
        message_dict["name"] = message.additional_kwargs["name"]
    return message_dict
