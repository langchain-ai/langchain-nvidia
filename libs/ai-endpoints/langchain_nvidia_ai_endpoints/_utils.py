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

    In LangChain 1.0, message.content can be:
    - A string (traditional)
    - A list of content blocks (new in v1.0)
    - None

    This function converts list content to string or None as needed.
    For multimodal content (images, etc.), returns the list as-is.
    """
    if content is None or isinstance(content, str):
        return content

    # If content is a list of blocks
    if isinstance(content, list):
        # Check if this is multimodal content (has non-text blocks)
        has_non_text_blocks = any(
            isinstance(block, dict) and block.get("type") not in ("text", None)
            for block in content
        )

        # If multimodal (images, etc.), return as-is for API to handle
        if has_non_text_blocks:
            return content

        # For text-only blocks, extract and join text
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)

        # Return joined text or None if empty
        result = "".join(text_parts)
        return result if result else None

    # For other types, convert to string
    return str(content)


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
