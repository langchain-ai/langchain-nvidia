"""Compatibility facade for internal LangChain and Switchyard mappers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from langchain_core.messages import AIMessage, BaseMessage

from .request_mapper import SwitchyardRequestMapper
from .response_mapper import SwitchyardResponseMapper


def request_from_langchain(
    messages: Sequence[BaseMessage],
    *,
    tools: Sequence[object],
    tool_choice: object | None,
    model_settings: Mapping[str, object],
    stop: list[str] | None,
) -> dict[str, object]:
    """Build a buffered Switchyard request from a LangChain model call."""
    return SwitchyardRequestMapper.to_switchyard(
        messages,
        tools=tools,
        tool_choice=tool_choice,
        model_settings=model_settings,
        stop=stop,
    )


def messages_from_request(request: Mapping[str, object]) -> list[BaseMessage]:
    """Convert a Switchyard request into target LangChain messages."""
    return SwitchyardRequestMapper.from_switchyard(request).messages


def model_options_from_request(
    request: Mapping[str, object],
) -> tuple[list[dict[str, object]], object | None, dict[str, object]]:
    """Extract LangChain tools, tool choice, and options from a Switchyard request."""
    invocation = SwitchyardRequestMapper.from_switchyard(request)
    options = dict(invocation.options)
    if invocation.stop is not None:
        options["stop"] = invocation.stop
    return invocation.tools, invocation.tool_choice, options


def response_from_ai_message(message: AIMessage, *, model_name: str) -> dict[str, object]:
    """Convert a target LangChain response into a Switchyard response."""
    return SwitchyardResponseMapper.to_switchyard(message, model_name=model_name)


def ai_message_from_response(
    response: Mapping[str, object],
    decisions: Sequence[Mapping[str, object]],
) -> AIMessage:
    """Convert a routed Switchyard response into a LangChain response."""
    return SwitchyardResponseMapper.from_switchyard(response, decisions=decisions)


__all__ = [
    "ai_message_from_response",
    "messages_from_request",
    "model_options_from_request",
    "request_from_langchain",
    "response_from_ai_message",
]
