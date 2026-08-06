"""Define validated arguments for a target LangChain model invocation."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import BaseMessage


@dataclass(slots=True)
class LangChainInvocation:
    """Hold validated arguments for one target LangChain model call.

    Attributes:
        messages: Ordered target messages, with Switchyard instructions prepended.
        tools: OpenAI-compatible function definitions accepted by ``bind_tools``.
        tool_choice: The optional LangChain tool-selection policy.
        options: Per-call keyword arguments for ``BaseChatModel.ainvoke``.
        stop: Optional stop sequences passed separately to ``ainvoke``.
    """

    messages: list[BaseMessage]
    tools: list[dict[str, object]]
    tool_choice: object | None
    options: dict[str, object]
    stop: list[str] | None
