"""Delegate buffered chat generation to a libsy algorithm."""

from __future__ import annotations

import asyncio
import contextvars
from collections.abc import Callable, Sequence
from typing import Any, cast

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, PrivateAttr

from .request_mapper import SwitchyardRequestMapper
from .response_mapper import SwitchyardResponseMapper


class _SwitchyardChatModel(BaseChatModel):
    """Present an opaque libsy algorithm as one buffered LangChain chat model."""

    algorithm: Any = Field(exclude=True)
    _sync_runner: asyncio.Runner = PrivateAttr(default_factory=asyncio.Runner)

    @property
    def _llm_type(self) -> str:
        return "switchyard-libsy"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        """Bind Deep Agent tools to the Switchyard request produced at generation time."""
        converted = [convert_to_openai_tool(tool) for tool in tools]
        return self.bind(tools=converted, tool_choice=tool_choice, **kwargs)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools_value = kwargs.pop("tools", [])
        if not isinstance(tools_value, Sequence) or isinstance(
            tools_value, (str, bytes, bytearray)
        ):
            raise ValueError("tools must be a sequence")
        tool_choice = kwargs.pop("tool_choice", None)
        request = SwitchyardRequestMapper.to_switchyard(
            messages,
            tools=cast(Sequence[object], tools_value),
            tool_choice=tool_choice,
            model_settings=kwargs,
            stop=stop,
        )
        decisions, response = await self.algorithm.run(request)
        message = SwitchyardResponseMapper.from_switchyard(
            response,
            decisions=decisions,
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self._sync_runner.run(
                self._agenerate(messages, stop=stop, **kwargs),
                context=contextvars.copy_context(),
            )
        raise RuntimeError(
            "synchronous Switchyard routing cannot run inside an active event loop; "
            "use await agent.ainvoke(...) instead"
        )
