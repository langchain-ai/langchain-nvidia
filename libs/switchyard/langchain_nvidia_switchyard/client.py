"""Adapt LangChain chat models to libsy's Python client contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from .request_mapper import SwitchyardRequestMapper
from .response_mapper import SwitchyardResponseMapper


class LangChainLlmClient:
    """Use a LangChain chat model as a target for a libsy algorithm."""

    def __init__(self, model: BaseChatModel) -> None:
        """Store the target model without opening or owning its transport."""
        self.model = model

    def _model_name(self) -> str:
        for attribute in ("model_name", "model"):
            value = getattr(self.model, attribute, None)
            if isinstance(value, str) and value:
                return value
        return self.model._llm_type

    async def call(
        self,
        request: Mapping[str, object],
    ) -> Mapping[str, object]:
        """Invoke the target with a buffered Switchyard request."""
        invocation = SwitchyardRequestMapper.from_switchyard(request)

        model: Any = self.model
        if invocation.tools:
            model = self.model.bind_tools(
                invocation.tools,
                tool_choice=cast(Any, invocation.tool_choice),
            )
        response = await model.ainvoke(
            invocation.messages,
            stop=invocation.stop,
            **invocation.options,
        )
        if not isinstance(response, AIMessage):
            raise ValueError(f"target returned {type(response).__name__} instead of AIMessage")
        return SwitchyardResponseMapper.to_switchyard(
            response,
            model_name=self._model_name(),
        )
