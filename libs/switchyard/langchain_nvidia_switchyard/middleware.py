"""Preserve the LangChain handler chain while routing through libsy."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from switchyard.libsy import Algorithm

from .routed_chat_model import _SwitchyardChatModel


class SwitchyardRoutingMiddleware(AgentMiddleware[Any, Any, Any]):
    """Replace each LangChain model call with a caller-configured libsy algorithm."""

    def __init__(self, algorithm: Algorithm) -> None:
        """Create middleware around an already configured Python-bound algorithm."""
        super().__init__()
        self._model = _SwitchyardChatModel(algorithm=algorithm)

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        """Route a synchronous buffered model call and preserve inner middleware."""
        return handler(request.override(model=self._model))

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        """Route an asynchronous buffered model call and preserve inner middleware."""
        return await handler(request.override(model=self._model))
