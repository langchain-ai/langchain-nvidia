"""Test handler-preserving Switchyard routing middleware."""

from __future__ import annotations

from typing import Any

import pytest
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.outputs import ChatResult

from langchain_nvidia_switchyard import SwitchyardRoutingMiddleware


class UnusedChatModel(BaseChatModel):
    """Base request model that fails if the middleware does not replace it."""

    @property
    def _llm_type(self) -> str:
        return "unused"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise AssertionError("the original Deep Agent model was invoked")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise AssertionError("the original Deep Agent model was invoked")


class RecordingAlgorithm:
    """Structural stand-in for the opaque native Algorithm handle."""

    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    async def run(
        self,
        request: dict[str, object],
        headers: object | None = None,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        self.requests.append(request)
        return (
            [
                {
                    "selected_model": "efficient",
                    "reasoning": "test routing selected efficient",
                }
            ],
            {
                "id": "routed-1",
                "model": "provider/efficient",
                "outputs": [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "routed response"}],
                        "stop_reason": "end_turn",
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
            },
        )


def _request() -> ModelRequest:
    return ModelRequest(
        model=UnusedChatModel(),
        messages=[HumanMessage("hello")],
        tools=[],
        model_settings={},
    )


async def test_async_middleware_routes_through_replacement_model_and_handler() -> None:
    algorithm = RecordingAlgorithm()
    middleware = SwitchyardRoutingMiddleware(algorithm)
    handled_models: list[BaseChatModel] = []

    async def handler(request: ModelRequest) -> ModelResponse:
        handled_models.append(request.model)
        message = await request.model.ainvoke(request.messages)
        return ModelResponse(result=[message])

    response = await middleware.awrap_model_call(_request(), handler)

    assert len(handled_models) == 1
    assert not isinstance(handled_models[0], UnusedChatModel)
    assert response.result[0].content == [{"type": "text", "text": "routed response"}]
    assert response.result[0].response_metadata["switchyard"] == {
        "decisions": [
            {
                "selected_model": "efficient",
                "reasoning": "test routing selected efficient",
            }
        ],
        "selected_model": "efficient",
    }
    assert algorithm.requests[0]["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hello"}]}
    ]


async def test_bound_tools_reach_the_neutral_algorithm_request() -> None:
    algorithm = RecordingAlgorithm()
    middleware = SwitchyardRoutingMiddleware(algorithm)
    tool = {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up text.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    request = ModelRequest(
        model=UnusedChatModel(),
        messages=[HumanMessage("hello")],
        tools=[tool],
        tool_choice="required",
        model_settings={"temperature": 0.3},
    )

    async def handler(routed_request: ModelRequest) -> ModelResponse:
        bound = routed_request.model.bind_tools(
            routed_request.tools,
            tool_choice=routed_request.tool_choice,
        )
        message = await bound.ainvoke(
            routed_request.messages,
            **routed_request.model_settings,
        )
        return ModelResponse(result=[message])

    await middleware.awrap_model_call(request, handler)

    assert algorithm.requests[0]["tools"] == [
        {
            "name": "lookup",
            "description": "Look up text.",
            "parameters": {"type": "object", "properties": {}},
        }
    ]
    assert algorithm.requests[0]["tool_choice"] == {"type": "required"}
    assert algorithm.requests[0]["sampling"] == {"temperature": 0.3}


def test_sync_middleware_runs_the_same_async_algorithm() -> None:
    algorithm = RecordingAlgorithm()
    middleware = SwitchyardRoutingMiddleware(algorithm)

    def handler(request: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[request.model.invoke(request.messages)])

    response = middleware.wrap_model_call(_request(), handler)

    assert response.result[0].text == "routed response"
    assert len(algorithm.requests) == 1


async def test_sync_model_call_inside_event_loop_directs_user_to_ainvoke() -> None:
    middleware = SwitchyardRoutingMiddleware(RecordingAlgorithm())

    def handler(request: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[request.model.invoke(request.messages)])

    with pytest.raises(RuntimeError, match=r"active event loop.*await agent\.ainvoke"):
        middleware.wrap_model_call(_request(), handler)
