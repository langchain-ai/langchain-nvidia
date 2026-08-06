"""Hermetic tests for every currently Python-bound libsy algorithm."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

from deepagents import create_deep_agent
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from switchyard.libsy import LlmTarget, TaskClassifierConfig, algorithms

from langchain_nvidia_switchyard import LangChainLlmClient, SwitchyardRoutingMiddleware


class StaticChatModel(BaseChatModel):
    """Return one literal response while recording complete model options."""

    model_name: str
    response_text: str
    response_tool_name: str | None = None
    response_tool_args: dict[str, object] = Field(default_factory=dict)
    calls: list[dict[str, object]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "static"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        return self.bind(tools=list(tools), tool_choice=tool_choice, **kwargs)

    def _result(
        self, messages: list[BaseMessage], stop: list[str] | None, **kwargs: Any
    ) -> ChatResult:
        self.calls.append({"messages": messages, "stop": stop, **kwargs})
        tool_calls = []
        if self.response_tool_name is not None:
            tool_calls.append(
                {
                    "id": "call-structured-output",
                    "name": self.response_tool_name,
                    "args": self.response_tool_args,
                    "type": "tool_call",
                }
            )
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        self.response_text,
                        tool_calls=tool_calls,
                        response_metadata={"finish_reason": "stop"},
                    )
                )
            ]
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._result(messages, stop, **kwargs)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._result(messages, stop, **kwargs)


class ContactInfo(BaseModel):
    """Structured response used to verify both LangChain strategies."""

    name: str
    email: str


async def _run_algorithm(
    algorithm: Any,
    messages: list[AnyMessage],
) -> AIMessage:
    middleware = SwitchyardRoutingMiddleware(algorithm)
    request = ModelRequest(
        model=StaticChatModel(model_name="unused", response_text="unused"),
        messages=messages,
        tools=[],
    )

    async def handler(routed: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[await routed.model.ainvoke(routed.messages)])

    response = await middleware.awrap_model_call(request, handler)
    return cast(AIMessage, response.result[0])


async def test_noop_runs_through_the_generic_middleware() -> None:
    message = await _run_algorithm(algorithms.noop(), [HumanMessage("hello")])

    assert message.text == "OK"
    assert message.response_metadata["switchyard"]["selected_model"] == "auto"


async def test_seeded_random_runs_a_langchain_target() -> None:
    target = StaticChatModel(model_name="efficient-provider", response_text="efficient")
    algorithm = algorithms.random(
        [LlmTarget("efficient", LangChainLlmClient(target))],
        seed=42,
    )

    message = await _run_algorithm(algorithm, [HumanMessage("hello")])

    assert message.text == "efficient"
    assert message.response_metadata["switchyard"]["selected_model"] == "efficient"
    assert len(target.calls) == 1


async def test_llm_task_classifier_runs_judge_then_efficient_target() -> None:
    judge = StaticChatModel(
        model_name="judge-provider",
        response_text=(
            '{"crux":"bounded task","primary_rule":"SUP-1",'
            '"capability_boundary":"supported","p_solve":0.9}'
        ),
    )
    efficient = StaticChatModel(model_name="efficient-provider", response_text="efficient")
    capable = StaticChatModel(model_name="capable-provider", response_text="capable")
    algorithm = algorithms.llm_task_classifier(
        LlmTarget("judge", LangChainLlmClient(judge)),
        LlmTarget("efficient", LangChainLlmClient(efficient)),
        LlmTarget("capable", LangChainLlmClient(capable)),
        config=TaskClassifierConfig(0.5),
    )

    message = await _run_algorithm(algorithm, [HumanMessage("Say hello.")])

    assert message.text == "efficient"
    assert len(judge.calls) == 1
    assert len(efficient.calls) == 1
    assert capable.calls == []


async def test_stage_router_uses_deep_agent_failed_tool_history() -> None:
    efficient = StaticChatModel(model_name="efficient-provider", response_text="efficient")
    capable = StaticChatModel(model_name="capable-provider", response_text="capable")
    algorithm = algorithms.stage_router(
        LlmTarget("capable", LangChainLlmClient(capable)),
        LlmTarget("efficient", LangChainLlmClient(efficient)),
        picker="efficient_first",
        confidence_threshold=0.5,
        recent_window=3,
    )

    simple = await _run_algorithm(algorithm, [HumanMessage("Say hello.")])
    failed = await _run_algorithm(
        algorithm,
        [
            HumanMessage("Fix the tests."),
            AIMessage(
                "",
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "run_check",
                        "args": {"path": "tests"},
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                "fatal runtime error: out of memory",
                tool_call_id="call-1",
                status="error",
            ),
        ],
    )

    assert simple.text == "efficient"
    assert simple.response_metadata["switchyard"]["selected_model"] == "efficient"
    assert failed.text == "capable"
    assert failed.response_metadata["switchyard"]["selected_model"] == "capable"


async def test_real_deep_agent_binds_its_tools_to_the_selected_target() -> None:
    target = StaticChatModel(model_name="efficient-provider", response_text="finished")
    algorithm = algorithms.random(
        [LlmTarget("efficient", LangChainLlmClient(target))],
        seed=7,
    )
    agent = create_deep_agent(
        model=StaticChatModel(model_name="unused", response_text="unused"),
        middleware=[SwitchyardRoutingMiddleware(algorithm)],
    )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Reply with a short greeting."}]}
    )

    final = cast(AIMessage, result["messages"][-1])
    assert final.text == "finished"
    assert final.response_metadata["switchyard"]["selected_model"] == "efficient"
    tools = target.calls[0]["tools"]
    assert isinstance(tools, list)
    assert {tool["function"]["name"] for tool in tools} >= {
        "read_file",
        "write_file",
        "task",
    }


async def test_real_deep_agent_raw_schema_uses_portable_tool_strategy() -> None:
    target = StaticChatModel(
        model_name="efficient-provider",
        response_text="",
        response_tool_name="ContactInfo",
        response_tool_args={
            "name": "Ada Lovelace",
            "email": "ada@example.com",
        },
    )
    algorithm = algorithms.random(
        [LlmTarget("efficient", LangChainLlmClient(target))],
        seed=11,
    )
    agent = create_deep_agent(
        model=StaticChatModel(model_name="unused", response_text="unused"),
        middleware=[SwitchyardRoutingMiddleware(algorithm)],
        response_format=ContactInfo,
    )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Return Ada's contact details."}]}
    )

    assert result["structured_response"] == ContactInfo(
        name="Ada Lovelace",
        email="ada@example.com",
    )
    final = next(
        message for message in reversed(result["messages"]) if isinstance(message, AIMessage)
    )
    assert final.response_metadata["switchyard"]["selected_model"] == "efficient"
    assert target.calls[0]["tool_choice"] == "required"
    assert "response_format" not in target.calls[0]


async def test_real_deep_agent_explicit_provider_strategy_reaches_target() -> None:
    target = StaticChatModel(
        model_name="efficient-provider",
        response_text='{"name":"Ada Lovelace","email":"ada@example.com"}',
    )
    algorithm = algorithms.random(
        [LlmTarget("efficient", LangChainLlmClient(target))],
        seed=13,
    )
    agent = create_deep_agent(
        model=StaticChatModel(model_name="unused", response_text="unused"),
        middleware=[SwitchyardRoutingMiddleware(algorithm)],
        response_format=ProviderStrategy(ContactInfo),
    )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Return Ada's contact details."}]}
    )

    assert result["structured_response"] == ContactInfo(
        name="Ada Lovelace",
        email="ada@example.com",
    )
    assert target.calls[0]["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "ContactInfo",
            "schema": {
                "description": ("Structured response used to verify both LangChain strategies."),
                "properties": {
                    "name": {"title": "Name", "type": "string"},
                    "email": {"title": "Email", "type": "string"},
                },
                "required": ["name", "email"],
                "title": "ContactInfo",
                "type": "object",
            },
        },
    }
