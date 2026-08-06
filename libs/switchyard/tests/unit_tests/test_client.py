"""Test adaptation of LangChain chat models to libsy clients."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from pydantic import Field

from langchain_nvidia_switchyard import LangChainLlmClient


class RecordingChatModel(BaseChatModel):
    """Deterministic model that records the real LangChain invocation boundary."""

    model_name: str = "provider/fake-target"
    received_messages: list[list[BaseMessage]] = Field(default_factory=list)
    received_options: list[dict[str, object]] = Field(default_factory=list)
    response_message: BaseMessage = Field(
        default_factory=lambda: AIMessage(
            "target response",
            id="response-1",
            response_metadata={"finish_reason": "stop"},
            usage_metadata={
                "input_tokens": 8,
                "output_tokens": 2,
                "total_tokens": 10,
            },
        )
    )

    @property
    def _llm_type(self) -> str:
        return "recording"

    def bind_tools(
        self,
        tools: Any,
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[Any, AIMessage]:
        return self.bind(tools=tools, tool_choice=tool_choice, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise AssertionError("the libsy client must use the async model path")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.received_messages.append(messages)
        self.received_options.append({**kwargs, "stop": stop})
        return ChatResult(generations=[ChatGeneration(message=self.response_message)])


async def test_client_invokes_target_with_algorithm_transformed_request() -> None:
    model = RecordingChatModel()
    client = LangChainLlmClient(model)
    request = {
        "instructions": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Injected by the router."}],
            }
        ],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        "tools": [
            {
                "name": "lookup",
                "description": "Look up text.",
                "parameters": {"type": "object", "properties": {}},
                "strict": True,
            }
        ],
        "tool_choice": {"type": "tool", "data": {"name": "lookup"}},
        "sampling": {"temperature": 0.1},
        "output": {"max_output_tokens": 64},
        "extensions": {"fields": {"stop": ["DONE"]}},
    }

    response = await client.call(request)

    assert [message.content for message in model.received_messages[0]] == [
        "Injected by the router.",
        [{"type": "text", "text": "hello"}],
    ]
    assert model.received_options == [
        {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Look up text.",
                        "parameters": {"type": "object", "properties": {}},
                        "strict": True,
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "lookup"}},
            "temperature": 0.1,
            "max_tokens": 64,
            "stop": ["DONE"],
        }
    ]
    assert response == {
        "id": "response-1",
        "model": "provider/fake-target",
        "outputs": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "target response"}],
                "stop_reason": "end_turn",
            }
        ],
        "usage": {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10},
    }


async def test_client_requires_an_ai_message() -> None:
    model = RecordingChatModel(response_message=HumanMessage("wrong role"))
    client = LangChainLlmClient(model)

    with pytest.raises(ValueError, match="target returned HumanMessage instead of AIMessage"):
        await client.call(
            {"messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]}
        )
