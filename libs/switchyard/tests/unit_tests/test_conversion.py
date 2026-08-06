"""Test the LangChain and libsy neutral-dictionary boundary."""

from __future__ import annotations

from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from langchain_nvidia_switchyard.conversion import (
    ai_message_from_response,
    messages_from_request,
    model_options_from_request,
    request_from_langchain,
    response_from_ai_message,
)


def test_request_preserves_instructions_tool_history_and_model_options() -> None:
    messages = [
        SystemMessage("Be concise."),
        HumanMessage(content=[{"type": "text", "text": "Run the check."}]),
        AIMessage(
            content="",
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
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "run_check",
                "description": "Run a test path.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                "strict": True,
            },
        }
    ]

    result = request_from_langchain(
        messages,
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "run_check"}},
        model_settings={
            "temperature": 0.2,
            "top_p": 0.8,
            "max_tokens": 64,
            "reasoning": {"effort": "low"},
        },
        stop=["DONE"],
    )

    assert result == {
        "model": "auto",
        "instructions": [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Be concise."}],
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Run the check."}],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "call-1",
                        "name": "run_check",
                        "arguments": {"path": "tests"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_call_id": "call-1",
                        "content": [
                            {
                                "type": "text",
                                "text": "fatal runtime error: out of memory",
                            }
                        ],
                        "is_error": True,
                    }
                ],
            },
        ],
        "tools": [
            {
                "name": "run_check",
                "description": "Run a test path.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
                "strict": True,
            }
        ],
        "tool_choice": {"type": "tool", "data": {"name": "run_check"}},
        "sampling": {"temperature": 0.2, "top_p": 0.8},
        "output": {"max_output_tokens": 64},
        "reasoning": {"effort": "low"},
        "stream": False,
        "extensions": {"fields": {"stop": ["DONE"]}},
    }


@pytest.mark.parametrize(
    ("choice", "expected"),
    [
        ("auto", {"type": "auto"}),
        ("required", {"type": "required"}),
        ("any", {"type": "required"}),
        ("none", {"type": "none"}),
        ("run_check", {"type": "tool", "data": {"name": "run_check"}}),
    ],
)
def test_tool_choice_strings_are_normalized(choice: str, expected: dict[str, object]) -> None:
    result = request_from_langchain(
        [HumanMessage("hello")],
        tools=[],
        tool_choice=choice,
        model_settings={},
        stop=None,
    )

    assert result["tool_choice"] == expected


def test_request_rejects_multimodal_content() -> None:
    with pytest.raises(ValueError, match=r"messages\[0\]\.content\[0\].*not supported"):
        request_from_langchain(
            [HumanMessage(content=[{"type": "image", "url": "https://example.test/x"}])],
            tools=[],
            tool_choice=None,
            model_settings={},
            stop=None,
        )


def test_request_preserves_provider_response_format() -> None:
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "contact_info",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
                "additionalProperties": False,
            },
        },
    }

    result = request_from_langchain(
        [HumanMessage("Return a contact.")],
        tools=[],
        tool_choice=None,
        model_settings={"response_format": response_format},
        stop=None,
    )

    assert result["output"] == {"response_format": response_format}


def test_request_rejects_non_mapping_response_format() -> None:
    with pytest.raises(
        ValueError,
        match="model_settings.response_format must be a mapping",
    ):
        request_from_langchain(
            [HumanMessage("hello")],
            tools=[],
            tool_choice=None,
            model_settings={"response_format": "json"},
            stop=None,
        )


def test_neutral_request_becomes_target_messages_tools_and_options() -> None:
    request: dict[str, object] = {
        "instructions": [
            {
                "role": "developer",
                "content": [{"type": "text", "text": "Follow the rubric."}],
            }
        ],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hello"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "call-2",
                        "name": "lookup",
                        "arguments": {"query": "rust"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_call_id": "call-2",
                        "content": [{"type": "text", "text": "result"}],
                        "is_error": False,
                    }
                ],
            },
        ],
        "tools": [
            {
                "name": "lookup",
                "description": "Look up text.",
                "parameters": {"type": "object", "properties": {}},
                "strict": False,
            }
        ],
        "tool_choice": {"type": "tool", "data": {"name": "lookup"}},
        "sampling": {"temperature": 0.1, "top_p": 0.9},
        "output": {
            "max_output_tokens": 32,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "decision", "schema": {"type": "object"}},
            },
        },
        "reasoning": {"effort": "medium"},
        "extensions": {"fields": {"stop": ["STOP"]}},
    }

    messages = messages_from_request(request)
    tools, tool_choice, options = model_options_from_request(request)

    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content == "Follow the rubric."
    assert messages[0].additional_kwargs == {"__openai_role__": "developer"}
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == [{"type": "text", "text": "hello"}]
    assert isinstance(messages[2], AIMessage)
    assert messages[2].tool_calls == [
        {
            "id": "call-2",
            "name": "lookup",
            "args": {"query": "rust"},
            "type": "tool_call",
        }
    ]
    assert isinstance(messages[3], ToolMessage)
    assert messages[3].content == [{"type": "text", "text": "result"}]
    assert messages[3].status == "success"
    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up text.",
                "parameters": {"type": "object", "properties": {}},
                "strict": False,
            },
        }
    ]
    assert tool_choice == {"type": "function", "function": {"name": "lookup"}}
    assert options == {
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": 32,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "decision", "schema": {"type": "object"}},
        },
        "reasoning": {"effort": "medium"},
        "stop": ["STOP"],
    }


def test_ai_message_becomes_neutral_response_with_usage() -> None:
    message = AIMessage(
        content="",
        id="response-1",
        tool_calls=[
            {
                "id": "call-3",
                "name": "search",
                "args": {"query": "switchyard"},
                "type": "tool_call",
            }
        ],
        response_metadata={"finish_reason": "tool_calls"},
        usage_metadata={
            "input_tokens": 12,
            "output_tokens": 4,
            "total_tokens": 18,
            "input_token_details": {"cache_read": 2},
            "output_token_details": {"reasoning": 2},
        },
    )

    assert response_from_ai_message(message, model_name="efficient") == {
        "id": "response-1",
        "model": "efficient",
        "outputs": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "call-3",
                        "name": "search",
                        "arguments": {"query": "switchyard"},
                    }
                ],
                "stop_reason": "tool_use",
            }
        ],
        "usage": {
            "input_tokens": 12,
            "output_tokens": 4,
            "total_tokens": 18,
            "cached_input_tokens": 2,
            "reasoning_tokens": 2,
        },
    }


def test_ai_message_reasoning_round_trips_through_switchyard_content() -> None:
    message = AIMessage(
        "visible answer",
        additional_kwargs={"reasoning_content": "private reasoning"},
    )

    response = response_from_ai_message(message, model_name="efficient")

    outputs = cast(list[dict[str, object]], response["outputs"])
    assert outputs[0]["content"] == [
        {"type": "reasoning", "text": "private reasoning"},
        {"type": "text", "text": "visible answer"},
    ]

    restored = ai_message_from_response(
        {
            **response,
            "outputs": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "reasoning",
                            "text": "private reasoning",
                            "signature": "provider-signature",
                        },
                        {"type": "text", "text": "visible answer"},
                    ],
                    "stop_reason": "end_turn",
                }
            ],
        },
        [],
    )

    assert restored.content_blocks == [
        {
            "type": "reasoning",
            "reasoning": "private reasoning",
            "extras": {"signature": "provider-signature"},
        },
        {"type": "text", "text": "visible answer"},
    ]


def test_neutral_response_becomes_ai_message_with_decision_trace() -> None:
    response: dict[str, Any] = {
        "id": "response-2",
        "model": "provider-model",
        "outputs": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "done"}],
                "stop_reason": "end_turn",
            }
        ],
        "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
    }
    decisions = [
        {"selected_model": "judge", "reasoning": "score request"},
        {"selected_model": "capable", "reasoning": "score exceeded threshold"},
    ]

    message = ai_message_from_response(response, decisions)

    assert message.content == [{"type": "text", "text": "done"}]
    assert message.id == "response-2"
    assert message.response_metadata == {
        "model_name": "provider-model",
        "finish_reason": "stop",
        "switchyard": {
            "decisions": decisions,
            "selected_model": "capable",
        },
    }
    assert message.usage_metadata == {
        "input_tokens": 5,
        "output_tokens": 1,
        "total_tokens": 6,
    }


def test_neutral_response_requires_assistant_content() -> None:
    with pytest.raises(ValueError, match="no assistant text or tool calls"):
        ai_message_from_response(
            {
                "model": "empty",
                "outputs": [{"role": "assistant", "content": [], "stop_reason": "end_turn"}],
            },
            [],
        )
