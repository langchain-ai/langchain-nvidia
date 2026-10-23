"""Regression coverage for assistant tool calls in outgoing conversations."""

import json
from copy import deepcopy
from typing import Any

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    ToolMessage,
)
from langchain_core.messages.tool import ToolCall

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_ai_endpoints._utils import convert_message_to_dict

from .conftest import MockHTTP


@pytest.mark.parametrize("content", ["", [], [{"type": "text", "text": ""}], "Working"])
@pytest.mark.parametrize("count", [1, 2])
def test_normalized_tool_calls(content: Any, count: int) -> None:
    """Normalized calls retain order, IDs, names and JSON argument values."""
    all_calls: list[ToolCall] = [
        {
            "id": "call_read",
            "name": "read_file",
            "args": {"path": '资料/"notes".md', "options": {"lines": [1, 2]}},
            "type": "tool_call",
        },
        {
            "id": "call_search",
            "name": "search",
            "args": {"query": "line\nnext", "exact": True, "limit": None},
            "type": "tool_call",
        },
    ]
    calls = all_calls[:count]
    message = AIMessage(content=content, tool_calls=calls)
    original = deepcopy(message.model_dump())
    assert "tool_calls" not in message.additional_kwargs

    result = convert_message_to_dict(message)

    assert result["role"] == "assistant"
    assert result["content"] == ("Working" if content == "Working" else None)
    assert len(result["tool_calls"]) == count
    for actual, expected in zip(result["tool_calls"], calls):
        assert set(actual) == {"id", "type", "function"}
        assert actual["id"] == expected["id"]
        assert actual["type"] == "function"
        assert set(actual["function"]) == {"name", "arguments"}
        assert actual["function"]["name"] == expected["name"]
        assert isinstance(actual["function"]["arguments"], str)
        assert json.loads(actual["function"]["arguments"]) == expected["args"]
    assert message.model_dump() == original
    assert convert_message_to_dict(message) == result


@pytest.mark.parametrize("arguments", ['{ "path" : "README.md" }', "not-json"])
def test_legacy_raw_tool_calls(arguments: str) -> None:
    """Legacy calls, including unparseable arguments, retain their wire data."""
    raw = [
        {
            "id": "call_legacy",
            "type": "function",
            "function": {"name": "read_file", "arguments": arguments},
        }
    ]
    message = AIMessage(content="", additional_kwargs={"tool_calls": raw})

    assert convert_message_to_dict(message) == {
        "role": "assistant",
        "content": None,
        "tool_calls": raw,
    }


def test_legacy_function_call() -> None:
    function_call = {"name": "read_file", "arguments": '{"path":"README.md"}'}
    message = AIMessage(content="", additional_kwargs={"function_call": function_call})

    assert convert_message_to_dict(message) == {
        "role": "assistant",
        "content": None,
        "function_call": function_call,
    }


def test_raw_tool_calls_take_precedence() -> None:
    """Keep existing raw precedence without duplicating or rewriting calls."""
    raw = [
        {
            "id": "call_raw",
            "type": "function",
            "function": {"name": "raw_name", "arguments": '{ "value" : 1 }'},
            "extra": "preserve-provider-data",
        }
    ]
    message = AIMessage(
        content="Working",
        tool_calls=[{"id": "call_normalized", "name": "other", "args": {"value": 2}}],
        additional_kwargs={"tool_calls": raw},
    )
    original = deepcopy(message.model_dump())

    assert convert_message_to_dict(message) == {
        "role": "assistant",
        "content": "Working",
        "tool_calls": raw,
    }
    assert message.model_dump() == original


@pytest.mark.parametrize("raw", [[], None])
def test_empty_legacy_tool_calls(raw: Any) -> None:
    """Preserve the existing behavior of explicitly supplied empty raw data."""
    message = AIMessage(content="", additional_kwargs={"tool_calls": raw})

    assert convert_message_to_dict(message) == {
        "role": "assistant",
        "content": None,
        "tool_calls": raw,
    }


@pytest.mark.parametrize(
    "content, expected", [("Hello", "Hello"), ("", ""), ([], None)]
)
def test_message_without_tool_calls(content: Any, expected: Any) -> None:
    assert convert_message_to_dict(AIMessage(content=content)) == {
        "role": "assistant",
        "content": expected,
    }


@pytest.mark.parametrize("use_async", [False, True], ids=["invoke", "ainvoke"])
@pytest.mark.parametrize("from_chunks", [False, True], ids=["message", "stream-chunks"])
async def test_normalized_tool_calls_in_followup_request(
    use_async: bool, from_chunks: bool, mock_http: MockHTTP
) -> None:
    """Assistant calls and matching tool results survive the actual request path."""
    assistant: AIMessage
    if from_chunks:
        assistant = AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"index": 0, "id": "call_1", "name": "read_file", "args": '{"path":'}
            ],
        ) + AIMessageChunk(
            content="",
            tool_call_chunks=[
                {"index": 0, "id": None, "name": None, "args": '"README.md"}'}
            ],
        )
    else:
        assistant = AIMessage(
            content="",
            tool_calls=[
                {"id": "call_1", "name": "read_file", "args": {"path": "README.md"}}
            ],
        )
    assert "tool_calls" not in assistant.additional_kwargs
    response = {"choices": [{"message": {"role": "assistant", "content": "Done"}}]}
    mock_http.requests.post(
        "https://integrate.api.nvidia.com/v1/chat/completions", json=response
    )
    mock_http.set_post(json_body=response)
    model = ChatNVIDIA(model="nvidia/nemotron-3.5-super-vl-120b-a12b", api_key="BOGUS")
    history = [
        HumanMessage(content="Read README.md"),
        assistant,
        ToolMessage(content="File contents", tool_call_id="call_1"),
    ]

    if use_async:
        result = await model.ainvoke(history)
        payload = mock_http.history[-1].kwargs["json"]
    else:
        result = model.invoke(history)
        assert mock_http.requests.last_request is not None
        payload = mock_http.requests.last_request.json()

    assert result.content == "Done"
    assert payload["messages"][1] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": json.dumps({"path": "README.md"}),
                },
            }
        ],
    }
    assert payload["messages"][2] == {
        "role": "tool",
        "content": "File contents",
        "tool_call_id": "call_1",
    }
