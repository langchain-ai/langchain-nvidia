import json
import warnings
from functools import reduce
from operator import add
from typing import Any, List

import pytest
import requests_mock
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool

from langchain_nvidia_ai_endpoints import ChatNVIDIA


def xxyyzz_func(a: int, b: int) -> int:
    """xxyyzz two numbers"""
    return 42


class xxyyzz_cls(BaseModel):
    """xxyyzz two numbers"""

    a: int = Field(..., description="First number")
    b: int = Field(..., description="Second number")


@tool
def xxyyzz_tool(
    a: int = Field(..., description="First number"),
    b: int = Field(..., description="Second number"),
) -> int:
    """xxyyzz two numbers"""
    return 42


@pytest.mark.parametrize(
    "tools, choice",
    [
        ([xxyyzz_func], "xxyyzz_func"),
        ([xxyyzz_cls], "xxyyzz_cls"),
        ([xxyyzz_tool], "xxyyzz_tool"),
    ],
    ids=["func", "cls", "tool"],
)
def test_bind_tool_and_select(tools: Any, choice: str) -> None:
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=".*not known to support tools.*"
    )
    ChatNVIDIA(api_key="BOGUS").bind_tools(tools=tools, tool_choice=choice)


@pytest.mark.parametrize(
    "tools, choice",
    [
        ([], "wrong"),
        ([xxyyzz_func], "wrong_xxyyzz_func"),
        ([xxyyzz_cls], "wrong_xxyyzz_cls"),
        ([xxyyzz_tool], "wrong_xxyyzz_tool"),
    ],
    ids=["empty", "func", "cls", "tool"],
)
def test_bind_tool_and_select_negative(tools: Any, choice: str) -> None:
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=".*not known to support tools.*"
    )
    with pytest.raises(ValueError) as e:
        ChatNVIDIA(api_key="BOGUS").bind_tools(tools=tools, tool_choice=choice)
    assert "not found in the tools list" in str(e.value)


@pytest.fixture
def mock_v1_models(requests_mock: requests_mock.Mocker) -> None:
    requests_mock.get("https://integrate.api.nvidia.com/v1/models", json={"data": []})


###
# the invoke/stream response_parsing tests are here because of a bug in the
# server response where "" was returned as the arguments for the tool call.
# we're verifying expected results parse correctly.
###


@pytest.mark.parametrize(
    "arguments",
    [
        r"{}",
        # r"",
        r'{"input": 3}',
    ],
    ids=[
        "no-args-oai",
        #  "no-args-nim",
        "one-arg-int",
    ],
)
def test_invoke_response_parsing(
    requests_mock: requests_mock.Mocker,
    mock_v1_models: None,
    arguments: str,
) -> None:
    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        json={
            "id": "chatcmpl-ID",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "BOGUS",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "tool-ID",
                                "type": "function",
                                "function": {
                                    "name": "magic",
                                    "arguments": arguments,
                                },
                            }
                        ],
                    },
                    "logprobs": None,
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 22,
                "completion_tokens": 20,
                "total_tokens": 42,
            },
            "system_fingerprint": None,
        },
    )

    llm = ChatNVIDIA(api_key="BOGUS")
    response = llm.invoke("What's the magic?")
    assert isinstance(response, AIMessage)
    assert response.tool_calls
    assert response.tool_calls[0]["name"] == "magic"
    assert response.tool_calls[0]["args"] == json.loads(arguments)


@pytest.mark.parametrize(
    "argument_chunks",
    [
        [
            r'"{}"',
        ],
        [
            r'""',
        ],
        [
            r'"{\""',
            r'"input\""',
            r'"\":"',
            r"3",
            r'"}"',
        ],
        [r'"{\"intput\": 3}"'],
    ],
    ids=["no-args-oai", "no-args-nim", "one-arg-int-oai", "one-arg-int-nim"],
)
def test_stream_response_parsing(
    requests_mock: requests_mock.Mocker,
    mock_v1_models: None,
    argument_chunks: List[str],
) -> None:
    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        text="\n\n".join(
            [
                'data: {"id":"ID0","object":"chat.completion.chunk","created":1234567890,"model":"BOGUS","system_fingerprint":null,"choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"ID1","type":"function","function":{"name":"magic","arguments":""}}]},"logprobs":null,"finish_reason":null}]}',  # noqa: E501
                *[
                    f'data: {{"id":"ID0","object":"chat.completion.chunk","created":1234567890,"model":"BOGUS","system_fingerprint":null,"choices":[{{"index":0,"delta":{{"tool_calls":[{{"index":0,"function":{{"arguments":{argument}}}}}]}},"logprobs":null,"finish_reason":null}}]}}'  # noqa: E501
                    for argument in argument_chunks
                ],
                'data: {"id":"ID0","object":"chat.completion.chunk","created":1234567890,"model":"BOGUS","system_fingerprint":null,"choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"tool_calls"}]}',  # noqa: E501
                "data: [DONE]",
            ]
        ),
    )

    llm = ChatNVIDIA(api_key="BOGUS")
    response = reduce(add, llm.stream("What's the magic?"))
    assert isinstance(response, AIMessageChunk)
    assert response.tool_calls
    assert response.tool_calls[0]["name"] == "magic"
