import json
import warnings
from functools import reduce
from operator import add
from typing import Annotated, Any, List, Optional

import pytest
import requests_mock
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from langchain_nvidia_ai_endpoints import ChatNVIDIA


def xxyyzz_func(a: int, b: int) -> int:
    """xxyyzz two numbers"""
    return 42


class xxyyzz_cls(BaseModel):
    """xxyyzz two numbers"""

    a: int = Field(..., description="First number")
    b: int = Field(..., description="Second number")


@tool
def xxyyzz_tool_field(
    a: int = Field(..., description="First number"),
    b: int = Field(..., description="Second number"),
) -> int:
    """xxyyzz two numbers"""
    return 42


@tool
def xxyyzz_tool_annotated(
    a: Annotated[int, "First number"],
    b: Annotated[int, "Second number"],
) -> int:
    """xxyyzz two numbers"""
    return 42


@pytest.mark.parametrize(
    "tools, choice",
    [
        ([xxyyzz_func], "xxyyzz_func"),
        ([xxyyzz_cls], "xxyyzz_cls"),
        ([xxyyzz_tool_field], "xxyyzz_tool_field"),
        ([xxyyzz_tool_annotated], "xxyyzz_tool_annotated"),
    ],
    ids=["func", "cls", "tool_field", "tool_annotated"],
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
        ([xxyyzz_tool_field], "wrong_xxyyzz_tool_field"),
        ([xxyyzz_tool_annotated], "wrong_xxyyzz_tool_annotated"),
    ],
    ids=["empty", "func", "cls", "tool_field", "tool_annotated"],
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
            r'"3"',
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
                'data: {"id":"ID0","object":"chat.completion.chunk","created":1234567890,"model":"BOGUS","choices":[],"usage":{"prompt_tokens":20,"total_tokens":42,"completion_tokens":22}}',  # noqa: E501
                "data: [DONE]",
            ]
        ),
    )

    llm = ChatNVIDIA(api_key="BOGUS")
    response = reduce(add, llm.stream("What's the magic?"))
    assert isinstance(response, AIMessageChunk)
    assert response.tool_calls
    assert response.tool_calls[0]["name"] == "magic"


def test_regression_parsing_human_ai_tool_invoke(
    requests_mock: requests_mock.Mocker,
) -> None:
    """
    a bug existed in the inference for sequence -
     0. messages = [human message]
     1. messages.append(llm.invoke(messages))
     2. llm.invoke(messages) <- raised ValueError: Message ... has no content
    """
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
                                    "arguments": "{}",
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
    messages: List[BaseMessage] = [HumanMessage("THIS IS IGNORED")]
    response0 = llm.invoke(messages)
    messages.append(response0)
    messages.append(ToolMessage(content="SO IS THIS", tool_call_id="BOGUS"))
    llm.invoke(messages)


def test_regression_ai_null_content(
    requests_mock: requests_mock.Mocker,
) -> None:
    requests_mock.post("https://integrate.api.nvidia.com/v1/chat/completions", json={})
    llm = ChatNVIDIA(api_key="BOGUS")
    assistant = AIMessage(content="SKIPPED")
    assistant.content = None  # type: ignore
    llm.invoke([assistant])
    llm.stream([assistant])


def test_stream_usage_metadata(
    requests_mock: requests_mock.Mocker,
) -> None:
    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        text="\n\n".join(
            [
                r'data: {"id":"ID0","object":"chat.completion.chunk","created":1234567890,"model":"BOGUS","system_fingerprint":null,"usage":null,"choices":[{"index":0,"delta":{"role":"assistant","content":null},"logprobs":null,"finish_reason":null}]}',  # noqa: E501
                r'data: {"id":"ID0","object":"chat.completion.chunk","created":1234567890,"model":"BOGUS","system_fingerprint":null,"usage":null,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"ID1","type":"function","function":{"name":"magic_function","arguments":""}}]},"logprobs":null,"finish_reason":null}]}',  # noqa: E501
                r'data: {"id":"ID0","object":"chat.completion.chunk","created":1234567890,"model":"BOGUS","system_fingerprint":null,"usage":null,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"in"}}]},"logprobs":null,"finish_reason":null}]}',  # noqa: E501
                r'data: {"id":"ID0","object":"chat.completion.chunk","created":1234567890,"model":"BOGUS","system_fingerprint":null,"usage":null,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"put\":"}}]},"logprobs":null,"finish_reason":null}]}',  # noqa: E501
                r'data: {"id":"ID0","object":"chat.completion.chunk","created":1234567890,"model":"BOGUS","system_fingerprint":null,"usage":null,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":" 3}"}}]},"logprobs":null,"finish_reason":null}]}',  # noqa: E501
                r'data: {"id":"ID0","object":"chat.completion.chunk","created":1234567890,"model":"BOGUS","system_fingerprint":null,"choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"tool_calls"}],"usage":null}',  # noqa: E501
                r'data: {"id":"ID0","object":"chat.completion.chunk","created":1234567890,"model":"BOGUS","system_fingerprint":null,"choices":[],"usage":{"prompt_tokens":76,"completion_tokens":29,"total_tokens":105}}',  # noqa: E501
                r"data: [DONE]",
            ]
        ),
    )

    llm = ChatNVIDIA(api_key="BOGUS")
    response = reduce(add, llm.stream("IGNROED"))
    assert isinstance(response, AIMessage)
    assert response.usage_metadata is not None
    assert response.usage_metadata["input_tokens"] == 76
    assert response.usage_metadata["output_tokens"] == 29
    assert response.usage_metadata["total_tokens"] == 105


@pytest.mark.parametrize(
    "strict",
    [False, None, "BOGUS"],
)
def test_strict_warns(strict: Optional[bool]) -> None:
    warnings.filterwarnings("error")  # no warnings should be raised

    # acceptable warnings
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=".*not known to support.*"
    )

    # warnings under test
    strict_warning = ".*`strict` parameter is not necessary.*"
    warnings.filterwarnings("default", category=UserWarning, message=strict_warning)

    with pytest.warns(UserWarning, match=strict_warning):
        ChatNVIDIA(api_key="BOGUS").bind_tools(
            tools=[xxyyzz_tool_annotated],
            strict=strict,
        )


@pytest.mark.parametrize(
    "strict",
    [True, None],
    ids=["strict-True", "no-strict"],
)
def test_strict_no_warns(strict: Optional[bool]) -> None:
    warnings.filterwarnings("error")  # no warnings should be raised

    # acceptable warnings
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=".*not known to support.*"
    )

    ChatNVIDIA(api_key="BOGUS").bind_tools(
        tools=[xxyyzz_tool_annotated],
        **({"strict": strict} if strict is not None else {}),
    )


def test_json_mode(
    requests_mock: requests_mock.Mocker,
    mock_v1_models: None,
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
                        "content": '{"a": 1}',
                    },
                    "logprobs": None,
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

    llm = ChatNVIDIA(api_key="BOGUS").bind(response_format={"type": "json_object"})
    response = llm.invoke("Return this as json: {'a': 1}")
    assert isinstance(response, AIMessage)
    assert json.loads(str(response.content)) == {"a": 1}
