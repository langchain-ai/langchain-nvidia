from typing import Any, Optional, cast

import pytest
import requests_mock
from langchain_core.messages import AIMessage, BaseMessageChunk, HumanMessage

# from langchain_core.messages.ai import UsageMetadata
from langchain_nvidia_ai_endpoints import ChatNVIDIA

mock_response = {
    "id": "chat-c891882b0c4448a5b258c63d2b031c82",
    "object": "chat.completion",
    "created": 1729173278,
    "model": "meta/llama-3.2-3b-instruct",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "A simple yet"},
            "logprobs": "",
            "finish_reason": "tool_calls",
            "stop_reason": "",
        }
    ],
    "usage": {"prompt_tokens": 12, "total_tokens": 15, "completion_tokens": 3},
    "prompt_logprobs": "",
}


@pytest.fixture
def mock_local_models_metadata(requests_mock: requests_mock.Mocker) -> None:
    mock_response["tool_calls"] = (
        [
            {
                "id": "tool-ID",
                "type": "function",
                "function": {
                    "name": "magic",
                    "arguments": [],
                },
            }
        ],
    )
    requests_mock.post("http://localhost:8888/v1/chat/completions", json=mock_response)


@pytest.fixture
def mock_local_models_stream_metadata(requests_mock: requests_mock.Mocker) -> None:
    response_contents = "\n\n".join(
        [
            'data: {"id":"chatcmpl-ID0","object":"chat.completion.chunk","created":1721155403,"model":"magic-model","system_fingerprint":null,"choices":[{"index":0,"delta":{"role":"assistant","content":null},"logprobs":null,"model_name":"dummy","finish_reason":null}]}',  # noqa: E501
            'data: {"id":"chatcmpl-ID0","object":"chat.completion.chunk","created":1721155403,"model":"magic-model","system_fingerprint":null,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_ID0","type":"function","function":{"name":"xxyyzz","arguments":""}}]},"logprobs":null,"model_name":"dummy","finish_reason":null}]}',  # noqa: E501
            'data: {"id":"chatcmpl-ID0","object":"chat.completion.chunk","created":1721155403,"model":"magic-model","system_fingerprint":null,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"a\\""}}]},"logprobs":null, "model_name":"dummy","finish_reason":null}]}',  # noqa: E501
            'data: {"id":"chatcmpl-ID0","object":"chat.completion.chunk","created":1721155403,"model":"magic-model","system_fingerprint":null,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":": 11,"}}]},"logprobs":null,"model_name":"dummy","finish_reason":null}]}',  # noqa: E501
            'data: {"id":"chatcmpl-ID0","object":"chat.completion.chunk","created":1721155403,"model":"magic-model","system_fingerprint":null,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":" \\"b\\": "}}]},"logprobs":null,"model_name":"dummy","finish_reason":null}]}',  # noqa: E501
            'data: {"id":"chatcmpl-ID0","object":"chat.completion.chunk","created":1721155403,"model":"magic-model","system_fingerprint":null,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"3}"}}]},"logprobs":null,"model_name":"dummy","finish_reason":null}]}',  # noqa: E501
            'data: {"id":"chatcmpl-ID0","object":"chat.completion.chunk","created":1721155403,"model":"magic-model","system_fingerprint":null,"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_ID1","type":"function","function":{"name":"zzyyxx","arguments":""}}]},"logprobs":null,"model_name":"dummy","finish_reason":null}]}',  # noqa: E501
            'data: {"id":"chatcmpl-ID0","object":"chat.completion.chunk","created":1721155403,"model":"magic-model","system_fingerprint":null,"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\\"a\\""}}]},"logprobs":null,"model_name":"dummy","finish_reason":null}]}',  # noqa: E501
            'data: {"id":"chatcmpl-ID0","object":"chat.completion.chunk","created":1721155403,"model":"magic-model","system_fingerprint":null,"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":": 5, "}}]},"logprobs":null,"model_name":"dummy","finish_reason":null}]}',  # noqa: E501
            'data: {"id":"chatcmpl-ID0","object":"chat.completion.chunk","created":1721155403,"model":"magic-model","system_fingerprint":null,"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\\"b\\": 3"}}]},"logprobs":null,"model_name":"dummy","finish_reason":null}]}',  # noqa: E501
            'data: {"id":"chatcmpl-ID0","object":"chat.completion.chunk","created":1721155403,"model":"magic-model","system_fingerprint":null,"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"}"}}]},"logprobs":null,"model_name":"dummy","finish_reason":null}]}',  # noqa: E501
            'data: {"id":"chatcmpl-ID0","object":"chat.completion.chunk","created":1721155403,"model":"magic-model","system_fingerprint":null,"choices":[{"index":0,"delta":{},"logprobs":null,"model_name":"dummy","finish_reason":"tool_calls"}]}',  # noqa: E501
        ]
    )
    requests_mock.post(
        "http://localhost:8888/v1/chat/completions",
        text=response_contents,
    )


def response_metadata_checks(result: Any) -> None:
    assert isinstance(result, AIMessage)
    assert result.response_metadata
    assert all(
        k in result.response_metadata for k in ("model_name", "role", "token_usage")
    )

    assert isinstance(result.content, str)
    assert result.response_metadata.get("model_name") is not None

    if result.usage_metadata is not None:
        assert isinstance(result.usage_metadata, dict)
        usage_metadata = result.usage_metadata

        assert usage_metadata["input_tokens"] > 0
        assert usage_metadata["output_tokens"] > 0
        assert usage_metadata["total_tokens"] > 0


def test_response_metadata(mock_local_models_metadata: None) -> None:
    llm = ChatNVIDIA(base_url="http://localhost:8888/v1")
    result = llm.invoke([HumanMessage(content="I'm PickleRick")])
    response_metadata_checks(result)


async def test_async_response_metadata(mock_local_models_metadata: None) -> None:
    llm = ChatNVIDIA(base_url="http://localhost:8888/v1")
    result = await llm.ainvoke([HumanMessage(content="I'm PickleRick")], logprobs=True)
    response_metadata_checks(result)


def test_response_metadata_streaming(mock_local_models_stream_metadata: None) -> None:
    llm = ChatNVIDIA(base_url="http://localhost:8888/v1")
    full: Optional[BaseMessageChunk] = None
    for chunk in llm.stream("I'm Pickle Rick"):
        assert isinstance(chunk.content, str)
        full = chunk if full is None else full + chunk
    assert all(
        k in cast(BaseMessageChunk, full).response_metadata
        for k in ("model_name", "finish_reason")
    )


async def test_async_response_metadata_streaming(
    mock_local_models_stream_metadata: None,
) -> None:
    llm = ChatNVIDIA(base_url="http://localhost:8888/v1")
    full: Optional[BaseMessageChunk] = None
    async for chunk in llm.astream("I'm Pickle Rick"):
        assert isinstance(chunk.content, str)
        full = chunk if full is None else full + chunk
    assert all(
        k in cast(BaseMessageChunk, full).response_metadata
        for k in ("model_name", "finish_reason")
    )


def test_stream_tool_calls(
    mock_local_models_stream_metadata: None,
) -> None:
    llm = ChatNVIDIA(base_url="http://localhost:8888/v1")
    generator = llm.stream(
        "What is 11 xxyyzz 3 zzyyxx 5?",
    )
    response = next(generator)
    for chunk in generator:
        response += chunk
    assert isinstance(response, AIMessage)
    assert len(response.tool_calls) == 2
    tool_call0 = response.tool_calls[0]
    assert tool_call0["name"] == "xxyyzz"
    assert tool_call0["args"] == {"b": 3, "a": 11}
    tool_call1 = response.tool_calls[1]
    assert tool_call1["name"] == "zzyyxx"
    assert tool_call1["args"] == {"b": 3, "a": 5}
