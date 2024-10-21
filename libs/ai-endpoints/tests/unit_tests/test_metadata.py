from typing import Optional, cast

import pytest
import requests_mock
from langchain_core.messages import BaseMessageChunk, HumanMessage

from langchain_nvidia_ai_endpoints import ChatNVIDIA


@pytest.fixture
def mock_local_models_metadata(requests_mock: requests_mock.Mocker) -> None:
    requests_mock.post(
        "http://localhost:8888/v1/chat/completions",
        json={
            "id": "unknown_model",
            "created": 1234567890,
            "object": "chat.completion",
            "model": "mock-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "WORKED"},
                },
            ],
            "response_metadata": {
                "role": "user",
                "token_usage": {
                    "prompt_tokens": 20,
                    "total_tokens": 486,
                    "completion_tokens": 466,
                },
                "finish_reason": "stop",
                "model_name": "meta",
            },
        },
    )


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


async def test_ainvoke(mock_local_models_metadata: None) -> None:
    """Test invoke tokens from ChatNVIDIA."""
    llm = ChatNVIDIA(model="unknown_model", base_url="http://localhost:8888/v1/")

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)
    assert result.response_metadata.get("model_name") is not None


def test_invoke(mock_local_models_metadata: None) -> None:
    """Test invoke tokens from ChatNVIDIA."""
    llm = ChatNVIDIA(base_url="http://localhost:8888/v1")

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)
    assert result.response_metadata.get("model_name") is not None


def test_response_metadata(mock_local_models_metadata: None) -> None:
    llm = ChatNVIDIA(base_url="http://localhost:8888/v1")
    result = llm.invoke([HumanMessage(content="I'm PickleRick")])
    assert result.response_metadata
    assert all(k in result.response_metadata for k in ("model_name", "role"))


async def test_async_response_metadata(mock_local_models_metadata: None) -> None:
    llm = ChatNVIDIA(base_url="http://localhost:8888/v1")
    result = await llm.ainvoke([HumanMessage(content="I'm PickleRick")], logprobs=True)
    assert result.response_metadata
    assert all(
        k in result.response_metadata
        for k in (
            "model_name",
            "role",
        )
    )


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
