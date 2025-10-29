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


@pytest.mark.parametrize(
    "error_chunks,expected_error_type,expected_error_msg",
    [
        pytest.param(
            [
                'data: {"id":"id1","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}',  # noqa: E501
                'data: {"error": {"object": "error", "message": "Some Error Occurred", "type": "BadRequestError"}}',  # noqa: E501
                "data: [DONE]",
            ],
            "BadRequestError",
            "Some Error Occurred",
            id="error_after_content",
        ),
        pytest.param(
            [
                'data: {"error": {"object": "error", "message": "Some Error Occurred", "type": "InternalError"}}',  # noqa: E501
                "data: [DONE]",
            ],
            "InternalError",
            "Some Error Occurred",
            id="error_in_first_chunk",
        ),
    ],
)
def test_stream_error_handling(
    requests_mock: requests_mock.Mocker,
    error_chunks: list,
    expected_error_type: str,
    expected_error_msg: str,
) -> None:
    """Test that streaming properly raises errors."""
    response_text = "\n\n".join(error_chunks)
    requests_mock.post(
        "http://localhost:8888/v1/chat/completions",
        text=response_text,
    )

    llm = ChatNVIDIA(base_url="http://localhost:8888/v1")

    with pytest.raises(Exception) as exc_info:
        list(llm.stream("Test message"))

    error_msg = str(exc_info.value)
    assert error_msg == f"{expected_error_type}: {expected_error_msg}"


@pytest.mark.parametrize(
    "response_chunks,should_succeed,description",
    [
        pytest.param(
            [
                'data: {"id":"id1","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}',  # noqa: E501
                'data: {"id":"id1","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"content":"An error "},"finish_reason":null}]}',  # noqa: E501
                'data: {"id":"id1","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"content":"occurred in "},"finish_reason":null}]}',  # noqa: E501
                'data: {"id":"id1","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"content":"the system."},"finish_reason":null}]}',  # noqa: E501
                'data: {"id":"id1","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"content":""},"finish_reason":"stop"}]}',  # noqa: E501
                "data: [DONE]",
            ],
            True,
            "'error' field not in message",
            id="error_field_not_in_message",
        ),
        pytest.param(
            [
                'data: {"id":"id2","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}],"error":"something"}',  # noqa: E501
                "data: [DONE]",
            ],
            True,
            "'error' key present but 'choices' also present",
            id="has_error_key_and_choices",
        ),
        pytest.param(
            [
                'data: {"error": {"object": "not_error", "message": "test"}}',  # noqa: E501
                "data: [DONE]",
            ],
            True,
            "'error' key without 'choices' but object != 'error'",
            id="error_key_no_choices_wrong_object",
        ),
        pytest.param(
            [
                'data: {"error": "not a dict"}',  # noqa: E501
                "data: [DONE]",
            ],
            True,
            "'error' value is not a dict",
            id="error_not_dict",
        ),
        pytest.param(
            [
                'data: {"error": {"message": "test", "type": "SomeError"}}',  # noqa: E501
                "data: [DONE]",
            ],
            True,
            "'error' dict without 'object' field",
            id="error_missing_object_field",
        ),
    ],
)
def test_stream_non_error_conditions(
    requests_mock: requests_mock.Mocker,
    response_chunks: list,
    should_succeed: bool,
    description: str,
) -> None:
    """Test conditions that should not trigger error raising."""
    response_text = "\n\n".join(response_chunks)
    requests_mock.post(
        "http://localhost:8888/v1/chat/completions",
        text=response_text,
    )

    llm = ChatNVIDIA(base_url="http://localhost:8888/v1")

    # Should not raise an exception
    chunks = list(llm.stream("Test message"))
    assert len(chunks) > 0
