import warnings

import pytest
import requests_mock
from langchain_core.messages import AIMessage

from langchain_nvidia_ai_endpoints import ChatNVIDIA


@pytest.fixture(autouse=True)
def mock_v1_models(requests_mock: requests_mock.Mocker) -> None:
    requests_mock.get(
        "https://integrate.api.nvidia.com/v1/models",
        json={
            "data": [
                {
                    "id": "magic-model",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "OWNER",
                    "root": "magic-model",
                },
            ]
        },
    )


def test_invoke_parallel_tool_calls(requests_mock: requests_mock.Mocker) -> None:
    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        json={
            "id": "cmpl-100f0463deb8421480ab18ed32cb2581",
            "object": "chat.completion",
            "created": 1721154188,
            "model": "magic-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "chatcmpl-tool-7980a682cc24446a8da9148c2c3e37ce",
                                "type": "function",
                                "function": {
                                    "name": "xxyyzz",
                                    "arguments": '{"a": 11, "b": 3}',
                                },
                            },
                            {
                                "id": "chatcmpl-tool-299964d0c5fe4fc1b917c8eaabd1cda2",
                                "type": "function",
                                "function": {
                                    "name": "zzyyxx",
                                    "arguments": '{"a": 11, "b": 5}',
                                },
                            },
                        ],
                    },
                    "logprobs": None,
                    "finish_reason": "tool_calls",
                    "stop_reason": None,
                }
            ],
            "usage": {
                "prompt_tokens": 194,
                "total_tokens": 259,
                "completion_tokens": 65,
            },
        },
    )

    warnings.filterwarnings("ignore", r".*Found magic-model in available_models.*")
    llm = ChatNVIDIA(model="magic-model")
    response = llm.invoke(
        "What is 11 xxyyzz 3 zzyyxx 5?",
    )
    assert isinstance(response, AIMessage)
    assert len(response.tool_calls) == 2
    tool_call0 = response.tool_calls[0]
    assert tool_call0["name"] == "xxyyzz"
    assert tool_call0["args"] == {"b": 3, "a": 11}
    tool_call1 = response.tool_calls[1]
    assert tool_call1["name"] == "zzyyxx"
    assert tool_call1["args"] == {"b": 5, "a": 11}


def test_stream_parallel_tool_calls_A(requests_mock: requests_mock.Mocker) -> None:
    response_contents = "\n\n".join(
        [
            'data: {"id":"chatcmpl-ID0","object":"chat.completion.chunk","created":1721155403,"model":"magic-model","system_fingerprint":null,"choices":[{"index":0,"delta":{"role":"assistant","content":null},"logprobs":null,"finish_reason":null}]}',  # noqa: E501
            'data: {"id":"chatcmpl-ID0","object":"chat.completion.chunk","created":1721155403,"model":"magic-model","system_fingerprint":null,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_ID0","type":"function","function":{"name":"xxyyzz","arguments":""}}]},"logprobs":null,"finish_reason":null}]}',  # noqa: E501
            'data: {"id":"chatcmpl-ID0","object":"chat.completion.chunk","created":1721155403,"model":"magic-model","system_fingerprint":null,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"a\\""}}]},"logprobs":null,"finish_reason":null}]}',  # noqa: E501
            'data: {"id":"chatcmpl-ID0","object":"chat.completion.chunk","created":1721155403,"model":"magic-model","system_fingerprint":null,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":": 11,"}}]},"logprobs":null,"finish_reason":null}]}',  # noqa: E501
            'data: {"id":"chatcmpl-ID0","object":"chat.completion.chunk","created":1721155403,"model":"magic-model","system_fingerprint":null,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":" \\"b\\": "}}]},"logprobs":null,"finish_reason":null}]}',  # noqa: E501
            'data: {"id":"chatcmpl-ID0","object":"chat.completion.chunk","created":1721155403,"model":"magic-model","system_fingerprint":null,"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"3}"}}]},"logprobs":null,"finish_reason":null}]}',  # noqa: E501
            'data: {"id":"chatcmpl-ID0","object":"chat.completion.chunk","created":1721155403,"model":"magic-model","system_fingerprint":null,"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_ID1","type":"function","function":{"name":"zzyyxx","arguments":""}}]},"logprobs":null,"finish_reason":null}]}',  # noqa: E501
            'data: {"id":"chatcmpl-ID0","object":"chat.completion.chunk","created":1721155403,"model":"magic-model","system_fingerprint":null,"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\\"a\\""}}]},"logprobs":null,"finish_reason":null}]}',  # noqa: E501
            'data: {"id":"chatcmpl-ID0","object":"chat.completion.chunk","created":1721155403,"model":"magic-model","system_fingerprint":null,"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":": 5, "}}]},"logprobs":null,"finish_reason":null}]}',  # noqa: E501
            'data: {"id":"chatcmpl-ID0","object":"chat.completion.chunk","created":1721155403,"model":"magic-model","system_fingerprint":null,"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\\"b\\": 3"}}]},"logprobs":null,"finish_reason":null}]}',  # noqa: E501
            'data: {"id":"chatcmpl-ID0","object":"chat.completion.chunk","created":1721155403,"model":"magic-model","system_fingerprint":null,"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"}"}}]},"logprobs":null,"finish_reason":null}]}',  # noqa: E501
            'data: {"id":"chatcmpl-ID0","object":"chat.completion.chunk","created":1721155403,"model":"magic-model","system_fingerprint":null,"choices":[{"index":0,"delta":{},"logprobs":null,"finish_reason":"tool_calls"}]}',  # noqa: E501
            "data: [DONE]",
        ]
    )

    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        text=response_contents,
    )

    warnings.filterwarnings("ignore", r".*Found magic-model in available_models.*")
    llm = ChatNVIDIA(model="magic-model")
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


def test_stream_parallel_tool_calls_B(requests_mock: requests_mock.Mocker) -> None:
    response_contents = "\n\n".join(
        [
            'data: {"id":"cmpl-call_ID0","object":"chat.completion.chunk","created":1721155320,"model":"magic-model","choices":[{"index":0,"delta":{"role":"assistant","content":null},"logprobs":null,"finish_reason":null}]}',  # noqa: E501
            'data: {"id":"cmpl-call_ID0","object":"chat.completion.chunk","created":1721155320,"model":"magic-model","choices":[{"index":0,"delta":{"role":null,"content":null,"tool_calls":[{"index":0,"id":"chatcmpl-tool-IDA","type":"function","function":{"name":"xxyyzz","arguments":"{\\"a\\": 11, \\"b\\": 3}"}},{"index":1,"id":"chatcmpl-tool-IDB","type":"function","function":{"name":"zzyyxx","arguments":"{\\"a\\": 11, \\"b\\": 5}"}}]},"logprobs":null,"finish_reason":"tool_calls","stop_reason":null}]}',  # noqa: E501
            "data: [DONE]",
        ]
    )

    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        text=response_contents,
    )

    warnings.filterwarnings("ignore", r".*Found magic-model in available_models.*")
    llm = ChatNVIDIA(model="magic-model")
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
    assert tool_call1["args"] == {"b": 5, "a": 11}
