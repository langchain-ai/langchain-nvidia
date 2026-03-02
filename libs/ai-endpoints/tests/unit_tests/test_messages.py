import warnings
from typing import Any

import pytest
import requests_mock
from langchain_core.messages import AIMessage, HumanMessage

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_ai_endpoints._utils import convert_message_to_dict

from .conftest import MockHTTP


def test_invoke_aimessage_content_none(requests_mock: requests_mock.Mocker) -> None:
    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        json={
            "id": "mock-id",
            "created": 1234567890,
            "object": "chat.completion",
            "model": "mock-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "WORKED"},
                }
            ],
        },
    )

    empty_aimessage = AIMessage(content="EMPTY")
    empty_aimessage.content = None  # type: ignore

    llm = ChatNVIDIA(api_key="BOGUS")
    response = llm.invoke([empty_aimessage])
    request = requests_mock.request_history[0]
    assert request.method == "POST"
    assert request.url == "https://integrate.api.nvidia.com/v1/chat/completions"
    message = request.json()["messages"][0]
    assert "content" in message and message["content"] != "EMPTY"
    assert "content" in message and message["content"] is None
    assert isinstance(response, AIMessage)
    assert response.content == "WORKED"


@pytest.mark.asyncio
async def test_ainvoke_aimessage_content_none(mock_http: MockHTTP) -> None:
    mock_http.set_post(
        json_body={
            "id": "mock-id",
            "created": 1234567890,
            "object": "chat.completion",
            "model": "mock-model",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "WORKED"}}
            ],
        }
    )

    empty_aimessage = AIMessage(content="EMPTY")
    empty_aimessage.content = None  # type: ignore

    llm = ChatNVIDIA(api_key="BOGUS")
    response = await llm.ainvoke([empty_aimessage])
    request = mock_http.history[0]
    assert request.method == "POST"
    assert request.url == "https://integrate.api.nvidia.com/v1/chat/completions"
    payload = request.kwargs.get("json", {})
    messages = payload.get("messages", [{}])
    message = messages[0]
    assert "content" in message and message["content"] != "EMPTY"
    assert "content" in message and message["content"] is None
    assert isinstance(response, AIMessage)
    assert response.content == "WORKED"


# --- convert_message_to_dict: reasoning field forwarding ---


def test_convert_message_to_dict_preserves_reasoning_content() -> None:
    """reasoning_content in additional_kwargs is forwarded when from API."""
    msg = AIMessage(
        content="The answer is 4",
        additional_kwargs={
            "reasoning_content": "Let me think... 2+2=4",
            "_reasoning_api_fields": ["reasoning_content"],
        },
    )
    out = convert_message_to_dict(msg)
    assert out["role"] == "assistant"
    assert out["content"] == "The answer is 4"
    assert out["reasoning_content"] == "Let me think... 2+2=4"


def test_convert_message_to_dict_preserves_reasoning() -> None:
    """reasoning in additional_kwargs is forwarded when from API."""
    msg = AIMessage(
        content="The answer is 4",
        additional_kwargs={
            "reasoning": "Let me think... 2+2=4",
            "_reasoning_api_fields": ["reasoning"],
        },
    )
    out = convert_message_to_dict(msg)
    assert out["role"] == "assistant"
    assert out["content"] == "The answer is 4"
    assert out["reasoning"] == "Let me think... 2+2=4"


def test_convert_message_to_dict_preserves_both_reasoning_fields() -> None:
    """Both reasoning and reasoning_content are forwarded when from API."""
    msg = AIMessage(
        content="The answer is 4",
        additional_kwargs={
            "reasoning": "thinking text",
            "reasoning_content": "thinking text",
            "_reasoning_api_fields": ["reasoning_content", "reasoning"],
        },
    )
    out = convert_message_to_dict(msg)
    assert out["role"] == "assistant"
    assert out["content"] == "The answer is 4"
    assert out["reasoning"] == "thinking text"
    assert out["reasoning_content"] == "thinking text"


def test_convert_message_to_dict_no_reasoning_when_absent() -> None:
    """No reasoning fields added when not in additional_kwargs."""
    msg = AIMessage(content="Hello")
    out = convert_message_to_dict(msg)
    assert "reasoning" not in out
    assert "reasoning_content" not in out


def test_convert_message_to_dict_no_reasoning_fields_when_from_tags() -> None:
    """Reasoning fields NOT forwarded when reasoning came from tags (no API flag)."""
    msg = AIMessage(
        content="<think>My reasoning</think>The answer is 4",
        additional_kwargs={
            "reasoning_content": "My reasoning",
            "reasoning": "My reasoning",
        },
    )
    out = convert_message_to_dict(msg)
    assert out["content"] == "<think>My reasoning</think>The answer is 4"
    assert "reasoning_content" not in out
    assert "reasoning" not in out


# --- End-to-end: reasoning from separate API fields (round-trip) ---


def test_reasoning_fields_roundtrip_in_multiturn(
    requests_mock: requests_mock.Mocker,
) -> None:
    """Both reasoning and reasoning_content from separate API fields survive
    round-trip in multi-turn conversation."""
    turn1_response = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "BOGUS",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The answer is 4.",
                    "reasoning": "The user asked 2+2. That equals 4.",
                    "reasoning_content": "The user asked 2+2. That equals 4.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }

    turn2_response = {
        "id": "chatcmpl-2",
        "object": "chat.completion",
        "created": 1234567891,
        "model": "BOGUS",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "That gives 12.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 30,
            "completion_tokens": 10,
            "total_tokens": 40,
        },
    }

    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        [{"json": turn1_response}, {"json": turn2_response}],
    )

    llm = ChatNVIDIA(api_key="BOGUS")

    # Turn 1
    response1 = llm.invoke([HumanMessage(content="What is 2+2?")])

    assert response1.additional_kwargs["reasoning_content"] == (
        "The user asked 2+2. That equals 4."
    )
    assert response1.additional_kwargs["reasoning"] == (
        "The user asked 2+2. That equals 4."
    )

    # Turn 2: send response1 back as history
    llm.invoke(
        [
            HumanMessage(content="What is 2+2?"),
            response1,
            HumanMessage(content="Multiply that by 3"),
        ]
    )

    # Verify the turn 2 API request includes reasoning fields
    turn2_request = requests_mock.request_history[1].json()
    assistant_msg = turn2_request["messages"][1]

    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["content"] == "The answer is 4."
    assert assistant_msg["reasoning"] == "The user asked 2+2. That equals 4."
    assert assistant_msg["reasoning_content"] == ("The user asked 2+2. That equals 4.")


def test_reasoning_content_only_roundtrip(
    requests_mock: requests_mock.Mocker,
) -> None:
    """Model returns only reasoning_content (no reasoning field)."""
    turn1_response = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "BOGUS",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "The answer is 4.",
                    "reasoning_content": "Let me think step by step.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }

    turn2_response = {
        "id": "chatcmpl-2",
        "object": "chat.completion",
        "created": 1234567891,
        "model": "BOGUS",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "That gives 12.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 30,
            "completion_tokens": 10,
            "total_tokens": 40,
        },
    }

    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        [{"json": turn1_response}, {"json": turn2_response}],
    )

    llm = ChatNVIDIA(api_key="BOGUS")
    response1 = llm.invoke([HumanMessage(content="What is 2+2?")])

    assert response1.additional_kwargs["reasoning_content"] == (
        "Let me think step by step."
    )

    # Turn 2
    llm.invoke(
        [
            HumanMessage(content="What is 2+2?"),
            response1,
            HumanMessage(content="Multiply that by 3"),
        ]
    )

    turn2_request = requests_mock.request_history[1].json()
    assistant_msg = turn2_request["messages"][1]

    assert assistant_msg["reasoning_content"] == "Let me think step by step."


def test_reasoning_api_fields_flag_not_leaked_to_payload() -> None:
    """Internal _reasoning_api_fields flag must not appear in serialized message."""
    msg = AIMessage(
        content="The answer is 4",
        additional_kwargs={
            "reasoning_content": "thinking",
            "reasoning": "thinking",
            "_reasoning_api_fields": ["reasoning_content", "reasoning"],
        },
    )
    out = convert_message_to_dict(msg)
    assert "_reasoning_api_fields" not in out


# --- End-to-end: reasoning from <think> tags (NOT sent as separate fields) ---


def test_think_tags_not_sent_as_separate_reasoning_fields(
    requests_mock: requests_mock.Mocker,
) -> None:
    """When reasoning comes from <think> tags, it stays in content and is NOT
    sent as separate reasoning/reasoning_content fields in follow-up requests."""
    turn1_response = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "BOGUS",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "<think>My reasoning</think>The answer is 4.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }

    turn2_response = {
        "id": "chatcmpl-2",
        "object": "chat.completion",
        "created": 1234567891,
        "model": "BOGUS",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "That gives 12.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 30,
            "completion_tokens": 10,
            "total_tokens": 40,
        },
    }

    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        [{"json": turn1_response}, {"json": turn2_response}],
    )

    llm = ChatNVIDIA(api_key="BOGUS")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        response1 = llm.invoke([HumanMessage(content="What is 2+2?")])

    # Reasoning is in additional_kwargs for user access
    assert response1.additional_kwargs["reasoning_content"] == "My reasoning"
    assert response1.additional_kwargs["reasoning"] == "My reasoning"
    # Tags are preserved in content
    assert "<think>" in response1.content

    # Turn 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        llm.invoke(
            [
                HumanMessage(content="What is 2+2?"),
                response1,
                HumanMessage(content="Multiply that by 3"),
            ]
        )

    turn2_request = requests_mock.request_history[1].json()
    assistant_msg = turn2_request["messages"][1]

    # Reasoning round-trips via tags in content, not as separate fields
    assert "<think>" in assistant_msg["content"]
    assert "reasoning_content" not in assistant_msg
    assert "reasoning" not in assistant_msg


# --- Multimodal content preservation ---


def test_convert_message_to_dict_preserves_image_url_content() -> None:
    """Multimodal content with image_url is preserved as list."""
    content: Any = [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
    ]
    msg = HumanMessage(content=content)
    out = convert_message_to_dict(msg)
    assert out["role"] == "user"
    assert out["content"] == content


def test_convert_message_to_dict_preserves_image_content() -> None:
    """Multimodal content with image block type is preserved as list."""
    content: Any = [
        {"type": "text", "text": "Describe this."},
        {"type": "image", "image": {"data": "base64..."}},
    ]
    msg = HumanMessage(content=content)
    out = convert_message_to_dict(msg)
    assert out["role"] == "user"
    assert out["content"] == content


def test_convert_message_to_dict_preserves_video_url_content() -> None:
    """Multimodal content with video_url is preserved as list."""
    content: Any = [
        {"type": "text", "text": "What happens in this video?"},
        {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}},
    ]
    msg = HumanMessage(content=content)
    out = convert_message_to_dict(msg)
    assert out["role"] == "user"
    assert out["content"] == content


def test_convert_message_to_dict_preserves_video_content() -> None:
    """Multimodal content with video block type is preserved as list."""
    content: Any = [
        {"type": "text", "text": "Describe this."},
        {"type": "video", "video": {"data": "base64..."}},
    ]
    msg = HumanMessage(content=content)
    out = convert_message_to_dict(msg)
    assert out["role"] == "user"
    assert out["content"] == content
