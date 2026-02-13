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
