"""Test ChatNVIDIA chat model."""

import warnings

import pytest
from langchain_core.messages import BaseMessage, HumanMessage

from langchain_nvidia_ai_endpoints.chat_models import ChatNVIDIA


def test_chat_ai_endpoints_context_message(qa_model: str, mode: dict) -> None:
    """Test wrapper with context message."""
    chat = ChatNVIDIA(model=qa_model, max_tokens=36).mode(**mode)
    context_message = BaseMessage(
        content="Once upon a time there was a little langchainer", type="context"
    )
    human_message = HumanMessage(content="What was there once upon a time?")
    response = chat.invoke([context_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_image_in_models(image_in_model: str, mode: dict) -> None:
    try:
        chat = ChatNVIDIA(model=image_in_model).mode(**mode)
        response = chat.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Describe this image:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "tests/data/nvidia-picasso.jpg"},
                        },
                    ]
                )
            ]
        )
        assert isinstance(response, BaseMessage)
        assert isinstance(response.content, str)
    except TimeoutError as e:
        message = f"TimeoutError: {image_in_model} {e}"
        warnings.warn(message)
        pytest.skip(message)
