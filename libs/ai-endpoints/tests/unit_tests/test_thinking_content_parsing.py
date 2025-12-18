"""Tests for thinking content parsing and content_blocks generation."""

import warnings

import pytest
import requests_mock

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_ai_endpoints.chat_models import parse_thinking_content


@pytest.mark.parametrize(
    "response_content,expected_content,expected_reasoning,should_warn",
    [
        # Paired <think></think> tags
        (
            "<think>This is my reasoning.</think>This is the response.",
            "This is the response.",
            "This is my reasoning.",
            True,
        ),
        # Single </think> tag
        (
            "This is reasoning content</think>This is the actual response.",
            "This is the actual response.",
            "This is reasoning content",
            True,
        ),
        # No thinking tags
        (
            "Just a regular response",
            "Just a regular response",
            "",
            False,
        ),
        # Only thinking content, no response
        (
            "<think>Only reasoning here</think>",
            "",
            "Only reasoning here",
            True,
        ),
        # Whitespace handling with paired tags
        (
            "<think>\nReasoning with whitespace\n</think>\n\nResponse text\n",
            "Response text",
            "Reasoning with whitespace",
            True,
        ),
        # Whitespace handling with single tag
        (
            "\nReasoning\n</think>\n\n  Response  \n",
            "Response",
            "Reasoning",
            True,
        ),
        # Complex reasoning with newlines
        (
            (
                "<think>Step 1: analyze\nStep 2: process\n"
                "Step 3: conclude</think>Final answer"
            ),
            "Final answer",
            "Step 1: analyze\nStep 2: process\nStep 3: conclude",
            True,
        ),
    ],
    ids=[
        "paired-tags",
        "single-tag",
        "no-tags",
        "only-reasoning",
        "whitespace-paired",
        "whitespace-single",
        "complex-reasoning",
    ],
)
def test_thinking_content_parsing_and_blocks(
    requests_mock: requests_mock.Mocker,
    response_content: str,
    expected_content: str,
    expected_reasoning: str,
    should_warn: bool,
) -> None:
    """Test parsing and content_blocks generation for thinking content."""
    # Test 1: Parse function directly
    reasoning, content = parse_thinking_content(response_content)
    assert reasoning == expected_reasoning
    assert content == expected_content

    # Test 2: End-to-end with API call
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
                        "content": response_content,
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
            "system_fingerprint": None,
        },
    )

    llm = ChatNVIDIA(api_key="BOGUS")

    # Check for warning if reasoning is parsed from tags
    if should_warn:
        with pytest.warns(
            UserWarning,
            match="Reasoning content was parsed from model output",
        ):
            response = llm.invoke("Test message")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            response = llm.invoke("Test message")

    # Check content field
    assert response.content == expected_content

    # Check content_blocks and reasoning_content
    if expected_reasoning:
        assert response.additional_kwargs["reasoning_content"] == expected_reasoning
        assert hasattr(response, "content_blocks")
        assert response.content_blocks is not None

        # Check for reasoning block
        reasoning_blocks = [
            b for b in response.content_blocks if b.get("type") == "reasoning"
        ]
        assert len(reasoning_blocks) == 1
        assert reasoning_blocks[0]["reasoning"] == expected_reasoning  # type: ignore[typeddict-item]

        # Check for text block if there's content
        if expected_content:
            text_blocks = [
                b for b in response.content_blocks if b.get("type") == "text"
            ]
            assert len(text_blocks) == 1
            assert text_blocks[0]["text"] == expected_content  # type: ignore[typeddict-item]
    else:
        assert "reasoning_content" not in response.additional_kwargs

        # LangChain should auto-generate a single text block from string content
        assert hasattr(response, "content_blocks")

        # Should have no reasoning blocks
        reasoning_blocks = [
            b for b in response.content_blocks if b.get("type") == "reasoning"
        ]
        assert len(reasoning_blocks) == 0

        # Should have a text block
        text_blocks = [b for b in response.content_blocks if b.get("type") == "text"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == expected_content  # type: ignore[typeddict-item]


def test_content_blocks_with_reasoning_content_from_response(
    requests_mock: requests_mock.Mocker,
) -> None:
    """Test that reasoning_content from response is handled without warning."""
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
                        "content": "Response text",
                        "reasoning_content": "Reasoning from response",
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
            "system_fingerprint": None,
        },
    )

    llm = ChatNVIDIA(api_key="BOGUS")

    # Should not raise any warnings because reasoning comes from response field
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        response = llm.invoke("Test message")

    # Check content and additional_kwargs
    assert response.content == "Response text"
    assert response.additional_kwargs["reasoning_content"] == "Reasoning from response"

    # Check content_blocks
    assert hasattr(response, "content_blocks")
    assert response.content_blocks is not None

    # Check for reasoning block
    reasoning_blocks = [
        b for b in response.content_blocks if b.get("type") == "reasoning"
    ]
    assert len(reasoning_blocks) == 1
    assert reasoning_blocks[0]["reasoning"] == "Reasoning from response"  # type: ignore[typeddict-item]

    # Check for text block
    text_blocks = [b for b in response.content_blocks if b.get("type") == "text"]
    assert len(text_blocks) == 1
    assert text_blocks[0]["text"] == "Response text"  # type: ignore[typeddict-item]


def test_content_blocks_priority_response_over_tags(
    requests_mock: requests_mock.Mocker,
) -> None:
    """Test reasoning_content from response takes priority over tags without warning."""
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
                        "content": "<think>Tag reasoning</think>Response",
                        "reasoning_content": "Reasoning from response",
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
            "system_fingerprint": None,
        },
    )

    llm = ChatNVIDIA(api_key="BOGUS")

    # Should not warn because reasoning_content field takes priority
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        response = llm.invoke("Test message")

    # Content should be parsed from tags
    assert response.content == "Response"

    # Reasoning from response should take priority over tag-based reasoning
    assert response.additional_kwargs["reasoning_content"] == "Reasoning from response"

    # Check content_blocks
    reasoning_blocks = [
        b for b in response.content_blocks if b.get("type") == "reasoning"
    ]
    assert len(reasoning_blocks) == 1
    assert reasoning_blocks[0]["reasoning"] == "Reasoning from response"  # type: ignore[typeddict-item]

    text_blocks = [b for b in response.content_blocks if b.get("type") == "text"]
    assert len(text_blocks) == 1
    assert text_blocks[0]["text"] == "Response"  # type: ignore[typeddict-item]
