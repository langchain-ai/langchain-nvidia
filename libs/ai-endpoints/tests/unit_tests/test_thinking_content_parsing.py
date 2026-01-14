"""Tests for thinking content parsing and content_blocks generation."""

import warnings
from typing import Any

import pytest
import requests_mock
from pydantic import BaseModel

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_ai_endpoints.chat_models import parse_thinking_content


@pytest.mark.parametrize(
    "response_content,expected_content_without_tags,expected_reasoning,should_warn",
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
    expected_content_without_tags: str,
    expected_reasoning: str,
    should_warn: bool,
) -> None:
    """Test parsing and content_blocks generation for thinking content."""
    # Test 1: Parse function with remove_tags=False (backward compatible mode)
    reasoning, content_with_tags, content_without_tags = parse_thinking_content(
        response_content, remove_tags=False
    )
    assert reasoning == expected_reasoning
    # In backward compatible mode, tags are preserved in content_with_tags
    if expected_reasoning:
        # When there's reasoning, content_with_tags should contain the original
        assert content_with_tags == response_content
        # content_without_tags should match expected_content_without_tags
        assert content_without_tags == expected_content_without_tags
    else:
        # No reasoning, both should be the same as original
        assert content_with_tags == expected_content_without_tags
        assert content_without_tags == expected_content_without_tags

    # Test 2: Parse function with remove_tags=True (structured output mode)
    reasoning2, content_with_tags2, content_without_tags2 = parse_thinking_content(
        response_content, remove_tags=True
    )
    assert reasoning2 == expected_reasoning
    # When remove_tags=True, both should be without tags
    assert content_with_tags2 == expected_content_without_tags
    assert content_without_tags2 == expected_content_without_tags

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
    # In backward compatible mode (default), tags are kept in content with a warning
    if should_warn:
        with pytest.warns(
            UserWarning,
            match="Reasoning content was parsed from <think> tags in model output",
        ):
            response = llm.invoke("Test message")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            response = llm.invoke("Test message")

    # Check content field
    # In backward compatible mode, tags are preserved in content
    if expected_reasoning:
        # Content should contain the original response with tags preserved
        assert response.content == response_content
    else:
        assert response.content == expected_content_without_tags

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

        # Check for text block if there's content (should contain tags)
        text_blocks = [b for b in response.content_blocks if b.get("type") == "text"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == response_content  # type: ignore[typeddict-item]
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
        assert text_blocks[0]["text"] == expected_content_without_tags  # type: ignore[typeddict-item]


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

    # In backward compatible mode, tags are preserved in content
    assert response.content == "<think>Tag reasoning</think>Response"

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
    # In backward compatible mode, content contains tags
    assert text_blocks[0]["text"] == "<think>Tag reasoning</think>Response"  # type: ignore[typeddict-item]


@pytest.mark.parametrize(
    "mode,response_content,expected_content,expected_result",
    [
        pytest.param(
            "nvext",
            '<think>Reasoning here</think>{"answer": "42"}',
            '{"answer": "42"}',
            None,  # No parsing, just check raw response
            id="nvext-guided-json",
        ),
        pytest.param(
            "pydantic",
            '<think>Thinking...</think>{"value": 42}',
            '{"value": 42}',
            42,  # Expected parsed value
            id="with-structured-output-pydantic",
        ),
        pytest.param(
            "enum",
            "<think>Blue is the answer</think>blue",
            "blue",
            "blue",  # Expected enum value
            id="with-structured-output-enum",
        ),
    ],
)
def test_structured_output_removes_think_tags(
    requests_mock: requests_mock.Mocker,
    mode: str,
    response_content: str,
    expected_content: str,
    expected_result: Any,
) -> None:
    """Test that <think> tags are removed in structured output mode."""
    import enum

    class SimpleModel(BaseModel):
        value: int

    class Color(enum.Enum):
        RED = "red"
        BLUE = "blue"

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
                    "message": {"role": "assistant", "content": response_content},
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "system_fingerprint": None,
        },
    )

    llm = ChatNVIDIA(api_key="BOGUS")

    with pytest.warns(
        UserWarning,
        match="Reasoning content with <think> tags was detected in",
    ):
        if mode == "nvext":
            # Direct nvext usage
            llm_with_nvext = ChatNVIDIA(
                api_key="BOGUS", nvext={"guided_json": {"type": "object"}}
            )
            response = llm_with_nvext.invoke("Test")
            assert response.content == expected_content
            assert response.additional_kwargs["reasoning_content"] == "Reasoning here"
        elif mode == "pydantic":
            result = llm.with_structured_output(SimpleModel).invoke("Test")
            assert isinstance(result, SimpleModel)
            assert result.value == expected_result
        elif mode == "enum":
            result = llm.with_structured_output(Color).invoke("Test")
            assert isinstance(result, Color)
            assert result.value == expected_result
