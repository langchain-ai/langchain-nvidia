import inspect
from typing import Callable

import pytest
from langchain_core.messages import AIMessage

from langchain_nvidia_ai_endpoints import ChatNVIDIA


def is_async_func(func: Callable) -> bool:
    """Check if a function is an async function."""
    return inspect.iscoroutinefunction(func)


# TODO: Implement this for streaming
# def do_stream(llm: ChatNVIDIA, msg: str) -> AIMessageChunk:
#     pass


def do_invoke(llm: ChatNVIDIA, msg: str) -> AIMessage:
    return llm.invoke(msg)  # type: ignore[return-value]


async def do_ainvoke(llm: ChatNVIDIA, msg: str) -> AIMessage:
    return await llm.ainvoke(msg)  # type: ignore[return-value]


def check_reasoning_content(
    response: AIMessage, should_have_reasoning: bool = True
) -> None:
    """Check if response has reasoning content in the expected format.

    Args:
        response: The response to check
        should_have_reasoning: Whether the response should have reasoning content
    """
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)

    # Check for content_blocks with reasoning
    has_reasoning_block = (
        hasattr(response, "content_blocks")
        and response.content_blocks is not None
        and len(response.content_blocks) > 0
        and response.content_blocks[0].get("type") == "reasoning"
    )

    if should_have_reasoning:
        assert has_reasoning_block, "No reasoning content found in content_blocks"
        # Content should not contain think tags
        assert (
            "<think>" not in response.content
        ), "Content should not contain think tags"
    else:
        assert (
            not has_reasoning_block
        ), "Found reasoning content when it should not be present"


@pytest.mark.parametrize(
    "func",
    [do_invoke, do_ainvoke],
    ids=["invoke", "ainvoke"],
)
async def test_thinking_mode_enabled(
    thinking_model: str,
    mode: dict,
    func: Callable,
) -> None:
    """Test that thinking mode can be enabled and adds the correct system message."""

    llm = ChatNVIDIA(model=thinking_model, **mode).with_thinking_mode(enabled=True)

    if is_async_func(func):
        response = await func(
            llm, "Explain step by step how to solve the equation: 3x + 5 = 20"
        )
    else:
        response = func(
            llm, "Explain step by step how to solve the equation: 3x + 5 = 20"
        )

    check_reasoning_content(response, should_have_reasoning=True)


@pytest.mark.parametrize(
    "func",
    [do_invoke, do_ainvoke],
    ids=["invoke", "ainvoke"],
)
async def test_thinking_mode_disabled(
    thinking_model: str,
    mode: dict,
    func: Callable,
) -> None:
    """Test that thinking mode can be disabled."""

    llm = ChatNVIDIA(model=thinking_model, **mode).with_thinking_mode(enabled=False)

    if is_async_func(func):
        response = await func(llm, "What is the capital of France?")
    else:
        response = func(llm, "What is the capital of France?")

    assert len(response.content) > 0
    check_reasoning_content(response, should_have_reasoning=False)


@pytest.mark.parametrize(
    "func",
    [do_invoke, do_ainvoke],
    ids=["invoke", "ainvoke"],
)
async def test_thinking_mode_default(
    thinking_model: str, mode: dict, func: Callable
) -> None:
    """Test that model works without explicitly setting thinking mode."""

    llm = ChatNVIDIA(model=thinking_model, **mode)
    prompt = (
        "John is taller than Mike. Mike is taller than Sara. " "Who is the tallest?"
    )

    if is_async_func(func):
        response = await func(llm, prompt)
    else:
        response = func(llm, prompt)

    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


@pytest.mark.parametrize(
    "func",
    [do_invoke, do_ainvoke],
    ids=["invoke", "ainvoke"],
)
async def test_thinking_mode_unsupported_model(
    thinking_model: str, mode: dict, func: Callable
) -> None:
    """Test that thinking mode is handled gracefully for unsupported models."""
    unsupported_model = "meta/llama3-8b-instruct"
    llm = ChatNVIDIA(model=unsupported_model, **mode).with_thinking_mode(enabled=True)

    if is_async_func(func):
        response = await func(llm, "What is 2+2?")
    else:
        response = func(llm, "What is 2+2?")

    assert len(response.content) > 0
    check_reasoning_content(response, should_have_reasoning=False)
