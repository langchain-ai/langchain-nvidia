from typing import Callable

import pytest
from langchain_core.messages import AIMessage

from langchain_nvidia_ai_endpoints import ChatNVIDIA

# TODO: Implement this for streaming
# def do_stream(llm: ChatNVIDIA, msg: str) -> AIMessageChunk:
#     pass


def do_invoke(llm: ChatNVIDIA, msg: str) -> AIMessage:
    return llm.invoke(msg)


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
    
    has_think_tag = '<think>' in response.content
    has_metadata_reasoning = bool(
        response.response_metadata.get('reasoning_content')
    )
    
    if should_have_reasoning:
        assert has_think_tag or has_metadata_reasoning, (
            "No reasoning content found in either think tag or metadata"
        )
    else:
        assert not has_think_tag and not has_metadata_reasoning, (
            "Found reasoning content when it should not be present"
        )


@pytest.mark.parametrize("func", [do_invoke], ids=["invoke"])
def test_thinking_mode_enabled(
    thinking_model: str, mode: dict, func: Callable,
) -> None:
    """Test that thinking mode can be enabled and adds the correct system message."""
    llm = ChatNVIDIA(model=thinking_model, **mode).with_thinking_mode(enabled=True)
    response = func(
        llm,
        "Explain step by step how to solve the equation: 3x + 5 = 20"
    )
    check_reasoning_content(response, should_have_reasoning=True)


@pytest.mark.parametrize("func", [do_invoke], ids=["invoke"])
def test_thinking_mode_disabled(
    thinking_model: str, mode: dict, func: Callable,
) -> None:
    """Test that thinking mode can be disabled."""
    llm = ChatNVIDIA(model=thinking_model, **mode).with_thinking_mode(enabled=False)
    response = func(llm, "What is the capital of France?")
    assert len(response.content) > 0
    check_reasoning_content(response, should_have_reasoning=False)


@pytest.mark.parametrize("func", [do_invoke], ids=["invoke"])
def test_thinking_mode_default(thinking_model: str, mode: dict, func: Callable) -> None:
    """Test that model works without explicitly setting thinking mode."""
    llm = ChatNVIDIA(model=thinking_model, **mode)
    prompt = (
        "John is taller than Mike. Mike is taller than Sara. "
        "Who is the tallest?"
    )
    response = func(llm, prompt)
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


@pytest.mark.parametrize("func", [do_invoke], ids=["invoke"])
def test_thinking_mode_unsupported_model(
    thinking_model: str, mode: dict, func: Callable
) -> None:
    """Test that thinking mode is handled gracefully for unsupported models."""
    unsupported_model = "meta/llama3-8b-instruct"  
    llm = ChatNVIDIA(
        model=unsupported_model, **mode
    ).with_thinking_mode(enabled=True)
    response = func(llm, "What is 2+2?")
    assert len(response.content) > 0
    check_reasoning_content(response, should_have_reasoning=False)