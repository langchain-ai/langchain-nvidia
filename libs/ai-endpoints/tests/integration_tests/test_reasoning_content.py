from typing import Callable

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from langchain_nvidia_ai_endpoints import ChatNVIDIA


def do_stream(llm: ChatNVIDIA, msg: str) -> AIMessageChunk:
    generator = llm.stream(msg)
    response = next(generator)
    for chunk in generator:
        assert isinstance(chunk, AIMessageChunk)
        response += chunk
    return response


def do_invoke(llm: ChatNVIDIA, msg: str) -> AIMessage:
    return llm.invoke(msg)  # type: ignore[return-value]


@pytest.mark.parametrize(
    "func",
    [do_invoke, do_stream],
    ids=["invoke", "stream"],
)
def test_reasoning_content_exposed(
    reasoning_model: str, mode: dict, func: Callable
) -> None:
    llm = ChatNVIDIA(model=reasoning_model, temperature=0, **mode)
    resp = func(llm, "Say hello")
    assert isinstance(resp, (AIMessage, AIMessageChunk))
    rc = resp.additional_kwargs.get("reasoning_content")
    assert isinstance(rc, str) and rc != ""
