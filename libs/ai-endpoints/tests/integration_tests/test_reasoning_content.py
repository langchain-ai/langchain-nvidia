from typing import Callable

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessageChunk

from langchain_nvidia_ai_endpoints import ChatNVIDIA


def do_stream(llm: ChatNVIDIA, msg: str) -> BaseMessageChunk:
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
    prompt = (
        "Solve this step by step: A train leaves station A at 2 PM traveling at "
        "60 mph. Another train leaves station B at 3 PM traveling at 80 mph "
        "towards station A. If the stations are 300 miles apart, when will "
        "they meet? Show your complete reasoning process."
    )
    resp = func(llm, prompt)
    assert isinstance(resp, (AIMessage, BaseMessageChunk))
    rc = resp.additional_kwargs.get("reasoning_content")
    assert isinstance(rc, str) and rc != ""
