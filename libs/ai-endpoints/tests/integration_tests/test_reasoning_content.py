from typing import Union

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
)

from langchain_nvidia_ai_endpoints import ChatNVIDIA


@pytest.mark.parametrize(
    "func",
    ["invoke", "stream", "ainvoke", "astream"],
)
async def test_reasoning_content_exposed(
    reasoning_model: str, mode: dict, func: str
) -> None:
    llm = ChatNVIDIA(model=reasoning_model, temperature=0, **mode)
    prompt = (
        "Solve this step by step: A train leaves station A at 2 PM traveling at "
        "60 mph. Another train leaves station B at 3 PM traveling at 80 mph "
        "towards station A. If the stations are 300 miles apart, when will "
        "they meet? Show your complete reasoning process."
    )

    resp: Union[BaseMessage, BaseMessageChunk]
    if func == "invoke":
        resp = llm.invoke(prompt)
    elif func == "stream":
        generator = llm.stream(prompt)
        resp = next(generator)
        for chunk in generator:
            assert isinstance(chunk, AIMessageChunk)
            resp += chunk
    elif func == "ainvoke":
        resp = await llm.ainvoke(prompt)
    else:  # astream
        async_generator = llm.astream(prompt)
        resp = await async_generator.__anext__()
        async for chunk in async_generator:
            assert isinstance(chunk, AIMessageChunk)
            resp += chunk
    assert isinstance(resp, (AIMessage, BaseMessageChunk))
    rc = resp.additional_kwargs.get("reasoning_content")
    assert isinstance(rc, str) and rc != ""
