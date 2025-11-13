import inspect
import json
from functools import reduce
from operator import add
from typing import Any, Callable, List

import pytest
import requests_mock

from langchain_nvidia_ai_endpoints import NVIDIA

from .conftest import MockHTTP


def invoke(llm: NVIDIA, prompt: str, **kwargs: Any) -> str:
    return llm.invoke(prompt, **kwargs)


def stream(llm: NVIDIA, prompt: str, **kwargs: Any) -> str:
    return reduce(add, llm.stream(prompt, **kwargs))


async def ainvoke(llm: NVIDIA, prompt: str, **kwargs: Any) -> str:
    return await llm.ainvoke(prompt, **kwargs)


async def astream(llm: NVIDIA, prompt: str, **kwargs: Any) -> str:
    chunks = []
    async for chunk in llm.astream(prompt, **kwargs):
        chunks.append(chunk)
    return reduce(add, chunks)


mock_response = {
    "id": "ID",
    "object": "text_completion",
    "created": 1234567890,
    "model": "BOGUS",
    "choices": [
        {
            "index": 0,
            "text": "COMPLETION",
        }
    ],
    "usage": {"prompt_tokens": 7, "total_tokens": 207, "completion_tokens": 200},
}


@pytest.fixture(scope="function")
def mock_v1_completions_invoke(
    requests_mock: requests_mock.Mocker,
) -> requests_mock.Mocker:
    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/completions",
        json=mock_response,
    )
    return requests_mock


@pytest.fixture(scope="function")
def mock_v1_completions_stream(
    requests_mock: requests_mock.Mocker,
) -> requests_mock.Mocker:
    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/completions",
        text="\n\n".join(
            [
                f"data: {json.dumps(mock_response)}",
                "data: [DONE]",
            ]
        ),
    )
    return requests_mock


@pytest.fixture(scope="function")
def mock_v1_completions_ainvoke(
    mock_http: MockHTTP,
) -> MockHTTP:
    mock_http.set_post(json_body=mock_response)
    return mock_http


@pytest.fixture(scope="function")
def mock_v1_completions_astream(
    mock_http: MockHTTP,
) -> MockHTTP:
    mock_http.set_post(
        text_body="\n\n".join(
            [
                f"data: {json.dumps(mock_response)}",
                "data: [DONE]",
            ]
        ),
    )
    return mock_http


def is_async_func(func: Callable) -> bool:
    """Check if function is async"""
    return inspect.iscoroutinefunction(func)


def is_stream_func(func: Callable) -> bool:
    """Check if function is stream (async or sync)"""
    return func in [stream, astream]


def reset_async_mock(mock: MockHTTP, is_stream: bool) -> None:
    """Reset async mock for the next request."""
    if is_stream:
        mock.set_post(
            text_body="\n\n".join(
                [
                    f"data: {json.dumps(mock_response)}",
                    "data: [DONE]",
                ]
            ),
        )
    else:
        mock.set_post(json_body=mock_response)


@pytest.mark.parametrize(
    "param, value",
    [
        ("frequency_penalty", [0.25, 0.5, 0.75]),
        ("max_tokens", [2, 32, 512]),
        ("presence_penalty", [0.25, 0.5, 0.75]),
        ("seed", [1, 1234, 4321]),
        ("stop", ["Hello", "There", "World"]),
        ("temperature", [0, 0.5, 1]),
        ("top_p", [0, 0.5, 1]),
        ("best_of", [1, 5, 10]),
        ("echo", [True, False, True]),
        ("logit_bias", [{"hello": 1.0}, {"there": 1.0}, {"world": 1.0}]),
        ("logprobs", [1, 2, 3]),
        ("n", [1, 2, 3]),
        ("suffix", ["Hello", "There", "World"]),
        ("user", ["Bob", "Alice", "Eve"]),
    ],
)
@pytest.mark.parametrize(
    "func, mock_name",
    [
        (invoke, "mock_v1_completions_invoke"),
        (stream, "mock_v1_completions_stream"),
        (ainvoke, "mock_v1_completions_ainvoke"),
        (astream, "mock_v1_completions_astream"),
    ],
    ids=["invoke", "stream", "ainvoke", "astream"],
)
@pytest.mark.asyncio
async def test_params(
    param: str,
    value: List[Any],
    func: Callable,
    mock_name: str,
    request: pytest.FixtureRequest,
) -> None:
    """
    This tests the following...
     - priority order (init -> bind -> infer)
     - param passed to init, bind, invoke / stream / ainvoke / astream
    ...for each known Completion API param.
    """

    mock = request.getfixturevalue(mock_name)

    init, bind, infer = value
    is_async = is_async_func(func)
    is_stream = is_stream_func(func)

    llm = NVIDIA(api_key="BOGUS", **{param: init})
    if is_async:
        await func(llm, "IGNORED")
        request_payload = mock.aio.post.call_args.kwargs.get("json", {})
    else:
        func(llm, "IGNORED")
        request_payload = mock.last_request.json()
    assert param in request_payload
    assert request_payload[param] == init

    if is_async:
        reset_async_mock(mock, is_stream)

    bound_llm = llm.bind(**{param: bind})
    if is_async:
        await func(bound_llm, "IGNORED")
        request_payload = mock.aio.post.call_args.kwargs.get("json", {})
    else:
        func(bound_llm, "IGNORED")
        request_payload = mock.last_request.json()
    assert param in request_payload
    assert request_payload[param] == bind

    if is_async:
        reset_async_mock(mock, is_stream)

    if is_async:
        await func(bound_llm, "IGNORED", **{param: infer})
        request_payload = mock.aio.post.call_args.kwargs.get("json", {})
    else:
        func(bound_llm, "IGNORED", **{param: infer})
        request_payload = mock.last_request.json()
    assert param in request_payload
    assert request_payload[param] == infer


@pytest.mark.parametrize(
    "func, mock_name",
    [
        (invoke, "mock_v1_completions_invoke"),
        (stream, "mock_v1_completions_stream"),
        (ainvoke, "mock_v1_completions_ainvoke"),
        (astream, "mock_v1_completions_astream"),
    ],
    ids=["invoke", "stream", "ainvoke", "astream"],
)
@pytest.mark.asyncio
async def test_params_unknown(
    func: Callable,
    mock_name: str,
    request: pytest.FixtureRequest,
) -> None:
    mock = request.getfixturevalue(mock_name)
    is_async = is_async_func(func)
    is_stream = is_stream_func(func)

    with pytest.warns(UserWarning) as record:
        llm = NVIDIA(api_key="BOGUS", init_unknown="INIT")
    assert len(record) == 1
    assert "Unrecognized, ignored arguments: {'init_unknown'}" in str(record[0].message)

    with pytest.warns(UserWarning) as record:
        if is_async:
            await func(llm, "IGNORED", arg_unknown="ARG")
        else:
            func(llm, "IGNORED", arg_unknown="ARG")
    assert len(record) == 1
    assert "Unrecognized, ignored arguments: {'arg_unknown'}" in str(record[0].message)

    bound_llm = llm.bind(bind_unknown="BIND")

    if is_async:
        reset_async_mock(mock, is_stream)

    with pytest.warns(UserWarning) as record:
        if is_async:
            await func(bound_llm, "IGNORED")
        else:
            func(bound_llm, "IGNORED")
    assert len(record) == 1
    assert "Unrecognized, ignored arguments: {'bind_unknown'}" in str(record[0].message)


def test_identifying_params() -> None:
    llm = NVIDIA(api_key="BOGUS")
    assert set(llm._identifying_params.keys()) == {"model", "base_url"}
