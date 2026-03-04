"""Tests for inference_priority decorator / context manager."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from requests_mock import Mocker

from langchain_nvidia_ai_endpoints import (
    ChatNVIDIA,
    ChatNVIDIADynamo,
    get_inference_priority,
    inference_priority,
)

# ── fixtures ──────────────────────────────────────────────────────────────────

MOCK_COMPLETION = {
    "id": "ID0",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "mock-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
        }
    ],
}


@pytest.fixture(autouse=True)
def mock_v1_models(requests_mock: Mocker) -> None:
    requests_mock.get(
        "https://integrate.api.nvidia.com/v1/models",
        json={"data": [{"id": "mock-model"}]},
    )


@pytest.fixture(autouse=True)
def mock_completions(requests_mock: Mocker) -> None:
    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        json=MOCK_COMPLETION,
    )


def _make_dynamo(**kwargs: Any) -> ChatNVIDIADynamo:
    kwargs.setdefault("model", "mock-model")
    kwargs.setdefault("nvidia_api_key", "nvapi-test")
    return ChatNVIDIADynamo(**kwargs)


def _last_body(requests_mock: Mocker) -> dict:
    """Return the JSON body of the last POST request."""
    last = requests_mock.last_request
    assert last is not None
    return json.loads(last.body)


# ── basic context propagation ─────────────────────────────────────────────────


def test_decorator_sets_priority(requests_mock: Mocker) -> None:
    llm = _make_dynamo()

    @inference_priority(priority=10)
    def _run() -> None:
        llm.invoke("hello")

    _run()
    hints = _last_body(requests_mock)["nvext"]["agent_hints"]
    assert hints["priority"] == 10


def test_context_manager_sets_priority(requests_mock: Mocker) -> None:
    llm = _make_dynamo()

    with inference_priority(priority=7):
        llm.invoke("hello")

    hints = _last_body(requests_mock)["nvext"]["agent_hints"]
    assert hints["priority"] == 7


def test_no_decorator_uses_instance_default(requests_mock: Mocker) -> None:
    llm = _make_dynamo(priority=3)
    llm.invoke("hello")
    hints = _last_body(requests_mock)["nvext"]["agent_hints"]
    assert hints["priority"] == 3


# ── precedence ────────────────────────────────────────────────────────────────


def test_explicit_kwarg_beats_decorator(requests_mock: Mocker) -> None:
    llm = _make_dynamo()

    @inference_priority(priority=10)
    def _run() -> None:
        llm.invoke("hello", priority=99)

    _run()
    hints = _last_body(requests_mock)["nvext"]["agent_hints"]
    assert hints["priority"] == 99


def test_decorator_beats_instance_default(requests_mock: Mocker) -> None:
    llm = _make_dynamo(priority=1)

    @inference_priority(priority=10)
    def _run() -> None:
        llm.invoke("hello")

    _run()
    hints = _last_body(requests_mock)["nvext"]["agent_hints"]
    assert hints["priority"] == 10


# ── nesting ───────────────────────────────────────────────────────────────────


def test_inner_decorator_replaces_outer(requests_mock: Mocker) -> None:
    llm = _make_dynamo()

    @inference_priority(priority=10)
    def _outer() -> None:
        @inference_priority(priority=1)
        def _inner() -> None:
            llm.invoke("hello")

        _inner()

    _outer()
    hints = _last_body(requests_mock)["nvext"]["agent_hints"]
    assert hints["priority"] == 1


def test_outer_restored_after_inner_exits(requests_mock: Mocker) -> None:
    llm = _make_dynamo()
    priorities: list[int] = []

    @inference_priority(priority=10)
    def _outer() -> None:
        llm.invoke("first")
        priorities.append(_last_body(requests_mock)["nvext"]["agent_hints"]["priority"])

        with inference_priority(priority=1):
            llm.invoke("second")
            priorities.append(
                _last_body(requests_mock)["nvext"]["agent_hints"]["priority"]
            )

        llm.invoke("third")
        priorities.append(_last_body(requests_mock)["nvext"]["agent_hints"]["priority"])

    _outer()
    assert priorities == [10, 1, 10]


def test_context_cleared_after_exit() -> None:
    with inference_priority(priority=5):
        pass

    assert get_inference_priority() is None


# ── async ─────────────────────────────────────────────────────────────────────


async def test_async_decorator(requests_mock: Mocker) -> None:
    """Async decorator sets ContextVar — sync invoke verifies the payload."""
    llm = _make_dynamo()

    @inference_priority(priority=8)
    async def _run() -> None:
        llm.invoke("hello")  # sync invoke; requests_mock handles it

    await _run()
    hints = _last_body(requests_mock)["nvext"]["agent_hints"]
    assert hints["priority"] == 8


async def test_concurrent_async_tasks_isolated() -> None:
    """ContextVar is per-task — concurrent coroutines don't interfere."""

    @inference_priority(priority=10)
    async def _high() -> int:
        await asyncio.sleep(0)
        assert get_inference_priority() == 10
        return get_inference_priority()  # type: ignore[return-value]

    @inference_priority(priority=1)
    async def _low() -> int:
        await asyncio.sleep(0)
        assert get_inference_priority() == 1
        return get_inference_priority()  # type: ignore[return-value]

    high, low = await asyncio.gather(_high(), _low())
    assert high == 10
    assert low == 1


# ── non-Dynamo LLM unaffected ────────────────────────────────────────────────


def test_plain_chat_nvidia_unaffected(requests_mock: Mocker) -> None:
    """ChatNVIDIA has no ``priority`` field — context should not inject."""
    llm = ChatNVIDIA(model="mock-model", nvidia_api_key="nvapi-test")

    with inference_priority(priority=10):
        result = llm.invoke("hello")

    assert result.content == "hi"
    body = _last_body(requests_mock)
    assert "priority" not in body


# ── streaming ─────────────────────────────────────────────────────────────────


def test_streaming_picks_up_context(requests_mock: Mocker) -> None:
    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        text=(
            'data: {"id":"ID0","object":"chat.completion.chunk","created":1234567890,'
            '"model":"mock-model","choices":[{"index":0,"delta":{"role":"assistant",'
            '"content":"hi"},"logprobs":null,"finish_reason":null}]}\n\n'
            'data: {"id":"ID0","object":"chat.completion.chunk","created":1234567890,'
            '"model":"mock-model","choices":[{"index":0,"delta":{"role":null,'
            '"content":""},"logprobs":null,"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        ),
    )
    llm = _make_dynamo()

    with inference_priority(priority=5):
        list(llm.stream("hello"))

    body = _last_body(requests_mock)
    hints = body["nvext"]["agent_hints"]
    assert hints["priority"] == 5


# ── validation ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "value",
    [-1, "high", 3.5, None],
)
def test_invalid_priority_raises(value: Any) -> None:
    with pytest.raises((ValueError, TypeError)):
        inference_priority(priority=value)


def test_valid_priority_zero() -> None:
    """Zero is a valid non-negative int."""
    with inference_priority(priority=0) as p:
        assert p == 0


# ── public accessor ───────────────────────────────────────────────────────────


def test_get_inference_priority_unset() -> None:
    assert get_inference_priority() is None


def test_get_inference_priority_in_context() -> None:
    with inference_priority(priority=42):
        assert get_inference_priority() == 42
    assert get_inference_priority() is None


# ── other Dynamo fields unaffected ────────────────────────────────────────────


def test_osl_iat_latency_untouched_by_decorator(requests_mock: Mocker) -> None:
    """The decorator only sets priority; osl/iat/latency_sensitivity come
    from the instance or per-invocation kwargs."""
    llm = _make_dynamo(osl=256, iat=100, latency_sensitivity=0.5)

    @inference_priority(priority=10)
    def _run() -> None:
        llm.invoke("hello")

    _run()
    hints = _last_body(requests_mock)["nvext"]["agent_hints"]
    assert hints["priority"] == 10
    assert hints["osl"] == 256
    assert hints["iat"] == 100
    assert hints["latency_sensitivity"] == 0.5
