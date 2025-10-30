import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Generator, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
import requests_mock

from langchain_nvidia_ai_endpoints import (
    NVIDIA,
    ChatNVIDIA,
    NVIDIAEmbeddings,
    NVIDIARerank,
)
from langchain_nvidia_ai_endpoints._common import _NVIDIAClient
from langchain_nvidia_ai_endpoints._statics import MODEL_TABLE


@pytest.fixture(
    params=[
        ChatNVIDIA,
        NVIDIAEmbeddings,
        NVIDIARerank,
        NVIDIA,
    ]
)
def public_class(request: pytest.FixtureRequest) -> type:
    return request.param


@pytest.fixture
def empty_v1_models(requests_mock: requests_mock.Mocker) -> None:
    requests_mock.get("https://integrate.api.nvidia.com/v1/models", json={"data": []})


@pytest.fixture
def mock_model() -> str:
    return "mock-model"


@pytest.fixture(autouse=True)
def mock_v1_models(requests_mock: requests_mock.Mocker, mock_model: str) -> None:
    requests_mock.get(
        re.compile(".*/v1/models"),
        json={
            "data": [
                {"id": mock_model},
            ]
        },
    )


@pytest.fixture(autouse=True)
def reset_model_table() -> Generator[None, None, None]:
    """
    Reset MODEL_TABLE between tests.
    """
    original = MODEL_TABLE.copy()
    yield
    MODEL_TABLE.clear()
    MODEL_TABLE.update(original)


@pytest.fixture
def mock_streaming_response(
    requests_mock: requests_mock.Mocker, mock_model: str
) -> Callable:
    def builder(chunks: List[str]) -> None:
        requests_mock.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            text="\n\n".join(
                [
                    'data: {"id":"ID0","object":"chat.completion.chunk","created":1234567890,"model":"bogus","choices":[{"index":0,"delta":{"role":"assistant","content":null},"logprobs":null,"finish_reason":null}]}',  # noqa: E501
                    *[
                        f'data: {{"id":"ID0","object":"chat.completion.chunk","created":1234567890,"model":"bogus","choices":[{{"index":0,"delta":{{"role":null,"content":"{content}"}},"logprobs":null,"finish_reason":null}}]}}'  # noqa: E501
                        for content in chunks
                    ],
                    'data: {"id":"ID0","object":"chat.completion.chunk","created":1234567890,"model":"bogus","choices":[{"index":0,"delta":{"role":null,"content":""},"logprobs":null,"finish_reason":"stop","stop_reason":null}]}',  # noqa: E501
                    "data: [DONE]",
                ]
            ),
        )

    return builder


def _make_async_response(
    *,
    status: int = 200,
    headers: Optional[dict] = None,
    json_body: Optional[dict] = None,
    text_body: Optional[str] = None,
) -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.headers = headers or {}
    if json_body is not None:
        payload = json.dumps(json_body)
        resp.json = AsyncMock(return_value=json_body)
        resp.text = AsyncMock(return_value=payload)
        resp.read = AsyncMock(return_value=payload.encode())
    else:
        resp.json = AsyncMock(side_effect=Exception("no json"))
        resp.text = AsyncMock(return_value=text_body or "")
        resp.read = AsyncMock(return_value=(text_body or "").encode())

        if text_body and "\n\n" in text_body:
            reader = MagicMock()
            stream_lines = [
                line.encode() + b"\n" for line in text_body.split("\n\n") if line
            ]
            it = iter(stream_lines)

            async def _readline() -> bytes:
                try:
                    return next(it)
                except StopIteration:
                    return b""

            reader.readline = AsyncMock(side_effect=_readline)
            resp.content = reader
    return resp


@dataclass
class HTTPRequest:
    """Represents a captured HTTP request."""

    method: str
    url: str
    kwargs: dict


@dataclass
class MockHTTP:
    """HTTP mocking fixture for tests."""

    requests: requests_mock.Mocker
    aio: MagicMock
    history: list[HTTPRequest]

    def set_post(self, **kwargs: Any) -> None:
        self.aio.post.return_value = _make_async_response(**kwargs)

    def set_get(self, **kwargs: Any) -> None:
        self.aio.get.return_value = _make_async_response(**kwargs)


@pytest.fixture
def mock_http(
    monkeypatch: pytest.MonkeyPatch,
    requests_mock: requests_mock.Mocker,
    mock_model: str,
) -> MockHTTP:
    """Dependency-injected HTTP mocks.
    - Injects AsyncMock aiohttp.ClientSession for async POST/GET/stream
    - Provides helpers to configure responses per-test
    """
    aio_sess = MagicMock()
    aio_sess.close = AsyncMock()
    aio_sess.post = AsyncMock()
    aio_sess.get = AsyncMock()
    aio_sess.post.return_value = _make_async_response(json_body={})
    aio_sess.get.return_value = _make_async_response(json_body={})

    # Keep a custom async request history
    _history: list[HTTPRequest] = []

    async def _post_side_effect(*args: Any, **kwargs: Any) -> MagicMock:
        url = kwargs.get("url") or (args[0] if args else "")
        _history.append(HTTPRequest("POST", str(url), kwargs))
        return aio_sess.post.return_value

    async def _get_side_effect(*args: Any, **kwargs: Any) -> MagicMock:
        url = kwargs.get("url") or (args[0] if args else "")
        _history.append(HTTPRequest("GET", str(url), kwargs))
        return aio_sess.get.return_value

    aio_sess.post.side_effect = _post_side_effect
    aio_sess.get.side_effect = _get_side_effect

    monkeypatch.setattr(
        _NVIDIAClient, "_create_async_session", lambda self: aio_sess, raising=True
    )

    return MockHTTP(requests_mock, aio_sess, _history)
