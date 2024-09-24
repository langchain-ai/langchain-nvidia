import re
from typing import Callable, List

import pytest
import requests_mock

from langchain_nvidia_ai_endpoints import (
    NVIDIA,
    ChatNVIDIA,
    NVIDIAEmbeddings,
    NVIDIARerank,
)


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
