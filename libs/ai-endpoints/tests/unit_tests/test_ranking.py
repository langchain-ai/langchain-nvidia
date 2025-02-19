import warnings
from typing import Any, Literal, Optional

import httpx
import pytest
from langchain_core.documents import Document
from pytest_httpx import HTTPXMock
from requests_mock import Mocker

from langchain_nvidia_ai_endpoints import NVIDIARerank


@pytest.fixture
def mock_v1_ranking(requests_mock: Mocker) -> None:
    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/ranking",
        json={
            "rankings": [
                {"index": 0, "logit": 4.2},
            ]
        },
    )


@pytest.mark.parametrize(
    "truncate",
    [
        None,
        "END",
        "NONE",
    ],
)
def test_truncate(
    requests_mock: Mocker,
    truncate: Optional[Literal["END", "NONE"]],
    mock_v1_ranking: None,
) -> None:
    truncate_param = {}
    if truncate:
        truncate_param = {"truncate": truncate}
    warnings.filterwarnings(
        "ignore", ".*Found mock-model in available_models.*"
    )  # expect to see this warning
    client = NVIDIARerank(api_key="BOGUS", model="mock-model", **truncate_param)
    response = client.compress_documents(
        documents=[Document(page_content="Nothing really.")], query="What is it?"
    )

    assert len(response) == 1

    assert requests_mock.last_request is not None
    request_payload = requests_mock.last_request.json()
    if truncate is None:
        assert "truncate" not in request_payload
    else:
        assert "truncate" in request_payload
        assert request_payload["truncate"] == truncate


@pytest.mark.parametrize("truncate", [True, False, 1, 0, 1.0, "START", "BOGUS"])
def test_truncate_invalid(truncate: Any, mock_v1_ranking: None) -> None:
    with pytest.raises(ValueError):
        NVIDIARerank(truncate=truncate)


class CustomTestClient(httpx.Client):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.custom_headers = {"X-Custom-Header": "test-value"}

    def request(self, *args, **kwargs) -> httpx.Response:
        # Add custom headers to each request
        if "headers" in kwargs:
            kwargs["headers"].update(self.custom_headers)
        else:
            kwargs["headers"] = self.custom_headers
        return super().request(*args, **kwargs)


def test_rerank_with_custom_client(httpx_mock: HTTPXMock, mock_model: str) -> None:
    """
    Test that both available models (GET request) and compress_documents (POST request)
    """
    httpx_mock.add_response(
        method="GET",
        url="https://integrate.api.nvidia.com/v1/models",
        json={
            "data": [
                {
                    "id": mock_model,
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "OWNER",
                    "type": "reranking",
                }
            ]
        },
        headers={"Authorization": "Bearer BOGUS", "X-Custom-Header": "test-value"},
    )

    custom_client = CustomTestClient()
    client = NVIDIARerank(api_key="BOGUS", model=mock_model, http_client=custom_client)

    _ = client.available_models

    calls = httpx_mock.get_requests()
    models_requests = [req for req in calls if "/v1/models" in str(req.url)]
    assert len(models_requests) > 0, "No requests made to /models endpoint"
    for req in models_requests:
        assert req.headers.get("authorization") == "Bearer BOGUS"
        assert req.headers.get("x-custom-header") == "test-value"

    httpx_mock.add_response(
        method="POST",
        url="https://integrate.api.nvidia.com/v1/ranking",
        json={"rankings": [{"index": 0, "logit": 5.5}]},
    )

    compress_response = client.compress_documents(
        documents=[Document(page_content="Test compress.")], query="What is compress?"
    )
    assert (
        len(compress_response) == 1
    ), "Expected one ranking result from compress_documents"

    post_calls = [
        req
        for req in httpx_mock.get_requests()
        if req.method == "POST" and "/v1/ranking" in str(req.url)
    ]
    assert len(post_calls) > 0, "No POST requests made to /ranking endpoint"
    for req in post_calls:
        assert req.headers.get("authorization") == "Bearer BOGUS"
        assert req.headers.get("x-custom-header") == "test-value"


def test_rerank_client_error(httpx_mock: HTTPXMock, mock_model: str) -> None:
    httpx_mock.add_response(
        method="GET",
        url="https://integrate.api.nvidia.com/v1/models",
        status_code=401,
        json={"detail": "Invalid API key"},
    )

    custom_client = CustomTestClient()

    with pytest.raises(Exception) as exc_info:
        NVIDIARerank(api_key="INVALID", model=mock_model, http_client=custom_client)

    assert "Invalid API key" in str(exc_info.value)
