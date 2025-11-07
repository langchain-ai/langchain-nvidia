import warnings
from typing import Any, Literal, Optional

import pytest
from langchain_core.documents import Document
from requests_mock import Mocker

from langchain_nvidia_ai_endpoints import NVIDIARerank

from .conftest import MockHTTP


@pytest.fixture(autouse=True)
def mock_v1_models(requests_mock: Mocker) -> None:
    requests_mock.get(
        "https://integrate.api.nvidia.com/v1/models",
        json={
            "data": [
                {
                    "id": "mock-model",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "OWNER",
                }
            ]
        },
    )


@pytest.fixture(autouse=True)
def mock_v1_ranking(requests_mock: Mocker) -> None:
    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/ranking",
        json={
            "rankings": [
                {"index": 0, "logit": 4.2},
            ]
        },
    )


@pytest.fixture(autouse=True)
def mock_v1_aranking(mock_http: MockHTTP) -> None:
    mock_http.set_post(
        json_body={
            "rankings": [
                {"index": 0, "logit": 4.2},
            ]
        }
    )


@pytest.mark.parametrize(
    "truncate",
    [
        None,
        "END",
        "NONE",
    ],
)
@pytest.mark.parametrize(
    "func",
    ["compress_documents", "acompress_documents"],
)
@pytest.mark.asyncio
async def test_truncate(
    requests_mock: Mocker,
    mock_http: MockHTTP,
    truncate: Optional[Literal["END", "NONE"]],
    func: str,
) -> None:
    truncate_param = {}
    if truncate:
        truncate_param = {"truncate": truncate}
    warnings.filterwarnings(
        "ignore", ".*Found mock-model in available_models.*"
    )  # expect to see this warning
    client = NVIDIARerank(api_key="BOGUS", model="mock-model", **truncate_param)
    if func == "acompress_documents":
        response = await client.acompress_documents(
            documents=[Document(page_content="Nothing really.")], query="What is it?"
        )
    else:
        response = client.compress_documents(
            documents=[Document(page_content="Nothing really.")], query="What is it?"
        )

    assert len(response) == 1

    if func == "acompress_documents":
        assert len(mock_http.history) > 0
        last_request = mock_http.history[-1]
        assert last_request is not None
        request_payload = last_request.kwargs.get("json", {})
    else:
        assert requests_mock.last_request is not None
        request_payload = requests_mock.last_request.json()

    if truncate is None:
        assert "truncate" not in request_payload
    else:
        assert "truncate" in request_payload
        assert request_payload["truncate"] == truncate


@pytest.mark.parametrize("truncate", [True, False, 1, 0, 1.0, "START", "BOGUS"])
def test_truncate_invalid(truncate: Any) -> None:
    with pytest.raises(ValueError):
        NVIDIARerank(truncate=truncate)


@pytest.mark.parametrize(
    "func",
    ["compress_documents", "acompress_documents"],
)
@pytest.mark.asyncio
async def test_default_headers(
    requests_mock: Mocker,
    mock_http: MockHTTP,
    func: str,
) -> None:
    """Test that default_headers are passed to requests."""
    warnings.filterwarnings("ignore", ".*Found mock-model in available_models.*")
    client = NVIDIARerank(
        api_key="BOGUS", model="mock-model", default_headers={"X-Test": "test"}
    )
    assert client.default_headers == {"X-Test": "test"}

    if func == "acompress_documents":
        _ = await client.acompress_documents(
            documents=[Document(page_content="Nothing really.")], query="What is it?"
        )
        assert len(mock_http.history) > 0
        last_request = mock_http.history[-1]
        assert last_request is not None
        assert last_request.kwargs.get("headers", {})["X-Test"] == "test"
    else:
        _ = client.compress_documents(
            documents=[Document(page_content="Nothing really.")], query="What is it?"
        )
        assert requests_mock.last_request is not None
        assert requests_mock.last_request.headers["X-Test"] == "test"


@pytest.mark.parametrize(
    "func",
    ["compress_documents", "acompress_documents"],
)
@pytest.mark.asyncio
async def test_extra_headers(
    requests_mock: Mocker,
    mock_http: MockHTTP,
    func: str,
) -> None:
    """Test backward compatibility: extra_headers deprecated, issues warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        client = NVIDIARerank(
            api_key="BOGUS", model="mock-model", extra_headers={"X-Test": "test"}
        )
        # Check deprecation warning was issued
        deprecation_warnings = [
            warning for warning in w if issubclass(warning.category, DeprecationWarning)
        ]
        assert len(deprecation_warnings) == 1
        assert str(deprecation_warnings[0].message) == (
            "The 'extra_headers' parameter is deprecated and will be removed "
            "in a future version. Please use 'default_headers' instead."
        )

    # Verify it still works (copied to default_headers)
    assert client.default_headers == {"X-Test": "test"}

    if func == "acompress_documents":
        _ = await client.acompress_documents(
            documents=[Document(page_content="Nothing really.")], query="What is it?"
        )
        assert len(mock_http.history) > 0
        last_request = mock_http.history[-1]
        assert last_request is not None
        assert last_request.kwargs.get("headers", {})["X-Test"] == "test"
    else:
        _ = client.compress_documents(
            documents=[Document(page_content="Nothing really.")], query="What is it?"
        )
        assert requests_mock.last_request is not None
        assert requests_mock.last_request.headers["X-Test"] == "test"
