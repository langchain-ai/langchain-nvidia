import warnings
from typing import Any, Dict, List, Literal, Optional

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


def _get_passages(requests_mock: Mocker) -> List[Dict[str, Any]]:
    """Helper to extract passages from the last sync request payload."""
    assert requests_mock.last_request is not None
    return requests_mock.last_request.json()["passages"]


def test_text_only_payload_has_no_image_key(requests_mock: Mocker) -> None:
    """Text-only documents should produce passages with only a 'text' key."""
    warnings.filterwarnings("ignore", ".*Found mock-model in available_models.*")
    client = NVIDIARerank(api_key="BOGUS", model="mock-model")
    client.compress_documents(
        documents=[
            Document(page_content="first passage"),
            Document(page_content="second passage"),
        ],
        query="test query",
    )
    passages = _get_passages(requests_mock)
    assert len(passages) == 2
    for p in passages:
        assert "text" in p
        assert "image" not in p
    assert passages[0]["text"] == "first passage"
    assert passages[1]["text"] == "second passage"


def test_text_and_image_payload(requests_mock: Mocker) -> None:
    """Documents with image metadata should include an 'image' field in the payload."""
    warnings.filterwarnings("ignore", ".*Found mock-model in available_models.*")
    client = NVIDIARerank(api_key="BOGUS", model="mock-model")
    b64_image = "data:image/jpeg;base64,1234567abc"
    client.compress_documents(
        documents=[
            Document(page_content="a caption", metadata={"image": b64_image}),
        ],
        query="test query",
    )
    passages = _get_passages(requests_mock)
    assert len(passages) == 1
    assert passages[0]["text"] == "a caption"
    assert passages[0]["image"] == b64_image


def test_image_only_payload(requests_mock: Mocker) -> None:
    """Documents with empty text and image metadata should still work."""
    warnings.filterwarnings("ignore", ".*Found mock-model in available_models.*")
    client = NVIDIARerank(api_key="BOGUS", model="mock-model")
    b64_image = "data:image/png;base64,1234567abc"
    client.compress_documents(
        documents=[
            Document(page_content="", metadata={"image": b64_image}),
        ],
        query="test query",
    )
    passages = _get_passages(requests_mock)
    assert len(passages) == 1
    assert passages[0]["text"] == ""
    assert passages[0]["image"] == b64_image


def test_invalid_image_raises(requests_mock: Mocker) -> None:
    """A bogus metadata['image'] value should raise ValueError."""
    warnings.filterwarnings("ignore", ".*Found mock-model in available_models.*")
    client = NVIDIARerank(api_key="BOGUS", model="mock-model")
    with pytest.raises(ValueError):
        client.compress_documents(
            documents=[
                Document(
                    page_content="test passage",
                    metadata={"image": "an image"},
                ),
            ],
            query="test query",
        )


def test_image_on_non_vlm_model_warns(requests_mock: Mocker) -> None:
    """Image metadata on a non-VLM ranking model should emit a UserWarning."""
    warnings.filterwarnings("ignore", ".*Found mock-model in available_models.*")
    client = NVIDIARerank(api_key="BOGUS", model="mock-model")
    # Simulate a resolved text-only ranking model
    assert client._client.model is not None
    client._client.model.model_type = "ranking"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        client.compress_documents(
            documents=[
                Document(
                    page_content="test passage",
                    metadata={"image": "data:image/png;base64,1234567abc"},
                ),
            ],
            query="test query",
        )
    assert any(
        "not known to support image" in str(warning.message) for warning in w
    ), "expected a warning about image support"


def test_image_on_ranking_vlm_model_no_warning(requests_mock: Mocker) -> None:
    """Image metadata on a ranking-vlm model should not emit the guardrail warning."""
    warnings.filterwarnings("ignore", ".*Found mock-model in available_models.*")
    client = NVIDIARerank(api_key="BOGUS", model="mock-model")
    assert client._client.model is not None
    client._client.model.model_type = "ranking-vlm"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        client.compress_documents(
            documents=[
                Document(
                    page_content="test passage",
                    metadata={"image": "data:image/png;base64,1234567abc"},
                ),
            ],
            query="test query",
        )
    assert not any(
        "not known to support image" in str(warning.message) for warning in w
    )


def test_mixed_text_and_image_payload(requests_mock: Mocker) -> None:
    """A batch with both text-only and text+image docs produces correct payloads."""
    warnings.filterwarnings("ignore", ".*Found mock-model in available_models.*")
    client = NVIDIARerank(api_key="BOGUS", model="mock-model")
    b64_image = "data:image/jpeg;base64,1234567abc"
    client.compress_documents(
        documents=[
            Document(page_content="text only"),
            Document(page_content="with image", metadata={"image": b64_image}),
            Document(page_content="another text only"),
        ],
        query="test query",
    )
    passages = _get_passages(requests_mock)
    assert len(passages) == 3
    # first: text-only
    assert passages[0]["text"] == "text only"
    assert "image" not in passages[0]
    # second: text+image
    assert passages[1]["text"] == "with image"
    assert passages[1]["image"] == b64_image
    # third: text-only
    assert passages[2]["text"] == "another text only"
    assert "image" not in passages[2]
