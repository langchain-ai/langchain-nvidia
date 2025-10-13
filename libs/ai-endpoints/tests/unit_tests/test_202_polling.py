import warnings

import pytest
import requests_mock
from aioresponses import aioresponses
from langchain_core.messages import AIMessage
from yarl import URL

from langchain_nvidia_ai_endpoints import ChatNVIDIA


def test_polling_auth_header(
    requests_mock: requests_mock.Mocker,
    mock_model: str,
) -> None:
    infer_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    polling_url = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/test-request-id"

    requests_mock.post(
        infer_url, status_code=202, headers={"NVCF-REQID": "test-request-id"}, json={}
    )

    requests_mock.get(
        polling_url,
        status_code=200,
        json={
            "id": "mock-id",
            "created": 1234567890,
            "object": "chat.completion",
            "model": mock_model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "WORKED"},
                }
            ],
        },
    )

    warnings.filterwarnings("ignore", r".*type is unknown and inference may fail.*")
    client = ChatNVIDIA(model=mock_model, api_key="BOGUS")
    response = client.invoke("IGNORED")

    # expected behavior -
    #  - first a GET request to /v1/models to check the model exists
    #  - second a POST request to /v1/chat/completions
    #  - third a GET request to /v2/nvcf/pexec/status/test-request-id
    # we want to check on the second and third requests

    assert len(requests_mock.request_history) == 3

    infer_request = requests_mock.request_history[-2]
    assert infer_request.method == "POST"
    assert infer_request.url == infer_url
    assert infer_request.headers["Authorization"] == "Bearer BOGUS"

    poll_request = requests_mock.request_history[-1]
    assert poll_request.method == "GET"
    assert poll_request.url == polling_url
    assert poll_request.headers["Authorization"] == "Bearer BOGUS"

    assert isinstance(response, AIMessage)
    assert response.content == "WORKED"


@pytest.mark.asyncio
async def test_async_polling_auth_header(mock_model: str) -> None:
    infer_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    polling_url = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/test-request-id"

    with aioresponses() as m:
        m.post(
            infer_url, status=202, headers={"NVCF-REQID": "test-request-id"}, payload={}
        )

        m.get(
            polling_url,
            status=200,
            payload={
                "id": "mock-id",
                "created": 1234567890,
                "object": "chat.completion",
                "model": mock_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "WORKED"},
                    }
                ],
            },
        )

        warnings.filterwarnings("ignore", r".*type is unknown and inference may fail.*")
        client = ChatNVIDIA(model=mock_model, api_key="BOGUS")
        response = await client.ainvoke("IGNORED")

        # expected behavior -
        #  - first a POST request to /v1/chat/completions
        #  - second a GET request to /v2/nvcf/pexec/status/test-request-id
        # we want to check on both requests

        # Get requests from aioresponses - dict with (method, URL) as keys
        post_requests = m.requests.get(("POST", URL(infer_url)), [])
        poll_requests = m.requests.get(("GET", URL(polling_url)), [])

        assert (
            post_requests and poll_requests
        ), "Both POST and GET requests should be made"

        infer_request = post_requests[0]
        assert (
            infer_request.kwargs.get("headers", {}).get("Authorization")
            == "Bearer BOGUS"
        )

        poll_request = poll_requests[0]
        assert (
            poll_request.kwargs.get("headers", {}).get("Authorization")
            == "Bearer BOGUS"
        )

        assert isinstance(response, AIMessage)
        assert response.content == "WORKED"
