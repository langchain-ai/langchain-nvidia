import warnings

import requests_mock
from langchain_core.messages import AIMessage

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
