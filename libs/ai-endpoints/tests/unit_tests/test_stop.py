import pytest
from requests_mock import Mocker

from langchain_nvidia_ai_endpoints import ChatNVIDIA


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
def mock_v1_chat_completions(requests_mock: Mocker) -> None:
    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        json={
            "id": "mock-id",
            "created": 1234567890,
            "object": "chat.completion",
            "model": "mock-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Ok"},
                }
            ],
        },
    )


def test_stop_property_invoke(requests_mock: Mocker) -> None:
    expected_stop = ["STOP"]

    client = ChatNVIDIA(model="mock-model", api_key="mocked", stop=expected_stop)
    response = client.invoke("Ok?")

    assert response.content == "Ok"

    assert requests_mock.last_request is not None
    request_payload = requests_mock.last_request.json()
    assert "stop" in request_payload
    assert request_payload["stop"] == expected_stop


def test_stop_parameter_invoke(requests_mock: Mocker) -> None:
    expected_stop = ["STOP"]

    client = ChatNVIDIA(model="mock-model", api_key="mocked")
    response = client.invoke("Ok?", stop=expected_stop)

    assert response.content == "Ok"

    assert requests_mock.last_request is not None
    request_payload = requests_mock.last_request.json()
    assert "stop" in request_payload
    assert request_payload["stop"] == expected_stop


def test_stop_override_invoke(requests_mock: Mocker) -> None:
    expected_stop = ["STOP"]

    client = ChatNVIDIA(model="mock-model", api_key="mocked", stop=["GO"])
    response = client.invoke("Ok?", stop=expected_stop)

    assert response.content == "Ok"

    assert requests_mock.last_request is not None
    request_payload = requests_mock.last_request.json()
    assert "stop" in request_payload
    assert request_payload["stop"] == expected_stop


def test_stop_property_stream(requests_mock: Mocker) -> None:
    expected_stop = ["STOP"]

    client = ChatNVIDIA(model="mock-model", api_key="mocked", stop=expected_stop)
    response = client.stream("Ok?")

    assert next(response).content == "Ok"

    assert requests_mock.last_request is not None
    request_payload = requests_mock.last_request.json()
    assert "stop" in request_payload
    assert request_payload["stop"] == expected_stop


def test_stop_parameter_stream(requests_mock: Mocker) -> None:
    expected_stop = ["STOP"]

    client = ChatNVIDIA(model="mock-model", api_key="mocked")
    response = client.stream("Ok?", stop=expected_stop)

    assert next(response).content == "Ok"

    assert requests_mock.last_request is not None
    request_payload = requests_mock.last_request.json()
    assert "stop" in request_payload
    assert request_payload["stop"] == expected_stop


def test_stop_override_stream(requests_mock: Mocker) -> None:
    expected_stop = ["STOP"]

    client = ChatNVIDIA(model="mock-model", api_key="mocked", stop=["GO"])
    response = client.stream("Ok?", stop=expected_stop)

    assert next(response).content == "Ok"

    assert requests_mock.last_request is not None
    request_payload = requests_mock.last_request.json()
    assert "stop" in request_payload
    assert request_payload["stop"] == expected_stop
