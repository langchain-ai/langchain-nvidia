import pytest
import requests_mock

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank


@pytest.fixture(
    params=[
        ChatNVIDIA,
        NVIDIAEmbeddings,
        NVIDIARerank,
    ]
)
def public_class(request: pytest.FixtureRequest) -> type:
    return request.param


@pytest.fixture
def empty_v1_models(requests_mock: requests_mock.Mocker) -> None:
    requests_mock.get("https://integrate.api.nvidia.com/v1/models", json={"data": []})
