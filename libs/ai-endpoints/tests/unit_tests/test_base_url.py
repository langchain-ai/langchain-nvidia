import os
import pytest
from requests_mock import Mocker
from typing import Any
from .test_api_key import no_env_var

@pytest.fixture(autouse=True)
def mock_v1_local_models(requests_mock: Mocker, base_url: str) -> None:
    os.environ["NVIDIA_API_KEY"] = "DUMMY_KEY"
    requests_mock.get(
        f"{base_url}/models",
        json={
            "data": [
                {
                    "id": "model1",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "OWNER",
                    "root": "model1",
                },
            ]
        },
    )

@pytest.mark.parametrize(
    "base_url",["bogus"],
)
def test_create_without_base_url(public_class: type) -> None:
   with no_env_var("NVIDIA_BASE_URL"):
       assert public_class().base_url == "https://integrate.api.nvidia.com/v1"

@pytest.mark.parametrize("base_url, param", 
                         [("https://ai.api.nvidia.com/v1","nvidia_base_url"),
                          ("https://ai.api.nvidia.com/v1", "base_url")])
def test_create_with_base_url(public_class: type, base_url: str, param: str) -> None:
    with no_env_var("NVIDIA_BASE_URL"):
        assert public_class(**{param: base_url}).base_url == "https://ai.api.nvidia.com/v1"

@pytest.mark.parametrize(
    "base_url",["bogus"],
)
def test_base_url_priority(public_class: type) -> None:
    with no_env_var("NVIDIA_BASE_URL"):
        os.environ["NVIDIA_BASE_URL"] = "https://ai.api.nvidia.com/v1"
        assert public_class().base_url == "https://ai.api.nvidia.com/v1"
        assert public_class(nvidia_base_url="https://ai.api.nvidia.com/v1").base_url == "https://ai.api.nvidia.com/v1"
        assert public_class(base_url="https://ai.api.nvidia.com/v1").base_url == "https://ai.api.nvidia.com/v1"
        assert public_class(base_url="https://integrate.api.nvidia.com/v1", nvidia_base_url="https://ai.api.nvidia.com/v1").base_url == "https://ai.api.nvidia.com/v1"

@pytest.mark.parametrize(

    "base_url",
    [
        "bogus",
        "http:/",
        "http://",
        "http:/oops",
    ],
)
def test_param_base_url_negative(public_class: type, base_url: str) -> None:
    with no_env_var("NVIDIA_BASE_URL"):
        with pytest.raises(ValueError):
            public_class(base_url=base_url)

@pytest.mark.parametrize(
    "base_url",
    ["https://integrate.api.nvidia.com/v1", "https://ai.api.nvidia.com/v1"],
)
def test_param_base_url_hosted(public_class: type, base_url: str) -> None:
    with no_env_var("NVIDIA_BASE_URL"):
        client = public_class(base_url=base_url)
        assert client._client.is_hosted

@pytest.mark.parametrize(
    "base_url",
    [
        "https://localhost",
        "http://localhost:8888",
        "http://0.0.0.0:8888",
    ],
)
def test_param_base_url_not_hosted(public_class: type, base_url: str) -> None:
    with no_env_var("NVIDIA_BASE_URL") and pytest.warns(UserWarning):
        client = public_class(base_url=base_url)
        assert not client._client.is_hosted 
