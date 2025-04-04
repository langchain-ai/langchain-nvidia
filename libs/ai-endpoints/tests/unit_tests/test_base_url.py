import os
import re
import warnings
from typing import Any

import pytest
from requests_mock import Mocker

from .test_api_key import no_env_var


@pytest.fixture(autouse=True)
def mock_v1_local_models(requests_mock: Mocker) -> None:
    requests_mock.get(
        re.compile(r".*/models"),
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


def test_create_without_base_url(public_class: type) -> None:
    with no_env_var("NVIDIA_BASE_URL"):
        x = public_class(api_key="BOGUS")
        assert x.base_url == "https://integrate.api.nvidia.com/v1"
        assert x._client.base_url == "https://integrate.api.nvidia.com/v1"


@pytest.mark.parametrize(
    "base_url, param",
    [("https://test_url/v1", "nvidia_base_url"), ("https://test_url/v1", "base_url")],
)
def test_create_with_base_url(public_class: type, base_url: str, param: str) -> None:
    with no_env_var("NVIDIA_BASE_URL"):
        assert public_class(model="model1", **{param: base_url}).base_url == base_url


def test_base_url_priority(public_class: type) -> None:
    ENV_URL = "https://ENV/v1"
    NV_PARAM_URL = "https://NV_PARAM/v1"
    PARAM_URL = "https://PARAM/v1"

    def get_base_url(**kwargs: Any) -> str:
        return public_class(model="model1", **kwargs).base_url

    with no_env_var("NVIDIA_BASE_URL"):
        os.environ["NVIDIA_BASE_URL"] = ENV_URL
        assert get_base_url() == ENV_URL
        assert get_base_url(nvidia_base_url=NV_PARAM_URL) == NV_PARAM_URL
        assert get_base_url(base_url=PARAM_URL) == PARAM_URL
        assert (
            get_base_url(base_url=PARAM_URL, nvidia_base_url=NV_PARAM_URL)
            == NV_PARAM_URL
        )


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
        with pytest.raises(ValueError) as e:
            public_class(base_url=base_url)
        assert "Invalid base_url" in str(e.value)


@pytest.mark.parametrize(
    "base_url",
    ["https://integrate.api.nvidia.com/v1", "https://ai.api.nvidia.com/v1"],
)
def test_param_base_url_hosted(public_class: type, base_url: str) -> None:
    with no_env_var("NVIDIA_BASE_URL"):
        client = public_class(api_key="BOGUS", base_url=base_url)
        assert client._client.is_hosted


@pytest.mark.parametrize(
    "base_url",
    [
        "https://localhost",
        "http://localhost:8888",
        "http://0.0.0.0:8888/v1",
        "http://0.0.0.0:8888/v1/",
        "http://blah/some/other/path/v1",
    ],
)
def test_param_base_url_not_hosted(public_class: type, base_url: str) -> None:
    warnings.filterwarnings("ignore", r".*does not contain /v1.*")
    with no_env_var("NVIDIA_BASE_URL"):
        client = public_class(model="model1", base_url=base_url)
        assert not client._client.is_hosted


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:8888/embeddings",
        "http://0.0.0.0:8888/rankings",
        "http://localhost:8888/embeddings/",
        "http://0.0.0.0:8888/rankings/",
        "http://localhost:8888/chat/completions",
    ],
)
def test_expect_warn(public_class: type, base_url: str) -> None:
    with pytest.warns(UserWarning) as record:
        public_class(model="model1", base_url=base_url)
    assert len(record) == 1
    assert "does not contain /v1" in str(record[0].message)


def test_default_hosted(public_class: type) -> None:
    x = public_class(api_key="BOGUS")
    assert x._client.is_hosted


@pytest.mark.parametrize(
    "base_url",
    [
        "http://host/path0/path1/path2/v1",
        "http://host:123/path0/path1/path2/v1/",
    ],
)
def test_proxy_base_url(
    public_class: type, base_url: str, requests_mock: Mocker
) -> None:
    with no_env_var("NVIDIA_BASE_URL"):
        client = public_class(model="model1", base_url=base_url)
        assert base_url.startswith(client.base_url)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://host/path0/path1/path2/v1",
        "http://host:123/path0/path1/path2/v1/",
    ],
)
def test_proxy_base_url_models(
    public_class: type, base_url: str, requests_mock: Mocker
) -> None:
    with no_env_var("NVIDIA_BASE_URL"):
        client = public_class(model="model1", base_url=base_url)
        client.available_models
        models_url = base_url.rstrip("/") + "/models"
        assert requests_mock.last_request
        assert requests_mock.last_request.url == models_url
