from urllib.parse import urlparse, urlunparse

import pytest
from requests_mock import Mocker


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
    with pytest.raises(ValueError):
        public_class(base_url=base_url)


@pytest.mark.parametrize(
    "base_url",
    ["https://integrate.api.nvidia.com/v1", "https://ai.api.nvidia.com/v1"],
)
def test_param_base_url_hosted(public_class: type, base_url: str) -> None:
    client = public_class(base_url=base_url)
    assert client._client.is_hosted


@pytest.fixture()
def mock_v1_local_models(requests_mock: Mocker, base_url: str) -> None:
    requests_mock.get(
        f"{base_url}/v1/models",
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


@pytest.fixture()
def mock_v1_local_models2(requests_mock: Mocker, base_url: str) -> None:
    result = urlparse(base_url)
    base_url = urlunparse((result.scheme, result.netloc, "v1", "", "", ""))
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


# @pytest.fixture(autouse=True)
# def mock_local_health(requests_mock: Mocker, base_url: str) -> None:
#     result = urlparse(base_url)
#     base_url = urlunparse((result.scheme, result.netloc, "v1", "", "", ""))
#     requests_mock.get(
#         f"{base_url}/health/live",
#         json={"object": "health-response", "message": "Service is live."},
#     )


# @pytest.fixture
# def mock_local_unhealthy(requests_mock: Mocker, base_url: str) -> None:
#     result = urlparse(base_url)
#     base_url = urlunparse((result.scheme, result.netloc, "v1", "", "", ""))
#     requests_mock.get(f"{base_url}/health/live", status_code=404)


@pytest.mark.parametrize(
    "base_url",
    [
        "https://localhost",
        "http://localhost:8888",
        "http://0.0.0.0:8888",
    ],
)
def test_param_base_url_not_hosted(
    public_class: type, base_url: str, mock_v1_local_models: None
) -> None:
    with pytest.warns(UserWarning):
        client = public_class(base_url=base_url)
        assert not client._client.is_hosted


# test case for base_url warnings
@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:8888/embeddings",
        "http://0.0.0.0:8888/ranking",
        "http://localhost:8888/v1/chat/completion",
    ],
)
def test_base_url_warning_not_hosted(
    public_class: type, base_url: str, mock_v1_local_models2: None
) -> None:
    with pytest.raises(ValueError):
        public_class(base_url=base_url)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:8888",
    ],
)
def test_base_url_unhealthy_not_hosted(
    public_class: type, base_url: str, mock_v1_local_models: None
) -> None:
    with pytest.warns(UserWarning):
        public_class(base_url=base_url)
