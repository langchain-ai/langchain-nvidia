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


@pytest.fixture(autouse=True)
def mock_v1_local_models(requests_mock: Mocker, base_url: str) -> None:
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
    "base_url",
    [
        "https://localhost",
        "http://localhost:8888",
        "http://0.0.0.0:8888",
    ],
)
def test_param_base_url_not_hosted(public_class: type, base_url: str) -> None:
    client = public_class(base_url=base_url)
    assert not client._client.is_hosted
