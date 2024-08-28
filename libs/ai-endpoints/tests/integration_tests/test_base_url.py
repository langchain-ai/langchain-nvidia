import re
from typing import Any

import pytest
from requests.exceptions import ConnectionError
from requests_mock import Mocker


# Fixture setup /v1/chat/completions endpoints
@pytest.fixture()
def mock_endpoints(requests_mock: Mocker) -> None:
    for endpoint in [
        "/v1/embeddings",
        "/v1/chat/completions",
        "/v1/ranking",
        "/v1/completions",
    ]:
        requests_mock.post(
            re.compile(f".*{endpoint}"),
            exc=ConnectionError(f"Mocked ConnectionError for {endpoint}"),
        )
    requests_mock.get(
        re.compile(".*/v1/models"),
        json={
            "data": [
                {
                    "id": "not-a-model",
                    "object": "model",
                    "created": 1234567890,
                    "owned_by": "OWNER",
                },
            ]
        },
    )


# Test function using the mock_endpoints fixture
@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:12321",
        "http://localhost:12321/v1",
    ],
)
def test_endpoint_unavailable(
    public_class: type,
    base_url: str,
    contact_service: Any,
    mock_endpoints: None,  # Inject the mock_endpoints fixture
) -> None:
    # we test this with a bogus model because users should supply
    # a model when using their own base_url
    client = public_class(model="not-a-model", base_url=base_url)
    with pytest.raises(ConnectionError) as e:
        contact_service(client)
    assert "Mocked ConnectionError for" in str(e.value)


# todo: move this to be a unit test
