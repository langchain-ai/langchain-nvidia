from typing import Any

import pytest
from requests.exceptions import ConnectionError
from requests_mock import Mocker


# Fixture setup (mocking /v1/health/live and /v1/chat/completions endpoints)
@pytest.fixture()
def mock_endpoints(requests_mock: Mocker, base_url: str) -> None:
    # Mock the /v1/health/live endpoint
    requests_mock.get(
        f"{base_url}/v1/health/live",
        json={"object": "health-response", "message": "Service is live."},
    )

    for endpoint in ["/v1/embeddings", "/v1/chat/completions", "/v1/ranking"]:
        requests_mock.post(
            f"{base_url}{endpoint}",
            exc=ConnectionError(f"Mocked ConnectionError for {endpoint}"),
        )


# Test function using the mock_endpoints fixture
@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:12321",
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
    with pytest.raises(ConnectionError):
        contact_service(client)
