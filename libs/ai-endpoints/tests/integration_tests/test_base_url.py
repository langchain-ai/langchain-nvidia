from typing import Any

import pytest
from requests.exceptions import ConnectionError


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
) -> None:
    # we test this with a bogus model because users should supply
    # a model when using their own base_url
    # as endpoint is unavailable /models call would fail to validate the model
    with pytest.raises(ConnectionError) as excinfo:
        client = public_class(model="not-a-model", base_url=base_url)
        contact_service(client)
        conn_err = str(excinfo.value)
        assert (
            "Failed to establish a new connection: [Errno 111] Connection refused"
            in conn_err
        )
        assert "Max retries exceeded with url: /models" in conn_err
