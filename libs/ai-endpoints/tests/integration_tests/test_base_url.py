import pytest
from requests.exceptions import ConnectionError

from .test_api_key import contact_service


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:12321",
    ],
)
def test_endpoint_unavailable(
    public_class: type,
    base_url: str,
) -> None:
    # we test this with a bogus model because users should supply
    # a model when using their own base_url
    client = public_class(model="not-a-model", base_url=base_url)
    with pytest.raises(ConnectionError):
        contact_service(client)
