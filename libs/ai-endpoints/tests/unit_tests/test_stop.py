import warnings
from typing import Optional, Sequence, Union

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


@pytest.mark.parametrize(
    "prop_stop, param_stop, expected_stop",
    [
        (None, ["PARAM"], ["PARAM"]),
        (None, "PARAM", "PARAM"),
        (["PROP"], None, ["PROP"]),
        (["PROP"], ["PARAM"], ["PARAM"]),
        (["PROP"], "PARAM", "PARAM"),
        (None, None, None),
    ],
    ids=[
        "parameter_seq",
        "parameter_str",
        "property",
        "override_seq",
        "override_str",
        "absent",
    ],
)
@pytest.mark.parametrize("func_name", ["invoke", "stream"])
def test_stop(
    requests_mock: Mocker,
    prop_stop: Optional[Sequence[str]],
    param_stop: Optional[Union[str, Sequence[str]]],
    expected_stop: Union[str, Sequence[str]],
    func_name: str,
) -> None:
    """
    Users can pass `stop` as a property of the client or as a parameter to the
    `invoke` or `stream` methods. The value passed as a parameter should
    override the value passed as a property.

    Also, the `stop` parameter can be a str or Sequence[str], while the `stop`
    property is always a Sequence[str].
    """
    # `**(dict(stop=...) if ... else {})` is a clever way to avoid passing stop
    # if the value is None
    warnings.filterwarnings(
        "ignore", ".*Found mock-model in available_models.*"
    )  # expect to see this warning
    client = ChatNVIDIA(
        model="mock-model",
        api_key="mocked",
        **(dict(stop=prop_stop) if prop_stop else {}),
    )
    # getattr(client, func_name) is a clever way to call a method by name
    response = getattr(client, func_name)(
        "Ok?", **(dict(stop=param_stop) if param_stop else {})
    )
    # the `stream` method returns a generator, so we need to call `next` to get
    # the actual response
    if func_name == "stream":  # one step too clever parameterizing the function name
        response = next(response)

    assert response.content == "Ok"

    assert requests_mock.last_request is not None
    request_payload = requests_mock.last_request.json()
    if expected_stop:
        assert "stop" in request_payload
        assert request_payload["stop"] == expected_stop
    else:
        assert "stop" not in request_payload
