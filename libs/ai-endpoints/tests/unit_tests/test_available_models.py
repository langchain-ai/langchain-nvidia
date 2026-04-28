import re
import warnings
from typing import Any

import requests_mock as rm

from langchain_nvidia_ai_endpoints import ChatNVIDIA, Model, register_model
from langchain_nvidia_ai_endpoints._common import _NVIDIASyncClient


def test_model_listing(public_class: Any, mock_model: str) -> None:
    warnings.filterwarnings("ignore", message=f"Default model is set as: {mock_model}")
    # we set base_url to avoid having results filtered by the public_class name
    models = public_class.get_available_models(base_url="https://mock/v1")
    assert any(model.id == mock_model for model in models)


def test_model_listing_hosted(
    public_class: Any,
    mock_model: str,
) -> None:
    model = Model(
        id=mock_model,
        model_type={
            "ChatNVIDIA": "chat",
            "NVIDIAEmbeddings": "embedding",
            "NVIDIARerank": "ranking",
            "NVIDIA": "completions",
        }[public_class.__name__],
        client=public_class.__name__,
        endpoint="BOGUS",
    )
    register_model(model)
    models = public_class.get_available_models()
    assert any(model.id == mock_model for model in models)


def test_single_models_call_per_instance(
    requests_mock: rm.Mocker,
) -> None:
    """Constructing a public class should hit /v1/models and emit each warning
    at most once.

    Regression test for the sync/async client split: `_build_clients` runs
    `_finalize` once on the sync client and copies resolved fields plus
    `_available_models` to the async client. If that ever regresses to
    instantiating two fully-validated clients, /v1/models would be fetched
    twice and `_finalize` warnings would double up.
    """
    unknown_model = "test-org/unknown-model"
    requests_mock.get(
        re.compile(".*/v1/models"),
        json={"data": [{"id": "test-org/something-else"}]},
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ChatNVIDIA(model=unknown_model, api_key="BOGUS")

    models_calls = [
        req for req in requests_mock.request_history if req.path.endswith("/v1/models")
    ]
    assert (
        len(models_calls) == 1
    ), f"expected 1 /v1/models call, got {len(models_calls)}"

    unknown_warnings = [w for w in caught if "is unknown" in str(w.message)]
    assert len(unknown_warnings) == 1, (
        f"expected 1 'is unknown' warning, got {len(unknown_warnings)}: "
        f"{[str(w.message) for w in unknown_warnings]}"
    )


def test_duplicate_models_in_api_response(
    requests_mock: rm.Mocker,
) -> None:
    """API returning duplicate model entries should not crash ChatNVIDIA.

    Regression test for the case where /v1/models returns the same model id
    more than once (e.g. nvidia/nemotron-3-super-120b-a12b). Previously this
    triggered an AssertionError in _NVIDIASyncClient.__init__.
    """
    duplicate_model = "test-org/duplicate-model"
    requests_mock.get(
        re.compile(".*/v1/models"),
        json={
            "data": [
                {"id": duplicate_model},
                {"id": duplicate_model},
                {"id": "test-org/unique-model"},
            ]
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client = ChatNVIDIA(model=duplicate_model)
    assert client.model == duplicate_model


def test_duplicate_models_deduplicated_in_available_models(
    requests_mock: rm.Mocker,
) -> None:
    """available_models should contain each model id exactly once."""
    duplicate_model = "test-org/duplicate-model"
    requests_mock.get(
        re.compile(".*/v1/models"),
        json={
            "data": [
                {"id": duplicate_model},
                {"id": duplicate_model},
                {"id": "test-org/unique-model"},
            ]
        },
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        client = ChatNVIDIA(model=duplicate_model)
    # Access the internal client's cached model list (API-only, before
    # get_available_models merges in the static MODEL_TABLE)
    internal: _NVIDIASyncClient = client._client  # type: ignore[attr-defined]
    ids = [m.id for m in internal.available_models]
    assert ids.count(duplicate_model) == 1
    assert ids.count("test-org/unique-model") == 1
    assert len(ids) == 2
