from typing import Any

import pytest
import requests_mock


def test_available_models(public_class: type, mode: dict) -> None:
    models = public_class(**mode).available_models
    assert models
    assert isinstance(models, list)
    assert len(models) >= 1
    assert all(isinstance(model.id, str) for model in models)
    assert all(model.model_type is not None for model in models)
    assert all(model.client == public_class.__name__ for model in models)


def test_get_available_models(public_class: Any, mode: dict) -> None:
    models = public_class.get_available_models(**mode)
    assert isinstance(models, list)
    assert len(models) >= 1
    assert all(isinstance(model.id, str) for model in models)
    assert all(model.model_type is not None for model in models)
    assert all(model.client == public_class.__name__ for model in models)


# todo: turn this into a unit test
def test_available_models_cached(public_class: type, mode: dict) -> None:
    if public_class.__name__ == "NVIDIARerank" and "base_url" not in mode:
        pytest.skip("There is no listing service for hosted ranking NIMs")
    with requests_mock.Mocker(real_http=True) as mock:
        client = public_class()
        assert not mock.called
        client.available_models
        assert mock.called
        client.available_models
        assert mock.call_count == 1
