import warnings

import pytest

from langchain_nvidia_ai_endpoints import Model, register_model


@pytest.mark.parametrize(
    "model_type, client",
    [
        ("chat", "NVIDIAEmbeddings"),
        ("chat", "NVIDIARerank"),
        ("vlm", "NVIDIAEmbeddings"),
        ("vlm", "NVIDIARerank"),
        ("embeddings", "ChatNVIDIA"),
        ("embeddings", "NVIDIARerank"),
        ("ranking", "ChatNVIDIA"),
        ("ranking", "NVIDIAEmbeddings"),
    ],
)
def test_mismatched_type_client(model_type: str, client: str) -> None:
    with pytest.raises(ValueError) as e:
        register_model(
            Model(
                id=f"{model_type}-{client}",
                model_type=model_type,
                client=client,
                endpoint="BOGUS",
            )
        )
    assert "not supported" in str(e.value)


def test_duplicate_model_warns() -> None:
    model = Model(id="registered-model", endpoint="BOGUS")
    register_model(model)
    with pytest.warns(UserWarning) as record:
        register_model(model)
    assert len(record) == 1
    assert isinstance(record[0].message, UserWarning)
    assert "already registered" in str(record[0].message)
    assert "Overriding" in str(record[0].message)


def test_registered_model_usable(public_class: type) -> None:
    model_type = {
        "ChatNVIDIA": "chat",
        "NVIDIAEmbeddings": "embedding",
        "NVIDIARerank": "ranking",
    }[public_class.__name__]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        id = f"registered-model-{model_type}"
        model = Model(
            id=id,
            model_type=model_type,
            client=public_class.__name__,
            endpoint="BOGUS",
        )
        register_model(model)
        x = public_class(model=id, nvidia_api_key="a-bogus-key")
        assert x.model == id


def test_registered_model_without_client_usable(public_class: type) -> None:
    id = f"test/no-client-{public_class.__name__}"
    model = Model(id=id, endpoint="BOGUS")
    register_model(model)
    # todo: this should warn that the model is known but type is not
    #       and therefore inference may not work
    # Marking this as failed
    with pytest.warns(UserWarning):
        public_class(model=id, nvidia_api_key="a-bogus-key")


def test_missing_endpoint() -> None:
    with pytest.raises(ValueError) as e:
        register_model(
            Model(id="missing-endpoint", model_type="chat", client="ChatNVIDIA")
        )
    assert "does not have an endpoint" in str(e.value)
