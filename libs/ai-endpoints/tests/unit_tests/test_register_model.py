import warnings

import pytest

from langchain_nvidia_ai_endpoints import (
    NVIDIA,
    ChatNVIDIA,
    Model,
    NVIDIAEmbeddings,
    NVIDIARerank,
    register_model,
)


@pytest.mark.parametrize(
    "model_type, client",
    [
        ("chat", "NVIDIAEmbeddings"),
        ("chat", "NVIDIARerank"),
        ("chat", "NVIDIA"),
        ("vlm", "NVIDIAEmbeddings"),
        ("vlm", "NVIDIARerank"),
        ("vlm", "NVIDIA"),
        ("embedding", "ChatNVIDIA"),
        ("embedding", "NVIDIARerank"),
        ("embedding", "NVIDIA"),
        ("ranking", "ChatNVIDIA"),
        ("ranking", "NVIDIAEmbeddings"),
        ("ranking", "NVIDIA"),
        ("completions", "ChatNVIDIA"),
        ("completions", "NVIDIAEmbeddings"),
        ("completions", "NVIDIARerank"),
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
        "NVIDIA": "completions",
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
    with pytest.warns(UserWarning) as record:
        public_class(model=id, nvidia_api_key="a-bogus-key")
    assert len(record) == 1
    assert isinstance(record[0].message, UserWarning)
    assert "Unable to determine validity" in str(record[0].message)


def test_missing_endpoint() -> None:
    with pytest.raises(ValueError) as e:
        register_model(
            Model(id="missing-endpoint", model_type="chat", client="ChatNVIDIA")
        )
    assert "does not have an endpoint" in str(e.value)


def test_registered_model_is_available() -> None:
    register_model(
        Model(
            id="test/chat",
            model_type="chat",
            client="ChatNVIDIA",
            endpoint="BOGUS",
        )
    )
    register_model(
        Model(
            id="test/embedding",
            model_type="embedding",
            client="NVIDIAEmbeddings",
            endpoint="BOGUS",
        )
    )
    register_model(
        Model(
            id="test/rerank",
            model_type="ranking",
            client="NVIDIARerank",
            endpoint="BOGUS",
        )
    )
    register_model(
        Model(
            id="test/completions",
            model_type="completions",
            client="NVIDIA",
            endpoint="BOGUS",
        )
    )
    chat_models = ChatNVIDIA.get_available_models(api_key="BOGUS")
    embedding_models = NVIDIAEmbeddings.get_available_models(api_key="BOGUS")
    ranking_models = NVIDIARerank.get_available_models(api_key="BOGUS")
    completions_models = NVIDIA.get_available_models(api_key="BOGUS")

    assert "test/chat" in [model.id for model in chat_models]
    assert "test/chat" not in [model.id for model in embedding_models]
    assert "test/chat" not in [model.id for model in ranking_models]
    assert "test/chat" not in [model.id for model in completions_models]

    assert "test/embedding" not in [model.id for model in chat_models]
    assert "test/embedding" in [model.id for model in embedding_models]
    assert "test/embedding" not in [model.id for model in ranking_models]
    assert "test/embedding" not in [model.id for model in completions_models]

    assert "test/rerank" not in [model.id for model in chat_models]
    assert "test/rerank" not in [model.id for model in embedding_models]
    assert "test/rerank" in [model.id for model in ranking_models]
    assert "test/rerank" not in [model.id for model in completions_models]

    assert "test/completions" not in [model.id for model in chat_models]
    assert "test/completions" not in [model.id for model in embedding_models]
    assert "test/completions" not in [model.id for model in ranking_models]
    assert "test/completions" in [model.id for model in completions_models]


def test_registered_model_without_client_is_not_listed(public_class: type) -> None:
    model_name = f"test/{public_class.__name__}"
    register_model(Model(id=model_name, endpoint="BOGUS"))
    models = public_class.get_available_models(api_key="BOGUS")  # type: ignore
    assert model_name not in [model.id for model in models]
