import warnings
from typing import Any

import pytest

from langchain_nvidia_ai_endpoints import (
    ChatNVIDIA,
    Model,
    NVIDIAEmbeddings,
    NVIDIARerank,
    register_model,
)


#
# if this test is failing it may be because the function uuids have changed.
# you will have to find the new ones from https://api.nvcf.nvidia.com/v2/nvcf/functions
#
@pytest.mark.parametrize(
    "client, id, endpoint",
    [
        (
            ChatNVIDIA,
            "meta/llama3-8b-instruct",
            "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/a5a3ad64-ec2c-4bfc-8ef7-5636f26630fe",
        ),
        (
            NVIDIAEmbeddings,
            "NV-Embed-QA",
            "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/09c64e32-2b65-4892-a285-2f585408d118",
        ),
        (
            NVIDIARerank,
            "nv-rerank-qa-mistral-4b:1",
            "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/0bf77f50-5c35-4488-8e7a-f49bb1974af6",
        ),
    ],
)
def test_registered_model_functional(
    client: type, id: str, endpoint: str, contact_service: Any
) -> None:
    model = Model(id=id, endpoint=endpoint)
    warnings.filterwarnings(
        "ignore", r".*is already registered.*"
    )  # intentionally overridding known models
    warnings.filterwarnings(
        "ignore", r".*Unable to determine validity of.*"
    )  # we aren't passing client & type to Model()
    register_model(model)
    contact_service(client(model=id))


# todo: make this a unit test
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
    chat_models = ChatNVIDIA.get_available_models()
    embedding_models = NVIDIAEmbeddings.get_available_models()
    ranking_models = NVIDIARerank.get_available_models()

    assert "test/chat" in [model.id for model in chat_models]
    assert "test/chat" not in [model.id for model in embedding_models]
    assert "test/chat" not in [model.id for model in ranking_models]

    assert "test/embedding" not in [model.id for model in chat_models]
    assert "test/embedding" in [model.id for model in embedding_models]
    assert "test/embedding" not in [model.id for model in ranking_models]

    assert "test/rerank" not in [model.id for model in chat_models]
    assert "test/rerank" not in [model.id for model in embedding_models]
    assert "test/rerank" in [model.id for model in ranking_models]


def test_registered_model_without_client_is_not_listed(public_class: type) -> None:
    model_name = f"test/{public_class.__name__}"
    register_model(Model(id=model_name, endpoint="BOGUS"))
    models = public_class.get_available_models()
    assert model_name not in [model.id for model in models]
