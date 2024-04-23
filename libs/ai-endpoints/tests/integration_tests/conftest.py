import warnings
from typing import List

import pytest

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_nvidia_ai_endpoints._common import Model


def get_mode(config: pytest.Config) -> dict:
    nim_endpoint = config.getoption("--nim-endpoint")
    if nim_endpoint:
        return dict(mode="nim", base_url=nim_endpoint)
    return {}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--chat-model-id",
        action="store",
        help="Run tests for a specific chat model",
    )
    parser.addoption(
        "--embedding-model-id",
        action="store",
        help="Run tests for a specific embedding model",
    )
    parser.addoption(
        "--rerank-model-id",
        action="store",
        help="Run tests for a specific rerank model",
    )
    parser.addoption(
        "--all-models",
        action="store_true",
        help="Run tests across all models",
    )
    parser.addoption(
        "--nim-endpoint",
        type=str,
        help="Run tests using NIM mode",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    mode = get_mode(metafunc.config)

    def get_all_models() -> List[Model]:
        return ChatNVIDIA().mode(**mode).get_available_models(list_all=True, **mode)

    if "chat_model" in metafunc.fixturenames:
        models = [metafunc.config.getoption("chat_model_id", ChatNVIDIA._default_model)]
        if metafunc.config.getoption("all_models"):
            models = [
                model.id for model in get_all_models() if model.model_type == "chat"
            ]
        metafunc.parametrize("chat_model", models, ids=models)

    if "rerank_model" in metafunc.fixturenames:
        models = ["ai-rerank-qa-mistral-4b"]
        if model := metafunc.config.getoption("rerank_model_id"):
            models = [model]
        # nim-mode reranking does not support model listing
        if metafunc.config.getoption("all_models"):
            if mode.get("mode", None) == "nim":
                warnings.warn(
                    "Skipping model listing for Rerank "
                    "with --nim-endpoint, not supported"
                )
            else:
                models = [
                    model.id
                    for model in get_all_models()
                    if model.model_type == "ranking"
                ]
        metafunc.parametrize("rerank_model", models, ids=models)

    if "image_in_model" in metafunc.fixturenames:
        models = ["ai-fuyu-8b"]
        if metafunc.config.getoption("all_models"):
            models = [
                model.id for model in get_all_models() if model.model_type == "image_in"
            ]
        metafunc.parametrize("image_in_model", models, ids=models)

    if "qa_model" in metafunc.fixturenames:
        models = ["nemotron_qa_8b"]
        if metafunc.config.getoption("all_models"):
            models = [
                model.id for model in get_all_models() if model.model_type == "qa"
            ]
        metafunc.parametrize("qa_model", models, ids=models)

    if "embedding_model" in metafunc.fixturenames:
        models = [NVIDIAEmbeddings._default_model]
        if metafunc.config.getoption("embedding_model_id"):
            models = [metafunc.config.getoption("embedding_model_id")]
        if metafunc.config.getoption("all_models"):
            if mode.get("mode", None) == "nim":
                # there is no guarantee the NIM will return a known model name,
                # so we just grab all models and assume they are embeddings
                models = [model.id for model in get_all_models()]
            else:
                models = [
                    model.id
                    for model in get_all_models()
                    if model.model_type == "embedding"
                ]
        metafunc.parametrize("embedding_model", models, ids=models)


@pytest.fixture
def mode(request: pytest.FixtureRequest) -> dict:
    return get_mode(request.config)
