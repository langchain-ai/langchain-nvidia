from typing import Any, List

import pytest
from langchain_core.documents import Document

from langchain_nvidia_ai_endpoints import (
    NVIDIA,
    ChatNVIDIA,
    NVIDIAEmbeddings,
    NVIDIARerank,
)
from langchain_nvidia_ai_endpoints._statics import MODEL_TABLE, Model
from langchain_nvidia_ai_endpoints.chat_models import (
    _DEFAULT_MODEL_NAME as DEFAULT_CHAT_MODEL,
)
from langchain_nvidia_ai_endpoints.embeddings import (
    _DEFAULT_MODEL_NAME as DEFAULT_EMBEDDINGS_MODEL,
)
from langchain_nvidia_ai_endpoints.llm import (
    _DEFAULT_MODEL_NAME as DEFAULT_COMPLETIONS_MODEL,
)
from langchain_nvidia_ai_endpoints.reranking import (
    _DEFAULT_MODEL_NAME as DEFAULT_RERANKING_MODEL,
)


def get_mode(config: pytest.Config) -> dict:
    nim_endpoint = config.getoption("--nim-endpoint")
    if nim_endpoint:
        return dict(base_url=nim_endpoint)
    return {}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--chat-model-id",
        action="store",
        nargs="+",
        help="Run tests for a specific chat model or list of models",
    )
    parser.addoption(
        "--tool-model-id",
        action="store",
        nargs="+",
        help="Run tests for a specific chat models that support tool calling",
    )
    parser.addoption(
        "--structured-model-id",
        action="store",
        nargs="+",
        help="Run tests for a specific models that support structured output",
    )
    parser.addoption(
        "--thinking-model-id",
        action="store",
        nargs="+",
        help="Run tests for a specific models that support thinking mode",
    )
    parser.addoption(
        "--qa-model-id",
        action="store",
        nargs="+",
        help="Run tests for a specific qa model or list of models",
    )
    parser.addoption(
        "--completions-model-id",
        action="store",
        nargs="+",
        help="Run tests for a specific completions model or list of models",
    )
    parser.addoption(
        "--embedding-model-id",
        action="store",
        nargs="+",
        help="Run tests for a specific embedding model or list of models",
    )
    parser.addoption(
        "--rerank-model-id",
        action="store",
        nargs="+",
        help="Run tests for a specific rerank model or list of models",
    )
    parser.addoption(
        "--vlm-model-id",
        action="store",
        nargs="+",
        help="Run tests for a specific vlm model or list of models",
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

    def get_all_known_models() -> List[Model]:
        return list(MODEL_TABLE.values())

    if "thinking_model" in metafunc.fixturenames:
        models = [
            "nvidia/llama-3.1-nemotron-nano-4b-v1.1",
            "nvidia/llama-3.1-nemotron-nano-8b-v1"
        ]
        if model_list := metafunc.config.getoption("thinking_model_id"):
            models = model_list
        if metafunc.config.getoption("all_models"):
            models = [
                model.id
                for model in ChatNVIDIA(**mode).available_models
                if model.supports_thinking
            ]
        metafunc.parametrize("thinking_model", models, ids=models)

    if "chat_model" in metafunc.fixturenames:
        models = [DEFAULT_CHAT_MODEL]
        if model_list := metafunc.config.getoption("chat_model_id"):
            models = model_list
        if metafunc.config.getoption("all_models"):
            models = [
                model.id
                for model in ChatNVIDIA(**mode).available_models
                if model.model_type == "chat"
            ]
        metafunc.parametrize("chat_model", models, ids=models)

    if "tool_model" in metafunc.fixturenames:
        models = ["meta/llama-3.3-70b-instruct"]
        if model_list := metafunc.config.getoption("tool_model_id"):
            models = model_list
        if metafunc.config.getoption("all_models"):
            models = [
                model.id
                for model in ChatNVIDIA(**mode).available_models
                if model.model_type == "chat" and model.supports_tools
            ]
        metafunc.parametrize("tool_model", models, ids=models)

    if "completions_model" in metafunc.fixturenames:
        models = [DEFAULT_COMPLETIONS_MODEL]
        if model_list := metafunc.config.getoption("completions_model_id"):
            models = model_list
        if metafunc.config.getoption("all_models"):
            models = [
                model.id
                for model in NVIDIA(**mode).available_models
                if model.model_type == "completions"
            ]
        metafunc.parametrize("completions_model", models, ids=models)

    if "structured_model" in metafunc.fixturenames:
        models = ["meta/llama-3.3-70b-instruct"]
        if model_list := metafunc.config.getoption("structured_model_id"):
            models = model_list
        if metafunc.config.getoption("all_models"):
            models = [
                model.id
                for model in ChatNVIDIA(**mode).available_models
                if model.supports_structured_output
            ]
        metafunc.parametrize("structured_model", models, ids=models)

    if "rerank_model" in metafunc.fixturenames:
        models = [DEFAULT_RERANKING_MODEL]
        if model_list := metafunc.config.getoption("rerank_model_id"):
            models = model_list
        if metafunc.config.getoption("all_models"):
            models = [model.id for model in NVIDIARerank(**mode).available_models]
        metafunc.parametrize("rerank_model", models, ids=models)

    if "vlm_model" in metafunc.fixturenames:
        models = ["meta/llama-3.2-11b-vision-instruct"]
        if model_list := metafunc.config.getoption("vlm_model_id"):
            models = model_list
        if metafunc.config.getoption("all_models"):
            models = [
                model.id
                for model in get_all_known_models()
                if model.model_type in {"vlm", "nv-vlm"}
            ]
        metafunc.parametrize("vlm_model", models, ids=models)

    if "qa_model" in metafunc.fixturenames:
        models = []
        if model_list := metafunc.config.getoption("qa_model_id"):
            models = model_list
        if metafunc.config.getoption("all_models"):
            models = [
                model.id
                for model in ChatNVIDIA(**mode).available_models
                if model.model_type == "qa"
            ]
        metafunc.parametrize("qa_model", models, ids=models)

    if "embedding_model" in metafunc.fixturenames:
        models = [DEFAULT_EMBEDDINGS_MODEL]
        if metafunc.config.getoption("all_models"):
            models = [model.id for model in NVIDIAEmbeddings(**mode).available_models]
        if model_list := metafunc.config.getoption("embedding_model_id"):
            models = model_list
        if metafunc.config.getoption("all_models"):
            models = [model.id for model in NVIDIAEmbeddings(**mode).available_models]
        metafunc.parametrize("embedding_model", models, ids=models)


@pytest.fixture
def mode(request: pytest.FixtureRequest) -> dict:
    return get_mode(request.config)


@pytest.fixture(
    params=[
        ChatNVIDIA,
        NVIDIAEmbeddings,
        NVIDIARerank,
        NVIDIA,
    ]
)
def public_class(request: pytest.FixtureRequest) -> type:
    return request.param


@pytest.fixture
def contact_service() -> Any:
    def _contact_service(instance: Any) -> None:
        if isinstance(instance, ChatNVIDIA):
            instance.invoke("Hello")
        elif isinstance(instance, NVIDIAEmbeddings):
            instance.embed_documents(["Hello"])
        elif isinstance(instance, NVIDIARerank):
            instance.compress_documents(
                documents=[Document(page_content="World")], query="Hello"
            )
        elif isinstance(instance, NVIDIA):
            instance.invoke("Hello")

    return _contact_service
