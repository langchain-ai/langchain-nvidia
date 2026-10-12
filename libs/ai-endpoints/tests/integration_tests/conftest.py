from typing import Any, Generator, List, Sequence, cast

import pytest
from langchain_core.documents import Document

from langchain_nvidia_ai_endpoints import (
    NVIDIA,
    ChatNVIDIA,
    NVIDIAEmbeddings,
    NVIDIARerank,
)
from langchain_nvidia_ai_endpoints._statics import (
    MODEL_TABLE,
    RANKING_VLM_MODEL_TABLE,
    Model,
)
from langchain_nvidia_ai_endpoints.llm import (
    _DEFAULT_MODEL_NAME as DEFAULT_COMPLETIONS_MODEL,
)
from langchain_nvidia_ai_endpoints.reranking import (
    _DEFAULT_VLM_MODEL_NAME as DEFAULT_RERANKING_VLM_MODEL,
)
from tests.integration_tests.smoke_models import (
    HOSTED_EOL_MODEL_IDS,
    SMOKE_CHAT_MODEL,
    SMOKE_EMBEDDING_MODEL,
    SMOKE_REASONING_MODEL,
    SMOKE_RERANKING_MODEL,
    SMOKE_VLM_MODEL,
)


def get_mode(config: pytest.Config) -> dict:
    nim_endpoint = config.getoption("--nim-endpoint")
    if nim_endpoint:
        return dict(base_url=nim_endpoint)
    return {}


def _model_id(model: str | Model) -> str:
    return model.id if isinstance(model, Model) else model


def _filter_hosted_eol_models(
    config: pytest.Config, models: Sequence[str | Model]
) -> List[str]:
    model_ids = [_model_id(model) for model in models]
    if config.getoption("--nim-endpoint"):
        return model_ids
    return [model for model in model_ids if model not in HOSTED_EOL_MODEL_IDS]


def _is_hosted_eol_error(config: pytest.Config, exc: BaseException) -> bool:
    if config.getoption("--nim-endpoint"):
        return False
    message = str(exc).lower()
    return (
        "[410]" in message
        and "gone" in message
        and ("end of life" in message or "no longer available" in message)
    )


def _is_hosted_unavailable_rerank_error(item: pytest.Item, exc: BaseException) -> bool:
    if item.config.getoption("--nim-endpoint") or "rerank_model" not in getattr(
        item, "fixturenames", ()
    ):
        return False
    message = str(exc).lower()
    return "[404]" in message and "not found" in message


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo
) -> Generator[None, None, None]:
    outcome = yield
    report = cast(Any, outcome).get_result()
    if call.when != "call" or not report.failed or call.excinfo is None:
        return
    if _is_hosted_eol_error(item.config, call.excinfo.value):
        report.outcome = "skipped"
        report.longrepr = (
            str(item.path),
            item.location[1],
            "Skipped: hosted model endpoint is retired or no longer available",
        )
    if _is_hosted_unavailable_rerank_error(item, call.excinfo.value):
        report.outcome = "skipped"
        report.longrepr = (
            str(item.path),
            item.location[1],
            "Skipped: hosted reranking model endpoint is unavailable",
        )


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--chat-model-id",
        action="store",
        nargs="+",
        help="Run tests for a specific chat model or list of models",
    )
    parser.addoption(
        "--reasoning-model-id",
        action="store",
        nargs="+",
        help=("Run reasoning-content tests for a specific model or list of models"),
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
        "--rerank-vlm-model-id",
        action="store",
        nargs="+",
        help="Run tests for a specific rerank VLM model or list of models",
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

    if "reasoning_model" in metafunc.fixturenames:
        models = [SMOKE_REASONING_MODEL]
        if model_list := metafunc.config.getoption("reasoning_model_id"):
            models = model_list
        else:
            models = _filter_hosted_eol_models(metafunc.config, models)
        metafunc.parametrize("reasoning_model", models, ids=models)

    if "thinking_model" in metafunc.fixturenames:
        models = [SMOKE_CHAT_MODEL]
        if model_list := metafunc.config.getoption("thinking_model_id"):
            models = model_list
        if metafunc.config.getoption("all_models"):
            models = [
                model.id
                for model in ChatNVIDIA(**mode).available_models
                if model.supports_thinking
            ]
            models = _filter_hosted_eol_models(metafunc.config, models)
        metafunc.parametrize("thinking_model", models, ids=models)

    if "chat_model" in metafunc.fixturenames:
        models = [SMOKE_CHAT_MODEL]
        if model_list := metafunc.config.getoption("chat_model_id"):
            models = model_list
        if metafunc.config.getoption("all_models"):
            models = [
                model.id
                for model in ChatNVIDIA(**mode).available_models
                if model.model_type == "chat"
            ]
            models = _filter_hosted_eol_models(metafunc.config, models)
        metafunc.parametrize("chat_model", models, ids=models)

    if "tool_model" in metafunc.fixturenames:
        models = [SMOKE_CHAT_MODEL]
        if model_list := metafunc.config.getoption("tool_model_id"):
            models = model_list
        if metafunc.config.getoption("all_models"):
            models = [
                model.id
                for model in ChatNVIDIA(**mode).available_models
                if model.model_type == "chat" and model.supports_tools
            ]
            models = _filter_hosted_eol_models(metafunc.config, models)
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
        models = [SMOKE_CHAT_MODEL]
        if model_list := metafunc.config.getoption("structured_model_id"):
            models = model_list
        if metafunc.config.getoption("all_models"):
            models = [
                model.id
                for model in ChatNVIDIA(**mode).available_models
                if model.supports_structured_output
            ]
            models = _filter_hosted_eol_models(metafunc.config, models)
        metafunc.parametrize("structured_model", models, ids=models)

    if "rerank_model" in metafunc.fixturenames:
        models = [SMOKE_RERANKING_MODEL]
        if model_list := metafunc.config.getoption("rerank_model_id"):
            models = model_list
        if metafunc.config.getoption("all_models"):
            models = [model.id for model in NVIDIARerank(**mode).available_models]
            models = _filter_hosted_eol_models(metafunc.config, models)
        metafunc.parametrize("rerank_model", models, ids=models)

    if "rerank_vlm_model" in metafunc.fixturenames:
        models = [DEFAULT_RERANKING_VLM_MODEL]
        if model_list := metafunc.config.getoption("rerank_vlm_model_id"):
            models = model_list
        if metafunc.config.getoption("all_models"):
            models = list(RANKING_VLM_MODEL_TABLE.keys())
        metafunc.parametrize("rerank_vlm_model", models, ids=models)

    if "vlm_model" in metafunc.fixturenames:
        models = [SMOKE_VLM_MODEL]
        if model_list := metafunc.config.getoption("vlm_model_id"):
            models = model_list
        if metafunc.config.getoption("all_models"):
            models = [
                model.id
                for model in get_all_known_models()
                if model.model_type in {"vlm", "nv-vlm"}
            ]
            models = _filter_hosted_eol_models(metafunc.config, models)
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
        models = [SMOKE_EMBEDDING_MODEL]
        if metafunc.config.getoption("all_models"):
            models = [model.id for model in NVIDIAEmbeddings(**mode).available_models]
            models = _filter_hosted_eol_models(metafunc.config, models)
        if model_list := metafunc.config.getoption("embedding_model_id"):
            models = model_list
        if metafunc.config.getoption("all_models"):
            models = [model.id for model in NVIDIAEmbeddings(**mode).available_models]
            models = _filter_hosted_eol_models(metafunc.config, models)
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
def smoke_model_kwargs(public_class: type, mode: dict) -> dict[str, str]:
    if "base_url" in mode:
        return {}
    smoke_kwargs: dict[Any, dict[str, str]] = {
        ChatNVIDIA: {"model": SMOKE_CHAT_MODEL},
        NVIDIAEmbeddings: {"model": SMOKE_EMBEDDING_MODEL},
        NVIDIARerank: {"model": SMOKE_RERANKING_MODEL},
    }
    return smoke_kwargs.get(public_class, {})


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
