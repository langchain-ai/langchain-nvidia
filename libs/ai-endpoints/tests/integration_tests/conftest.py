from langchain_nvidia_ai_endpoints import ChatNVIDIA

import pytest


def get_mode(config):
    nim_endpoint = config.getoption("--nim-endpoint")
    if nim_endpoint:
        return dict(mode="nim", base_url=nim_endpoint)
    return {}

def pytest_addoption(parser):
    parser.addoption("--chat-model-id", action="store", help="Run tests for a specific chat model",)
    parser.addoption("--all-models", action="store_true", help="Run tests across all models",)
    parser.addoption("--nim-endpoint", type=str, help="Run tests using NIM mode",)


def pytest_generate_tests(metafunc):
    mode = get_mode(metafunc.config)
    available_models = ChatNVIDIA().mode(**mode).get_available_models(list_all=True, **mode)

    if "chat_model" in metafunc.fixturenames:
        models = [metafunc.config.getoption("chat_model_id", "llama2_13b")]
        if metafunc.config.getoption("all_models"):
            models = [model.id for model in available_models if model.model_type == "chat"]
        metafunc.parametrize("chat_model", models, ids=models)

    if "image_in_model" in metafunc.fixturenames:
        models = ["fuyu_8b"]
        if metafunc.config.getoption("all_models"):
            models = [model.id for model in available_models if model.model_type == "image_in"]
        metafunc.parametrize("image_in_model", models, ids=models)

    if "qa_model" in metafunc.fixturenames:
        models = ["nemotron_qa_8b"]
        if metafunc.config.getoption("all_models"):
            models = [model.id for model in available_models if model.model_type == "qa"]
        metafunc.parametrize("qa_model", models, ids=models)

    if "embedding_model" in metafunc.fixturenames:
        models = ["nvolveqa_40k"]
        if metafunc.config.getoption("all_models"):
            models = [model.id for model in available_models if model.model_type == "embedding"]
        metafunc.parametrize("embedding_model", models, ids=models)


@pytest.fixture
def mode(request):
    return get_mode(request.config)
