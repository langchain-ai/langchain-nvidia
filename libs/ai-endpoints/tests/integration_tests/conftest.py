from langchain_nvidia_ai_endpoints import ChatNVIDIA

def pytest_addoption(parser):
    parser.addoption("--all-models", action="store_true", help="Run tests across all models",)


def pytest_generate_tests(metafunc):
    if "chat_model" in metafunc.fixturenames:
        models = ["llama2_13b"]
        if metafunc.config.getoption("all_models"):
            models = [model.id for model in ChatNVIDIA.get_available_models() if model.model_type == "chat"]
        metafunc.parametrize("chat_model", models, ids=models)
