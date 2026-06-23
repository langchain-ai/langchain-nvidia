from langchain_nvidia import __all__ as short_all
from langchain_nvidia_ai_endpoints import __all__ as long_all
from langchain_nvidia_ai_endpoints import __version__

EXPECTED_ALL = [
    "__version__",
    "ChatNVIDIA",
    "ChatNVIDIADynamo",
    "NVIDIAEmbeddings",
    "NVIDIARerank",
    "NVIDIA",
    "NVIDIARAGRetriever",
    "register_model",
    "Model",
    "inference_priority",
    "get_inference_priority",
]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(short_all)
    assert sorted(EXPECTED_ALL) == sorted(long_all)


def test_version_import() -> None:
    assert __version__ == "1.4.2"
