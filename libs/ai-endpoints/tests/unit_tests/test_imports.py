from langchain_nvidia import __all__ as short_all
from langchain_nvidia_ai_endpoints import __all__ as long_all

EXPECTED_ALL = [
    "ChatNVIDIA",
    "ChatNVIDIADynamo",
    "NVIDIAEmbeddings",
    "NVIDIARerank",
    "NVIDIA",
    "register_model",
    "Model",
]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(short_all)
    assert sorted(EXPECTED_ALL) == sorted(long_all)
