from langchain_core.load.dump import dumps
from langchain_core.load.load import loads

from langchain_nvidia_ai_endpoints import ChatNVIDIA


def test_serialize_chatnvidia() -> None:
    secret = "a-bogus-key"
    x = ChatNVIDIA(nvidia_api_key=secret)
    y = loads(
        dumps(x),
        secrets_map={"NVIDIA_API_KEY": secret},
        valid_namespaces=["langchain_nvidia_ai_endpoints"],
    )
    assert x == y
    assert isinstance(y, ChatNVIDIA)
