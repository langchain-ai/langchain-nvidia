import pickle

from langchain_core.load.dump import dumps
from langchain_core.load.load import loads

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings


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


def test_pickle_embeddings() -> None:
    x = NVIDIAEmbeddings()
    y = pickle.loads(pickle.dumps(x))
    assert x.model == y.model
    assert x.max_batch_size == y.max_batch_size
    assert isinstance(y, NVIDIAEmbeddings)
