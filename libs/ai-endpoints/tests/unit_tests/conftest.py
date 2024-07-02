import pytest

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank


@pytest.fixture(
    params=[
        ChatNVIDIA,
        NVIDIAEmbeddings,
        NVIDIARerank,
    ]
)
def public_class(request: pytest.FixtureRequest) -> type:
    return request.param
