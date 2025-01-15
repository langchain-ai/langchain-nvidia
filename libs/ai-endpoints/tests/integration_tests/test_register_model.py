import warnings
from typing import Any

import pytest

from langchain_nvidia_ai_endpoints import (
    NVIDIA,
    ChatNVIDIA,
    Model,
    NVIDIAEmbeddings,
    NVIDIARerank,
    register_model,
)


#
# if this test is failing it may be because the function uuids have changed.
# you will have to find the new ones from https://api.nvcf.nvidia.com/v2/nvcf/functions
#
@pytest.mark.parametrize(
    "client, id, endpoint",
    [
        (
            ChatNVIDIA,
            "meta/llama3-8b-instruct",
            "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/a5a3ad64-ec2c-4bfc-8ef7-5636f26630fe",
        ),
        (
            NVIDIAEmbeddings,
            "NV-Embed-QA",
            "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/09c64e32-2b65-4892-a285-2f585408d118",
        ),
        (
            NVIDIARerank,
            "nv-rerank-qa-mistral-4b:1",
            "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/0bf77f50-5c35-4488-8e7a-f49bb1974af6",
        ),
        (
            NVIDIA,
            "bigcode/starcoder2-15b",
            "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/d9cfe8a2-44df-44a0-ba51-3fc4a202c11c",
        ),
    ],
)
def test_registered_model_functional(
    client: type, id: str, endpoint: str, contact_service: Any
) -> None:
    model = Model(id=id, endpoint=endpoint)
    warnings.filterwarnings(
        "ignore", r".*is already registered.*"
    )  # intentionally overridding known models
    warnings.filterwarnings(
        "ignore", r".*Unable to determine validity of.*"
    )  # we aren't passing client & type to Model()
    register_model(model)
    contact_service(client(model=id))
