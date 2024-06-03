import warnings
from typing import Optional

from langchain_core.pydantic_v1 import BaseModel


#
# Model information
#  - id: unique identifier for the model, passed as model parameter for requests
#  - model_type: API type (chat, vlm, embedding, ranking, completion)
#  - client: client name
#  - endpoint: custom endpoint for the model
#  - aliases: list of aliases for the model
#
# All aliases are deprecated and will trigger a warning when used.
#
class Model(BaseModel):
    id: str
    model_type: Optional[str] = None
    client: Optional[str] = None
    endpoint: Optional[str] = None
    aliases: Optional[list] = None

    def __hash__(self) -> int:
        return hash(self.id)


CHAT_MODEL_TABLE = {
    "meta/codellama-70b": Model(
        id="meta/codellama-70b",
        model_type="chat",
        client="ChatNVIDIA",
        aliases=[
            "ai-codellama-70b",
            "playground_llama2_code_70b",
            "llama2_code_70b",
            "playground_llama2_code_34b",
            "llama2_code_34b",
            "playground_llama2_code_13b",
            "llama2_code_13b",
        ],
    ),
    "google/gemma-7b": Model(
        id="google/gemma-7b",
        model_type="chat",
        client="ChatNVIDIA",
        aliases=["ai-gemma-7b", "playground_gemma_7b", "gemma_7b"],
    ),
    "meta/llama2-70b": Model(
        id="meta/llama2-70b",
        model_type="chat",
        client="ChatNVIDIA",
        aliases=[
            "ai-llama2-70b",
            "playground_llama2_70b",
            "llama2_70b",
            "playground_llama2_13b",
            "llama2_13b",
        ],
    ),
    "mistralai/mistral-7b-instruct-v0.2": Model(
        id="mistralai/mistral-7b-instruct-v0.2",
        model_type="chat",
        client="ChatNVIDIA",
        aliases=["ai-mistral-7b-instruct-v2", "playground_mistral_7b", "mistral_7b"],
    ),
    "mistralai/mixtral-8x7b-instruct-v0.1": Model(
        id="mistralai/mixtral-8x7b-instruct-v0.1",
        model_type="chat",
        client="ChatNVIDIA",
        aliases=["ai-mixtral-8x7b-instruct", "playground_mixtral_8x7b", "mixtral_8x7b"],
    ),
    "google/codegemma-7b": Model(
        id="google/codegemma-7b",
        model_type="chat",
        client="ChatNVIDIA",
        aliases=["ai-codegemma-7b"],
    ),
    "google/gemma-2b": Model(
        id="google/gemma-2b",
        model_type="chat",
        client="ChatNVIDIA",
        aliases=["ai-gemma-2b", "playground_gemma_2b", "gemma_2b"],
    ),
    "google/recurrentgemma-2b": Model(
        id="google/recurrentgemma-2b",
        model_type="chat",
        client="ChatNVIDIA",
        aliases=["ai-recurrentgemma-2b"],
    ),
    "mistralai/mistral-large": Model(
        id="mistralai/mistral-large",
        model_type="chat",
        client="ChatNVIDIA",
        aliases=["ai-mistral-large"],
    ),
    "mistralai/mixtral-8x22b-instruct-v0.1": Model(
        id="mistralai/mixtral-8x22b-instruct-v0.1",
        model_type="chat",
        client="ChatNVIDIA",
        aliases=["ai-mixtral-8x22b-instruct"],
    ),
    "meta/llama3-8b-instruct": Model(
        id="meta/llama3-8b-instruct",
        model_type="chat",
        client="ChatNVIDIA",
        aliases=["ai-llama3-8b"],
    ),
    "meta/llama3-70b-instruct": Model(
        id="meta/llama3-70b-instruct",
        model_type="chat",
        client="ChatNVIDIA",
        aliases=["ai-llama3-70b"],
    ),
    "microsoft/phi-3-mini-128k-instruct": Model(
        id="microsoft/phi-3-mini-128k-instruct",
        model_type="chat",
        client="ChatNVIDIA",
        aliases=["ai-phi-3-mini"],
    ),
    "snowflake/arctic": Model(
        id="snowflake/arctic",
        model_type="chat",
        client="ChatNVIDIA",
        aliases=["ai-arctic"],
    ),
    "databricks/dbrx-instruct": Model(
        id="databricks/dbrx-instruct",
        model_type="chat",
        client="ChatNVIDIA",
        aliases=["ai-dbrx-instruct"],
    ),
    "microsoft/phi-3-mini-4k-instruct": Model(
        id="microsoft/phi-3-mini-4k-instruct",
        model_type="chat",
        client="ChatNVIDIA",
        aliases=["ai-phi-3-mini-4k", "playground_phi2", "phi2"],
    ),
    "seallms/seallm-7b-v2.5": Model(
        id="seallms/seallm-7b-v2.5",
        model_type="chat",
        client="ChatNVIDIA",
        aliases=["ai-seallm-7b"],
    ),
    "aisingapore/sea-lion-7b-instruct": Model(
        id="aisingapore/sea-lion-7b-instruct",
        model_type="chat",
        client="ChatNVIDIA",
        aliases=["ai-sea-lion-7b-instruct"],
    ),
    "microsoft/phi-3-small-8k-instruct": Model(
        id="microsoft/phi-3-small-8k-instruct",
        model_type="chat",
        client="ChatNVIDIA",
        aliases=["ai-phi-3-small-8k-instruct"],
    ),
    "microsoft/phi-3-small-128k-instruct": Model(
        id="microsoft/phi-3-small-128k-instruct",
        model_type="chat",
        client="ChatNVIDIA",
        aliases=["ai-phi-3-small-128k-instruct"],
    ),
    "microsoft/phi-3-medium-4k-instruct": Model(
        id="microsoft/phi-3-medium-4k-instruct",
        model_type="chat",
        client="ChatNVIDIA",
        aliases=["ai-phi-3-medium-4k-instruct"],
    ),
    "ibm/granite-8b-code-instruct": Model(
        id="ibm/granite-8b-code-instruct",
        model_type="chat",
        client="ChatNVIDIA",
        aliases=["ai-granite-8b-code-instruct"],
    ),
    "ibm/granite-34b-code-instruct": Model(
        id="ibm/granite-34b-code-instruct",
        model_type="chat",
        client="ChatNVIDIA",
        aliases=["ai-granite-34b-code-instruct"],
    ),
    "google/codegemma-1.1-7b": Model(
        id="google/codegemma-1.1-7b",
        model_type="chat",
        client="ChatNVIDIA",
        aliases=["ai-codegemma-1.1-7b"],
    ),
    "mediatek/breeze-7b-instruct": Model(
        id="mediatek/breeze-7b-instruct",
        model_type="chat",
        client="ChatNVIDIA",
        aliases=["ai-breeze-7b-instruct"],
    ),
}

VLM_MODEL_TABLE = {
    "adept/fuyu-8b": Model(
        id="adept/fuyu-8b",
        model_type="vlm",
        client="ChatNVIDIA",
        endpoint="https://ai.api.nvidia.com/v1/vlm/adept/fuyu-8b",
        aliases=["ai-fuyu-8b", "playground_fuyu_8b", "fuyu_8b"],
    ),
    "google/deplot": Model(
        id="google/deplot",
        model_type="vlm",
        client="ChatNVIDIA",
        endpoint="https://ai.api.nvidia.com/v1/vlm/google/deplot",
        aliases=["ai-google-deplot", "playground_deplot", "deplot"],
    ),
    "microsoft/kosmos-2": Model(
        id="microsoft/kosmos-2",
        model_type="vlm",
        client="ChatNVIDIA",
        endpoint="https://ai.api.nvidia.com/v1/vlm/microsoft/kosmos-2",
        aliases=["ai-microsoft-kosmos-2", "playground_kosmos_2", "kosmos_2"],
    ),
    "nvidia/neva-22b": Model(
        id="nvidia/neva-22b",
        model_type="vlm",
        client="ChatNVIDIA",
        endpoint="https://ai.api.nvidia.com/v1/vlm/nvidia/neva-22b",
        aliases=["ai-neva-22b", "playground_neva_22b", "neva_22b"],
    ),
    "google/paligemma": Model(
        id="google/paligemma",
        model_type="vlm",
        client="ChatNVIDIA",
        endpoint="https://ai.api.nvidia.com/v1/vlm/google/paligemma",
        aliases=["ai-google-paligemma"],
    ),
    "microsoft/phi-3-vision-128k-instruct": Model(
        id="microsoft/phi-3-vision-128k-instruct",
        model_type="vlm",
        client="ChatNVIDIA",
        endpoint="https://ai.api.nvidia.com/v1/vlm/microsoft/phi-3-vision-128k-instruct",
        aliases=["ai-phi-3-vision-128k-instruct"],
    ),
}

EMBEDDING_MODEL_TABLE = {
    "snowflake/arctic-embed-l": Model(
        id="snowflake/arctic-embed-l",
        model_type="embedding",
        client="NVIDIAEmbeddings",
        aliases=["ai-arctic-embed-l"],
    ),
    "NV-Embed-QA": Model(
        id="NV-Embed-QA",
        model_type="embedding",
        client="NVIDIAEmbeddings",
        endpoint="https://ai.api.nvidia.com/v1/retrieval/nvidia/embeddings",
        aliases=[
            "ai-embed-qa-4",
            "playground_nvolveqa_40k",
            "nvolveqa_40k",
        ],
    ),
}

RANKING_MODEL_TABLE = {
    "nv-rerank-qa-mistral-4b:1": Model(
        id="nv-rerank-qa-mistral-4b:1",
        model_type="ranking",
        client="NVIDIARerank",
        endpoint="https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking",
        aliases=["ai-rerank-qa-mistral-4b"],
    ),
}

COMPLETION_MODEL_TABLE = {
    "mistralai/mixtral-8x22b-v0.1": Model(
        id="mistralai/mixtral-8x22b-v0.1",
        model_type="completion",
        client="NVIDIA",
        aliases=["ai-mixtral-8x22b"],
    ),
}

MODEL_TABLE = {
    **CHAT_MODEL_TABLE,
    **VLM_MODEL_TABLE,
    **EMBEDDING_MODEL_TABLE,
    **RANKING_MODEL_TABLE,
}


def lookup_model(name: str) -> Optional[Model]:
    """
    Lookup a model by name, using only the table of known models.
    The name is either:
        - directly in the table
        - an alias in the table
        - not found (None)
    Callers can check to see if the name was an alias by
    comparing the result's id field to the name they provided.
    """
    model = None
    if not (model := MODEL_TABLE.get(name)):
        for mdl in MODEL_TABLE.values():
            if mdl.aliases and name in mdl.aliases:
                model = mdl
                break
    return model


def determine_model(name: str) -> Optional[Model]:
    """
    Determine the model to use based on a name, using
    only the table of known models.

    Raise a warning if the model is found to be
    an alias of a known model.

    If the model is not found, return None.
    """
    if model := lookup_model(name):
        # all aliases are deprecated
        if model.id != name:
            warnings.warn(
                f"Model {name} is deprecated. Using {model.id} instead.", UserWarning
            )
    return model
