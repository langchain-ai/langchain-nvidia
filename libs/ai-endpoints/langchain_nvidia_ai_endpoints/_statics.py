from typing import Optional

from langchain_core.pydantic_v1 import BaseModel


#
# Model information
#  - id: unique identifier for the model
#  - model_type: API type (chat, vlm, embedding, ranking, completion)
#  - model_name: name passed as model parameter for requests
#  - client: client name
#  - endpoint: API endpoint
#  - deprecated: whether the model is deprecated
#  - aliases: list of aliases for the model
#
# If a model is deprecated and the model_name is set, the model with
# the model_name is be used instead and a warning is emitted.
#
# If a model is deprecated and the model_name is not set, the model's
# endpoint is used and a warning is emitted. These models have no
# alternative and will be removed in the future.
#
# If a model is deprecated, the model_name is not set, and there
# is no endpoint, the model is not available and an error is raised.
# On 20 May 2024, this applies to smaug and mamba_chat.
#
# model = lookup_model(name)
# if model.deprecated:
#    if model.model_name:
#        model = lookup_model(model.model_name)
#        warn(f"Model {name} is deprecated. Using {model.id} instead.")
#    else:
#        if model.endpoint:
#            warn(f"Model {name} is deprecated.")
#        else:
#            raise ValueError(f"Model {name} is no longer available.")
#
# Model lookup by name -
#  if name in model_table, use it
#  if name not in model_table,
#     if name is alias for a model (there should only be one),
#        use model w/ alias
#        warn that the alias is deprecated
#     else
#        return None
#
#  caller of lookup can decide if None means an error (unknown model)
#  or if there is enough additional information to proceed, e.g.
#  name and endpoint to invoke.
#
class Model(BaseModel):
    id: str
    model_type: Optional[str] = None
    model_name: Optional[str] = None
    client: Optional[str] = None
    endpoint: Optional[str] = None
    deprecated: Optional[bool] = False
    aliases: Optional[list] = None


CHAT_MODEL_TABLE = {
    "playground_smaug_72b": Model(
        id="playground_smaug_72b",
        model_type="chat",
        model_name=None,
        client="ChatNVIDIA",
        endpoint=None,
        deprecated=True,
        aliases=["smaug_72b"],
    ),
    "playground_llama2_70b": Model(
        id="playground_llama2_70b",
        model_type="chat",
        model_name="meta/llama2-70b",
        client="ChatNVIDIA",
        endpoint="https://integrate.api.nvidia.com/v1",
        deprecated=True,
        aliases=["llama2_70b"],
    ),
    "playground_nemotron_qa_8b": Model(
        id="playground_nemotron_qa_8b",
        model_type="qa",
        model_name=None,
        client="ChatNVIDIA",
        endpoint="https://integrate.api.nvidia.com/v1",
        deprecated=True,
        aliases=["nemotron_qa_8b"],
    ),
    "playground_gemma_7b": Model(
        id="playground_gemma_7b",
        model_type="chat",
        model_name="google/gemma-7b",
        client="ChatNVIDIA",
        endpoint="https://integrate.api.nvidia.com/v1",
        deprecated=True,
        aliases=["gemma_7b"],
    ),
    "playground_mistral_7b": Model(
        id="playground_mistral_7b",
        model_type="chat",
        model_name="mistralai/mistral-7b-instruct-v0.2",
        client="ChatNVIDIA",
        endpoint="https://integrate.api.nvidia.com/v1",
        deprecated=True,
        aliases=["mistral_7b"],
    ),
    "playground_mamba_chat": Model(
        id="playground_mamba_chat",
        model_type="chat",
        model_name=None,
        client="ChatNVIDIA",
        endpoint=None,
        deprecated=True,
        aliases=["mamba_chat"],
    ),
    "playground_phi2": Model(
        id="playground_phi2",
        model_type="chat",
        model_name="microsoft/phi-3-mini-4k-instruct",
        client="ChatNVIDIA",
        endpoint="https://integrate.api.nvidia.com/v1",
        deprecated=True,
        aliases=["phi2"],
    ),
    "playground_nv_llama2_rlhf_70b": Model(
        id="playground_nv_llama2_rlhf_70b",
        model_type="chat",
        model_name=None,
        client="ChatNVIDIA",
        endpoint="https://integrate.api.nvidia.com/v1",
        deprecated=True,
        aliases=["nv_llama2_rlhf_70b"],
    ),
    "playground_yi_34b": Model(
        id="playground_yi_34b",
        model_type="chat",
        model_name=None,
        client="ChatNVIDIA",
        endpoint="https://integrate.api.nvidia.com/v1",
        deprecated=True,
        aliases=["yi_34b"],
    ),
    "playground_nemotron_steerlm_8b": Model(
        id="playground_nemotron_steerlm_8b",
        model_type="chat",
        model_name=None,
        client="ChatNVIDIA",
        endpoint="https://integrate.api.nvidia.com/v1",
        deprecated=True,
        aliases=["nemotron_steerlm_8b"],
    ),
    "playground_llama2_code_70b": Model(
        id="playground_llama2_code_70b",
        model_type="chat",
        model_name="meta/codellama-70b",
        client="ChatNVIDIA",
        endpoint="https://integrate.api.nvidia.com/v1",
        deprecated=True,
        aliases=["llama2_code_70b"],
    ),
    "playground_gemma_2b": Model(
        id="playground_gemma_2b",
        model_type="chat",
        model_name="google/gemma-2b",
        client="ChatNVIDIA",
        endpoint="https://integrate.api.nvidia.com/v1",
        deprecated=True,
        aliases=["gemma_2b"],
    ),
    "playground_mixtral_8x7b": Model(
        id="playground_mixtral_8x7b",
        model_type="chat",
        model_name="mistralai/mixtral-8x7b-instruct-v0.1",
        client="ChatNVIDIA",
        endpoint="https://integrate.api.nvidia.com/v1",
        deprecated=True,
        aliases=["mixtral_8x7b"],
    ),
    "playground_llama2_code_34b": Model(
        id="playground_llama2_code_34b",
        model_type="chat",
        model_name=None,
        client="ChatNVIDIA",
        endpoint="https://integrate.api.nvidia.com/v1",
        deprecated=True,
        aliases=["llama2_code_34b"],
    ),
    "playground_llama2_code_13b": Model(
        id="playground_llama2_code_13b",
        model_type="chat",
        model_name=None,
        client="ChatNVIDIA",
        endpoint="https://integrate.api.nvidia.com/v1",
        deprecated=True,
        aliases=["llama2_code_13b"],
    ),
    "playground_steerlm_llama_70b": Model(
        id="playground_steerlm_llama_70b",
        model_type="chat",
        model_name=None,
        client="ChatNVIDIA",
        endpoint="https://integrate.api.nvidia.com/v1",
        deprecated=True,
        aliases=["steerlm_llama_70b"],
    ),
    "playground_llama2_13b": Model(
        id="playground_llama2_13b",
        model_type="chat",
        model_name=None,
        client="ChatNVIDIA",
        endpoint="https://integrate.api.nvidia.com/v1",
        deprecated=True,
        aliases=["llama2_13b"],
    ),
    "meta/codellama-70b": Model(
        id="meta/codellama-70b",
        model_type="chat",
        model_name="meta/codellama-70b",
        client="ChatNVIDIA",
        endpoint="https://integrate.api.nvidia.com/v1",
        deprecated=False,
        aliases=["ai-codellama-70b"],
    ),
    "google/gemma-7b": Model(
        id="google/gemma-7b",
        model_type="chat",
        model_name="google/gemma-7b",
        client="ChatNVIDIA",
        endpoint="https://integrations.api.nvidia.com/v1",
        deprecated=False,
        aliases=["ai-gemma-7b"],
    ),
    "meta/llama2-70b": Model(
        id="meta/llama2-70b",
        model_type="chat",
        model_name="meta/llama2-70b",
        client="ChatNVIDIA",
        endpoint="https://integrate.api.nvidia.com/v1",
        deprecated=False,
        aliases=["ai-llama2-70b"],
    ),
    "mistralai/mistral-7b-instruct-v0.2": Model(
        id="mistralai/mistral-7b-instruct-v0.2",
        model_type="chat",
        model_name="mistralai/mistral-7b-instruct-v0.2",
        client="ChatNVIDIA",
        endpoint="https://integrate.api.nvidia.com/v1",
        deprecated=False,
        aliases=["ai-mistral-7b-instruct-v2"],
    ),
    "mistralai/mixtral-8x7b-instruct-v0.1": Model(
        id="mistralai/mixtral-8x7b-instruct-v0.1",
        model_type="chat",
        model_name="mistralai/mixtral-8x7b-instruct-v0.1",
        client="ChatNVIDIA",
        endpoint="https://integrate.api.nvidia.com/v1",
        deprecated=False,
        aliases=["ai-mixtral-8x7b-instruct"],
    ),
    "google/codegemma-7b": Model(
        id="google/codegemma-7b",
        model_type="chat",
        model_name="google/codegemma-7b",
        client="ChatNVIDIA",
        endpoint="https://integrations.api.nvidia.com/v1",
        deprecated=False,
        aliases=["ai-codegemma-7b"],
    ),
    "google/gemma-2b": Model(
        id="google/gemma-2b",
        model_type="chat",
        model_name="google/gemma-2b",
        client="ChatNVIDIA",
        endpoint="https://integrations.api.nvidia.com/v1",
        deprecated=False,
        aliases=["ai-gemma-2b"],
    ),
    "google/recurrentgemma-2b": Model(
        id="google/recurrentgemma-2b",
        model_type="chat",
        model_name="google/recurrentgemma-2b",
        client="ChatNVIDIA",
        endpoint="https://integrations.api.nvidia.com/v1",
        deprecated=False,
        aliases=["ai-recurrentgemma-2b"],
    ),
    "mistralai/mistral-large": Model(
        id="mistralai/mistral-large",
        model_type="chat",
        model_name="mistralai/mistral-large",
        client="ChatNVIDIA",
        endpoint="https://integrate.api.nvidia.com/v1",
        deprecated=False,
        aliases=["ai-mistral-large"],
    ),
    "mistralai/mixtral-8x22b-instruct-v0.1": Model(
        id="mistralai/mixtral-8x22b-instruct-v0.1",
        model_type="chat",
        model_name="mistralai/mixtral-8x22b-instruct-v0.1",
        client="ChatNVIDIA",
        endpoint="https://integrate.api.nvidia.com/v1",
        deprecated=False,
        aliases=["ai-mixtral-8x22b-instruct"],
    ),
    "meta/llama3-8b-instruct": Model(
        id="meta/llama3-8b-instruct",
        model_type="chat",
        model_name="meta/llama3-8b-instruct",
        client="ChatNVIDIA",
        endpoint="https://integrate.api.nvidia.com/v1",
        deprecated=False,
        aliases=["ai-llama3-8b-instruct"],
    ),
    "meta/llama3-70b-instruct": Model(
        id="meta/llama3-70b-instruct",
        model_type="chat",
        model_name="meta/llama3-70b-instruct",
        client="ChatNVIDIA",
        endpoint="https://integrate.api.nvidia.com/v1",
        deprecated=False,
        aliases=["ai-llama3-70b-instruct"],
    ),
    "microsoft/phi-3-mini-128k-instruct": Model(
        id="microsoft/phi-3-mini-128k-instruct",
        model_type="chat",
        model_name="microsoft/phi-3-mini-128k-instruct",
        client="ChatNVIDIA",
        endpoint="https://integrate.api.nvidia.com/v1",
        deprecated=False,
        aliases=["ai-phi-3-mini"],
    ),
    "snowflake/arctic": Model(
        id="snowflake/arctic",
        model_type="chat",
        model_name="snowflake/arctic",
        client="ChatNVIDIA",
        endpoint="https://integrate.api.nvidia.com/v1",
        deprecated=False,
        aliases=["ai-arctic"],
    ),
    "databricks/dbrx-instruct": Model(
        id="databricks/dbrx-instruct",
        model_type="chat",
        model_name="databricks/dbrx-instruct",
        client="ChatNVIDIA",
        endpoint="https://integrate.api.nvidia.com/v1",
        deprecated=False,
        aliases=["ai-dbrx-instruct"],
    ),
    "microsoft/phi-3-mini-4k-instruct": Model(
        id="microsoft/phi-3-mini-4k-instruct",
        model_type="chat",
        model_name="microsoft/phi-3-mini-4k-instruct",
        client="ChatNVIDIA",
        endpoint="https://integrate.api.nvidia.com/v1",
        deprecated=False,
        aliases=["ai-phi-3-mini-4k"],
    ),
    "seallms/seallm-7b-v2.5": Model(
        id="seallms/seallm-7b-v2.5",
        model_type="chat",
        model_name="seallms/seallm-7b-v2.5",
        client="ChatNVIDIA",
        endpoint="https://integrate.api.nvidia.com/v1",
        deprecated=False,
        aliases=["ai-seallm-7b"],
    ),
    "aisingapore/sea-lion-7b-instruct": Model(
        id="aisingapore/sea-lion-7b-instruct",
        model_type="chat",
        model_name="aisingapore/sea-lion-7b-instruct",
        client="ChatNVIDIA",
        endpoint="https://integrate.api.nvidia.com/v1",
        deprecated=False,
        aliases=["ai-sea-lion-7b-instruct"],
    ),
}

VLM_MODEL_TABLE = {
    "playground_fuyu_8b": Model(
        id="playground_fuyu_8b",
        model_type="vlm",
        model_name="adept/fuyu-8b",
        client="ChatNVIDIA",
        endpoint=None,
        deprecated=True,
        aliases=["fuyu_8b"],
    ),
    "playground_deplot": Model(
        id="playground_deplot",
        model_type="vlm",
        model_name="google/deplot",
        client="ChatNVIDIA",
        endpoint=None,
        deprecated=True,
        aliases=["deplot"],
    ),
    "playground_kosmos_2": Model(
        id="playground_kosmos_2",
        model_type="vlm",
        model_name="microsoft/kosmos-2",
        client="ChatNVIDIA",
        endpoint=None,
        deprecated=True,
        aliases=["kosmos_2"],
    ),
    "playground_neva_22b": Model(
        id="playground_neva_22b",
        model_type="vlm",
        model_name="nvidia/neva-22b",
        client="ChatNVIDIA",
        endpoint=None,
        deprecated=True,
        aliases=["neva_22b"],
    ),
    "adept/fuyu-8b": Model(
        id="adept/fuyu-8b",
        model_type="vlm",
        model_name="adept/fuyu-8b",
        client="ChatNVIDIA",
        endpoint="https://ai.api.nvidia.com/v1/vlm/adept/fuyu-8b",
        deprecated=False,
        aliases=["ai-fuyu-8b"],
    ),
    "google/deplot": Model(
        id="google/deplot",
        model_type="vlm",
        model_name="google/deplot",
        client="ChatNVIDIA",
        endpoint="https://ai.api.nvidia.com/v1/vlm/google/deplot",
        deprecated=False,
        aliases=["ai-google-deplot"],
    ),
    "microsoft/kosmos-2": Model(
        id="microsoft/kosmos-2",
        model_type="vlm",
        model_name="microsoft/kosmos-2",
        client="ChatNVIDIA",
        endpoint="https://ai.api.nvidia.com/v1/vlm/microsoft/kosmos-2",
        deprecated=False,
        aliases=["ai-microsoft-kosmos-2"],
    ),
    "nvidia/neva-22b": Model(
        id="nvidia/neva-22b",
        model_type="vlm",
        model_name="nvidia/neva-22b",
        client="ChatNVIDIA",
        endpoint="https://ai.api.nvidia.com/v1/vlm/nvidia/neva-22b",
        deprecated=False,
        aliases=["ai-neva-22b"],
    ),
    "google/paligemma": Model(
        id="google/paligemma",
        model_type="vlm",
        model_name="google/paligemma",
        client="ChatNVIDIA",
        endpoint="https://ai.api.nvidia.com/v1/vlm/google/paligemma",
        deprecated=False,
        aliases=["ai-google-paligemma"],
    ),
}

EMBEDDING_MODEL_TABLE = {
    "playground_nvolveqa_40k": Model(
        id="playground_nvolveqa_40k",
        model_type="embedding",
        model_name=None,
        client="NVIDIAEmbeddings",
        endpoint=None,
        deprecated=True,
        aliases=["nvolveqa_40k"],
    ),
    "NV-Embed-QA": Model(
        id="NV-Embed-QA",
        model_type="embedding",
        model_name="NV-Embed-QA",
        client="NVIDIAEmbeddings",
        endpoint="https://ai.api.nvidia.com/v1/retrieval/nvidia",
        deprecated=False,
        aliases=["ai-embed-qa-4"],
    ),
    "snowflake/arctic-embed-l": Model(
        id="snowflake/arctic-embed-l",
        model_type="embedding",
        model_name="snowflake/arctic-embed-l",
        client="NVIDIAEmbeddings",
        endpoint="https://integrate.api.nvidia.com/v1",
        deprecated=False,
        aliases=["ai-arctic-embed-l"],
    ),
}

RANKING_MODEL_TABLE = {
    "nv-rerank-qa-mistral-4b:1": Model(
        id="nv-rerank-qa-mistral-4b:1",
        model_type="ranking",
        model_name="nv-rerank-qa-mistral-4b:1",
        client="NVIDIARerank",
        endpoint="https://ai.api.nvidia.com/v1/retrieval/nvidia",
        deprecated=False,
        aliases=["ai-rerank-qa-mistral-4b"],
    ),
}

COMPLETION_MODEL_TABLE = {
    "playground_starcode_15b": Model(
        id="playground_starcode_15b",
        model_type="completion",
        model_name=None,
        client="NVIDIA",
        endpoint=None,
        deprecated=True,
        aliases=["starcode_15b"],
    ),
    "mistralai/mixtral-8x22b-v0.1": Model(
        id="mistralai/mixtral-8x22b-v0.1",
        model_type="completion",
        model_name="mistralai/mixtral-8x22b-v0.1",
        client="NVIDIA",
        endpoint="https://integrate.api.nvidia.com/v1",
        deprecated=False,
        aliases=["ai-mixtral-8x22b"],
    ),
}

MODEL_TABLE = {
    **CHAT_MODEL_TABLE,
    **VLM_MODEL_TABLE,
    **EMBEDDING_MODEL_TABLE,
    **RANKING_MODEL_TABLE,
}
