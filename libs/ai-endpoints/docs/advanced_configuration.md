# Advanced configuration

This page collects connector details that most applications do not need on day
one: how base URLs are resolved, the read-only Doctor preflight, connecting to
several self-hosted NIM microservices, custom inference endpoints, and the
optional maintainer-only compatibility evidence lookup. For a first working
example, start with the [package README](../README.md).

## About NVIDIA AI Foundation Models and NIM

The `langchain-nvidia-ai-endpoints` package contains LangChain integrations for chat models and embeddings powered by [NVIDIA AI Foundation Models](https://www.nvidia.com/en-us/ai-data-science/foundation-models/), and hosted on the [NVIDIA API Catalog](https://build.nvidia.com/).

A strong starting point is [Nemotron](https://www.nvidia.com/en-us/ai-data-science/foundation-models/nemotron/), NVIDIA's open model family built for agentic AI. Nemotron models use a hybrid Mamba-Transformer mixture-of-experts architecture that delivers leading benchmark accuracy with high throughput and up to 1M token context windows. Model weights, training data, and implementation recipes are published openly under the NVIDIA Open Model License.

NVIDIA AI Foundation models are community- and NVIDIA-built models that are optimized to deliver the best performance on NVIDIA-accelerated infrastructure. 
You can use the API to query live endpoints that are available on the NVIDIA API Catalog to get quick results from a DGX-hosted cloud compute environment. 
or you can download models from NVIDIA's API catalog with NVIDIA NIM, which is included with the NVIDIA AI Enterprise license. 
The ability to run models on-premises gives your enterprise ownership of your customizations and full control of your IP and AI application. 

NIM microservices are packaged as container images on a per model/model family basis 
and are distributed as NGC container images through the [NVIDIA NGC Catalog](https://catalog.ngc.nvidia.com/). 
At their core, NIM microservices are containers that provide interactive APIs for running inference on an AI Model. 

Use this documentation to learn how to install the `langchain-nvidia-ai-endpoints` package 
and use it for some common functionality for text-generative and embedding models.

## Get and verify an NVIDIA API key

To get access to the NVIDIA API Catalog, do the following:

1. Create a free account on the [NVIDIA API Catalog](https://build.nvidia.com/) and log in.
2. Click your profile icon, and then click **API Keys**. The **API Keys** page appears.
3. Click **Generate API Key**. The **Generate API Key** window appears.
4. Click **Generate Key**.  You should see **API Key Granted**, and your key appears.
5. Copy and save the key as `NVIDIA_API_KEY`.
6. To verify your key, use the following code.

    ```python
    import getpass
    import os

    if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
        nvidia_api_key = getpass.getpass("Enter your NVIDIA API key: ")
        assert nvidia_api_key.startswith("nvapi-"), f"{nvidia_api_key[:5]}... is not a valid key"
        os.environ["NVIDIA_API_KEY"] = nvidia_api_key
    ```

You can now use your key to access endpoints on the NVIDIA API Catalog.

## Choose a deployment and base URL

Hosted NVIDIA API Catalog calls default to `https://integrate.api.nvidia.com/v1`
and **require** `NVIDIA_API_KEY`. Set it in your environment (do not put it in
source code or a URL). A self-hosted NIM can use its deployment root
(`http://localhost:8000`) or its OpenAI-compatible API base
(`http://localhost:8000/v1`); both resolve to `/v1`. Set `NVIDIA_BASE_URL`
or pass `base_url=` to `ChatNVIDIA`, `NVIDIAEmbeddings`, `NVIDIA` (completions),
or `NVIDIARerank`. An explicit constructor `base_url` takes precedence over
`NVIDIA_BASE_URL`. Custom reverse-proxy prefixes are retained (for example,
`https://proxy.example/nim` becomes `https://proxy.example/nim/v1`).
Do **not** pass `/models`, `/chat/completions`, `/embeddings`, `/completions`,
`/ranking`, `/rankings`, or `/reranking` as the base URL: those are endpoint
paths, not deployment roots. The connector uses `GET <base>/models` for
discovery and appends `/chat/completions`, `/embeddings`, `/completions`, or
`/ranking` for the corresponding client. Self-hosted deployments may accept
anonymous access, but set `NVIDIA_API_KEY` if yours requires bearer auth.
Hosted catalog models with dedicated endpoints may not appear in `/models`;
their connector routing depends on the known hosted-model registry.

For TLS, model selection, and GPU/runtime considerations for hosted versus
self-hosted deployments, see the deployment table in the
[adoption guide](adoption_guide.md#choose-a-deployment).

## Check an endpoint with Doctor

Run the read-only Doctor preflight before sending prompts:

```bash
python -m langchain_nvidia_ai_endpoints doctor --model nvidia/nemotron-3-super-120b-a12b --capability chat
# Self-hosted NIM; no API key unless your deployment requires one
python -m langchain_nvidia_ai_endpoints doctor --base-url http://localhost:8000/v1 --model my-model --capability embeddings
```

Doctor uses `NVIDIA_BASE_URL` (or the hosted default), `NVIDIA_API_KEY`, and
`GET /v1/models`; it sends **no inference request or usage telemetry**.
`--capability` accepts `chat`, `embeddings`, `completions`, or `ranking`.
It checks `/models` model type metadata where available; unknown type is
reported as **UNKNOWN**, not verified by a generated response. It exits 0
when connectivity/auth/model listing checks pass and nonzero for missing
hosted auth, rejected keys, bad URLs, unavailable `/models`, missing models,
incompatible declared model types, network failures, or timeouts. Check
`--help` for options; use `--timeout 10` for a slow deployment. A preflight
pass is not proof that inference succeeds or that a framework is BCB-compatible.

## Self-host with NVIDIA NIM Microservices

When you are ready to deploy your AI application, you can self-host models with NVIDIA NIM. 
For more information, refer to [NVIDIA NIM Microservices](https://www.nvidia.com/en-us/ai-data-science/products/nim-microservices/).

The following code connects to locally hosted NIM Microservices.

```python
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank

# Connect to an chat NIM running at localhost:8000, and specify a model
llm = ChatNVIDIA(base_url="http://localhost:8000/v1", model="nvidia/nemotron-3-super-120b-a12b")

# Connect to an embedding NIM running at localhost:8080
embedder = NVIDIAEmbeddings(base_url="http://localhost:8080/v1")

# Connect to a reranking NIM running at localhost:2016
ranker = NVIDIARerank(base_url="http://localhost:2016/v1")
```

If your deployment exposes a direct inference URL instead of a `/v1/models`
listing plus a standard inference path, register it as a model endpoint rather
than passing the inference URL as `base_url`. A 404 from `base_url` plus
`/embeddings` usually means the deployment URL is already the full inference
endpoint.

```python
import os

from langchain_nvidia_ai_endpoints import Model, NVIDIAEmbeddings, register_model

register_model(
    Model(
        id="nvidia/llama-nemotron-embed-1b-v2",
        model_type="embedding",
        client="NVIDIAEmbeddings",
        endpoint="https://CUSTOM_DEPLOYMENT_LINK_WITHOUT_TRAILING_EMBEDDINGS",
    )
)

embedder = NVIDIAEmbeddings(
    model="nvidia/llama-nemotron-embed-1b-v2",
    api_key=os.environ.get("NVIDIA_API_KEY"),
)
```

## Optional BCB compatibility evidence (maintainers)

Optional BCB compatibility evidence is available from NIM OSS Manager's
**authenticated internal-alpha** API (not a public anonymous service).
It is never queried by default. If your team has approved, scoped read-only
Manager consumer access, set `NIM_OSS_MANAGER_TOKEN` in the environment and opt in explicitly:

```bash
python -m langchain_nvidia_ai_endpoints doctor --model nvidia/nemotron-3-super-120b-a12b --capability chat --manager-url https://your-manager-host
```

Doctor requests the Manager's read-only `bcb-public-v1alpha1` compatibility
summary for `langchain-nvidia` and that model. It does not transmit your
NVIDIA API key to Manager. Missing access, unknown responses, and stale
evidence are labeled **unavailable**, **unknown**, or **stale**, never
invented as compatible. An unavailable optional Manager lookup does not
turn a working endpoint into a failed connectivity check. Do not put
Manager tokens in command-line arguments or URLs.

For a maintainer-reviewed, framework-level compatibility snapshot, see
[NIM compatibility evidence](compatibility.md). The optional BCB
review artifact separates verified framework status from unknown individual
capabilities; it is not a live release approval or a public API.

## Further maintainer documentation

- [Adoption guide](adoption_guide.md): hosted-versus-local setup,
  streaming/tools, embeddings retrieval, migration from generic
  OpenAI-compatible clients, and scoped hosted recipe smoke results.
  Capabilities and compatibility still require deployment-specific evidence.
- [Non-text NIM modality guidance](non_text_nim_modalities.md): documented gaps
  in OCR, page-elements, table-structure, and offline ASR support (API shapes,
  proposed abstractions, and future validation cases). This is a maintainer
  proposal, not implemented modality support.
- [Reliability queue](reliability_backlog.md): open issues grouped by root
  cause, evidence, owner role, and next release or backlog decision; pending
  fixes are not claimed as released.
- [Model registry](model_registry.md): the reviewed registry of current and
  deprecated model IDs.
