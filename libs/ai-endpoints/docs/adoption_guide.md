# Adopt `langchain-nvidia-ai-endpoints` (maintainer-review draft)

**Human review is required before publication or upstream merge.** This guide describes connector behavior, not a certification of any model or deployment. Start with the endpoint you actually control; hosted catalog access and a self-hosted NIM are different paths. Install `langchain-nvidia-ai-endpoints` into your Python 3.10+ environment (`python -m pip install langchain-nvidia-ai-endpoints`). For development against this checkout, run `python -m pip install -e .` from `libs/ai-endpoints` instead. Keep credentials in an approved secret store/environment, never in a URL, notebook, repository, pasted diagnostic, or command argument.

## Choose a deployment

| | NVIDIA-hosted API Catalog | Your self-hosted NIM |
| --- | --- | --- |
| Access | Obtain an API key for your account at [build.nvidia.com](https://build.nvidia.com/) and supply `NVIDIA_API_KEY` from a secure store. | Deploy the appropriate NIM image/model under your own entitlement and infrastructure procedures. A local service may allow anonymous access; if your deployment requires bearer auth, supply its approved `NVIDIA_API_KEY`. Do not send a hosted key to an untrusted local/proxy endpoint. |
| URL | Omit `base_url` and unset any inherited `NVIDIA_BASE_URL` to use the connector default `https://integrate.api.nvidia.com/v1`. | Pass `base_url="http://localhost:8000/v1"` (or your actual deployment root/`/v1` URL). The connector appends `/v1` to a root, preserving a reverse-proxy prefix, and uses `/v1/models` for discovery. Do not pass `/models` or `/chat/completions` as `base_url`. `NVIDIA_BASE_URL` is a global alternative; explicit constructor `base_url` takes precedence. |
| Models | Select a **currently available** hosted chat or embedding model by its actual request ID on your account; the local reviewed registry helps route known hosted models, including some dedicated endpoints not listed by the shared `/models` API. Listing alone does not establish capability. | Use the exact served ID returned by this deployment's `GET /v1/models` and a separate embedding NIM/URL if needed. An NGC downloadable resource ID, registry entry, or hosted ID need not equal the deployed served ID. Do not assume an arbitrary OpenAI-compatible proxy exposes model listing. |
| TLS | Use HTTPS with normal certificate validation. | Plain HTTP is appropriate only for loopback or an approved isolated transport; use HTTPS and a trusted CA for remote/proxied NIM. The connector verifies TLS by default; for an organization-issued CA, configure trusted certificate material (for example, `verify_ssl="/path/to/approved-ca.pem"` on `ChatNVIDIA` or `NVIDIAEmbeddings`) rather than disabling verification. Doctor uses the normal `requests` CA trust configuration (including an approved `REQUESTS_CA_BUNDLE`); it does not accept a certificate-bypass flag. |
| GPU/runtime | Catalog service resources and behavior are operated by the hosted provider. | GPU SKU, driver, NIM image digest, runtime profile, memory needs and entitlement depend on **your actual deployment**. Record them from your deployment manifest, NGC release notes and operator evidence; connector metadata, a hosted catalog listing and Doctor cannot observe or attest to them. |

Choose IDs before running the recipes below. Set `NVIDIA_CHAT_MODEL` and `NVIDIA_EMBED_MODEL` to the **actual chat and embedding inference IDs** for your selected endpoint(s), not to examples from an unrelated catalog. The commands below contain no credential literals; load keys from your approved credential store before invoking Python. For hosted usage, leave `NVIDIA_CHAT_BASE_URL` and `NVIDIA_EMBED_BASE_URL` unset as well as `NVIDIA_BASE_URL`. For self-hosted usage, set each to the actual deployment root or `/v1` base (the two NIMs can be on different ports); unset `NVIDIA_BASE_URL` to avoid an inherited global override. With a shared credential, the connector reads `NVIDIA_API_KEY` when present. When the two destinations require different bearer tokens, unset `NVIDIA_API_KEY`, load `NVIDIA_CHAT_API_KEY` and `NVIDIA_EMBED_API_KEY` separately into the environment from your secure store, and let the examples pass each token only to its intended client. Doctor only reads `NVIDIA_API_KEY`; for preflights using separate keys, load the appropriate key as `NVIDIA_API_KEY` for each individual Doctor invocation, without printing it.

From `libs/ai-endpoints` in this checkout, check each endpoint without sending a prompt. For **hosted** (with `NVIDIA_API_KEY` loaded and `NVIDIA_BASE_URL` unset):

```sh
python -m langchain_nvidia_ai_endpoints doctor --model "$NVIDIA_CHAT_MODEL" --capability chat
python -m langchain_nvidia_ai_endpoints doctor --model "$NVIDIA_EMBED_MODEL" --capability embeddings
```

For **self-hosted** (with the corresponding base URL variables set):

```sh
python -m langchain_nvidia_ai_endpoints doctor --base-url "$NVIDIA_CHAT_BASE_URL" --model "$NVIDIA_CHAT_MODEL" --capability chat
python -m langchain_nvidia_ai_endpoints doctor --base-url "$NVIDIA_EMBED_BASE_URL" --model "$NVIDIA_EMBED_MODEL" --capability embeddings
```

Doctor's `GET /v1/models` checks reachability/auth and (when provided) model/type metadata; it makes **no inference call**. A known hosted model routed to a dedicated endpoint may not be in the shared `/models` response, so a missing entry is not a model-compatibility verdict; investigate its documented route and actual endpoint. An `UNKNOWN` capability means no usable model-type metadata, not unsupported. Run `python -m langchain_nvidia_ai_endpoints doctor --help` for options. Do not put your API key in command arguments. Doctor is included on the PR372 branch; it may not be available in an older published package.

## Recipe: stream chat, then handle an optional tool call

Save the following as `chat_recipe.py` outside the repository. Set `NVIDIA_CHAT_MODEL`, optionally `NVIDIA_CHAT_BASE_URL`, and the appropriate credential as described above, then run `python chat_recipe.py`. It streams by default. **Only after verifying tool support on this exact deployment**, set `NVIDIA_ENABLE_TOOLS=1` to run the second segment. Streaming support does not establish tool support. The tool is a local, read-only lookup; model output does not execute a tool by itself.

```python
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import ToolMessage

options = {"model": os.environ["NVIDIA_CHAT_MODEL"]}
if os.getenv("NVIDIA_CHAT_BASE_URL"):
    options["base_url"] = os.environ["NVIDIA_CHAT_BASE_URL"]
if os.getenv("NVIDIA_CHAT_API_KEY"):
    options["api_key"] = os.environ["NVIDIA_CHAT_API_KEY"]
chat = ChatNVIDIA(**options)

for chunk in chat.stream("Explain one benefit of retrieval in a sentence."):
    if chunk.content:
        print(chunk.content, end="", flush=True)
print()
if os.getenv("NVIDIA_ENABLE_TOOLS") != "1":
    print("Tool segment skipped; verify this model's tool support before enabling.")
    raise SystemExit(0)


# Proceed only if the selected deployment is documented/tested to support tools.
lookup = {
    "name": "lookup_policy",
    "description": "Read a policy statement by name from this example's local table.",
    "parameters": {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    },
}
policy = {"retention": "Keep only approved data for the approved retention period."}
question = "What is the retention policy? Use lookup_policy if available."
assistant = chat.bind_tools([lookup]).invoke(question)
if assistant.tool_calls:
    messages = [("human", question), assistant]
    for call in assistant.tool_calls:
        if call["name"] != "lookup_policy":
            raise ValueError("Unexpected tool requested")
        name = call["args"].get("name")
        result = policy.get(name, "No policy by that name in this example.")
        messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    answer = chat.invoke(messages)
else:
    answer = assistant
print(answer.content)
```

A model may refuse a tool, return no tool call, or reject the request entirely. Inspect its response and the deployment's own tool behavior; no registry flag, warning, or recipe guarantees it. Avoid feeding untrusted retrieved text or tool arguments into privileged side effects.

## Recipe: embeddings + retrieval + grounded chat

This small in-memory retrieval recipe uses `NVIDIAEmbeddings.embed_documents` (passage vectors) and `.embed_query` (query vector), then cosine similarity to select a passage. No vector database or hidden service is required. It still **calls real inference endpoints**: provide a working embedding NIM/model and chat NIM/model. Save it as `retrieval_recipe.py` outside the repository and run `python retrieval_recipe.py` with the variables above. Replace the sample passages with material you are permitted to send to the selected service.

```python
import math
import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

chat_options = {"model": os.environ["NVIDIA_CHAT_MODEL"]}
embed_options = {"model": os.environ["NVIDIA_EMBED_MODEL"]}
if os.getenv("NVIDIA_CHAT_BASE_URL"):
    chat_options["base_url"] = os.environ["NVIDIA_CHAT_BASE_URL"]
if os.getenv("NVIDIA_EMBED_BASE_URL"):
    embed_options["base_url"] = os.environ["NVIDIA_EMBED_BASE_URL"]
if os.getenv("NVIDIA_CHAT_API_KEY"):
    chat_options["api_key"] = os.environ["NVIDIA_CHAT_API_KEY"]
if os.getenv("NVIDIA_EMBED_API_KEY"):
    embed_options["api_key"] = os.environ["NVIDIA_EMBED_API_KEY"]
chat = ChatNVIDIA(**chat_options)
embedder = NVIDIAEmbeddings(**embed_options)

passages = [
    "The visitor desk opens at 09:00 on weekdays.",
    "The visitor desk closes at 17:00 on weekdays.",
    "The visitor desk is closed on weekends.",
]
question = "When does the visitor desk open on weekdays?"
doc_vectors = embedder.embed_documents(passages)
query_vector = embedder.embed_query(question)

def cosine(a, b):
    if len(a) != len(b):
        raise ValueError("Query and passage embedding dimensions differ")
    denominator = math.sqrt(sum(x * x for x in a) * sum(x * x for x in b))
    if not denominator:
        raise ValueError("Cannot compare zero-length embedding vectors")
    return sum(x * y for x, y in zip(a, b)) / denominator

best = max(range(len(passages)), key=lambda i: cosine(query_vector, doc_vectors[i]))
reply = chat.invoke(
    "Answer only from the passage below. If the passage does not answer the "
    f"question, say you do not know.\nPassage: {passages[best]}\nQuestion: {question}"
)
print(reply.content)
```

Check retrieval relevance and generated answers against your source data. This is a demonstration, not a measured quality claim. NVIDIA embedding requests carry different `input_type` values for passage (`embed_documents`) and query (`embed_query`); keep the same embedding model/deployment for both sides, and rebuild stored vectors after changing that model. Use a vector store and access-control boundaries appropriate to your production corpus rather than this process-local list.

## Migrating from a generic OpenAI-compatible client

- Keep the **same actual served model ID** and deployment root/`/v1` base; replace a generic chat client with `ChatNVIDIA(model=..., base_url=...)`, and a generic embedding client with `NVIDIAEmbeddings(model=..., base_url=...)`. The connector uses `NVIDIA_API_KEY` or the constructor `api_key` for bearer auth where required. A key for one provider/deployment is not automatically valid for another; do not pass an OpenAI key to the NVIDIA catalog or a hosted NVIDIA key to an untrusted proxy.
- The connector's `base_url` must point to a deployment root or `/v1`, **not** the full `/v1/chat/completions` or `/v1/embeddings` inference path. It appends those paths itself and expects `/v1/models` for discovery (not every generic OpenAI-compatible endpoint implements listing). A configured `NVIDIA_BASE_URL` may redirect an otherwise hosted call; set explicit base URLs for local endpoints and clear stale global settings when changing environments.
- `ChatNVIDIA` offers LangChain streaming and `bind_tools`; `NVIDIAEmbeddings` distinguishes query from passage vectors and supports model-dependent NVIDIA options such as `truncate` and `dimensions`. Request formats and extras such as NVIDIA-specific inference routing/profile handling are **provider/model dependent**. Do not copy generic `response_format`, tool, or embedding dimension assumptions unchanged; test the selected endpoint. An advisory built-in model registration can select a hosted endpoint, map an alias, or warn about deprecated/unknown capabilities; it does **not** test the served model, GPU, or capability. Custom chat-only inference routes can use `register_model(Model(...))` where there is no `/models`, but that is explicit, reviewed client configuration, not evidence of compatibility; see [registry guidance](model_registry.md).
- Preserve provenance separately: record whether the ID came from the hosted catalog or your specific downloadable deployment, the exact served ID, endpoint, image/version/digest and GPU/runtime evidence **if operator-verified**, and the observed inference behavior for each capability. See [reviewed registry/provenance](model_registry.md) and [NIM compatibility evidence](compatibility.md). Hosted listing, NGC metadata, and a local warning are not interchangeable proof.

The NIM OSS Manager `bcb-public-v1alpha1` evidence projection is **authenticated internal-alpha**, not a public support promise. Request a read-only, rate-limited consumer token scoped to the approved Manager public-v1 endpoints; never use the privileged controller token for Doctor or CI. Only authorized maintainers should opt in to Doctor's `--manager-url` with `NIM_OSS_MANAGER_TOKEN` loaded from an approved secret store (never a URL, argument, repository, or public CI secret). [Compatibility review](compatibility.md) describes the reviewed, non-publishing draft: exact target/model matching and fresh accepted framework checks can indicate a scoped **framework-level** result, not capability-level streaming/tools/embeddings/structured-output approval, GPU requirements, production readiness, or release approval. Missing, stale, inaccessible, ambiguous, or mismatched evidence remains **unknown/not verified**; no catalog metadata, Doctor preflight, or successful single prompt upgrades it. Human approval is required before publishing any excerpt or upstream merge.
