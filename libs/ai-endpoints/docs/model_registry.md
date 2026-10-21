# Reviewed NIM model registry

`langchain_nvidia_ai_endpoints/data/model_registry.json` is the **reviewed source of truth** for the connector's built-in model table. The Python tables inside `_statics.py` are generated; `MODEL_TABLE`, its category tables, alias lookup, user registration, and the `_INCLUDE_OPENAI` switch remain public runtime behavior. Do not change an identifier or alias solely because a catalog row changed: the built-in IDs also determine the telemetry allowlist.

## Schema (version 1)

Each `models` entry has `table` (one of the eight `_statics.py` category tables), `id` (request and lookup identifier), `model` (exact constructor fields, including the historic client, endpoint, aliases, deprecation and thinking settings), and `served_names` (reviewed request names; the canonical ID is mandatory). `deployment_type` is explicitly `unknown`, `hosted`, `downloadable`, or `both`: it must agree with the source entries marked `current` in `provenance`. `provenance.hosted` and `.downloadable` each carry `state` (`current`, `stale`, `unavailable`, `unknown`) and an HTTPS `url` for a known state. A catalog observation is only evidence for **its own** deployment source. All 162 migrated entries retain `deployment_type: unknown`; historical inclusion in a static table does not prove current hosted or downloadable availability.

`capabilities` gives `supports_tools`, `supports_structured_output` and `supports_thinking` as `supported`, `unsupported`, or `unknown`. `capability_provenance` contains one `{source, url}` per capability, where `source` is `unknown`, `legacy_static`, `hosted_catalog`, `ngc_catalog`, or `bcb`. Known capability states must match explicit runtime `model` flags; `legacy_static` means an explicit historical boolean **without independent evidence** and has a null URL. An absent historical flag has `unknown` state/source, not `unsupported`. Catalog and BCB evidence require an HTTPS link; these sources are distinct, and **no migrated claim is marked BCB**. `stale` is a reviewed marker, not an automatic deletion request. Historical `model` values remain verbatim, including explicit false flags and all legacy aliases.

The hosted catalog is the NVIDIA hosted `/v1/models` response, not the NGC downloadable catalog. NGC resources and their metadata must be exported to the normalized snapshot below; an NGC resource ID is not automatically a served inference ID. These two sources may disagree. Compatibility evidence belongs to the platform BCB harness, **not** to this registry or bot. Do not infer tool calling or structured output compatibility from descriptive catalog metadata.

## Regenerate and review

From `libs/ai-endpoints`:

```sh
python3 scripts/model_registry.py generate          # exits nonzero on generated drift
python3 scripts/model_registry.py generate --write  # only after reviewing JSON changes
```

Regeneration is deterministic and offline. Review the JSON diff and the Python diff together; a second `generate` must be diff-free. Preserve original IDs and aliases unless the change is explicitly reviewed against consumers and telemetry behavior. Run focused unit tests before requesting maintainer review.

## Registry updates

Registry validation and generation are offline. Catalog monitoring and internal
release/triage reporting are owned by the internal orchestration system, not
this public package or its GitHub workflows. A catalog listing alone is not
proof that an SDK capability works.

For a reviewed model change, regenerate the Python tables, inspect both diffs,
and run the registry, statics and telemetry unit tests. Preserve request IDs,
legacy aliases and unknown capability states unless verified evidence supports
a deliberate update. Changes still go through normal pull-request review.

## Public API identity review (2026-10-21)

The identity review in `tests/data/public_model_catalog_20261021.json` records
19 additions with their exact public NVIDIA request-ID and endpoint references.
The comparison used the official [LLM](https://docs.api.nvidia.com/nim/reference/llm-apis),
[retrieval](https://docs.api.nvidia.com/nim/reference/retrieval-apis), and
[visual](https://docs.api.nvidia.com/nim/reference/visual-models-apis) API indexes,
individual request specifications, model cards, and read-only hosted discovery.
Discovery observations alone were not promoted into the registry. Some index
labels differ from the documented request ID; existing canonical IDs and aliases
remain unchanged. Specialized APIs and candidates lacking verified compatible
request shapes were excluded.

The new entries add identity recognition, not certification of live inference,
tool calling, structured output, thinking, or all multimodal input paths.
Those capability states remain unknown. Hosted/downloadable availability also
remains unknown unless explicit evidence supports a different state. Public
free-endpoint deprecation is recorded separately from identity recognition;
deprecated public IDs may still be useful when classifying historical telemetry.
The tests exercise client requests and emitted identities with mocked responses.

Model ID recognition and family recognition are separate. For example,
`openai/gpt-oss-120b` and `nvidia/nv-embed-v1` are recognized IDs whose current
telemetry family is `unknown`. `embedding` is an operation/model type, not an
approved family value. Adding family enums requires a separate event-contract
change; this registry update preserves existing telemetry family behavior.
Neither a missing family nor a provider `4xx` establishes test traffic, EOL,
or the exact error cause. Adoption reporting should retain attempts, successes,
failures, and identity coverage as separate measures.
