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

## Small hosted onboarding batch (2026-10-29)

This batch adds only `nvidia/nemotron-3-embed-1b` to `NVIDIAEmbeddings` for
text query and passage embeddings. The [API reference](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-embed-1b-infer)
requires `input_type=query` for queries and `input_type=passage` for documents;
the existing connector supplies both correctly.

Live sync and async query/document calls returned HTTP 200, 2,048 finite,
nonzero dimensions and parsed input-token usage. The initial process-local
registration checks and repeated generated-default checks both passed. The
sanitized record is `tests/data/qualified_hosted_models_20261029.json`.

Tools, structured output and configurable thinking remain unknown. This batch
makes no claim about multimodal input, downloadable NIM compatibility, model
quality, latency SLOs or general availability for every account.

`poolside/laguna-xs-2.1` passed initial chat/streaming checks but returned HTTP
503 in the generated-default recheck and again in one bounded retry. It is held
out of this addition. `nvidia/nemotron-3.5-bolt-e4b` is also held: its API
reference specifies a staging-prefixed standalone request ID, which needs
clarification before a default registry entry is appropriate.

The earlier broad catalog proposal is not included. Deprecated free endpoints
must not be added to active hosted tests solely for historical telemetry
recognition. Hosted deprecation does not establish downloadable NIM deprecation.
Discovery and model cards nominate candidates; exact served identity, compatible
request/response contracts and live connector checks establish the limited scope
of each addition. Edit the reviewed JSON registry, then regenerate `_statics.py`.

The streaming-usage telemetry fix is reviewed separately and should merge first.
It fixes an existing cancellation regression in the telemetry tests and allows
the collector to retain usage from processed streaming chunks. Catalog and
streaming changes have disjoint diffs; combined acceptance is checked separately.
