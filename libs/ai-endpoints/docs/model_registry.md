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

## Hosted onboarding qualification (2026-10-29)

The reviewed batch contains five qualified entries. It retains the earlier
`nvidia/nemotron-3-embed-1b` addition and adds four more entries:

| Model | Connector | Live scope verified |
| --- | --- | --- |
| `nvidia/nemotron-3-embed-1b` | `NVIDIAEmbeddings` | Sync/async query and passage embeddings |
| `nvidia/llama-nemotron-embed-vl-1b-v2` | `NVIDIAEmbeddings` | Sync/async **text** query and passage embeddings |
| `nvidia/llama-3.1-nemotron-safety-guard-8b-v3` | `ChatNVIDIA` | Text invoke, async invoke, stream and async stream |
| `nvidia/riva-translate-4b-instruct-v2` | `ChatNVIDIA` | Text invoke, async invoke, stream and async stream |
| `nvidia/ising-calibration-1.5-31b` | `ChatNVIDIA`, VLM table | Text and synthetic single-image input in all four modes |

These entries use the standard NVIDIA hosted chat/embeddings routes. Their
individual API references are linked in the registry's hosted provenance. The
embedding API requires `input_type=query` for queries and `input_type=passage`
for documents; the existing connector supplies both correctly. Both embedding
models returned 2,048 finite, nonzero dimensions.

The final generated-default acceptance record is
`tests/data/qualified_hosted_models_e2e_20261029.json`: 24 successful real
inference checks for the five retained entries. Each check verifies the exact
request model and route, parsed response, and the real telemetry collector's
model ID, success count and token counts against returned measurements. The
collector posts to an in-memory sink; synthetic telemetry never reaches Hive or
production. Per-entry hashes tie the evidence to reviewed defaults. The tested
registry also contained Nano Omni, removed after its final check failed; retained
entry hashes are unchanged.

The image test uses a generated 64x64 red PNG, verifies the color response, and
makes no general vision-quality claim. The embedding model with `vl` in its name
is qualified for text only. Tools, structured output and configurable thinking
remain unknown. No claims are made for audio/video, downloadable deployments,
model quality, sustained availability or every account's access. Output usage is
checked against what the provider returns, not an independent tokenizer.

The earlier one-entry evidence remains in
`tests/data/qualified_hosted_models_20261029.json`. The complete candidate
review, including held and deprecated entries, is in
[hosted_model_qualification_20261029.md](hosted_model_qualification_20261029.md).

The streaming-usage telemetry fix is reviewed separately and should merge first.
Live qualification used that fix locally. Catalog and streaming changes have
disjoint diffs, and combined regression checks include both. A catalog listing
or an API reference alone does not establish live availability.
