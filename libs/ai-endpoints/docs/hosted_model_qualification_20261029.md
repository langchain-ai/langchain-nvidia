# Hosted model qualification — 2026-10-29

The earlier proposal contained 19 additions. This review qualifies five for a
limited hosted scope, holds seven after live failures, and leaves seven
deprecated hosted entries out. Four of the five qualified entries are new since
the earlier one-model batch. No existing upstream entry is removed or changed.

This work does **not** establish that the proposed models caused historical
unknown-model or token gaps. It establishes whether each candidate can be added
with evidence of connector compatibility and correct local telemetry attribution.

## Qualification method

1. Check the NVIDIA API reference for deployment route, exact served request ID,
   input format and the appropriate connector. Use model cards to check lifecycle;
   discovery alone is insufficient.
2. Generate provisional registry defaults locally. Keep advanced capability flags
   unknown. Exercise actual connector methods against NVIDIA hosted endpoints,
   using synthetic text and a generated image; never use customer content.
3. Check parsed results: nonempty chat content, image color identification, or
   finite, nonzero embedding vectors with consistent dimensions and batch size.
4. Exercise the real aggregate collector with an in-memory delivery sink. Verify
   model ID, request/success/failure counts, token presence and input/output sums
   against returned usage. No synthetic telemetry is published to Hive or Manager.
5. Stop a candidate after its first failed mode. Retain only candidates that also
   pass the final generated-default checks. Bind evidence to each registry entry
   with a SHA-256 hash. Run offline routing, identity and regression checks.

Each request has a bounded deadline; there is no retry loop. A timeout is an
incomplete qualification within that deadline, not proof of global unavailability.
HTTP status and exception class are recorded without response bodies, headers,
credentials or vectors. The separate streaming telemetry fix is present during
qualification; this catalog change depends on it merging first.

## Qualified entries

| Model | API reference | Final real checks |
| --- | --- | --- |
| `nvidia/nemotron-3-embed-1b` | [Text embeddings](https://docs.api.nvidia.com/nim/reference/nvidia-nemotron-3-embed-1b-infer) | Query, document batch, async query, async document batch |
| `nvidia/llama-nemotron-embed-vl-1b-v2` | [Text embeddings](https://docs.api.nvidia.com/nim/reference/nvidia-llama-nemotron-embed-vl-1b-v2-infer) | Query, document batch, async query, async document batch |
| `nvidia/llama-3.1-nemotron-safety-guard-8b-v3` | [Chat](https://docs.api.nvidia.com/nim/reference/nvidia-llama-3_1-nemotron-safety-guard-8b-v3-infer) | Invoke, async invoke, stream, async stream |
| `nvidia/riva-translate-4b-instruct-v2` | [Chat](https://docs.api.nvidia.com/nim/reference/nvidia-riva-translate-4b-instruct-v2-infer) | Invoke, async invoke, stream, async stream |
| `nvidia/ising-calibration-1.5-31b` | [Vision chat](https://docs.api.nvidia.com/nim/reference/nvidia-ising-calibration-1-5-31b-infer) | All four chat modes with text, then all four with a synthetic image |

All 24 retained checks returned HTTP 200 and preserved the exact model ID and
reported token counts in the local telemetry collector. Both embedding models
returned 2,048-dimensional vectors. The VL embedding model is tested for **text
only**. No tools, structured-output, configurable-thinking, audio/video,
downloadable-deployment, quality benchmark or sustained-availability claims are
introduced. A model's presence in the VLM table does not certify every modality.

Sanitized evidence: `tests/data/qualified_hosted_models_e2e_20261029.json`.
Offline tests check retained entry hashes and required modes, plus request routing
and collector identity. These tests do not make provider calls.

## Held after live checks

Sanitized failed checks are recorded in
`tests/data/held_hosted_models_20261029.json`.

| Candidate | Observation | Evidence needed to reconsider |
| --- | --- | --- |
| `nvidia/llama-3.2-nemoretriever-500m-rerank-v2` | Documented custom reranking route returned HTTP 410 | Confirm a current deployment/route with the service owner, then validate sync/async ranking and usage handling |
| `z-ai/glm-5.3` | Documented chat route returned HTTP 410 | Confirm current hosted availability and exact served ID before repeating qualification |
| `z-ai/glm-5.3-flash` | Initial text request reached the read deadline | Complete all four text modes under an agreed practical timeout |
| `google/diffusiongemma-26b-a4b-it` | Initial text request reached the read deadline | Validate documented generation settings, then complete text and image modes |
| `stepfun-ai/step-5-preview` | Initial text request reached the read deadline | Check documented reasoning/output budgets and expected latency, then complete text and image modes |
| `poolside/laguna-xs-2.1` | Initial text request returned HTTP 503, consistent with earlier failed rechecks | Restore a usable hosted endpoint and complete a fresh final-default qualification |
| `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | Initial six checks passed; final async text stream returned HTTP 200 with no parsed content or token usage | Investigate an empty/truncated SSE response versus client handling; complete a fresh final text/image matrix |

The Nano Omni collector correctly recorded missing measurements for the empty
stream. An HTTP 200 by itself is not a passing inference check. The available
evidence does not establish whether the empty response originated in the provider,
network or parser. Do not label the underlying model defective from this result.

## Deprecated hosted candidates

The NVIDIA Build model cards reviewed on 2026-10-29 mark these free endpoints
deprecated. No live inference is attempted against them as part of active hosted
onboarding:

- [meta/llama2-70b](https://build.nvidia.com/meta/llama2-70b)
- [moonshotai/kimi-k3](https://build.nvidia.com/moonshotai/kimi-k3)
- [nvidia/gliner-pii](https://build.nvidia.com/nvidia/gliner-pii)
- [nvidia/nemotron-content-safety-reasoning-4b](https://build.nvidia.com/nvidia/nemotron-content-safety-reasoning-4b)
- [sarvamai/sarvam-m](https://build.nvidia.com/sarvamai/sarvam-m)
- [stockmark/stockmark-2-100b-instruct](https://build.nvidia.com/stockmark/stockmark-2-100b-instruct)
- [thinkingmachines/inkling](https://build.nvidia.com/thinkingmachines/inkling)

A deprecated free endpoint does not establish that downloadable or partner
deployments are deprecated. Testing those requires an accessible supported
deployment, its exact served ID and endpoint, appropriate credentials and the
same connector checks. Historical model recognition is a separate review from
adding entries to an active inference catalog.

## Release and follow-up

Merge the separately reviewed streaming fix first, review the five-entry catalog
batch, and rerun the focused checks after rebasing. Keep this qualification local
until approved for publication. Once released, compare real incoming identity
coverage and successful-call token coverage; do not promise a numerical reduction
in unknowns from these synthetic checks. Provider-to-connector-to-local-collector
qualification is complete only for the scopes above; Hive/Airflow/Manager ingestion
and production availability remain separate acceptance checks.
