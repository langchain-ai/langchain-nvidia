# NIM compatibility evidence

This connector does not own or assert NIM compatibility. NIM OSS Manager BCB owns framework-level pass/fail, baseline acceptance, fingerprints and recommended actions. The [reviewed model registry](model_registry.md) owns static connector identities and catalog provenance, **not** validated behaviors. For a local preflight, see `python -m langchain_nvidia_ai_endpoints doctor --help`; Doctor's reachability check is not a compatibility test.

## Internal-alpha review artifact

Authorized maintainers can produce a **draft**, not a published badge or release gate, from Manager's authenticated non-mutating `bcb-public-v1alpha1` GET projections. The script uses `compatible` (scoped to `langchain-nvidia`), `recommend`, `badge` and aggregate `matrix-summary`; it neither calls internal routes nor fetches raw validation artifacts. The Manager API is **internal-alpha**, not anonymously public. Request an approved scoped, read-only consumer token; the privileged controller bearer token permits mutation and must not be used for this pilot. Supply `NIM_OSS_MANAGER_TOKEN` from an approved secure store for an authorized local run only, never in source, a query parameter, or a command argument.

Create a locally held, reviewed JSON selection array, for example with an existing registry ID and **an actual Manager target ID**:

```json
[
  {"model": "nvidia/nemotron-3-super-120b-a12b", "nim_target_id": 123, "deployment_type": "downloadable"}
]
```

The target ID above is illustrative **only**; replace it with the actual Manager target ID before running. `deployment_type` is maintainer-selected (hosted or downloadable); it is **not** inferred from the registry, catalog, or target image. Distinct deployments require distinct target IDs. A response with a mismatched target ID or served model name never becomes verified. The local JSON file is an input, not a compatibility attestation. Use an HTTPS Manager origin configured in your environment; do not commit live origins or tokens:

```sh
python3 scripts/bcb_compatibility_view.py \
  --manager-url "$NIM_OSS_MANAGER_URL" \
  --selection /path/to/reviewed-selection.json \
  --output /path/to/bcb-compatibility-review.md \
  --badges /path/to/bcb-compatibility-badges-review.json
```

From `libs/ai-endpoints`, use a scoped `NIM_OSS_MANAGER_TOKEN` and `NIM_OSS_MANAGER_URL` from an approved credential store. Do **not** configure the privileged controller token as a GitHub secret. The scheduled/dispatch workflow `bcb-compatibility-docs.yml` stays disabled until the Manager owner provisions and approves this repository's read-only consumer credential and sets `BCB_COMPATIBILITY_REVIEW_ENABLED=true`. On trusted `main`, it may read that approved secret plus `BCB_COMPATIBILITY_SELECTION_JSON` (dispatch may override the selection); absent configuration fails closed. It writes detailed Markdown and badge JSON only to the ephemeral runner, with no GitHub upload, commit, public badge or PR. Authorized operators must review locally and store detailed evidence solely in an approved internal channel.

## Reading the view

- **Verified** means all three Manager projections identify the selected target/model; scoped framework checks are active and passing; the accepted baseline is current; freshness is known/non-stale; accepted, latest and recommended fingerprints agree and are nonempty; and Manager recommends `no_action`. Downloadable evidence must also agree on its digest; hosted evidence need not have an image digest. The observation must be within `--max-age-days` (30 by default). This remains **framework-level**, not release approval or SDK capability certification.
- **Unknown** includes missing, unlinked, ambiguous, inaccessible, inconsistent, or undated evidence. Changed, failed, blocked, incompatible and stale are distinguished for maintainer triage but never promoted to verified. A failure to access Manager cannot produce a green badge.
- **Chat, streaming, tools, structured output, reasoning, VLM, embedding, rerank** SDK states remain **unknown** without an executed, package-bound SDK check. A separate native-probe table consumes bounded Manager case results only when target, result ID, fingerprint and execution time agree. Failed cases remain failed; empty, skipped, stale, contradictory or mismatched observations cannot become passing probes. A native probe pass, ChatOpenAI profile, wire adapter, replay fixture or catalog flag never becomes ChatNVIDIA support.
- **Tested SDK** stays unknown when exact SDK execution/version binding is absent. The renderer checkout version is explicitly labelled and never copied into this column. Attributable framework evidence shows measured installed-package inventory, exact captured check/profile scope and result IDs; merely installing the SDK does not prove it was executed.
- **Deployment** distinguishes maintainer selection from a target-bound measured executor. GPU/runtime remains **unavailable/redacted** unless authoritative measured evidence is projected; no GPU or performance requirement is inferred.
- **Provenance** includes the complete projected fingerprint, accepted/latest native result IDs and allowlisted private GitLab artifact/pipeline links when available. Missing values remain unknown. Manager-specific actions such as `run_baseline` survive classification; an action from a mismatched target does not.
- The matrix summary is aggregate across all returned pairs, **not** a selected model's verdict. Badges derive only from the same classified JSON rows as the table and require the same maintainer review. Re-run before use; stale screenshots and snapshots are not live evidence.

The detailed Markdown is private maintainer evidence, **not a public-safe support table**. It contains internal target/result IDs, digests, fingerprints, measured package inventory and allowlisted private links. Review target mapping, package applicability, timestamps, status, source ownership and data classification before publication. Do not upload this artifact or its badge JSON to a public store. Missing owner acceptance or real SDK-bound evidence is not supplied by a passed fixture, local smoke or current baseline; Manager's separately reviewed release gate and production checks remain authoritative.

