# NIM compatibility evidence

This connector does not own or assert NIM compatibility. NIM OSS Manager BCB owns framework-level pass/fail, baseline acceptance, fingerprints and recommended actions. The [reviewed model registry](model_registry.md) owns static connector identities and catalog provenance, **not** validated behaviors. For a local preflight, see `python -m langchain_nvidia_ai_endpoints doctor --help`; Doctor's reachability check is not a compatibility test.

## Internal-alpha review artifact

Authorized maintainers can produce a **draft**, not a published badge or release gate, from Manager's authenticated non-mutating `bcb-public-v1alpha1` GET projections. The script uses `compatible` (scoped to `langchain-nvidia`), `recommend`, `badge` and aggregate `matrix-summary`; it neither calls internal routes nor fetches raw validation artifacts. The Manager API is currently **internal-alpha**, not anonymously public. Its existing controller bearer token also permits mutation; there is no scoped read-only credential yet. Supply `NIM_OSS_MANAGER_TOKEN` from an approved secure store for an authorized local run only, never in source, a query parameter, or a command argument.

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

From `libs/ai-endpoints`, use `NIM_OSS_MANAGER_TOKEN` and `NIM_OSS_MANAGER_URL` from an approved credential store. Do **not** configure the privileged controller token as a GitHub secret. The scheduled/dispatch workflow `bcb-compatibility-docs.yml` stays disabled until the Manager owner provisions and approves a scoped read-only consumer identity and sets `BCB_COMPATIBILITY_REVIEW_ENABLED=true`. On trusted `main`, it may read that approved secret plus `BCB_COMPATIBILITY_SELECTION_JSON` (dispatch may override the selection); absent configuration fails closed. It writes detailed Markdown and badge JSON only to the ephemeral runner, with no GitHub upload, commit, public badge or PR. Authorized operators must review locally and store detailed evidence solely in an approved internal channel.

## Reading the view

- **Verified** means all three Manager projections identify the exact selected target and model; the scoped framework checks are active and passing, the accepted baseline is current, Manager freshness is known/non-stale, the latest observation agrees with the accepted fingerprint (and with the digest for downloadable targets; hosted endpoints need not have an image digest), and the framework observation has a timestamp not older than 30 days (adjustable via `--max-age-days`). This is **framework-level** only, not release-readiness or capability certification.
- **Unknown** includes missing, unlinked, ambiguous, inaccessible, inconsistent, or undated evidence. Changed, failed, blocked, incompatible and stale are distinguished for maintainer triage but never promoted to verified. A failure to access Manager cannot produce a green badge.
- **Chat, streaming, tools, structured output, reasoning, VLM, embedding, rerank** remain **unknown** until the public-alpha contract exposes attributable capability-specific results. A framework suite marked passed does not prove structured output: even `response_format_supported=true` can coexist with `valid_json=false`, or streaming may finish by length. No catalog boolean, historic static flag, or overall pass substitutes for that evidence.
- GPU/runtime observations are marked **unavailable/redacted** because these public-alpha projections do not expose them. No fabricated performance, version-specific GPU requirement, raw logs, private artifact link, contact, or telemetry is included. The package version is read from this checkout's `pyproject.toml`, **not** a Manager-tested version claim. The digest and fingerprint family, when shown, are Manager projection values. Source ownership and limitations accompany each row.
- The matrix summary is aggregate across all returned pairs, **not** a selected model's verdict. Badges derive only from the same classified JSON rows as the table and require the same maintainer review. Re-run before use; stale screenshots and snapshots are not live evidence.

The emitted Markdown escapes display values and strips arbitrary Manager text/links, but still contains internal target IDs, digests and fingerprints. Review exact target mapping, package-version applicability, timestamps, status and data classification before selectively publishing a static excerpt; never upload the private draft to a public artifact store. This pilot is nonblocking and cannot replace Manager's separately reviewed release gate or production deployment checks.
