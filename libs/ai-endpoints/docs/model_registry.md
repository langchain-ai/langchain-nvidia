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

## Drift report (no automatic publication)

```sh
python3 scripts/model_registry.py drift --hosted-file /path/hosted.json --ngc-file /path/ngc.json --output /path/model-registry-review.md
```

Omit `--hosted-file` to fetch the fixed `https://integrate.api.nvidia.com/v1/models` URL (15-second timeout); supply `NVIDIA_API_KEY` through the environment if authentication is required. The scheduled workflow reads the optional repository secret of that name and never prints its value. There is no automatic NGC fetch: supply a reviewed export from the NGC catalog or existing platform metadata harness. No credentials are included in snapshots or reports. A missing, malformed or empty source is called out as **unavailable** and exits with status 2 after writing a report. It never causes deletion proposals. Reports are proposed review artifacts, not registry changes or compatibility approvals.

For the existing downloadable NIM harness, save the job log outside the
repository and supply its GitLab job URL as evidence:

```sh
python3 scripts/model_registry.py drift --hosted-file /path/hosted.json --ngc-file /path/ngc.json \
  --harness-log /path/downloadable-job.log --harness-url https://gitlab-master.nvidia.com/group/project/-/jobs/42 \
  --output /path/model-registry-review.md
```

Only exact `UPSTREAM NEEDED: add '<model>' to langchain-nvidia _statics.py`
lines become linked **candidate additions**; duplicate or already-reviewed IDs
are ignored. The log itself is never copied into the report. A harness candidate
does not establish the served model name, NGC availability, or capabilities;
review the linked job and NGC metadata before changing reviewed JSON.

Hosted snapshot: `{"source_url":"https://integrate.api.nvidia.com/v1/models","data":[{"id":"nvidia/example"}]}`. NGC snapshot: `{"source_url":"https://catalog.ngc.nvidia.com/...","models":[{"id":"nvidia/example","url":"https://catalog.ngc.nvidia.com/..."}]}`. Optional normalized fields on either item: `served_names` and `aliases` (arrays of strings), `deprecated` (boolean), `replaces` (an existing registry ID, only when explicitly asserted upstream), and `capabilities` (map using the three capability names and the schema states). Attach per-item HTTPS `url` links when available; otherwise the snapshot's HTTPS `source_url` is used. Capture complete snapshots; filtered exports must not be treated as authoritative for removals.

The comparison proposes additions, explicitly linked renames, removals/staleness only for entries previously marked `current` in that source, source-specific deployment-type changes, alias/served-name changes, deprecations, and capability changes **with claimed provenance**. An absent capability stays unknown; a source outage is not evidence of removal. A catalog claim is never labeled BCB evidence, even when its state agrees with an existing legacy flag. Validate observed model behavior, served names and links, update the JSON deliberately, regenerate, review both diffs and request **maintainer approval** on the pull request. The scheduled GitHub workflow uploads a report for triage; it has read-only repository permission and cannot publish updates or open an unreviewed PR.

## Test plan for a model update

1. Run `python3 scripts/model_registry.py generate` to detect stale generated tables, then `generate --write` after editing reviewed JSON; repeat the check to prove stable regeneration.
2. Run `pytest tests/unit_tests/test_model_registry.py tests/unit_tests/test_statics.py tests/unit_tests/test_usage_telemetry.py` from this package with its development dependencies installed.
3. Check additions and aliases against actual served IDs; investigate every rename and removal against **both** source links. An unavailable/filtered source cannot establish removal.
4. Run appropriate hosted and downloadable inference scenarios and ask the platform BCB harness for compatibility evidence before approving capability changes. Record the per-capability provenance link accurately (`legacy_static`/catalog claims do not become `bcb` by agreement). No catalog field is a substitute for a passing behavior result.
5. Review the generated diff, historic aliases, telemetry identifier changes and the Markdown drift report in a pull request. A maintainer must explicitly approve the registry change; the bot never merges or publishes.
