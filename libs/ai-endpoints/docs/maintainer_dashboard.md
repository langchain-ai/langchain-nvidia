# Private connector maintainer dashboard

`python3 scripts/maintainer_dashboard.py` prints four **read-only Markdown views** to the
local terminal: Release Readiness, Model Registry Drift, Issue Hygiene and Adoption
Signals. It does not call Manager, fetch catalogs, change a registry, open a PR,
publish a report or write files. Run it in an authorized private workspace;
terminal output and the input artifacts contain internal evidence. **Do not
publish the view or input artifacts** until the designated release owner **and
security owner** explicitly approve that particular public/published surface.
Neither this view, a green CI run, nor a passing target-level BCB result grants
release approval or per-capability compatibility. The Manager `/api/bcb/public/v1`
contract is authenticated **internal-alpha**, not a public API.

From `libs/ai-endpoints`, after preparing authorized, local review artifacts:

```sh
python3 scripts/maintainer_dashboard.py \
  --review-file /private/review.json \
  --release-file /private/bcb-release-evidence.json \
  --hosted-file /private/hosted.json \
  --ngc-file /private/ngc.json
```

`--release-file` is the **JSON output** of `scripts/bcb_release_gate.py` (see
[bcb_release_gate.md](bcb_release_gate.md)); do not paste raw Manager responses
or privileged tokens into a dashboard input. That output contains internal IDs,
digests, fingerprints and GitLab evidence links and is classified here as
`internal_restricted` even though its upstream report has no classification
field. `--hosted-file` and `--ngc-file` are optional **local** complete catalog
snapshots with the exact shapes described in [model_registry.md](model_registry.md).
The dashboard invokes the same reviewed `model_registry.compare` drift logic
as the registry script; it never performs the registry script's optional network
fetch. Missing/malformed snapshots show unavailable sources and cannot imply
removal. The dashboard also never writes its view to disk. Redirect stdout only
into an approved internal review location if retention is explicitly authorized.

The `--review-file` is a JSON object with `schema_version: 1` and
`data_classification: "internal_restricted"`. Its schema is:

```json
{
  "schema_version": 1,
  "data_classification": "internal_restricted",
  "release_owner": "release-maintainer",
  "generated_pr_url": "https://github.com/langchain-ai/langchain-nvidia/pull/372",
  "issues": [
    {
      "issue_url": "https://github.com/langchain-ai/langchain-nvidia/issues/956",
      "priority": "P1",
      "affected_version": "0.4.0",
      "repro": "chat tools fails on retry",
      "owner": "connector-maintainer",
      "tests": "regression added",
      "planned_release": "0.4.1",
      "triage_url": "https://github.com/langchain-ai/langchain-nvidia/issues/956",
      "evidence_url": "https://gitlab-master.nvidia.com/team/bcb/-/jobs/62"
    }
  ],
  "adoption": {
    "downloads": {
      "status": "available",
      "count": 19,
      "observed_at": "2026-09-29T10:00:00Z",
      "source_url": "https://pypi.org/project/langchain-nvidia-ai-endpoints/"
    },
    "docs_traffic": {"status": "unavailable"},
    "cookbook_runs": {"status": "unavailable"},
    "github_issues": {"status": "unavailable"},
    "github_prs": {"status": "unavailable"},
    "model_usage_examples": {"status": "unavailable"}
  }
}
```

**All numbers, URLs and names above illustrate the input format, not measured
adoption, a real issue, an approved owner or a real generated PR.** Do not
reuse example counts or links as evidence. `generated_pr_url` is optional and
must reference an already-existing PR; the dashboard never creates one.
`issues` and `adoption` must be present, but may be empty. Issue fields are
short reviewed labels; absent or invalid fields are marked unavailable. `repro`
is a short, credential-free summary, **not** a raw log, stack trace or request.
Do not place raw diagnostics or secrets into artifacts. Metric keys are
`downloads`, `docs_traffic`, `cookbook_runs`, `github_issues`, `github_prs`,
and `model_usage_examples`. An available count requires an actual attributable
HTTPS source, a nonnegative integer `count`, and a recent ISO-8601 UTC
`observed_at`; otherwise the view withholds that count as unavailable or
stale/unknown. Zero is permitted only if the real source reports zero. Counts
have no inferred denominator or implied cross-source comparability. "Model
usage examples" counts reviewed examples, **not** actual model traffic.

The release view displays selected required capabilities as **scope, not
per-capability pass evidence**; latest result ID/status, target and gate
freshness, blockers, evidence and release-owner decision boundary. The owner
must inspect the complete BCB workflow and valid result back-reference before
claiming compatibility. The drift view groups additions, linked rename
proposals, deprecations/removals, aliases/served names, and catalog capability
claims (not BCB proof). All proposed changes require source verification,
behavior checks and maintainer review before registry edits. Issue gaps have
explicit actions. Adoption sources not supplied are always unavailable; the
view does not invent download, docs, cookbook, issue, PR or usage counts.

The CLI requires local reports, validates link schemes, hosts and paths,
displays only limited projected labels, and does not echo malformed JSON,
private paths or parse errors. External URLs are not dereferenced: validation
is syntactic and host allowlisting, **not** evidence authenticity. Source
owners must verify the underlying metrics and issue context; network-hostile
input, untrusted local files and public distribution remain outside this
internal review workflow. The default freshness bound is 14 days (adjust with
`--max-age-days` for an explicitly reviewed policy). Old/unknown data never
appears as a positive release assessment or current adoption count. A passing
assessment still says **review evidence (not approval)**: owner/security
approval and authorized release execution remain separate actions.
