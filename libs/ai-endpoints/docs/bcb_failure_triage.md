# BCB failure triage and maintainer drafts

This local, deterministic **review-only** tool consumes the sanitized
`bcb-public-v1alpha1` JSON report produced by `scripts/bcb_release_gate.py`.
It can also consume a `bcb-failure-triage-input-v1` wrapper containing that report
and a **manually reviewed, sanitized** result-bound diagnostic exported from a
scheduled BCB/connector check. No Manager fetch, inference request, CI upload,
issue/PR creation, branch push, merge, remediation trigger or release action
occurs. This is not an AI agent: confidence is the strength of explicit evidence,
not a model probability. Manager's public-alpha routes are authenticated,
**internal-alpha**, not a general public compatibility API. Never infer
compatibility from external CI success alone: the Manager result, accepted
baseline, workflow back-reference and full capability evidence require separate
owner review. Existing Manager Agent Pipeline and BCB remediation workflows
remain the only paths to prepare mutations, with their own approval gates.

From the repository root, after an authorized local release-evidence run:

```sh
python3 libs/ai-endpoints/scripts/bcb_failure_triage.py \
  --evidence /private/review/bcb-release-evidence.json \
  --json-output /private/review/triage.json \
  --markdown-output /private/review/triage.md
```

A release projection alone cannot diagnose root cause; its result is `unknown`
with `insufficient` confidence and no connector patch recommendation. For
scheduled failures, an authorized reviewer may create a **local** wrapper with
only the following allowlisted fields (illustrative IDs and URL, not real
compatibility evidence):

```json
{
  "contract_version": "bcb-failure-triage-input-v1",
  "connector_version": "0.9.0",
  "release_report": {
    "contract_version": "bcb-public-v1alpha1",
    "baseline_version": "reviewed-baseline",
    "targets": [
      {
        "nim_id": 17,
        "model_name": "reviewed-model",
        "deployment": "hosted",
        "status": "block",
        "reasons": ["latest_check_failed"],
        "accepted_fingerprint": "fp-1",
        "observed_fingerprint": "fp-1",
        "accepted_result": {"result_id": 30, "status": "passed", "pipeline_url": "https://gitlab-master.nvidia.com/team/-/pipelines/30"},
        "latest_result": {"result_id": 31, "status": "failed", "pipeline_url": "https://gitlab-master.nvidia.com/team/-/pipelines/31"}
      }
    ]
  },
  "failure": {
    "nim_id": 17,
    "result_id": 31,
    "status": "failed",
    "stage": "behavioral",
    "failure_category": "behavioral_failure",
    "evidence_url": "https://gitlab-master.nvidia.com/team/-/pipelines/31"
  },
  "comparison": {
    "prior_connector_version": "0.8.0",
    "prior_result_id": 30,
    "prior_status": "passed",
    "prior_fingerprint": "fp-1",
    "prior_evidence_url": "https://gitlab-master.nvidia.com/team/-/pipelines/30"
  }
}
```

Supply `failure`/`comparison` from actual reviewed Manager result and scheduled
job records, never from untrusted freeform output. `result_id` must refer to the
same target's latest Manager result. The tool checks this pointer but cannot
independently authenticate locally supplied artifacts, historical versions,
workflow back-references or the identity of an operator. The wrapper is not a
replacement for Manager's evidence contract: the nested release report retains
its contract version. A direct report is also accepted without a wrapper.

Classification rules are intentionally conservative:

- `credentials`: result-bound setup `authentication_failed`;
  `infrastructure`: setup `runner_unavailable` or `job_timeout`.
- `test_quality`: result-bound `artifact_schema_invalid`,
  `result_contract_missing`, or `test_fixture_defect` (not an SDK/model claim).
- `nim_behavior_delta`: behavioral failure, changed accepted versus latest
  fingerprint/digest, with allowlisted links to both results.
- `sdk_regression`: behavioral failure with a failing latest result and a linked
  prior passing **different connector version** on the same fingerprint; the
  prior result ID and evidence link must match the accepted result on that target.
- Anything unbound, ambiguous, stale, or missing evidence is `unknown`. Drift in
  a release report alone is not an attributed root cause.

The output includes owner/next action, constrained result IDs and GitLab links,
issue draft, conditional PR/patch **suggestion**, and release-note/docs-freshness
**review draft**. Missing model or version is shown as not established, rather
than invented. No patch is generated. Treat high confidence as a rule satisfied
by supplied evidence, **not** proof of causality or framework compatibility.
Only maintainers can review the actual BCB workflow and decide whether to issue,
patch, refresh docs, accept a baseline or request release-owner approval.

Security boundary: input is limited to 1 MiB; raw messages, logs, metadata and
unrecognized fields are discarded. Only short identifier labels and HTTPS
`gitlab-master.nvidia.com` path-only evidence URLs (no query, credentials,
fragments or redirects) reach output; snippets are rule-generated rather than
copied from raw logs. Output files are created exclusively with private `0600`
permissions and never overwrite existing files. Use a private review directory:
artifacts can contain internal model names, result IDs and links and must not be
committed or uploaded to public GitHub artifacts. Errors never echo input or
transport details. There are no network reads/writes or model calls, no API
credentials required, and no external publishing. If Manager ever adds an
approved model classifier, its actual call, provenance, data classification and
explicit owner-approved key must be separately implemented and reviewed; this
script makes no AI classification claim.
