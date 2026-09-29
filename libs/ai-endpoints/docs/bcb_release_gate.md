# BCB release evidence pilot

This opt-in, **nonblocking** pilot reads Manager's authenticated internal-alpha
`/api/bcb/public/v1/release-gate` (Manager's current default baseline, or an
explicit `--baseline-version` override) and
`/api/bcb/public/v1/recommend?nim_id=…` for selected hosted and downloadable NIMs.
It writes a sanitized JSON report and a Markdown summary. It does not call
inference, change Manager state, publish a package, or replace the release
owner's approval. Manager's existing bearer token is required; these are not
anonymous public APIs. No token or Manager URL is checked into this repository.

The release owner configures repository variables `BCB_MANAGER_URL` (HTTPS
origin), `BCB_HOSTED_NIM_ID`, `BCB_DOWNLOADABLE_NIM_ID`,
`BCB_HOSTED_REQUIRED_CAPABILITIES`, and
`BCB_DOWNLOADABLE_REQUIRED_CAPABILITIES` (distinct comma-separated capability
identifiers, e.g. `chat_basic,chat_tools`). IDs must identify the Manager NIM
targets associated with the chosen hosted and downloadable release scope;
they are **not** connector registry model IDs. Manager supports a scoped,
read-only public-v1 consumer credential; the platform owner must provision and
approve a separate `BCB_MANAGER_TOKEN` for this repository before enabling CI.
The privileged controller token can mutate state: never install it as a GitHub
Actions secret. The scheduled pilot stays disabled until that approved
consumer credential is configured and the release owner explicitly sets
`BCB_RELEASE_EVIDENCE_ENABLED=true`. Target selection and credential approval
are owner responsibilities, never inferred from the catalog.

Optional repository variable `BCB_BASELINE_VERSION` pins a baseline only when
the Manager owner confirms that version exists. The current Manager manifest
uses `release-readiness-2026-08-28-v2` as its default; `prod-v1` is **not** a
configured baseline version in that manifest. An unknown override warns rather
than silently falling back to another baseline. The report records the version
actually returned by Manager and warns if a selected target lies outside the
gate's declared scope.

For an authorized local run:

```sh
BCB_MANAGER_URL=https://manager.example.internal BCB_MANAGER_TOKEN="<from-secure-store>" \
BCB_HOSTED_NIM_ID=123 BCB_HOSTED_REQUIRED_CAPABILITIES=chat_basic,chat_tools \
BCB_DOWNLOADABLE_NIM_ID=456 BCB_DOWNLOADABLE_REQUIRED_CAPABILITIES=chat_basic \
python3 libs/ai-endpoints/scripts/bcb_release_gate.py \
  --json-output /tmp/bcb-release-evidence.json \
  --markdown-output /tmp/bcb-release-evidence.md
```

The sample URL and IDs above are placeholders, **not** live configuration.
An authorized operator can run the local pilot with Manager credentials obtained
from a secure store. Keep generated reports in an authorized internal review
channel: they contain internal model IDs, digests, fingerprints, and GitLab
evidence links. Never commit them or upload them as public GitHub artifacts.
The GitHub Action skips all PRs, runs on trusted `main` only with explicit owner
enablement, and writes its detailed report only into ephemeral runner storage;
its stdout records the nonblocking assessment, not the private evidence. A
missing configuration warns locally rather than claiming compatibility. The
pilot is not a dependency of the release/publish workflow.

The private report explicitly lists required capabilities, accepted/observed
digest where available (hosted endpoints may have no image digest), fingerprint,
freshness and `tested_at`, accepted/latest result IDs and allowlisted HTTPS
GitLab evidence pointers. Downloadable targets require a matching image digest;
hosted targets require matching fingerprints but not an image digest.
`pass` requires a ready Manager gate with complete passing counts and
checks, connector project and both selected targets in the gate scope, plus
both selected target baselines current, fresh and consistent with accepted
passing result evidence. Missing/stale/unknown data warns; changed or
failed evidence for a required target blocks its assessment. A blocked or
incomplete release-gate summary cannot produce `pass`. **A target-level pass is
not independent per-capability proof**: v1alpha1 release-gate projects only
aggregate counts and recommend projects target-level baseline data, not
per-capability check results. The owner must inspect the linked BCB workflow and
result to confirm every required capability and the complete evidence contract
before asserting framework compatibility. In particular a green external CI
run without an accepted BCB result and valid workflow back-reference is not
compatibility evidence. No unknown, missing, stale, changed or failed row can
be treated as compatible or release-ready.

**Promotion boundary:** only the designated release owner, after reviewing
Manager evidence, selected capability scope, applicable security and launch
requirements, and any warnings/blocks, may separately approve and execute
promotion. The pilot neither grants that approval nor blocks production
publication without the owner's separately authorized gate decision.
