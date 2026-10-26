# Public compatibility observation contract

This repository consumes reviewed compatibility observations. It does not
collect private evidence, connect to an internal service, approve captures,
or open publication PRs. Those operations belong in the private evidence
system, with its human approval and audit controls.

## Paired public output

A reviewed update changes these two files together:

- `docs/data/compatibility_public.json`: schema version 1 with a `rows` list.
- `docs/compatibility_state.md`: the corresponding observation table.

The empty initial snapshot intentionally makes no model-support claims. The
paths and contract in this change do not approve a future dataset. Source
owners and the appropriate publication/security reviewer must approve the
exact public projection before a maintainer reviews its PR.

## Allowed fields

Each row contains exactly:

| Field | Meaning |
| --- | --- |
| `model_id` | Canonical public model ID from the package registry. |
| `deployment` | `hosted` or `self_hosted`, established by the actual run. |
| `tested_sdk_version` | Version of the SDK that actually executed. |
| `tested_sdk_revision` | Full public Git commit SHA of that SDK checkout. |
| `observed_at` | Actual UTC completion time, ending in `Z`. |
| `capabilities` | Explicit `passed`, `failed` or `unknown` for every capability below. |
| `limitation_codes` | Nonempty list of the fixed public limitation codes below. |

The capability keys are `chat`, `streaming`, `tools`, `structured_output`,
`reasoning`, `vlm`, `embedding` and `rerank`. Unexecuted capabilities remain
`unknown`. Do not infer these results from a model catalog, installed package,
native/wire probe, replay fixture or a different SDK/deployment/model.

Every row includes `historical_observation`, `endpoint_availability_not_guaranteed`
and `untested_capabilities_unknown`. Additional permitted limitations are:

| Code | Meaning |
| --- | --- |
| `historical_observation` | Results describe only the recorded test time. |
| `endpoint_availability_not_guaranteed` | Availability may change after the run. |
| `untested_capabilities_unknown` | Missing evidence is not a support decision. |
| `native_baseline_not_accepted` | SDK results do not resolve the native baseline gate. |
| `self_hosted_not_tested` | A hosted observation does not validate a self-hosted NIM. |

Do not add internal target/result/workflow IDs, fingerprints, image digests,
private URLs or artifact paths, tokens, prompts, raw responses, reviewer names,
or freeform report messages. The snapshot has no arbitrary metadata fields.
Public model IDs, package versions and source revisions must themselves be
approved for disclosure; passing schema tests does not grant that approval.

## Review and validation

Generate both outputs in the private system from the same reviewed selection.
Keep the source evidence and approval records there. Provide only the resulting
public patch to the connector maintainer through the approved handoff.

Normal PR unit tests validate the allowlist, canonical identities, dates,
capability states and exact data/table agreement. They require neither private
credentials nor live NIM calls. They cannot verify the truth of an observation
or replace human source review. Run the focused check from `libs/ai-endpoints`:

```sh
poetry run pytest tests/unit_tests/test_public_compatibility.py
```

The public repository has no internal publication schedule, Manager secret,
privileged publication workflow, automatic merge or release gate for this data.
