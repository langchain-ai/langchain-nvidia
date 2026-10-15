"""Evidence-bound BCB triage must not turn setup failures into connector defects."""

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "bcb_failure_triage.py"
spec = importlib.util.spec_from_file_location("bcb_failure_triage", SCRIPT)
assert spec is not None and spec.loader is not None
triage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(triage)


def artifact(*, changed: bool = False) -> dict[str, Any]:
    return {
        "contract_version": "bcb-public-v1alpha1",
        "baseline_version": "baseline-1",
        "targets": [
            {
                "nim_id": 17,
                "model_name": "nemotron-model",
                "deployment": "hosted",
                "status": "block",
                "reasons": ["latest_check_failed"],
                "accepted_fingerprint": "fingerprint-1",
                "observed_fingerprint": "fingerprint-2" if changed else "fingerprint-1",
                "accepted_digest": "",
                "observed_digest": "",
                "accepted_result": {
                    "result_id": 30,
                    "status": "passed",
                    "pipeline_url": "https://gitlab-master.nvidia.com/team/-/pipelines/30",
                },
                "latest_result": {
                    "result_id": 31,
                    "status": "failed",
                    "pipeline_url": "https://gitlab-master.nvidia.com/team/-/pipelines/31",
                },
            }
        ],
    }


def wrapped(
    category: str = "behavioral_failure",
    stage: str = "behavioral",
    changed: bool = False,
) -> dict[str, Any]:
    return {
        "contract_version": "bcb-failure-triage-input-v1",
        "connector_version": "0.9.0",
        "release_report": artifact(changed=changed),
        "failure": {
            "nim_id": 17,
            "result_id": 31,
            "status": "failed",
            "stage": stage,
            "failure_category": category,
            "evidence_url": "https://gitlab-master.nvidia.com/team/-/pipelines/31",
        },
        "comparison": {
            "prior_connector_version": "0.8.0",
            "prior_result_id": 30,
            "prior_status": "passed",
            "prior_fingerprint": "fingerprint-1",
            "prior_evidence_url": "https://gitlab-master.nvidia.com/team/-/pipelines/30",
        },
    }


def decision(
    data: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    clean = triage.sanitize(data)
    assessment = triage.assess(clean)
    return clean, assessment, triage.drafts(clean, assessment)


def test_same_nim_fingerprint_prior_pass_and_current_failure_suggest_sdk_review() -> (
    None
):
    _, result, _ = decision(wrapped())
    assert result["classification"] == "sdk_regression"
    assert result["confidence"] == "high"
    assert result["target"]["model"] == "nemotron-model"


def test_changed_nim_fingerprint_routes_to_manager_instead_of_connector_patch() -> None:
    _, result, _ = decision(wrapped(changed=True))
    assert result["classification"] == "nim_behavior_delta"
    assert result["owner"] == "Manager/BCB owner"


@pytest.mark.parametrize(
    "category,stage,expected",
    [
        ("authentication_failed", "setup", "credentials"),
        ("runner_unavailable", "setup", "infrastructure"),
        ("job_timeout", "setup", "infrastructure"),
        ("artifact_schema_invalid", "setup", "test_quality"),
        ("result_contract_missing", "setup", "test_quality"),
        ("test_fixture_defect", "behavioral", "test_quality"),
    ],
)
def test_setup_and_evidence_defects_do_not_generate_connector_patch(
    category: str, stage: str, expected: str
) -> None:
    _, result, _ = decision(wrapped(category, stage, changed=True))
    assert result["classification"] == expected
    assert result["confidence"] == "medium"


def test_missing_or_unrelated_result_falls_back_without_claiming_compatibility() -> (
    None
):
    sample = wrapped()
    sample["failure"]["result_id"] = 99
    _, result, _ = decision(sample)
    assert result["classification"] == "unknown"
    assert result["target"] is None


def test_release_projection_alone_cannot_attribute_failure_cause() -> None:
    clean, result, _ = decision(artifact(changed=True))
    assert result["classification"] == "unknown"
    assert result["confidence"] == "insufficient"
    assert clean["failure"]["stage"] == "unknown"


def test_missing_comparison_and_missing_link_prevent_sdk_attribution() -> None:
    sample = wrapped()
    sample["comparison"][
        "prior_evidence_url"
    ] = "https://gitlab-master.nvidia.com/team?private_token=secret"
    _, result, drafts = decision(sample)
    assert result["classification"] == "unknown"
    assert "private_token" not in json.dumps(drafts)
    sample = wrapped()
    sample["comparison"]["prior_status"] = "unknown"
    assert decision(sample)[1]["classification"] == "unknown"
    sample = wrapped()
    sample["comparison"][
        "prior_evidence_url"
    ] = "https://gitlab-master.nvidia.com/other/-/pipelines/30"
    assert decision(sample)[1]["classification"] == "unknown"
    sample = wrapped()
    sample["comparison"]["prior_result_id"] = 999
    assert decision(sample)[1]["classification"] == "unknown"


def test_untrusted_fields_are_discarded_in_json_and_markdown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = wrapped()
    sample["failure"]["raw_log"] = "Bearer secret-input-123"
    sample["release_report"]["targets"][0][
        "model_name"
    ] = "model](https://evil.example/?token=secret)"
    sample["release_report"]["targets"][0]["latest_result"][
        "artifact_url"
    ] = "https://evil.example/artifact"
    source = tmp_path / "input.json"
    out_json = tmp_path / "private" / "triage.json"
    out_md = tmp_path / "private" / "triage.md"
    source.write_text(json.dumps(sample), encoding="utf-8")
    assert (
        triage.main(
            [
                "--evidence",
                str(source),
                "--json-output",
                str(out_json),
                "--markdown-output",
                str(out_md),
            ]
        )
        == 0
    )
    assert out_json.stat().st_mode & 0o777 == 0o600
    combined = out_json.read_text() + out_md.read_text() + capsys.readouterr().out
    assert "secret-input-123" not in combined
    assert "evil.example" not in combined
    assert "token=secret" not in combined


def test_existing_output_is_not_overwritten_or_exported(tmp_path: Path) -> None:
    source = tmp_path / "input.json"
    source.write_text(json.dumps(wrapped()), encoding="utf-8")
    existing = tmp_path / "triage.json"
    existing.write_text("retain existing", encoding="utf-8")
    assert (
        triage.main(
            [
                "--evidence",
                str(source),
                "--json-output",
                str(existing),
                "--markdown-output",
                str(tmp_path / "triage.md"),
            ]
        )
        == 2
    )
    assert existing.read_text(encoding="utf-8") == "retain existing"
    assert not (tmp_path / "triage.md").exists()


def test_unrelated_failure_does_not_attach_another_targets_identity() -> None:
    sample = wrapped(changed=True)
    sample["failure"]["result_id"] = 999
    _, result, drafts = decision(sample)
    assert result["target"] is None
    serialized = json.dumps(drafts)
    assert "nemotron-model" not in serialized
    assert "fingerprint-1" not in serialized
    assert "fingerprint-2" not in serialized


def test_unbound_prior_version_is_not_presented_as_comparable() -> None:
    sample = wrapped()
    sample["comparison"]["prior_result_id"] = 999
    sample["comparison"]["prior_connector_version"] = "unbound-prior-version"
    _, result, drafts = decision(sample)
    assert result["classification"] == "unknown"
    assert "unbound-prior-version" not in json.dumps(drafts)


def observed_failure(
    http_status: object = 410, execution_scope: str = "native_probe"
) -> dict[str, Any]:
    sample = wrapped()
    observation = {
        "http_status": http_status,
        "workflow_id": 42,
        "execution_scope": execution_scope,
    }
    sample["failure"].update(observation)
    sample["release_report"]["targets"][0]["latest_result"].update(observation)
    return sample


@pytest.mark.parametrize("scope", ["native_probe", "sdk_test", "unknown"])
def test_hosted_410_drafts_expose_the_bound_observation_without_a_baseline(
    scope: str,
) -> None:
    sample = observed_failure(execution_scope=scope)
    sample["connector_version"] = ""
    sample["comparison"] = {}
    sample["release_report"]["targets"][0]["accepted_result"] = {}
    sample["release_report"]["targets"][0]["accepted_fingerprint"] = ""
    _, result, drafts = decision(sample)
    assert result["classification"] == "target_unavailable"
    assert result["owner"] == "Endpoint/catalog owner"
    assert result["observation"] == {
        "result_id": 31,
        "http_status": 410,
        "workflow_id": 42,
        "execution_scope": scope,
    }
    for key in ("issue", "pr_patch_suggestion", "release_note_docs_freshness"):
        body = drafts[key]["body"]
        assert "410 Gone" in body
        assert "42" in body
        assert "nemotron-model" in body
        assert result["owner"] in body
        assert result["next_action"] in body
        assert "https://gitlab-master.nvidia.com/team/-/pipelines/31" in body
        if scope == "native_probe":
            assert "SDK execution not established" in body
    assert drafts["publishable"] is False
    assert drafts["owner_review_required"] is True


@pytest.mark.parametrize(
    "field,value",
    [
        ("nim_id", 99),
        ("result_id", 99),
        ("workflow_id", 99),
        ("http_status", 404),
        ("execution_scope", "sdk_test"),
        ("evidence_url", "https://gitlab-master.nvidia.com/team/-/pipelines/99"),
    ],
)
def test_observation_mismatches_cannot_classify_or_attach_http_context(
    field: str, value: object
) -> None:
    sample = observed_failure()
    sample["failure"][field] = value
    _, result, drafts = decision(sample)
    assert result["classification"] == "unknown"
    assert result["observation"] == {}
    assert "HTTP status: 410" not in json.dumps(drafts)
    assert "Manager workflow: 42" not in json.dumps(drafts)


@pytest.mark.parametrize("status", [401, 404, 500, None, True, "410", 999])
def test_only_an_integer_hosted_410_establishes_endpoint_unavailability(
    status: object,
) -> None:
    _, result, _ = decision(observed_failure(status))
    assert result["classification"] == "unknown"


def test_downloadable_410_does_not_get_hosted_endpoint_routing() -> None:
    sample = observed_failure()
    sample["release_report"]["targets"][0]["deployment"] = "downloadable"
    _, result, _ = decision(sample)
    assert result["classification"] == "unknown"


@pytest.mark.parametrize(
    "scope,expected", [("native_probe", "unknown"), ("sdk_test", "sdk_regression")]
)
def test_native_probe_cannot_establish_a_connector_version_regression(
    scope: str, expected: str
) -> None:
    _, result, _ = decision(observed_failure(200, scope))
    assert result["classification"] == expected


def test_passing_latest_result_cannot_support_a_failed_http_diagnostic() -> None:
    sample = observed_failure()
    sample["release_report"]["targets"][0]["latest_result"]["status"] = "passed"
    _, result, _ = decision(sample)
    assert result["classification"] == "unknown"
    assert result["observation"] == {}


def test_generic_runner_failure_does_not_establish_endpoint_unavailability() -> None:
    _, result, _ = decision(wrapped(category="runner_failure", stage="setup"))
    assert result["classification"] == "unknown"


def test_http_context_is_private_and_does_not_copy_raw_diagnostics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sample = observed_failure()
    sample["failure"]["raw_log"] = "Bearer secret-context-123"
    sample["failure"]["workflow_url"] = "https://evil.example/?token=secret"
    source = tmp_path / "input.json"
    source.write_text(json.dumps(sample), encoding="utf-8")
    output = tmp_path / "private"
    assert (
        triage.main(
            [
                "--evidence",
                str(source),
                "--json-output",
                str(output / "triage.json"),
                "--markdown-output",
                str(output / "triage.md"),
            ]
        )
        == 0
    )
    report = json.loads((output / "triage.json").read_text())
    assert report["triage"]["observation"]["http_status"] == 410
    combined = (output / "triage.json").read_text() + (output / "triage.md").read_text()
    assert "secret-context-123" not in combined
    assert "evil.example" not in combined
    assert "token=secret" not in combined
    assert "secret-context-123" not in capsys.readouterr().out
    for path in (output / "triage.json", output / "triage.md"):
        assert path.stat().st_mode & 0o777 == 0o600


def test_malformed_observation_fields_do_not_reach_the_drafts() -> None:
    sample = observed_failure("Bearer private-status", "private-scope")
    sample["failure"]["workflow_id"] = "private-workflow"
    sample["release_report"]["targets"][0]["latest_result"]["workflow_id"] = True
    _, result, drafts = decision(sample)
    assert result["classification"] == "unknown"
    assert result["observation"] == {}
    assert "private-status" not in json.dumps(drafts)
    assert "private-workflow" not in json.dumps(drafts)
    assert "private-scope" not in json.dumps(drafts)


def test_missing_diagnostic_scope_does_not_promote_a_native_result_to_sdk() -> None:
    sample = observed_failure(200)
    for key in ("http_status", "workflow_id", "execution_scope"):
        del sample["failure"][key]
    _, result, _ = decision(sample)
    assert result["classification"] == "unknown"


@pytest.mark.parametrize("field", ["http_status", "workflow_id", "execution_scope"])
def test_malformed_source_observation_cannot_support_routing(field: str) -> None:
    sample = observed_failure()
    sample["release_report"]["targets"][0]["latest_result"][field] = "private-invalid"
    _, result, drafts = decision(sample)
    assert result["classification"] == "unknown"
    assert result["observation"] == {}
    assert "private-invalid" not in json.dumps(drafts)
