"""Evidence-bound BCB triage must not turn setup failures into connector defects."""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "bcb_failure_triage.py"
spec = importlib.util.spec_from_file_location("bcb_failure_triage", SCRIPT)
assert spec is not None and spec.loader is not None
triage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(triage)


def artifact(*, changed=False):
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


def wrapped(category="behavioral_failure", stage="behavioral", changed=False):
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


def decision(data):
    clean = triage.sanitize(data)
    assessment = triage.assess(clean)
    return clean, assessment, triage.drafts(clean, assessment)


def test_same_nim_fingerprint_prior_pass_and_current_failure_suggest_sdk_review():
    _, result, drafts = decision(wrapped())
    assert result["classification"] == "sdk_regression"
    assert result["confidence"] == "high"
    assert result["target"]["model"] == "nemotron-model"
    assert "targeted connector fix" in drafts["pr_patch_suggestion"]["body"]
    assert drafts["publishable"] is False


def test_changed_nim_fingerprint_routes_to_manager_instead_of_connector_patch():
    _, result, drafts = decision(wrapped(changed=True))
    assert result["classification"] == "nim_behavior_delta"
    assert result["owner"] == "Manager/BCB owner"
    assert "existing remediation workflow" in result["next_action"]
    assert "Do not propose a connector patch" in drafts["pr_patch_suggestion"]["body"]


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
    category, stage, expected
):
    _, result, drafts = decision(wrapped(category, stage, changed=True))
    assert result["classification"] == expected
    assert result["confidence"] == "medium"
    assert "Do not propose a connector patch" in drafts["pr_patch_suggestion"]["body"]


def test_missing_or_unrelated_result_falls_back_without_claiming_compatibility():
    sample = wrapped()
    sample["failure"]["result_id"] = 99
    _, result, drafts = decision(sample)
    assert result["classification"] == "unknown"
    assert result["target"] is None
    assert "model not established" in drafts["issue"]["title"]
    assert (
        "No compatibility or release claim"
        in drafts["release_note_docs_freshness"]["body"]
    )


def test_release_projection_alone_cannot_attribute_failure_cause():
    clean, result, drafts = decision(artifact(changed=True))
    assert result["classification"] == "unknown"
    assert result["confidence"] == "insufficient"
    assert clean["failure"]["stage"] == "unknown"
    assert "Do not propose a connector patch" in drafts["pr_patch_suggestion"]["body"]


def test_missing_comparison_and_missing_link_prevent_sdk_attribution():
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


def test_untrusted_fields_are_discarded_in_json_and_markdown(tmp_path, capsys):
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
    assert "model not established" in combined
    assert "No AI model was called" in combined
    assert "Explicit maintainer review required" in combined


def test_existing_output_is_not_overwritten_or_exported(tmp_path):
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
