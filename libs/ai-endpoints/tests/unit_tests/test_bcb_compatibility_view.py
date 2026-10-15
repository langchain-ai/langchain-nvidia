"""Compatibility is a recent Manager framework claim, never a catalog capability."""

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/bcb_compatibility_view.py"
SPEC = importlib.util.spec_from_file_location("bcb_compatibility_view", SCRIPT)
assert SPEC and SPEC.loader
view = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(view)
NOW = datetime(2026, 9, 28, tzinfo=timezone.utc)


def evidence() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    target = {"id": 7, "model_name": "nvidia/example", "family": "nemotron"}
    freshness = {
        "state": "known",
        "stale": False,
        "observed_at": "2026-09-27T12:00:00Z",
    }
    result = {
        "result_id": 12,
        "status": "passed",
        "completed_at": "2026-09-27T12:00:00Z",
    }
    compatible = {
        "nim_target": target,
        "status": "compatible",
        "compatible": True,
        "freshness": freshness,
        "framework_results": [
            {
                "framework": "langchain-nvidia",
                "active": True,
                "compatible": True,
                "status": "passed",
            }
        ],
        "evidence": {"latest_result": result},
    }
    recommend = {
        "nim_target": target,
        "status": "compatible",
        "compatibility_status": "current",
        "freshness": freshness,
        "recommended_digest": "sha256:abcd",
        "recommended_action": "no_action",
        "tested_at": freshness["observed_at"],
        "fingerprint": {"family": "nemotron", "hash": "abc"},
        "accepted_baseline": {"digest": "sha256:abcd", "fingerprint_hash": "abc"},
        "latest_observation": {
            "digest": "sha256:abcd",
            "fingerprint_hash": "abc",
            "status": "passed",
        },
    }
    badge = {
        "nim_target": target,
        "status": "compatible",
        "compatible": True,
        "baseline_state": "current",
        "freshness": freshness,
    }
    return compatible, recommend, badge


ITEM = {
    "model": "nvidia/example",
    "nim_target_id": 7,
    "deployment_type": "downloadable",
}


def classify(
    compatible: dict[str, Any], recommend: dict[str, Any], badge: dict[str, Any]
) -> dict[str, Any]:
    return view.classify(ITEM, compatible, recommend, badge, now=NOW, max_age_days=30)


def test_verified_framework_never_becomes_verified_capability() -> None:
    c, r, b = evidence()
    row = classify(c, r, b)
    assert row["status"] == "verified"
    assert set(row["capabilities"]) == set(view.CAPABILITIES)
    assert set(row["capabilities"].values()) == {"unknown"}
    assert row["gpu_runtime"] == "unavailable/redacted"
    badges = view.badge_payload([row])
    assert badges["nvidia/example:7"]["message"] == "verified"
    assert "structured output" not in str(badges)


def test_hosted_framework_evidence_does_not_require_image_digest() -> None:
    compatible, recommend, badge = evidence()
    recommend["recommended_digest"] = ""
    recommend["accepted_baseline"]["digest"] = ""
    recommend["latest_observation"]["digest"] = ""
    hosted = {**ITEM, "deployment_type": "hosted"}

    assert (
        view.classify(hosted, compatible, recommend, badge, now=NOW, max_age_days=30)[
            "status"
        ]
        == "verified"
    )
    assert classify(compatible, recommend, badge)["status"] == "unknown"


@pytest.mark.parametrize(
    "change,expected",
    [
        ("wrong_target", "unknown"),
        ("wrong_model", "unknown"),
        ("missing_link", "unknown"),
        ("inactive_link", "unknown"),
        ("changed", "changed"),
        ("failed", "failed"),
        ("stale", "stale"),
        ("old_date", "stale"),
        ("future_date", "stale"),
        ("no_date", "unknown"),
        ("different_digest", "unknown"),
        ("not_compatible", "unknown"),
    ],
)
def test_latest_identity_freshness_and_failure_precedence(
    change: str, expected: str
) -> None:
    c, r, b = evidence()
    if change == "wrong_target":
        b["nim_target"] = {**b["nim_target"], "id": 8}
    elif change == "wrong_model":
        c["nim_target"] = {**c["nim_target"], "model_name": "other"}
    elif change == "missing_link":
        c["framework_results"] = []
    elif change == "inactive_link":
        c["framework_results"][0]["active"] = False
    elif change in ("changed", "failed"):
        r["compatibility_status"] = change
    elif change == "stale":
        b["freshness"] = {**b["freshness"], "stale": True}
    elif change in ("old_date", "future_date", "no_date"):
        c["freshness"] = {
            **c["freshness"],
            "observed_at": {
                "old_date": (NOW - timedelta(days=31)).isoformat(),
                "future_date": (NOW + timedelta(days=1)).isoformat(),
                "no_date": "",
            }[change],
        }
    elif change == "different_digest":
        r["latest_observation"]["digest"] = "sha256:changed"
    elif change == "not_compatible":
        c["compatible"] = False
    row = classify(c, r, b)
    assert row["status"] == expected
    assert row["recommended_action"] != "no_action"
    assert view.badge_payload([row])["nvidia/example:7"]["color"] != "brightgreen"


def test_private_render_escapes_values_and_does_not_verify_unknown_evidence() -> None:
    row = view.unavailable(ITEM)
    row["digest"] = "<private|value>"
    output = view.render(
        [row], {"summary": {"blocked_pairs": 2, "total_pairs": 3}}, "1.4.3", NOW, 30
    )
    assert "&lt;private&#124;value&gt;" in output
    assert "1 framework-verified" not in output
    assert "capability" in output
    assert "blocked pairs 2" in output
    assert "unavailable/redacted" in output


def test_selection_rejects_unreviewed_model_and_ambiguous_target() -> None:
    registry = {"nvidia/example": {"capabilities": {"supports_tools": "supported"}}}
    assert view.selection([ITEM], registry) == [ITEM]
    with pytest.raises(ValueError):
        view.selection([ITEM, ITEM], registry)
    with pytest.raises(ValueError):
        view.selection([{**ITEM, "model": "unreviewed/model"}], registry)
    with pytest.raises(ValueError):
        view.selection([{**ITEM, "deployment_type": "both"}], registry)


def test_missing_latest_or_disagreeing_projection_cannot_be_verified() -> None:
    c, r, b = evidence()
    r["latest_observation"] = None
    assert classify(c, r, b)["status"] == "unknown"
    c, r, b = evidence()
    r["freshness"]["observed_at"] = "2026-08-01T00:00:00Z"
    assert classify(c, r, b)["status"] == "stale"


def test_normalized_status_keeps_framework_verdict_not_capability() -> None:
    c, r, b = evidence()
    # Manager's v1alpha1 status allowlist normalizes raw \"success\" to \"unknown\";
    # its explicit compatible=True result remains the source-owned passing check.
    c["framework_results"][0]["status"] = "unknown"
    r["latest_observation"]["status"] = "unknown"
    row = classify(c, r, b)
    assert row["status"] == "verified"
    assert row["capabilities"]["structured output"] == "unknown"


@pytest.mark.parametrize("change", ["missing", "empty", "wrong_recommendation"])
def test_absent_or_inconsistent_fingerprints_cannot_produce_green_badge(
    change: str,
) -> None:
    c, r, b = evidence()
    if change == "wrong_recommendation":
        r["fingerprint"]["hash"] = "another"
    else:
        for observation in (r["accepted_baseline"], r["latest_observation"]):
            observation["fingerprint_hash"] = None if change == "missing" else ""
    row = classify(c, r, b)
    assert row["status"] == "unknown"
    assert view.badge_payload([row])["nvidia/example:7"]["color"] != "brightgreen"


def test_specific_source_action_survives_but_wrong_target_action_does_not() -> None:
    c, r, b = evidence()
    r["compatibility_status"] = "missing"
    r["recommended_action"] = "run_baseline"
    assert classify(c, r, b)["recommended_action"] == "run_baseline"
    b["nim_target"] = {**b["nim_target"], "id": 8}
    assert classify(c, r, b)["recommended_action"] == "review"


@pytest.mark.parametrize(
    "change,expected",
    [
        ("none", "passed"),
        ("wrong_result", "unknown"),
        ("wrong_target", "unknown"),
        ("wrong_hash", "unknown"),
        ("old_date", "unknown"),
        ("skipped_case", "unknown"),
        ("failed_case", "failed"),
    ],
)
def test_native_observations_require_result_target_hash_time_and_case_binding(
    change: str,
    expected: str,
) -> None:
    c, r, b = evidence()
    latest = c["evidence"]["latest_result"]
    latest.update(
        {
            "nim_target_id": 7,
            "fingerprint_hash": "abc",
            "probe_evidence": {
                "contract_version": "bcb-native-probe-evidence-v1",
                "suite_results": [
                    {
                        "suite": "tool_calling_conformance",
                        "status": "passed",
                        "cases": [{"case_id": "tool_shape", "status": "passed"}],
                    }
                ],
            },
        }
    )
    r["latest_observation"]["result_id"] = 12
    if change == "wrong_result":
        r["latest_observation"]["result_id"] = 13
    elif change == "wrong_target":
        latest["nim_target_id"] = 8
    elif change == "wrong_hash":
        latest["fingerprint_hash"] = "another"
    elif change == "old_date":
        latest["completed_at"] = (NOW - timedelta(days=31)).isoformat()
    elif change in ("skipped_case", "failed_case"):
        latest["probe_evidence"]["suite_results"][0]["cases"][0]["status"] = (
            "skipped" if change == "skipped_case" else "failed"
        )
    row = classify(c, r, b)
    assert row["native_probe_states"]["tools"] == expected
    assert row["capabilities"]["tools"] == "unknown"


def test_installed_sdk_inventory_does_not_certify_sdk_execution() -> None:
    c, r, b = evidence()
    c["framework_results"][0].update(
        {
            "project_target_id": 3,
            "validation_evidence": {
                "contract_version": "bcb-validation-evidence-v1",
                "result_id": 14,
                "project_target_id": 3,
                "nim_target_id": ITEM["nim_target_id"],
                "model_name": ITEM["model"],
                "target_snapshot": {
                    "id": ITEM["nim_target_id"],
                    "model_name": ITEM["model"],
                },
                "completed_at": "2026-09-27T12:00:00Z",
                "executor": "hosted_endpoint",
                "checks": [
                    {
                        "id": "langchain_chatopenai_tool_deserialization",
                        "status": "passed",
                        "evidence_class": "project_integration",
                        "package_versions": {"langchain-nvidia-ai-endpoints": "1.4.3"},
                    }
                ],
            },
        }
    )
    row = classify(c, r, b)
    assert row["tested_package_version"] == "unknown"
    assert row["capabilities"]["tools"] == "unknown"


def sdk_evidence() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    c, r, b = evidence()
    c["framework_results"][0].update(
        project_target_id=3,
        validation_evidence={
            "contract_version": "bcb-validation-evidence-v1",
            "result_id": 14,
            "project_target_id": 3,
            "nim_target_id": ITEM["nim_target_id"],
            "model_name": ITEM["model"],
            "target_snapshot": {
                "id": ITEM["nim_target_id"],
                "model_name": ITEM["model"],
            },
            "completed_at": "2026-09-27T12:01:00Z",
            "executor": "hosted_endpoint",
            "status": "passed",
            "evidence_class": "project_integration",
            "source_owner": "NIM OSS Manager BCB",
            "project_revision": view.SDK_REVISION,
            "fingerprint_hash": "a" * 64,
            "checks": [
                {
                    "id": "langchain_chatnvidia_chat_streaming",
                    "profile": view.SDK_PROFILE,
                    "profile_version": 1,
                    "status": "passed",
                    "evidence_class": "project_integration",
                    "completed_at": "2026-09-27T12:00:00Z",
                    "package_versions": dict(view.SDK_PACKAGES),
                    "structural_assertions": {
                        **dict.fromkeys(view.SDK_ASSERTIONS, True),
                        "chunk_count": 3,
                    },
                }
            ],
        },
    )
    return c, r, b


def test_version_bound_sdk_checks_certify_only_chat_and_streaming() -> None:
    row = view.classify(
        {**ITEM, "deployment_type": "hosted"}, *sdk_evidence(), now=NOW, max_age_days=30
    )
    assert row["tested_package_version"] == "1.5.0"
    assert row["sdk_result_ids"] == [14]
    assert row["capabilities"]["chat"] == "passed"
    assert row["capabilities"]["streaming"] == "passed"
    assert all(
        row["capabilities"][name] == "unknown"
        for name in view.CAPABILITIES
        if name not in {"chat", "streaming"}
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "sdk_flag",
        "version",
        "revision",
        "model",
        "target",
        "deployment",
        "failed",
        "stale",
        "future_check",
        "chunk_count",
        "owner",
    ],
)
def test_sdk_evidence_must_be_current_complete_and_bound(mutation: str) -> None:
    c, r, b = sdk_evidence()
    context = c["framework_results"][0]["validation_evidence"]
    check = context["checks"][0]
    if mutation == "sdk_flag":
        check["structural_assertions"]["sdk_executed"] = False
    elif mutation == "version":
        check["package_versions"]["langchain-nvidia-ai-endpoints"] = "0.0.1"
    elif mutation == "revision":
        context["project_revision"] = "b" * 40
    elif mutation == "model":
        context["model_name"] = "another/model"
    elif mutation == "target":
        context["target_snapshot"]["id"] = 999
    elif mutation == "deployment":
        context["executor"] = "downloadable_container"
    elif mutation == "failed":
        context["status"] = "failed"
    elif mutation == "stale":
        context["completed_at"] = "2020-01-01T00:00:00Z"
    elif mutation == "future_check":
        check["completed_at"] = "2026-10-01T00:00:00Z"
    elif mutation == "chunk_count":
        check["structural_assertions"]["chunk_count"] = True
    else:
        context["source_owner"] = "untrusted"
    row = view.classify(
        {**ITEM, "deployment_type": "hosted"}, c, r, b, now=NOW, max_age_days=30
    )
    assert row["tested_package_version"] == "unknown"
    assert all(state == "unknown" for state in row["capabilities"].values())
