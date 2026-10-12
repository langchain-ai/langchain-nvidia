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


def test_render_is_public_safe_and_unknown_rows_are_not_verified() -> None:
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
