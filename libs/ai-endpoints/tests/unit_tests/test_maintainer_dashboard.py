"""Private dashboard consumes local review data without publishing or fetching."""

import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "maintainer_dashboard.py"
sys.path.insert(0, str(SCRIPT.parent))
spec = importlib.util.spec_from_file_location("maintainer_dashboard", SCRIPT)
assert spec is not None and spec.loader is not None
view = importlib.util.module_from_spec(spec)
spec.loader.exec_module(view)

NOW = datetime.now(timezone.utc)
RECENT = (NOW - timedelta(hours=2)).isoformat()
OLD = (NOW - timedelta(days=30)).isoformat()
EVIDENCE = "https://gitlab-master.nvidia.com/team/bcb/-/jobs/62"
PR = "https://github.com/langchain-ai/langchain-nvidia/pull/372"
ISSUE = "https://github.com/langchain-ai/langchain-nvidia/issues/956"


def review() -> dict:
    return {
        "schema_version": 1,
        "data_classification": "internal_restricted",
        "release_owner": "release-maintainer",
        "generated_pr_url": PR,
        "issues": [
            {
                "issue_url": ISSUE,
                "priority": "P1",
                "affected_version": "0.4.0",
                "repro": "chat tools fails on retry",
                "owner": "connector-maintainer",
                "tests": "regression added",
                "planned_release": "0.4.1",
                "triage_url": ISSUE,
                "evidence_url": EVIDENCE,
            }
        ],
        "adoption": {
            "downloads": {
                "status": "available",
                "count": 19,
                "observed_at": RECENT,
                "source_url": "https://pypi.org/project/langchain-nvidia-ai-endpoints/",
            },
            "docs_traffic": {"status": "unavailable"},
            "cookbook_runs": {"status": "unavailable"},
        },
    }


def release() -> dict:
    def target(kind: str, identifier: int) -> dict:
        return {
            "deployment": kind,
            "nim_id": identifier,
            "status": "pass",
            "reasons": [],
            "required_capabilities": ["chat_basic", "chat_tools"],
            "freshness": {"state": "known", "stale": False},
            "tested_at": RECENT,
            "accepted_result": {
                "result_id": identifier + 60,
                "status": "passed",
                "job_url": EVIDENCE,
            },
            "latest_result": {
                "result_id": identifier + 60,
                "status": "passed",
                "job_url": EVIDENCE,
            },
        }

    return {
        "contract_version": "bcb-public-v1alpha1",
        "baseline_version": "release-readiness-2026-08-28-v2",
        "status": "pass",
        "promotion": "owner_approval_required",
        "collected_at": RECENT,
        "reasons": [],
        "gate": {"status": "pass", "evaluated_at": RECENT, "reasons": []},
        "targets": [target("hosted", 1), target("downloadable", 2)],
    }


def test_private_view_shows_review_actions_and_all_drift_categories() -> None:
    changes = [
        {
            "kind": kind,
            "id": "nvidia/example",
            "source": "hosted",
            "evidence": "https://integrate.api.nvidia.com/v1/models",
        }
        for kind in (
            "addition",
            "deprecation",
            "aliases",
            "capability",
            "removal/stale",
        )
    ]
    rendered = view.render(
        review(), release(), changes, {}, ["downloadable snapshot unavailable"], NOW, 14
    )
    assert "review evidence (not approval)" in rendered
    assert "owner/security approval required" in rendered
    assert "chat_basic, chat_tools" in rendered
    assert "#61 passed" in rendered
    assert EVIDENCE in rendered and PR in rendered
    assert "Additions (1)" in rendered and "Deprecations / removals (2)" in rendered
    assert (
        "Aliases / served names (1)" in rendered
        and "Capability flags (catalog claim only) (1)" in rendered
    )
    assert "downloadable snapshot unavailable" in rendered
    assert "P1" in rendered and "0.4.1" in rendered and ISSUE in rendered
    assert "| downloads | 19 |" in rendered
    assert "| docs traffic | unavailable |" in rendered
    assert "| github prs | unavailable |" in rendered


@pytest.mark.parametrize(
    "mutation",
    [
        "old_gate",
        "old_run",
        "stale_target",
        "unknown_target",
        "missing_target",
        "missing_evidence",
        "bad_contract",
    ],
)
def test_stale_unknown_and_missing_evidence_never_look_ready(mutation: str) -> None:
    artifact = release()
    if mutation == "old_gate":
        artifact["gate"]["evaluated_at"] = OLD
    elif mutation == "old_run":
        artifact["targets"][0]["tested_at"] = OLD
    elif mutation == "stale_target":
        artifact["targets"][0]["freshness"]["stale"] = True
    elif mutation == "unknown_target":
        artifact["targets"][0]["status"] = "warn"
    elif mutation == "missing_target":
        artifact["targets"].pop()
    elif mutation == "missing_evidence":
        artifact["targets"][0]["accepted_result"]["job_url"] = ""
    else:
        artifact["contract_version"] = "unexpected"
    text = view.render(review(), artifact, [], {}, [], NOW, 14)
    assert "HOLD — stale, unknown or blocked evidence" in text
    assert "review evidence (not approval)" not in text


def test_hostile_link_and_credential_fields_do_not_render() -> None:
    data = review()
    data["release_owner"] = "Bearer SECRET123"
    data[
        "generated_pr_url"
    ] = "https://github.com.evil.test/langchain-ai/langchain-nvidia/pull/372"
    data["issues"][0]["repro"] = "token abcdefgh123456"
    data["issues"][0]["evidence_url"] = "https://gitlab-master.nvidia.com@evil.test/x"
    data["adoption"]["downloads"][
        "source_url"
    ] = "https://pypi.org/project/example/?api_key=SECRET123"
    text = view.render(data, release(), [], {}, [], NOW, 14)
    assert "SECRET123" not in text and "abcdef" not in text and "evil.test" not in text
    assert "Generated review PR: unavailable" in text
    assert "| downloads | unavailable |" in text
    assert "fill repro" in text


def test_cli_uses_local_catalog_and_reports_missing_sources_without_writes(
    tmp_path: Path,
) -> None:
    paths = [tmp_path / name for name in ("review.json", "release.json", "hosted.json")]
    paths[0].write_text(json.dumps(review()), encoding="utf-8")
    paths[1].write_text(json.dumps(release()), encoding="utf-8")
    paths[2].write_text(
        json.dumps(
            {
                "source_url": "https://integrate.api.nvidia.com/v1/models",
                "data": [{"id": "nvidia/new-reviewed-candidate"}],
            }
        ),
        encoding="utf-8",
    )
    before = {path: path.read_bytes() for path in paths}
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--review-file",
            str(paths[0]),
            "--release-file",
            str(paths[1]),
            "--hosted-file",
            str(paths[2]),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "nvidia/new-reviewed-candidate" in result.stdout
    assert "downloadable snapshot unavailable" in result.stdout
    assert "PRIVATE REVIEW ONLY" in result.stdout
    assert before == {path: path.read_bytes() for path in paths}
    assert sorted(tmp_path.iterdir()) == sorted(paths)


def test_invalid_classification_fails_closed_without_echoing_private_input(
    tmp_path: Path,
) -> None:
    artifact = review()
    artifact["data_classification"] = "public"
    artifact["release_owner"] = "secret-test-data"
    input_path = tmp_path / "review.json"
    release_path = tmp_path / "release.json"
    input_path.write_text(json.dumps(artifact), encoding="utf-8")
    release_path.write_text(json.dumps(release()), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--review-file",
            str(input_path),
            "--release-file",
            str(release_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 2
    assert "secret-test-data" not in result.stdout + result.stderr
    assert "no view rendered" in result.stderr
