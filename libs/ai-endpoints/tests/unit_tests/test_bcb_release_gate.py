"""BCB report decisions are based on Manager evidence, not CI job success."""

import importlib.util
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "bcb_release_gate.py"
spec = importlib.util.spec_from_file_location("bcb_release_gate", SCRIPT)
assert spec is not None and spec.loader is not None
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


def gate() -> dict[str, Any]:
    return {
        "ok": True,
        "baseline_version": "release-readiness-2026-08-28-v2",
        "status": "decision_ready",
        "release_ready": True,
        "summary": {
            "numerator": 2,
            "denominator": 2,
            "decision_ready_pairs": 2,
            "blocked_pairs": 0,
            "stale_pairs": 0,
        },
        "scope": {"projects": ["langchain-nvidia"], "targets": [{"id": 1}, {"id": 2}]},
        "evidence": {"checks_count": 2, "evaluated_at": "2026-09-27T09:00:00Z"},
    }


def recommendation(nim_id: int) -> dict[str, Any]:
    return {
        "ok": True,
        "status": "compatible",
        "compatibility_status": "current",
        "nim_target": {"id": nim_id},
        "accepted_baseline": {
            "digest": "sha256:abc",
            "fingerprint_hash": "fp-abc",
            "fingerprint_family": "chat",
            "result_id": 61,
        },
        "latest_observation": {
            "digest": "sha256:abc",
            "fingerprint_hash": "fp-abc",
            "status": "passed",
            "result_id": 61,
        },
        "freshness": {"state": "known", "stale": False},
        "tested_at": "2026-09-27T09:00:00Z",
        "evidence": {
            "accepted_result": {
                "result_id": 61,
                "status": "passed",
                "pipeline_url": "https://gitlab-master.nvidia.com/a/-/pipelines/42",
            },
            "latest_result": {"result_id": 61, "status": "passed"},
        },
    }


def selected(
    kind: str, identifier: int, capabilities: str = "chat_basic,chat_tools"
) -> dict[str, Any]:
    return reporter.target_config(kind, str(identifier), capabilities)


def assessment(
    hosted: dict[str, Any] | None = None,
    downloadable: dict[str, Any] | None = None,
    gate_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    hosted = hosted if hosted is not None else recommendation(1)
    downloadable = downloadable if downloadable is not None else recommendation(2)
    return reporter.assemble(
        reporter.gate_report(gate_payload or gate()),
        [
            reporter.target_report(selected("hosted", 1), hosted),
            reporter.target_report(selected("downloadable", 2), downloadable),
        ],
        collected_at="2026-09-28T00:00:00Z",
    )


def test_fresh_accepted_evidence_for_both_targets_is_pass_without_promotion() -> None:
    result = assessment()
    assert result["status"] == "pass"
    assert result["promotion"] == "owner_approval_required"
    assert result["targets"][0]["required_capabilities"] == ["chat_basic", "chat_tools"]
    assert result["targets"][0]["accepted_digest"] == "sha256:abc"
    assert result["targets"][0]["accepted_result"]["result_id"] == 61


def test_hosted_fingerprint_only_baseline_can_pass_without_image_digest() -> None:
    target = recommendation(1)
    target["accepted_baseline"]["digest"] = ""
    target["latest_observation"]["digest"] = ""

    hosted = reporter.target_report(selected("hosted", 1), target)
    downloadable = reporter.target_report(selected("downloadable", 1), target)

    assert hosted["status"] == "pass"
    assert hosted["accepted_fingerprint"] == hosted["observed_fingerprint"]
    assert downloadable["status"] == "warn"
    assert "fingerprint_or_digest_missing" in downloadable["reasons"]


@pytest.mark.parametrize(
    "condition",
    ["missing", "stale", "unknown", "not_found", "no_fingerprint", "no_result"],
)
def test_missing_or_uncertain_required_evidence_never_passes(condition: str) -> None:
    target = recommendation(1)
    if condition in {"missing", "stale", "unknown"}:
        target["compatibility_status"] = condition
        target["status"] = condition
    elif condition == "not_found":
        target["ok"] = False
        target["nim_target"]["id"] = None
    elif condition == "no_fingerprint":
        target["accepted_baseline"]["fingerprint_hash"] = ""
    else:
        target["evidence"]["accepted_result"] = None
    assert assessment(hosted=target)["status"] == "warn"


@pytest.mark.parametrize(
    "condition", ["changed", "failed", "blocked", "digest_mismatch", "latest_failure"]
)
def test_changed_or_failed_required_target_blocks(condition: str) -> None:
    target = recommendation(1)
    if condition in {"changed", "failed", "blocked"}:
        target["compatibility_status"] = condition
        target["status"] = condition
    elif condition == "digest_mismatch":
        target["latest_observation"]["digest"] = "sha256:different"
    else:
        target["evidence"]["latest_result"]["status"] = "failed"
    result = assessment(hosted=target)
    assert result["status"] == "block"
    assert result["targets"][0]["status"] == "block"


def test_unselected_target_failure_does_not_block_selected_target() -> None:
    target = recommendation(99)
    target["compatibility_status"] = "failed"
    target["status"] = "failed"
    result = assessment(hosted=target)
    assert result["status"] == "warn"
    assert "target_not_found_or_mismatched" in result["targets"][0]["reasons"]


@pytest.mark.parametrize(
    "condition", ["blocked", "guard", "count_mismatch", "unknown", "stale", "no_checks"]
)
def test_gate_failures_cannot_be_green(condition: str) -> None:
    payload = gate()
    if condition == "blocked":
        payload["status"] = "blocked"
        payload["release_ready"] = False
        payload["summary"]["blocked_pairs"] = 1
    elif condition == "guard":
        payload["release_guards"] = [
            {"key": "post_v1_backlog", "status": "blocked", "count": 1}
        ]
    elif condition == "count_mismatch":
        payload["summary"]["numerator"] = 1
    elif condition == "unknown":
        payload["ok"] = False
        payload["status"] = "unknown_baseline"
    elif condition == "stale":
        payload["summary"]["stale_pairs"] = 1
    else:
        payload["evidence"]["checks_count"] = 0
    assert assessment(gate_payload=payload)["status"] == (
        "block" if condition in {"blocked", "guard"} else "warn"
    )


def test_unsafe_evidence_pointers_and_unexpected_fields_are_not_reported() -> None:
    target = recommendation(1)
    target["evidence"]["accepted_result"].update(
        {
            "pipeline_url": "https://gitlab-master.nvidia.com/a?access_token=secret",
            "job_url": "https://other.example.org/job",
            "artifact_url": "https://gitlab-master.nvidia.com/a/artifact",
            "authorization": "Bearer secret",
        }
    )
    target["raw_callback"] = {"token": "secret"}
    result = assessment(hosted=target)
    serialized = json.dumps(result) + reporter.markdown(result)
    assert "secret" not in serialized
    assert "other.example.org" not in serialized
    assert "https://gitlab-master.nvidia.com/a/artifact" in serialized


def test_cli_reads_only_public_routes_and_writes_artifacts_without_live_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []

    def fake_fetch(
        base: str, token: str, endpoint: str, query: dict[str, str]
    ) -> dict[str, Any]:
        calls.append((base, token, endpoint, query))
        payload = (
            gate()
            if endpoint == "release-gate"
            else recommendation(int(query["nim_id"]))
        )
        return {**payload, "contract_version": reporter.CONTRACT, "readonly": True}

    monkeypatch.setattr(reporter, "fetch", fake_fetch)
    monkeypatch.setenv("BCB_MANAGER_TOKEN", "fake-test-token")
    path = tmp_path / "evidence"
    assert (
        reporter.main(
            [
                "--manager-url",
                "https://manager.example.internal",
                "--hosted-nim-id",
                "1",
                "--hosted-required",
                "chat_basic,chat_tools",
                "--downloadable-nim-id",
                "2",
                "--downloadable-required",
                "chat_basic",
                "--json-output",
                str(path / "report.json"),
                "--markdown-output",
                str(path / "report.md"),
            ]
        )
        == 0
    )
    assert json.loads((path / "report.json").read_text())["status"] == "pass"
    assert "fake-test-token" not in (path / "report.md").read_text()
    assert [(endpoint, query) for _, _, endpoint, query in calls] == [
        ("release-gate", {}),
        ("recommend", {"nim_id": "1"}),
        ("recommend", {"nim_id": "2"}),
    ]


def test_override_and_unknown_baseline_preserve_requested_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = []

    def fake_fetch(
        base: str, token: str, endpoint: str, query: dict[str, str]
    ) -> dict[str, Any]:
        calls.append((endpoint, query))
        if endpoint == "release-gate":
            return {
                "contract_version": reporter.CONTRACT,
                "readonly": True,
                "ok": False,
                "status": "unknown_baseline",
                "baseline_version": query["baseline_version"],
            }
        return {
            **recommendation(int(query["nim_id"])),
            "contract_version": reporter.CONTRACT,
            "readonly": True,
        }

    monkeypatch.setattr(reporter, "fetch", fake_fetch)
    monkeypatch.setenv("BCB_MANAGER_TOKEN", "fake-test-token")
    output = tmp_path / "report.json"
    assert (
        reporter.main(
            [
                "--manager-url",
                "https://manager.example.internal",
                "--baseline-version",
                "retired-v1",
                "--hosted-nim-id",
                "1",
                "--hosted-required",
                "chat_basic",
                "--downloadable-nim-id",
                "2",
                "--downloadable-required",
                "chat_basic",
                "--json-output",
                str(output),
                "--markdown-output",
                str(tmp_path / "report.md"),
            ]
        )
        == 0
    )
    result = json.loads(output.read_text())
    assert result["status"] == "warn"
    assert result["baseline_version"] == "retired-v1"
    assert calls[0] == ("release-gate", {"baseline_version": "retired-v1"})


def test_ready_gate_cannot_cover_target_outside_scope() -> None:
    payload = gate()
    payload["scope"]["targets"] = [{"id": 1}]
    result = assessment(gate_payload=payload)
    assert result["status"] == "warn"
    assert "downloadable_not_in_gate_scope" in result["reasons"]


def test_missing_token_produces_warning_artifact_without_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("BCB_MANAGER_TOKEN", raising=False)
    result_file = tmp_path / "report.json"
    assert (
        reporter.main(
            [
                "--manager-url",
                "https://manager.example.internal",
                "--json-output",
                str(result_file),
                "--markdown-output",
                str(tmp_path / "report.md"),
            ]
        )
        == 0
    )
    assert json.loads(result_file.read_text())["status"] == "warn"


def test_rejects_unreviewed_capabilities_and_partial_selection() -> None:
    with pytest.raises(ValueError):
        selected("hosted", 1, "chat_basic,chat_basic")
    with pytest.raises(ValueError):
        reporter.target_config("downloadable", "2", "")
    with pytest.raises(ValueError):
        selected("hosted", 1, "chat_basic;injected")


def test_redirect_never_forwards_bearer_token() -> None:
    observed = []

    class Redirect(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            observed.append((self.path, self.headers.get("Authorization")))
            self.send_response(302)
            self.send_header("Location", "/untrusted")
            self.end_headers()

        def log_message(self, *args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Redirect)
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        with pytest.raises(
            RuntimeError, match="release-gate request unavailable"
        ) as error:
            reporter.fetch(
                f"http://127.0.0.1:{server.server_port}",
                "test-bearer",
                "release-gate",
                {},
            )
        assert observed == [("/api/bcb/public/v1/release-gate", "Bearer test-bearer")]
        assert "test-bearer" not in str(error.value)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_unknown_baseline_http_404_keeps_safe_response_and_warns() -> None:
    class Missing(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            assert self.path.endswith("?baseline_version=retired-v1")
            body = json.dumps(
                {
                    "contract_version": reporter.CONTRACT,
                    "readonly": True,
                    "endpoint": "release-gate",
                    "api_scope": "bcb-public-alpha",
                    "api_visibility": "internal-alpha",
                    "public_api": False,
                    "ok": False,
                    "status": "unknown_baseline",
                    "baseline_version": "retired-v1",
                }
            ).encode()
            self.send_response(404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Missing)
    thread = Thread(target=server.serve_forever)
    thread.start()
    try:
        payload = reporter.fetch(
            f"http://127.0.0.1:{server.server_port}",
            "test-bearer",
            "release-gate",
            {"baseline_version": "retired-v1"},
        )
        outcome = reporter.gate_report(payload, requested_version="retired-v1")
        assert outcome["status"] == "warn"
        assert outcome["baseline_version"] == "retired-v1"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
