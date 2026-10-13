"""Consumer-visible privacy and evidence boundaries of offline publication."""

import copy
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/bcb_compatibility_publish.py"
SPEC = importlib.util.spec_from_file_location("bcb_compatibility_publish", SCRIPT)
assert SPEC and SPEC.loader
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)
NOW = datetime(2026, 10, 13, 12, tzinfo=timezone.utc)
MODEL = "nvidia/nemotron-3-super-120b-a12b"
IDS = {MODEL, "a/b-c", "a-b/c"}


def decision(**changes: Any) -> dict[str, Any]:
    row = {
        "model": MODEL,
        "nim_target_id": 94721,
        "deployment_type": "hosted",
        "deployment_evidence": "hosted",
        "status": "verified",
        "freshness": "current",
        "tested_at": publisher.stamp(NOW - timedelta(hours=1)),
        "capabilities": {key: "verified" for key in publisher.CAPABILITIES},
        "tested_package_version": "0.99.1",
        "tested_packages": {"langchain-nvidia-ai-endpoints": ["0.99.1"]},
        "framework_checks": [{"id": "chat_openai", "status": "passed"}],
        "native_probe_states": {key: "passed" for key in publisher.CAPABILITIES},
        "digest": "sha256:private-container-secret",
        "fingerprint_hash": "private-fingerprint-secret",
        "result_ids": {"latest_result": 984731},
        "evidence_links": [{"url": "https://private.example/job/984731"}],
        "source_owner": "private-person-name",
        "limitations": "private-freeform-text with [link](https://private.example)",
        "gpu_runtime": "private-gpu-runtime",
    }
    row.update(changes)
    return {
        "schema_version": 1,
        "framework": "langchain-nvidia",
        "generated_at": publisher.stamp(NOW),
        "max_age_days": 30,
        "rows": [row],
    }


def public(**changes: Any) -> dict[str, Any]:
    return publisher.export_public(decision(**changes), IDS, now=NOW)


def package(tmp_path: Path) -> Path:
    root = tmp_path / "repo" / "libs" / "ai-endpoints"
    root.mkdir(parents=True)
    (root / "README.md").write_text(
        "# Connector\n\nUnrelated introduction.\n\n"
        "## Installation\n\nKeep this section.\n",
        encoding="utf-8",
    )
    return root


def apply_patch(root: Path, patch: str, tmp_path: Path) -> None:
    patch_path = tmp_path / "changes.patch"
    patch_path.write_text(patch, encoding="utf-8")
    subprocess.run(
        ["git", "apply", str(patch_path)],
        cwd=root.parents[1],
        check=True,
        capture_output=True,
        text=True,
    )


def test_private_decision_details_never_enter_snapshot_docs_badges_or_body(
    tmp_path: Path,
) -> None:
    data = public()
    patch, body = publisher.prepare_bundle(data, IDS, root=package(tmp_path), now=NOW)
    surfaces = publisher.json_text(data) + patch + body
    for forbidden in (
        "94721",
        "984731",
        "private-container-secret",
        "private-fingerprint-secret",
        "private.example",
        "private-person-name",
        "private-freeform-text",
        "private-gpu-runtime",
        "chat_openai",
        "0.99.1",
        "tested_packages",
        "result_ids",
        "fingerprint_hash",
        "nim_target_id",
    ):
        assert forbidden not in surfaces
    row = data["rows"][0]
    assert row["framework_baseline"] == "verified"
    assert row["tested_sdk_version"] is None
    assert set(row["capabilities"].values()) == {"unknown"}
    badges = [
        json.loads(text)
        for name, text in publisher.render_files(data).items()
        if name.startswith(publisher.BADGE_DIR)
    ]
    assert all(
        badge["color"] not in {"green", "brightgreen", "success"} for badge in badges
    )
    assert all("SDK unknown" in badge["message"] for badge in badges)


@pytest.mark.parametrize(
    ("changes", "baseline", "freshness", "deployment"),
    [
        ({"tested_at": "unavailable"}, "unknown", "unknown", "hosted"),
        (
            {"tested_at": publisher.stamp(NOW - timedelta(days=31))},
            "stale",
            "stale",
            "hosted",
        ),
        (
            {"tested_at": publisher.stamp(NOW + timedelta(hours=1))},
            "unknown",
            "unknown",
            "hosted",
        ),
        ({"deployment_evidence": "unknown"}, "unknown", "current", "unknown"),
        ({"deployment_evidence": "downloadable"}, "unknown", "current", "downloadable"),
        ({"freshness": "unknown"}, "unknown", "unknown", "hosted"),
    ],
)
def test_absent_stale_and_inconsistent_evidence_cannot_claim_verified_support(
    changes: dict[str, Any],
    baseline: str,
    freshness: str,
    deployment: str,
) -> None:
    row = public(**changes)["rows"][0]
    assert row["framework_baseline"] == baseline
    assert row["freshness"] == freshness
    assert row["deployment"] == deployment
    assert set(row["capabilities"].values()) == {"unknown"}
    assert publisher.badge(row)["color"] not in {"green", "brightgreen"}


def test_current_snapshot_expires_at_prepare_without_silently_mutating_reviewed_data(
    tmp_path: Path,
) -> None:
    data = public()
    original = copy.deepcopy(data)
    with pytest.raises(ValueError):
        publisher.prepare_bundle(
            data, IDS, root=package(tmp_path), now=NOW + timedelta(days=31)
        )
    assert data == original


@pytest.mark.parametrize("injected", ["private_link", "approved", "package_inventory"])
def test_unknown_public_fields_cannot_bypass_privacy_or_external_approval(
    tmp_path: Path,
    injected: str,
) -> None:
    data = public()
    data[injected] = "https://private.example/token=secret"
    with pytest.raises(ValueError):
        publisher.prepare_bundle(data, IDS, root=package(tmp_path), now=NOW)


def test_nested_private_fields_freeform_limitations_and_sdk_claims_are_rejected(
    tmp_path: Path,
) -> None:
    root = package(tmp_path)
    altered = []
    data = public()
    data["rows"][0]["pipeline_url"] = "https://private.example"
    altered.append(data)
    data = public()
    data["rows"][0]["limitation_codes"].append("[leak](https://private.example)")
    altered.append(data)
    data = public()
    data["rows"][0]["tested_sdk_version"] = "0.99.1"
    altered.append(data)
    data = public()
    data["rows"][0]["capabilities"]["tools"] = "verified"
    altered.append(data)
    data = public()
    data["source_owner"] = "private-person-name"
    altered.append(data)
    for snapshot in altered:
        with pytest.raises(ValueError):
            publisher.prepare_bundle(snapshot, IDS, root=root, now=NOW)


@pytest.mark.parametrize(
    "model", ["../escape", "unreviewed/model", "a/[link](secret)", "a\\b", "a\nb"]
)
def test_model_identity_requires_safe_exact_reviewed_canonical_id(model: str) -> None:
    with pytest.raises(ValueError):
        publisher.export_public(decision(model=model), IDS, now=NOW)


def test_duplicate_decisions_do_not_collapse_targets_into_a_false_public_claim() -> (
    None
):
    data = decision()
    duplicate = copy.deepcopy(data["rows"][0])
    duplicate["nim_target_id"] = 1234
    data["rows"].append(duplicate)
    with pytest.raises(ValueError):
        publisher.export_public(data, IDS, now=NOW)


def test_safe_badge_filenames_are_injective_across_model_and_deployment() -> None:
    rows = [
        {"model_id": model, "deployment": deployment}
        for model in ("a/b-c", "a-b/c")
        for deployment in ("hosted", "downloadable")
    ]
    names = [publisher.badge_name(row) for row in rows]
    assert len(set(names)) == len(rows)
    assert all(Path(name).name == name and len(name) < 255 for name in names)


def test_patch_applies_all_surfaces_and_regenerates_to_exact_noop(
    tmp_path: Path,
) -> None:
    root = package(tmp_path)
    source = (root / "README.md").read_text(encoding="utf-8")
    data = public()
    patch, _ = publisher.prepare_bundle(data, IDS, root=root, now=NOW)
    assert (root / "README.md").read_text(encoding="utf-8") == source
    assert not (root / publisher.SNAPSHOT_PATH).exists()
    apply_patch(root, patch, tmp_path)
    generated = publisher.render_files(data)
    assert all(
        (root / path).read_text(encoding="utf-8") == text
        for path, text in generated.items()
    )
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert readme.endswith(source.partition("\n")[2])
    assert readme.count(publisher.BEGIN) == 1
    assert publisher.prepare_bundle(data, IDS, root=root, now=NOW)[0] == ""
    badges = list((root / publisher.BADGE_DIR).glob("*.json"))
    assert len(badges) == 2
    assert all(json.loads(path.read_text())["schemaVersion"] == 1 for path in badges)


def test_updated_snapshot_removes_only_obsolete_previously_managed_badge_files(
    tmp_path: Path,
) -> None:
    root = package(tmp_path)
    first = public()
    patch, _ = publisher.prepare_bundle(first, IDS, root=root, now=NOW)
    apply_patch(root, patch, tmp_path)
    badge = root / publisher.BADGE_DIR / publisher.badge_name(first["rows"][0])
    unrelated = root / publisher.BADGE_DIR / "manual.json"
    unrelated.write_text('{"manual": true}\n', encoding="utf-8")
    second = copy.deepcopy(first)
    second["rows"] = []
    patch, body = publisher.prepare_bundle(second, IDS, root=root, now=NOW)
    apply_patch(root, patch, tmp_path)
    assert not badge.exists()
    assert unrelated.read_text() == '{"manual": true}\n'
    assert MODEL in body and "removed" in body
    assert publisher.prepare_bundle(second, IDS, root=root, now=NOW)[0] == ""


def test_expired_previous_baseline_can_be_replaced_with_reviewed_stale_state(
    tmp_path: Path,
) -> None:
    root = package(tmp_path)
    first = public()
    apply_patch(
        root, publisher.prepare_bundle(first, IDS, root=root, now=NOW)[0], tmp_path
    )
    later = NOW + timedelta(days=31)
    stale = publisher.export_public(decision(), IDS, now=later)
    patch, body = publisher.prepare_bundle(stale, IDS, root=root, now=later)
    apply_patch(root, patch, tmp_path)
    assert "verified → stale" in body
    assert publisher.prepare_bundle(stale, IDS, root=root, now=later)[0] == ""


@pytest.mark.parametrize(
    "collision", [publisher.DOC_PATH, publisher.BADGE_DIR + "/summary.json"]
)
def test_unmanaged_file_collisions_are_not_overwritten(
    tmp_path: Path, collision: str
) -> None:
    root = package(tmp_path)
    path = root / collision
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Maintainer-owned content\n", encoding="utf-8")
    with pytest.raises(ValueError):
        publisher.prepare_bundle(public(), IDS, root=root, now=NOW)
    assert path.read_text() == "Maintainer-owned content\n"
    assert not (root / publisher.SNAPSHOT_PATH).exists()


def test_modified_managed_file_blocks_whole_regeneration(tmp_path: Path) -> None:
    root = package(tmp_path)
    first = public()
    apply_patch(
        root, publisher.prepare_bundle(first, IDS, root=root, now=NOW)[0], tmp_path
    )
    target = root / publisher.DOC_PATH
    target.write_text(target.read_text() + "Manually reviewed note\n", encoding="utf-8")
    with pytest.raises(ValueError):
        publisher.prepare_bundle(first, IDS, root=root, now=NOW)
    assert target.read_text().endswith("Manually reviewed note\n")


def test_source_symlink_collision_is_rejected_without_touching_target(
    tmp_path: Path,
) -> None:
    root = package(tmp_path)
    private = tmp_path / "private.txt"
    private.write_text("private-data\n", encoding="utf-8")
    (root / "docs").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        publisher.prepare_bundle(public(), IDS, root=root, now=NOW)
    assert private.read_text() == "private-data\n"


def test_output_bundle_rolls_back_first_output_if_second_cannot_be_created(
    tmp_path: Path,
) -> None:
    root = package(tmp_path)
    directory = tmp_path / "new-output-bundle"
    first = directory / "publication.patch"
    second = directory / ("b" * 256)
    with pytest.raises(OSError):
        publisher.write_bundle(
            [(first, "paired patch"), (second, "paired body")], inputs=[], root=root
        )
    assert not first.exists()
    assert directory.is_dir()
    assert list(directory.iterdir()) == []


def test_output_bundle_preserves_existing_files_and_rejects_aliases(
    tmp_path: Path,
) -> None:
    root = package(tmp_path)
    directory = tmp_path / "existing-output-bundle"
    directory.mkdir()
    first = directory / "publication.patch"
    second = directory / "body.md"
    second.write_text("existing reviewer notes\n", encoding="utf-8")
    with pytest.raises(ValueError):
        publisher.write_bundle(
            [(first, "patch"), (second, "body")], inputs=[], root=root
        )
    assert not first.exists()
    assert second.read_text() == "existing reviewer notes\n"
    first = tmp_path / "new-alias-bundle" / "publication.patch"
    with pytest.raises(ValueError):
        publisher.write_bundle(
            [(first, "patch"), (first, "body")], inputs=[], root=root
        )
    with pytest.raises(ValueError):
        publisher.write_bundle([(first, "patch")], inputs=[first], root=root)
    with pytest.raises(ValueError):
        publisher.write_bundle(
            [(root / "new-bundle" / "new.json", "data")], inputs=[], root=root
        )


def test_duplicate_json_keys_cannot_hide_private_snapshot_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "snapshot.json"
    path.write_text(
        '{"source_owner": "private-name", "source_owner": "safe-role"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        publisher.load_json(path)


@pytest.mark.parametrize("command", ["export-public", "prepare-pr"])
def test_cli_rejection_does_not_echo_private_parse_content_or_create_outputs(
    tmp_path: Path,
    command: str,
) -> None:
    source = tmp_path / "private-input.json"
    secret = "PRIVATE_CREDENTIAL_VALUE_DO_NOT_ECHO"
    source.write_text('{"secret":"' + secret + '",broken-json}', encoding="utf-8")
    first = tmp_path / "new-cli-bundle" / "output"
    second = tmp_path / "new-cli-bundle" / "body"
    arguments = (
        ["--decisions", str(source), "--output", str(first)]
        if command == "export-public"
        else [
            "--reviewed-snapshot",
            str(source),
            "--patch-output",
            str(first),
            "--pr-body-output",
            str(second),
        ]
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), command, *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert secret not in result.stdout + result.stderr
    assert str(source) not in result.stdout + result.stderr
    assert not first.exists() and not second.exists()


def sdk_decision(*, capabilities: tuple[str, ...] = ("tools",)) -> dict[str, Any]:
    data = decision()
    row = data["rows"][0]
    row["sdk_execution"] = {
        "contract_version": "bcb-sdk-execution-v1",
        "package": "langchain-nvidia-ai-endpoints",
        "version": "0.4.1",
        "model_name": row["model"],
        "nim_target_id": row["nim_target_id"],
        "result_id": 3175,
        "fingerprint_hash": row["fingerprint_hash"],
        "digest": None,
        "deployment": row["deployment_evidence"],
        "completed_at": publisher.stamp(NOW - timedelta(minutes=30)),
        "checks": [
            {
                "capability": capability,
                "client": publisher.SDK_CHECKS[capability][0],
                "id": publisher.SDK_CHECKS[capability][1],
                "status": "passed",
            }
            for capability in capabilities
        ],
    }
    row["framework_checks"] = [
        {
            "id": publisher.SDK_CHECKS[capability][1],
            "status": "passed",
            "evidence_class": "project_integration",
            "result_id": 3175,
            "completed_at": publisher.stamp(NOW - timedelta(minutes=30)),
            "package_versions": {"langchain-nvidia-ai-endpoints": "0.4.1"},
        }
        for capability in capabilities
    ]
    return data


def test_bound_sdk_feature_execution_is_distinct_from_framework_baseline_and_inventory(
    tmp_path: Path,
) -> None:
    data = publisher.export_public(sdk_decision(), IDS, now=NOW)
    row = data["rows"][0]
    assert row["tested_sdk_version"] == "0.4.1"
    assert row["sdk_observed_at"] != row["observed_at"]
    assert row["sdk_freshness"] == "current"
    assert row["capabilities"]["tools"] == "verified"
    assert row["capabilities"]["embedding"] == "unknown"
    assert publisher.badge(row)["color"] == "orange"
    patch, body = publisher.prepare_bundle(data, IDS, root=package(tmp_path), now=NOW)
    assert "0.4.1" in patch and "tools: verified" in body
    assert "private-fingerprint-secret" not in patch + body
    assert "984731" not in patch + body
    assert "3175" not in patch + body
    assert "0.99.1" not in patch + body


def test_green_requires_current_sdk_execution_not_just_framework_baseline(
    tmp_path: Path,
) -> None:
    data = publisher.export_public(
        sdk_decision(capabilities=publisher.CAPABILITIES),
        IDS,
        now=NOW,
    )
    row = data["rows"][0]
    publisher.validate_public(data, IDS, now=NOW)
    assert publisher.badge(row)["color"] == "green"
    summary = json.loads(
        publisher.render_files(data)[publisher.BADGE_DIR + "/summary.json"]
    )
    assert summary["color"] == "green"
    row["framework_baseline"] = "unknown"
    assert publisher.badge(row)["color"] != "green"
    row["framework_baseline"] = "verified"
    row["capabilities"]["tools"] = "failed"
    assert publisher.badge(row)["color"] == "red"
    with pytest.raises(ValueError):
        publisher.prepare_bundle(
            data, IDS, root=package(tmp_path), now=NOW + timedelta(days=31)
        )


def test_sdk_failure_and_expiration_are_rendered_as_feature_states() -> None:
    private = sdk_decision(capabilities=("tools", "streaming"))
    private["rows"][0]["sdk_execution"]["checks"][0]["status"] = "failed"
    private["rows"][0]["framework_checks"][0]["status"] = "failed"
    failed = publisher.export_public(private, IDS, now=NOW)
    assert failed["rows"][0]["capabilities"]["tools"] == "failed"
    assert publisher.badge(failed["rows"][0])["color"] == "red"
    stale = publisher.export_public(private, IDS, now=NOW + timedelta(days=31))
    row = stale["rows"][0]
    assert row["tested_sdk_version"] == "0.4.1"
    assert row["sdk_freshness"] == "stale"
    assert row["capabilities"]["tools"] == "stale"
    assert row["capabilities"]["streaming"] == "stale"
    assert row["capabilities"]["chat"] == "unknown"
    assert publisher.badge(row)["color"] != "green"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("package", "openai"),
        ("version", "https://private.example/secret"),
        ("model_name", "different/model"),
        ("nim_target_id", True),
        ("nim_target_id", 42),
        ("result_id", 42),
        ("fingerprint_hash", "different-private-fingerprint"),
        ("deployment", "downloadable"),
        ("completed_at", publisher.stamp(NOW + timedelta(minutes=1))),
        ("completed_at", None),
    ],
)
def test_incomplete_or_contradictory_private_sdk_binding_is_not_exported_as_support(
    field: str,
    value: Any,
) -> None:
    private = sdk_decision()
    private["rows"][0]["sdk_execution"][field] = value
    row = publisher.export_public(private, IDS, now=NOW)["rows"][0]
    assert row["tested_sdk_version"] is None
    assert row["sdk_observed_at"] is None
    assert set(row["capabilities"].values()) == {"unknown"}
    assert "sdk_evidence_inconsistent" in row["limitation_codes"]
    assert publisher.badge(row)["color"] != "green"


@pytest.mark.parametrize(
    "contradiction", ["native-client", "wrong-id", "duplicate", "no-outcome"]
)
def test_private_sdk_checks_require_explicit_sdk_identities_and_unambiguous_outcomes(
    contradiction: str,
) -> None:
    private = sdk_decision()
    execution = private["rows"][0]["sdk_execution"]
    if contradiction == "native-client":
        execution["checks"][0]["client"] = "ChatOpenAI"
    elif contradiction == "wrong-id":
        execution["checks"][0]["id"] = "tool_calling_conformance"
    elif contradiction == "duplicate":
        execution["checks"].append(copy.deepcopy(execution["checks"][0]))
    else:
        execution["checks"][0]["status"] = "unknown"
    row = publisher.export_public(private, IDS, now=NOW)["rows"][0]
    assert set(row["capabilities"].values()) == {"unknown"}
    assert row["tested_sdk_version"] is None


def test_downloadable_sdk_evidence_requires_matching_immutable_digest() -> None:
    private = sdk_decision()
    row = private["rows"][0]
    row["deployment_type"] = row["deployment_evidence"] = "downloadable"
    row["sdk_execution"]["deployment"] = "downloadable"
    row["sdk_execution"]["digest"] = row["digest"]
    bound = publisher.export_public(private, IDS, now=NOW)
    assert bound["rows"][0]["capabilities"]["tools"] == "verified"
    row["sdk_execution"]["digest"] = "sha256:different-private-digest"
    inconsistent = publisher.export_public(private, IDS, now=NOW)
    assert inconsistent["rows"][0]["capabilities"]["tools"] == "unknown"
    assert "different-private-digest" not in publisher.json_text(inconsistent)


@pytest.mark.parametrize(
    "missing", ["tested_sdk_version", "deployment", "sdk_observed_at"]
)
def test_reviewed_public_verified_features_require_package_deployment_and_sdk_timestamp(
    tmp_path: Path,
    missing: str,
) -> None:
    data = publisher.export_public(sdk_decision(), IDS, now=NOW)
    data["rows"][0][missing] = "unknown" if missing == "deployment" else None
    with pytest.raises(ValueError):
        publisher.prepare_bundle(data, IDS, root=package(tmp_path), now=NOW)


def test_reviewed_public_sdk_verified_state_cannot_use_stale_or_unknown_freshness(
    tmp_path: Path,
) -> None:
    root = package(tmp_path)
    for freshness in ("stale", "unknown"):
        data = publisher.export_public(sdk_decision(), IDS, now=NOW)
        data["rows"][0]["sdk_freshness"] = freshness
        with pytest.raises(ValueError):
            publisher.prepare_bundle(data, IDS, root=root, now=NOW)


@pytest.mark.parametrize(
    "failure",
    [
        "missing-framework-check",
        "native-result-id",
        "unrelated-framework-result",
        "native-evidence-class",
        "wrong-package-version",
        "wrong-completion-time",
        "contradictory-outcome",
        "duplicate-framework-check",
    ],
)
def test_sdk_execution_requires_its_matching_framework_run_check(
    failure: str,
) -> None:
    private = sdk_decision()
    row = private["rows"][0]
    assert row["sdk_execution"]["result_id"] != row["result_ids"]["latest_result"]
    captured = row["framework_checks"][0]
    if failure == "missing-framework-check":
        row["framework_checks"] = []
    elif failure == "native-result-id":
        row["sdk_execution"]["result_id"] = row["result_ids"]["latest_result"]
    elif failure == "unrelated-framework-result":
        captured["result_id"] = 3176
    elif failure == "native-evidence-class":
        captured["evidence_class"] = "native_probe"
    elif failure == "wrong-package-version":
        captured["package_versions"] = {"langchain-nvidia-ai-endpoints": "0.99.1"}
    elif failure == "wrong-completion-time":
        captured["completed_at"] = publisher.stamp(NOW - timedelta(minutes=31))
    elif failure == "contradictory-outcome":
        captured["status"] = "failed"
    else:
        row["framework_checks"].append(copy.deepcopy(captured))
    exported = publisher.export_public(private, IDS, now=NOW)["rows"][0]
    assert exported["tested_sdk_version"] is None
    assert set(exported["capabilities"].values()) == {"unknown"}
    assert publisher.badge(exported)["color"] != "green"


def test_output_bundle_creates_private_namespace_and_rejects_reuse(
    tmp_path: Path,
) -> None:
    root = package(tmp_path)
    input_path = tmp_path / "reviewed-public.json"
    input_path.write_text(publisher.json_text(public()), encoding="utf-8")
    directory = tmp_path / "proposal"
    patch_path = directory / "publication.patch"
    body_path = directory / "body.md"
    patch, body = publisher.prepare_bundle(public(), IDS, root=root, now=NOW)
    publisher.write_bundle(
        [(patch_path, patch), (body_path, body)],
        inputs=[input_path],
        root=root,
    )
    assert directory.stat().st_mode & 0o077 == 0
    assert patch_path.stat().st_mode & 0o077 == 0
    assert body_path.stat().st_mode & 0o077 == 0
    apply_patch(root, patch_path.read_text(encoding="utf-8"), tmp_path)
    assert publisher.prepare_bundle(public(), IDS, root=root, now=NOW)[0] == ""
    with pytest.raises(ValueError):
        publisher.write_bundle(
            [(directory / "extra.patch", patch)], inputs=[input_path], root=root
        )
    assert not (directory / "extra.patch").exists()


def test_output_bundle_rejects_shared_or_symlinked_namespaces(tmp_path: Path) -> None:
    root = package(tmp_path)
    first = tmp_path / "proposal" / "patch"
    second = tmp_path / "different" / "body"
    with pytest.raises(ValueError):
        publisher.write_bundle(
            [(first, "patch"), (second, "body")], inputs=[], root=root
        )
    target = tmp_path / "absent-target"
    (tmp_path / "proposal").symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError):
        publisher.write_bundle([(first, "patch")], inputs=[], root=root)
    assert not target.exists()


def test_rollback_uses_pinned_private_directory_not_replaced_parent_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = package(tmp_path)
    directory = tmp_path / "proposal"
    moved = tmp_path / "moved-private-proposal"
    first = directory / "patch"
    failing_name = "b" * 256
    second = directory / failing_name
    real_open = publisher.os.open

    def replace_before_failure(
        path: Any,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if path == failing_name:
            directory.rename(moved)
            directory.mkdir()
            (directory / "patch").write_text(
                "Other invocation's output\n", encoding="utf-8"
            )
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(publisher.os, "open", replace_before_failure)
    with pytest.raises(OSError):
        publisher.write_bundle(
            [(first, "paired patch"), (second, "paired body")], inputs=[], root=root
        )
    assert (directory / "patch").read_text() == "Other invocation's output\n"
    assert directory.is_dir() and moved.is_dir()
    assert list(moved.iterdir()) == []
