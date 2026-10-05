"""Offline coverage of the reviewed registry and review-only drift reporter."""

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from langchain_nvidia_ai_endpoints._statics import MODEL_TABLE, determine_model
from langchain_nvidia_ai_endpoints._telemetry import _APPROVED_MODEL_LOOKUP

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/model_registry.py"
SPEC = importlib.util.spec_from_file_location("model_registry", SCRIPT)
assert SPEC and SPEC.loader
registry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(registry)


def fixture_model(identifier: str, *, current: bool = False) -> dict[str, Any]:
    return {
        "table": "CHAT_MODEL_TABLE",
        "id": identifier,
        "model": {
            "id": identifier,
            "model_type": "chat",
            "client": "ChatNVIDIA",
            "aliases": ["old-alias"],
        },
        "served_names": [identifier],
        "deployment_type": "hosted" if current else "unknown",
        "capabilities": {field: "unknown" for field in registry.CAPABILITIES},
        "capability_provenance": {
            field: {"source": "unknown", "url": None} for field in registry.CAPABILITIES
        },
        "provenance": {
            "hosted": {
                "state": "current" if current else "unknown",
                "url": "https://example.com/old",
            },
            "downloadable": {"state": "unknown", "url": None},
        },
        "stale": False,
    }


def test_legacy_alias_and_telemetry_identity_survive_registry_migration() -> None:
    with pytest.warns(UserWarning, match="deprecated"):
        gemma = determine_model("ai-gemma-7b")
    assert gemma is MODEL_TABLE["google/gemma-7b"]
    assert _APPROVED_MODEL_LOOKUP["ai-gemma-7b"] == "google/gemma-7b"
    assert MODEL_TABLE["nvidia/nemotron-3-ultra-550b-a55b"].supports_thinking is True


def test_drift_proposes_addition_rename_and_source_specific_staleness() -> None:
    old = fixture_model("nvidia/old", current=True)
    changes = registry.compare(
        [old],
        {
            "nvidia/new": {
                "id": "nvidia/new",
                "replaces": "nvidia/old",
                "url": "https://example.com/new",
            },
            "nvidia/added": {"id": "nvidia/added", "url": "https://example.com/add"},
        },
        None,
        complete_sources=frozenset({"hosted"}),
    )
    assert {(change["kind"], change["id"]) for change in changes} == {
        ("rename", "nvidia/new"),
        ("addition", "nvidia/added"),
        ("removal/stale", "nvidia/old"),
    }
    assert all(
        c["suggested_deployment_type"] == "hosted"
        for c in changes
        if c["kind"] in ("addition", "rename")
    )
    assert (
        next(c for c in changes if c["kind"] == "removal/stale")["after"] == "unknown"
    )
    assert not any(change["source"] == "downloadable" for change in changes)


def test_drift_detects_alias_deprecation_served_name_and_capability_changes() -> None:
    row = fixture_model("nvidia/old")
    observed = {
        "nvidia/old": {
            "id": "nvidia/old",
            "aliases": ["new-alias"],
            "served_names": ["nvidia/old", "served-alias"],
            "deprecated": True,
            "capabilities": {"supports_tools": "supported"},
            "url": "https://example.com/details",
        }
    }
    changes = registry.compare([row], None, observed)
    assert {change["kind"] for change in changes} == {
        "deployment_type",
        "aliases",
        "served_names",
        "deprecation",
        "capability",
    }
    capability = next(c for c in changes if c["kind"] == "capability")
    assert capability["before"] == "unknown"
    assert capability["provenance_after"] == {
        "source": "ngc_catalog",
        "url": "https://example.com/details",
    }
    assert capability["claim_only"] is True
    deployment = next(c for c in changes if c["kind"] == "deployment_type")
    assert deployment["after"] == "downloadable"
    assert all(c["source"] == "downloadable" for c in changes)


def test_missing_or_empty_source_does_not_propose_removals(tmp_path: Path) -> None:
    missing, reason, complete = registry.snapshot(tmp_path / "missing.json", "hosted")
    assert missing is None and "missing.json" in reason and not complete
    empty = tmp_path / "empty.json"
    empty.write_text(
        json.dumps({"source_url": "https://example.com/models", "data": []})
    )
    unavailable, reason, complete = registry.snapshot(empty, "hosted")
    assert unavailable is None and "empty catalog" in reason and not complete
    assert (
        registry.compare([fixture_model("nvidia/old", current=True)], unavailable, None)
        == []
    )
    assert "Unavailable" in registry.report([], {}, [reason])


def test_snapshot_uses_item_evidence_and_rejects_duplicate_ids(tmp_path: Path) -> None:
    snapshot = tmp_path / "ngc.json"
    snapshot.write_text(
        json.dumps(
            {
                "source_url": "https://catalog.ngc.nvidia.com/models",
                "models": [
                    {"id": "nvidia/old", "url": "https://catalog.ngc.nvidia.com/old"}
                ],
            }
        )
    )
    rows, _, complete = registry.snapshot(snapshot, "downloadable")
    assert rows is not None
    assert rows["nvidia/old"]["url"] == "https://catalog.ngc.nvidia.com/old"
    assert not complete
    snapshot.write_text(
        json.dumps(
            {
                "source_url": "https://catalog.ngc.nvidia.com/models",
                "models": [{"id": "same"}, {"id": "same"}],
            }
        )
    )
    rows, reason, complete = registry.snapshot(snapshot, "downloadable")
    assert rows is None and "duplicate id" in reason and not complete


def test_partial_inventory_never_proposes_false_removals(tmp_path: Path) -> None:
    row = fixture_model("nvidia/old")
    row["provenance"]["downloadable"] = {
        "state": "current",
        "url": "https://example.com/ngc/old",
    }
    snapshot = tmp_path / "ngc.json"
    payload = {
        "source_url": "https://example.com/ngc",
        "models": [{"id": "nvidia/new", "url": "https://example.com/ngc/new"}],
    }
    snapshot.write_text(json.dumps(payload))
    observed, _, complete = registry.snapshot(snapshot, "downloadable")
    assert not complete
    assert [(c["kind"], c["id"]) for c in registry.compare([row], None, observed)] == [
        ("addition", "nvidia/new")
    ]
    payload["complete_inventory"] = True
    snapshot.write_text(json.dumps(payload))
    observed, _, complete = registry.snapshot(snapshot, "downloadable")
    assert complete
    changes = registry.compare(
        [row], None, observed, complete_sources=frozenset({"downloadable"})
    )
    assert {(c["kind"], c["id"]) for c in changes} == {
        ("addition", "nvidia/new"),
        ("removal/stale", "nvidia/old"),
    }
    payload["complete_inventory"] = "true"
    snapshot.write_text(json.dumps(payload))
    invalid, reason, complete = registry.snapshot(snapshot, "downloadable")
    assert invalid is None and "complete_inventory" in reason and not complete


def test_registry_rejects_unreviewed_capability_change(tmp_path: Path) -> None:
    row = fixture_model("nvidia/old")
    row["capabilities"]["supports_tools"] = "supported"
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"schema_version": 1, "models": [row]}))
    with pytest.raises(ValueError, match="Capability disagrees"):
        registry.load_registry(path)


def test_capability_claim_preserves_legacy_origin_until_reviewed() -> None:
    row = fixture_model("nvidia/old", current=True)
    row["model"]["supports_tools"] = True
    row["capabilities"]["supports_tools"] = "supported"
    row["capability_provenance"]["supports_tools"] = {
        "source": "legacy_static",
        "url": None,
    }
    changes = registry.compare(
        [row],
        {
            "nvidia/old": {
                "id": "nvidia/old",
                "url": "https://example.com/catalog",
                "capabilities": {"supports_tools": "supported"},
            }
        },
        None,
    )
    change = next(c for c in changes if c["kind"] == "capability provenance")
    assert change["before"] == change["after"] == "supported"
    assert change["provenance_before"]["source"] == "legacy_static"
    assert change["provenance_after"]["source"] == "hosted_catalog"
    assert change["claim_only"] is True


def test_deployment_type_tracks_both_sources_and_source_specific_removal(
    tmp_path: Path,
) -> None:
    row = fixture_model("nvidia/old", current=True)
    row["provenance"]["downloadable"] = {
        "state": "current",
        "url": "https://example.com/ngc",
    }
    row["deployment_type"] = "both"
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"schema_version": 1, "models": [row]}))
    assert registry.load_registry(path)[0]["deployment_type"] == "both"
    changes = registry.compare(
        [row],
        {},
        {"nvidia/old": {"id": "nvidia/old", "url": "https://example.com/ngc"}},
        complete_sources=frozenset({"hosted"}),
    )
    assert len(changes) == 1
    assert changes[0]["source"] == "hosted"
    assert changes[0]["after"] == "downloadable"


def test_registry_rejects_unsubstantiated_type_and_capability_provenance(
    tmp_path: Path,
) -> None:
    row = fixture_model("nvidia/old")
    path = tmp_path / "registry.json"
    row["deployment_type"] = "hosted"
    path.write_text(json.dumps({"schema_version": 1, "models": [row]}))
    with pytest.raises(ValueError, match="Deployment type disagrees"):
        registry.load_registry(path)
    row["deployment_type"] = "unknown"
    row["model"]["supports_tools"] = True
    row["capabilities"]["supports_tools"] = "supported"
    row["capability_provenance"]["supports_tools"] = {"source": "bcb", "url": None}
    path.write_text(json.dumps({"schema_version": 1, "models": [row]}))
    with pytest.raises(ValueError, match="Invalid capability provenance"):
        registry.load_registry(path)


def test_downloadable_harness_upstream_needed_becomes_linked_review_candidate(
    tmp_path: Path,
) -> None:
    log = tmp_path / "downloadable.log"
    log.write_text(
        "  ⚑  UPSTREAM NEEDED: add 'nvidia/new-nim' to langchain-nvidia _statics.py\n"
        "  ⚑  UPSTREAM NEEDED: add 'nvidia/new-nim' to langchain-nvidia _statics.py\n"
        "  ⚑  UPSTREAM NEEDED: add 'nvidia/known' to langchain-nvidia _statics.py\n"
        "  ⚑  UPSTREAM NEEDED: add 'nvidia/evil`link' to langchain-nvidia _statics.py\n"
    )
    candidates = registry.harness_candidates(
        log, "https://gitlab-master.nvidia.com/example/-/jobs/42", {"nvidia/known"}
    )
    assert [item["id"] for item in candidates] == ["nvidia/new-nim"]
    assert candidates[0]["suggested_deployment_type"] == "downloadable"
    assert "https://gitlab-master.nvidia.com/example/-/jobs/42" in registry.report(
        candidates, {}, []
    )
