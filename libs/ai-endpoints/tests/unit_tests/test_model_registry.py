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


def test_registry_generation_remains_offline_and_does_not_modify_inputs() -> None:
    before = {path: path.read_bytes() for path in (registry.REGISTRY, registry.STATICS)}
    assert registry.generate(False) == 0
    assert before == {path: path.read_bytes() for path in before}
    assert not hasattr(registry, "snapshot")
    assert not hasattr(registry, "compare")


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


def test_registry_rejects_unreviewed_capability_change(tmp_path: Path) -> None:
    row = fixture_model("nvidia/old")
    row["capabilities"]["supports_tools"] = "supported"
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"schema_version": 1, "models": [row]}))
    with pytest.raises(ValueError, match="Capability disagrees"):
        registry.load_registry(path)


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
