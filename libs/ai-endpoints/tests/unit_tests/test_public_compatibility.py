"""Validate public output only; never read private evidence or call a service."""

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ROOT = Path(__file__).resolve().parents[2]
CAPABILITIES = (
    "chat",
    "streaming",
    "tools",
    "structured_output",
    "reasoning",
    "vlm",
    "embedding",
    "rerank",
)
REQUIRED_LIMITATIONS = {
    "historical_observation",
    "endpoint_availability_not_guaranteed",
    "untested_capabilities_unknown",
}
LIMITATIONS = REQUIRED_LIMITATIONS | {
    "native_baseline_not_accepted",
    "self_hosted_not_tested",
}
BEGIN = "<!-- BEGIN REVIEWED COMPATIBILITY OBSERVATIONS -->"
END = "<!-- END REVIEWED COMPATIBILITY OBSERVATIONS -->"
HEADER = (
    "| Model | Deployment | SDK version | SDK revision | Observed at (UTC) | "
    "Chat | Streaming | Tools | Structured output | Reasoning | VLM | "
    "Embedding | Rerank | Limitations |"
)
SEPARATOR = "| " + " | ".join(["---"] * 14) + " |"
EMPTY = "No approved public observations have been published. No support claim is made."


def unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate public field")
        result[key] = value
    return result


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    model_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9_.-]+$")
    deployment: Literal["hosted", "self_hosted"]
    tested_sdk_version: str = Field(
        pattern=r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+|\.post\d+|\.dev\d+)?$"
    )
    tested_sdk_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    observed_at: str
    capabilities: dict[str, Literal["passed", "failed", "unknown"]]
    limitation_codes: list[str]

    @field_validator("observed_at")
    @classmethod
    def check_timestamp(cls, value: str) -> str:
        if not value.endswith("Z"):
            raise ValueError("Observation must have a UTC timestamp")
        observed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if observed > datetime.now(timezone.utc):
            raise ValueError("Observation cannot be in the future")
        return value

    @field_validator("capabilities")
    @classmethod
    def check_capabilities(cls, value: dict[str, Any]) -> dict[str, Any]:
        if set(value) != set(CAPABILITIES):
            raise ValueError("All public capability keys must be explicit")
        return value

    @field_validator("limitation_codes")
    @classmethod
    def check_limitations(cls, value: list[str]) -> list[str]:
        codes = set(value)
        if (
            not REQUIRED_LIMITATIONS <= codes <= LIMITATIONS
            or len(codes) != len(value)
            or value != sorted(value)
        ):
            raise ValueError("Expected sorted, unique public limitation codes")
        return value

    @model_validator(mode="after")
    def check_deployment_limitations(self) -> "Observation":
        if self.deployment == "hosted" and "self_hosted_not_tested" not in (
            self.limitation_codes
        ):
            raise ValueError("Hosted evidence must not imply self-hosted support")
        return self


class Snapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    rows: list[Observation]

    @field_validator("schema_version", mode="before")
    @classmethod
    def check_schema_version(cls, value: Any) -> Any:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("Schema version must be an integer")
        return value

    @model_validator(mode="after")
    def check_identities(self) -> "Snapshot":
        registry = json.loads(
            (
                ROOT / "langchain_nvidia_ai_endpoints/data/model_registry.json"
            ).read_text()
        )
        identifiers = {row["id"] for row in registry["models"]}
        keys = []
        for row in self.rows:
            if row.model_id not in identifiers:
                raise ValueError("Expected canonical registry identity")
            keys.append((row.model_id, row.deployment))
        if len(keys) != len(set(keys)) or keys != sorted(keys):
            raise ValueError("Expected sorted unique model/deployment rows")
        return self


def expected_table(snapshot: Snapshot) -> str:
    lines = [HEADER, SEPARATOR]
    for row in snapshot.rows:
        cells = [
            f"`{row.model_id}`",
            row.deployment,
            row.tested_sdk_version,
            f"`{row.tested_sdk_revision}`",
            row.observed_at,
            *(row.capabilities[key] for key in CAPABILITIES),
            ", ".join(f"`{code}`" for code in row.limitation_codes),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    if not snapshot.rows:
        lines.extend(["", EMPTY])
    return "\n".join(lines)


def test_public_snapshot_and_table_agree() -> None:
    data = json.loads(
        (ROOT / "docs/data/compatibility_public.json").read_text(),
        object_pairs_hook=unique_keys,
    )
    snapshot = Snapshot.model_validate(data)
    document = (ROOT / "docs/compatibility_state.md").read_text()
    assert document.count(BEGIN) == document.count(END) == 1
    assert document.index(BEGIN) < document.index(END)
    actual = document.split(BEGIN)[1].split(END)[0].strip()
    assert actual == expected_table(snapshot)


def sample() -> dict[str, Any]:
    """Synthetic contract example, never a published support observation."""
    return {
        "schema_version": 1,
        "rows": [
            {
                "model_id": "nvidia/nemotron-3.5-lightning-30b-a3b",
                "deployment": "hosted",
                "tested_sdk_version": "1.5.0",
                "tested_sdk_revision": "a" * 40,
                "observed_at": "2020-01-01T00:00:00Z",
                "capabilities": dict.fromkeys(CAPABILITIES, "unknown"),
                "limitation_codes": sorted(
                    REQUIRED_LIMITATIONS | {"self_hosted_not_tested"}
                ),
            }
        ],
    }


@pytest.mark.parametrize("state", ["passed", "failed", "unknown"])
def test_public_contract_keeps_capability_outcomes_independent(state: str) -> None:
    data = sample()
    data["rows"][0]["capabilities"]["chat"] = state
    snapshot = Snapshot.model_validate(data)
    assert snapshot.rows[0].capabilities["streaming"] == "unknown"
    assert f"| {state} | unknown |" in expected_table(snapshot)


@pytest.mark.parametrize(
    "field",
    [
        "metadata",
        "nim_target_id",
        "workflow_id",
        "fingerprint",
        "token",
        "url",
        "reviewer",
    ],
)
@pytest.mark.parametrize("level", ["snapshot", "row"])
def test_public_contract_rejects_private_or_unexpected_fields(
    field: str, level: str
) -> None:
    data = sample()
    target = data if level == "snapshot" else data["rows"][0]
    target[field] = "private-value"
    with pytest.raises(ValueError):
        Snapshot.model_validate(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_id", "https://private.example/model"),
        ("model_id", "unregistered/model"),
        ("deployment", "downloadable"),
        ("tested_sdk_version", "unknown"),
        ("tested_sdk_revision", "main"),
        ("observed_at", "2020-01-01"),
        ("observed_at", "2020-01-01T00:00:00+01:00"),
        ("observed_at", "2999-01-01T00:00:00Z"),
        ("capabilities", {"chat": "passed"}),
        ("capabilities", dict.fromkeys(CAPABILITIES, "verified")),
        ("limitation_codes", []),
        ("limitation_codes", ["freeform private report"]),
        ("limitation_codes", sorted(REQUIRED_LIMITATIONS)),
    ],
)
def test_public_contract_rejects_invalid_or_overstated_rows(
    field: str, value: Any
) -> None:
    data = sample()
    data["rows"][0][field] = value
    with pytest.raises(ValueError):
        Snapshot.model_validate(data)


def test_public_contract_rejects_duplicate_rows_and_json_fields() -> None:
    data = sample()
    data["rows"].append(deepcopy(data["rows"][0]))
    with pytest.raises(ValueError):
        Snapshot.model_validate(data)
    with pytest.raises(ValueError, match="Duplicate"):
        json.loads(
            '{"schema_version": 1, "schema_version": 2}', object_pairs_hook=unique_keys
        )


@pytest.mark.parametrize("version", [True, "1", 2])
def test_public_contract_rejects_invalid_schema_version(version: Any) -> None:
    with pytest.raises(ValueError):
        Snapshot.model_validate({"schema_version": version, "rows": []})
