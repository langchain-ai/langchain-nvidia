# The review CLI deliberately reports status on stdout/stderr.
# ruff: noqa: T201
"""Export a public-safe UNAPPROVED snapshot and prepare a review-only paired patch.

Neither command approves data, applies a patch, calls a service, or publishes anything.
Framework baselines and SDK feature execution are separate claims. Only explicitly
bound SDK execution records can establish a tested version or feature state;
installed inventory and native/wire probes cannot establish SDK support.
Owner and security review of the exact snapshot is an external dispatch prerequisite.
"""

import argparse
import difflib
import importlib.util
import json
import os
import re
import sys
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PATH = "libs/ai-endpoints"
SNAPSHOT_PATH = "docs/data/bcb_compatibility_public.json"
DOC_PATH = "docs/compatibility_state.md"
BADGE_DIR = "docs/compatibility_badges"
CONTRACT = "bcb-public-compatibility-v1"
FRAMEWORK = "langchain-nvidia"
SOURCE_OWNER = "framework-evidence owners and connector maintainers"
CAPABILITIES = (
    "chat",
    "streaming",
    "tools",
    "structured output",
    "reasoning",
    "VLM",
    "embedding",
    "rerank",
)
BASELINES = {
    "verified",
    "unknown",
    "stale",
    "changed",
    "failed",
    "blocked",
    "incompatible",
}
DEPLOYMENTS = {"hosted", "downloadable", "unknown"}
SDK_STATES = {"verified", "failed", "stale", "unknown"}
SDK_PACKAGE = "langchain-nvidia-ai-endpoints"
SDK_VERSION = re.compile(
    r"[0-9]+(?:\.[0-9]+){1,3}(?:(?:a|b|rc)[0-9]+|\.post[0-9]+|\.dev[0-9]+)?"
)
SDK_CHECKS = {
    "chat": ("ChatNVIDIA", "sdk_chat"),
    "streaming": ("ChatNVIDIA", "sdk_streaming"),
    "tools": ("ChatNVIDIA", "sdk_tools"),
    "structured output": ("ChatNVIDIA", "sdk_structured_output"),
    "reasoning": ("ChatNVIDIA", "sdk_reasoning"),
    "VLM": ("ChatNVIDIA", "sdk_vlm"),
    "embedding": ("NVIDIAEmbeddings", "sdk_embedding"),
    "rerank": ("NVIDIARerank", "sdk_rerank"),
}
LIMITATIONS = {
    "framework_baseline_not_sdk_support": (
        "Framework baseline compatibility is not SDK feature verification."
    ),
    "sdk_execution_unavailable": (
        "Executed SDK feature evidence is unavailable; "
        "all SDK features remain unknown."
    ),
    "tested_version_unknown": (
        "No SDK version is explicitly bound to executed SDK feature evidence."
    ),
    "runtime_details_redacted": (
        "GPU, runtime, target identifiers and private evidence links "
        "are not published."
    ),
    "deployment_unknown": (
        "Measured deployment is unavailable; "
        "selection does not establish deployment."
    ),
    "observation_missing": "No attributable observation timestamp is available.",
    "evidence_stale": "Observation exceeds the configured freshness window.",
    "evidence_inconsistent": (
        "Evidence binding or freshness is inconsistent; " "no verified claim is made."
    ),
    "sdk_evidence_inconsistent": (
        "SDK execution identity or outcome binding is inconsistent; "
        "SDK support is unknown."
    ),
    "sdk_evidence_stale": (
        "Executed SDK feature observations exceed " "the configured freshness window."
    ),
    "sdk_features_partial": (
        "Only explicitly executed SDK features are classified; "
        "unexecuted features remain unknown."
    ),
}
REQUIRED_LIMITATIONS = {
    "framework_baseline_not_sdk_support",
    "runtime_details_redacted",
}
BEGIN = (
    "<!-- BEGIN GENERATED BCB COMPATIBILITY (scripts/bcb_compatibility_publish.py) -->"
)
END = "<!-- END GENERATED BCB COMPATIBILITY -->"
RAW_BASE = "https://raw.githubusercontent.com/langchain-ai/langchain-nvidia/main/libs/ai-endpoints/"
MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*(?:/[A-Za-z0-9][A-Za-z0-9_.-]*)*")
TOP_FIELDS = {
    "contract",
    "framework",
    "generated_at",
    "max_age_days",
    "source_owner",
    "rows",
}
ROW_FIELDS = {
    "model_id",
    "deployment",
    "framework_baseline",
    "observed_at",
    "freshness",
    "tested_sdk_version",
    "sdk_observed_at",
    "sdk_freshness",
    "capabilities",
    "limitation_codes",
}


def load_json(path: Path) -> Any:
    """Reject duplicate object keys rather than silently accepting the last value."""

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)


def json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def date(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else None


def stamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def registry_ids() -> set[str]:
    # Reuse the reviewed registry's complete validator without importing the SDK.
    spec = importlib.util.spec_from_file_location(
        "bcb_publish_registry", ROOT / "scripts/model_registry.py"
    )
    if spec is None or spec.loader is None:
        raise ValueError("Reviewed registry validator unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {row["id"] for row in module.load_registry()}


def identity(value: Any, ids: set[str]) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 100
        or MODEL_ID.fullmatch(value) is None
        or value not in ids
    ):
        raise ValueError("Expected safe canonical reviewed registry identity")
    return value


def sdk_projection(
    private: dict[str, Any],
    *,
    deployment: str,
    generated: datetime,
    now: datetime,
    max_age_days: int,
) -> dict[str, Any]:
    """Classify only explicitly bound SDK execution, never package inventory.

    The PRIVATE bcb-sdk-execution-v1 record names package, version, model_name,
    nim_target_id, result_id, fingerprint_hash, digest, deployment, completed_at
    and checks. Each check has a predefined id, capability, client and status
    (passed, failed or unknown). The execution record binds every check to the
    same measured target/framework-result/package/deployment observation. Every
    SDK check must agree with captured framework_checks project_integration
    evidence, including completion time and package version. Native result IDs
    are a different namespace. Downloadable execution binds the immutable digest.
    Duplicate or malformed checks invalidate the entire record.
    """
    unknown = {
        "tested_sdk_version": None,
        "sdk_observed_at": None,
        "sdk_freshness": "unknown",
        "capabilities": {capability: "unknown" for capability in CAPABILITIES},
        "limitation_codes": {"sdk_execution_unavailable", "tested_version_unknown"},
    }
    execution = private.get("sdk_execution")
    if execution is None:
        return unknown
    invalid = {
        **unknown,
        "limitation_codes": unknown["limitation_codes"] | {"sdk_evidence_inconsistent"},
    }
    if not isinstance(execution, dict):
        return invalid
    completed = date(execution.get("completed_at"))
    versions = execution.get("version")
    target = execution.get("nim_target_id")
    result = execution.get("result_id")
    framework_checks = private.get("framework_checks")
    fingerprint = execution.get("fingerprint_hash")
    checks = execution.get("checks")
    if not (
        execution.get("contract_version") == "bcb-sdk-execution-v1"
        and execution.get("package") == SDK_PACKAGE
        and isinstance(versions, str)
        and len(versions) <= 40
        and SDK_VERSION.fullmatch(versions)
        and execution.get("model_name") == private.get("model")
        and _integer(target)
        and target > 0
        and target == private.get("nim_target_id")
        and _integer(private.get("nim_target_id"))
        and _integer(result)
        and result > 0
        and isinstance(framework_checks, list)
        and isinstance(fingerprint, str)
        and fingerprint not in {"", "unknown", "—"}
        and fingerprint == private.get("fingerprint_hash")
        and deployment in {"hosted", "downloadable"}
        and execution.get("deployment") == deployment
        and private.get("deployment_type") == deployment
        and completed is not None
        and completed <= generated
        and execution.get("completed_at") == stamp(completed)
        and isinstance(checks, list)
        and 0 < len(checks) <= len(CAPABILITIES)
    ):
        return invalid
    if deployment == "downloadable" and not (
        isinstance(execution.get("digest"), str)
        and execution["digest"] not in {"", "unknown", "—"}
        and execution["digest"] == private.get("digest")
    ):
        return invalid
    capabilities = dict(unknown["capabilities"])
    seen = set()
    stale = completed < now - timedelta(days=max_age_days)
    for check in checks:
        if not isinstance(check, dict):
            return invalid
        capability = check.get("capability")
        if (
            not isinstance(capability, str)
            or capability not in SDK_CHECKS
            or capability in seen
        ):
            return invalid
        seen.add(capability)
        client, identifier = SDK_CHECKS[capability]
        if (
            check.get("client") != client
            or check.get("id") != identifier
            or check.get("status") not in ("passed", "failed", "unknown")
        ):
            return invalid
        attributed = [
            item
            for item in framework_checks
            if isinstance(item, dict)
            and item.get("id") == identifier
            and _integer(item.get("result_id"))
            and item["result_id"] == result
        ]
        if len(attributed) != 1:
            return invalid
        captured = attributed[0]
        packages = captured.get("package_versions")
        if not (
            captured.get("evidence_class") == "project_integration"
            and captured.get("status") == check["status"]
            and date(captured.get("completed_at")) == completed
            and isinstance(packages, dict)
            and packages.get(SDK_PACKAGE) == versions
        ):
            return invalid
        outcome = check["status"]
        capabilities[capability] = (
            "unknown"
            if outcome == "unknown"
            else "stale"
            if stale
            else "verified"
            if outcome == "passed"
            else "failed"
        )
    if set(capabilities.values()) == {"unknown"}:
        return invalid
    codes = {"sdk_evidence_stale"} if stale else set()
    if "unknown" in capabilities.values():
        codes.add("sdk_features_partial")
    return {
        "tested_sdk_version": versions,
        "sdk_observed_at": stamp(completed),
        "sdk_freshness": "stale" if stale else "current",
        "capabilities": capabilities,
        "limitation_codes": codes,
    }


def badge_name(row: dict[str, Any]) -> str:
    # Injective, reversible encoding: no slash replacement collisions or private hash.
    return f"model-{row['model_id'].encode('ascii').hex()}--{row['deployment']}.json"


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def export_public(decisions: Any, ids: set[str], *, now: datetime) -> dict[str, Any]:
    if (
        not isinstance(decisions, dict)
        or not _integer(decisions.get("schema_version"))
        or decisions["schema_version"] != 1
        or decisions.get("framework") != FRAMEWORK
        or not _integer(decisions.get("max_age_days"))
        or decisions["max_age_days"] <= 0
        or not isinstance(decisions.get("rows"), list)
    ):
        raise ValueError("Expected private normalized decision schema 1")
    generated = date(decisions.get("generated_at"))
    if generated is None or generated > now:
        raise ValueError("Expected nonfuture generation timestamp")
    age = decisions["max_age_days"]
    rows = []
    seen = set()
    for private in decisions["rows"]:
        if not isinstance(private, dict):
            raise ValueError("Expected decision row object")
        model = identity(private.get("model"), ids)
        measured = private.get("deployment_evidence")
        deployment = (
            measured
            if isinstance(measured, str) and measured in DEPLOYMENTS
            else "unknown"
        )
        key = (model, deployment)
        if key in seen:
            raise ValueError("Duplicate public model/deployment identity")
        seen.add(key)
        observed = date(private.get("tested_at"))
        codes = set(REQUIRED_LIMITATIONS)
        inconsistent = (observed is not None and observed > generated) or (
            deployment != "unknown" and private.get("deployment_type") != deployment
        )
        if observed is None:
            freshness = "unknown"
            codes.add("observation_missing")
        elif observed > generated:
            # An inconsistent future observation is not an actual tested timestamp.
            observed = None
            freshness = "unknown"
            codes.add("observation_missing")
        elif observed < now - timedelta(days=age):
            freshness = "stale"
            codes.add("evidence_stale")
        elif private.get("freshness") == "current":
            freshness = "current"
        else:
            freshness = "unknown"
            inconsistent = True
        if deployment == "unknown":
            codes.add("deployment_unknown")
        reported = private.get("status")
        baseline = (
            reported
            if isinstance(reported, str) and reported in BASELINES
            else "unknown"
        )
        if baseline == "verified" and (
            freshness != "current" or deployment == "unknown" or inconsistent
        ):
            baseline = "stale" if freshness == "stale" else "unknown"
        if baseline == "stale" and freshness != "stale":
            baseline = "unknown"
            inconsistent = True
        if inconsistent:
            codes.add("evidence_inconsistent")
            if baseline == "verified":
                baseline = "unknown"
        sdk = sdk_projection(
            private,
            deployment=deployment,
            generated=generated,
            now=now,
            max_age_days=age,
        )
        codes.update(sdk.pop("limitation_codes"))
        rows.append(
            {
                "model_id": model,
                "deployment": deployment,
                "framework_baseline": baseline,
                "observed_at": stamp(observed) if observed else None,
                "freshness": freshness,
                **sdk,
                "limitation_codes": sorted(codes),
            }
        )
    result = {
        "contract": CONTRACT,
        "framework": FRAMEWORK,
        "generated_at": stamp(now),
        "max_age_days": age,
        "source_owner": SOURCE_OWNER,
        "rows": sorted(rows, key=lambda row: (row["model_id"], row["deployment"])),
    }
    validate_public(result, ids, now=now)
    return result


def validate_public(data: Any, ids: set[str], *, now: datetime) -> dict[str, Any]:
    """Validate the exact public projection; owner/security review stays external.

    SDK feature states require a measured deployment, tested release version and
    their own actual observation timestamp/freshness. Public input cannot carry
    private evidence identifiers; review of the exact snapshot establishes its
    provenance before dispatch. A framework baseline alone never verifies SDK use.
    """
    if (
        not isinstance(data, dict)
        or set(data) != TOP_FIELDS
        or data.get("contract") != CONTRACT
        or data.get("framework") != FRAMEWORK
        or data.get("source_owner") != SOURCE_OWNER
        or not _integer(data.get("max_age_days"))
        or data["max_age_days"] <= 0
        or not isinstance(data.get("rows"), list)
    ):
        raise ValueError("Invalid public snapshot schema")
    generated = date(data["generated_at"])
    if generated is None or data["generated_at"] != stamp(generated) or generated > now:
        raise ValueError("Invalid public generation timestamp")
    seen = set()
    for row in data["rows"]:
        if not isinstance(row, dict) or set(row) != ROW_FIELDS:
            raise ValueError("Invalid public row fields")
        identity(row["model_id"], ids)
        if (
            not isinstance(row["deployment"], str)
            or row["deployment"] not in DEPLOYMENTS
        ):
            raise ValueError("Invalid measured deployment")
        key = (row["model_id"], row["deployment"])
        if key in seen:
            raise ValueError("Duplicate public model/deployment identity")
        seen.add(key)
        if (
            not isinstance(row["framework_baseline"], str)
            or row["framework_baseline"] not in BASELINES
        ):
            raise ValueError("Invalid framework baseline state")
        if row["freshness"] not in ("current", "stale", "unknown"):
            raise ValueError("Invalid observation freshness")
        capabilities = row["capabilities"]
        if (
            not isinstance(capabilities, dict)
            or set(capabilities) != set(CAPABILITIES)
            or any(
                not isinstance(value, str) or value not in SDK_STATES
                for value in capabilities.values()
            )
        ):
            raise ValueError("Invalid SDK capability states")
        sdk_version = row["tested_sdk_version"]
        sdk_observed = date(row["sdk_observed_at"])
        sdk_freshness = row["sdk_freshness"]
        if sdk_freshness not in ("current", "stale", "unknown"):
            raise ValueError("Invalid SDK freshness")
        if sdk_version is None:
            if (
                row["sdk_observed_at"] is not None
                or sdk_freshness != "unknown"
                or any(state != "unknown" for state in capabilities.values())
            ):
                raise ValueError("Absent tested SDK version cannot establish support")
        elif (
            not isinstance(sdk_version, str)
            or len(sdk_version) > 40
            or SDK_VERSION.fullmatch(sdk_version) is None
            or row["deployment"] == "unknown"
            or sdk_observed is None
            or row["sdk_observed_at"] != stamp(sdk_observed)
            or sdk_observed > generated
            or set(capabilities.values()) == {"unknown"}
        ):
            raise ValueError("Incomplete SDK version/deployment/observation binding")
        if sdk_freshness == "current" and (
            sdk_observed is None
            or sdk_observed < now - timedelta(days=data["max_age_days"])
            or "stale" in capabilities.values()
        ):
            raise ValueError("Current SDK evidence is absent or expired")
        if sdk_freshness == "stale" and (
            sdk_observed is None
            or sdk_observed >= now - timedelta(days=data["max_age_days"])
            or any(state in {"verified", "failed"} for state in capabilities.values())
        ):
            raise ValueError("Stale SDK execution cannot establish current support")
        if (
            any(state in {"verified", "failed"} for state in capabilities.values())
            and sdk_freshness != "current"
        ):
            raise ValueError(
                "Classified SDK features require current executed evidence"
            )
        if "stale" in capabilities.values() and sdk_freshness != "stale":
            raise ValueError("Stale SDK feature requires stale observation")
        codes = row["limitation_codes"]
        if (
            not isinstance(codes, list)
            or not all(isinstance(code, str) for code in codes)
            or codes != sorted(set(codes))
            or not set(codes) <= LIMITATIONS.keys()
            or not REQUIRED_LIMITATIONS <= set(codes)
        ):
            raise ValueError("Invalid predefined limitations")
        if (sdk_version is None) != ("tested_version_unknown" in codes):
            raise ValueError("SDK version limitation mismatch")
        if (sdk_version is None) != ("sdk_execution_unavailable" in codes):
            raise ValueError("SDK execution limitation mismatch")
        if (sdk_freshness == "stale") != ("sdk_evidence_stale" in codes):
            raise ValueError("SDK stale limitation mismatch")
        if sdk_version is not None and (
            ("unknown" in capabilities.values()) != ("sdk_features_partial" in codes)
            or "sdk_evidence_inconsistent" in codes
        ):
            raise ValueError("SDK feature limitation mismatch")
        observed = date(row["observed_at"])
        if row["observed_at"] is not None and (
            observed is None
            or row["observed_at"] != stamp(observed)
            or observed > generated
        ):
            raise ValueError("Invalid observation timestamp")
        if (observed is None) != ("observation_missing" in codes):
            raise ValueError("Observation limitation mismatch")
        if (row["deployment"] == "unknown") != ("deployment_unknown" in codes):
            raise ValueError("Deployment limitation mismatch")
        if row["freshness"] == "current" and (
            observed is None or observed < now - timedelta(days=data["max_age_days"])
        ):
            raise ValueError(
                "Current evidence is absent or expired; re-export for review"
            )
        if (row["freshness"] == "stale") != ("evidence_stale" in codes):
            raise ValueError("Stale limitation mismatch")
        if row["freshness"] == "stale" and (
            observed is None or observed >= now - timedelta(days=data["max_age_days"])
        ):
            raise ValueError("Stale observation mismatch")
        if row["framework_baseline"] == "verified" and (
            row["freshness"] != "current"
            or row["deployment"] == "unknown"
            or "evidence_inconsistent" in codes
        ):
            raise ValueError("Incomplete verified framework baseline binding")
        if row["framework_baseline"] == "stale" and row["freshness"] != "stale":
            raise ValueError("Stale baseline mismatch")
    if data["rows"] != sorted(
        data["rows"], key=lambda row: (row["model_id"], row["deployment"])
    ):
        raise ValueError("Public rows must use canonical model/deployment order")
    return data


def sdk_summary(row: dict[str, Any]) -> str:
    states = row["capabilities"].values()
    if "failed" in states:
        return "failed"
    if "stale" in states:
        return "stale"
    verified = sum(state == "verified" for state in states)
    return f"{verified}/{len(CAPABILITIES)} verified" if verified else "unknown"


def sdk_green(row: dict[str, Any]) -> bool:
    return (
        row["framework_baseline"] == "verified"
        and row["freshness"] == "current"
        and row["sdk_freshness"] == "current"
        and row["tested_sdk_version"] is not None
        and row["deployment"] != "unknown"
        and set(row["capabilities"].values()) == {"verified"}
    )


def badge(row: dict[str, Any]) -> dict[str, Any]:
    state = row["framework_baseline"]
    sdk = sdk_summary(row)
    return {
        "schemaVersion": 1,
        "label": "BCB framework / SDK",
        "message": f"{state} baseline / SDK {sdk}",
        "color": "green"
        if sdk_green(row)
        else (
            "red"
            if state in {"failed", "blocked", "incompatible"} or sdk == "failed"
            else (
                "orange"
                if state in {"verified", "stale", "changed"} or sdk != "unknown"
                else "lightgrey"
            )
        ),
    }


def badge_url(path: str) -> str:
    # Only hardcoded public repository paths produced by this generator are linked.
    return "https://img.shields.io/endpoint?url=" + quote(RAW_BASE + path, safe="")


def readme_block() -> str:
    return "\n".join(
        [
            BEGIN,
            "[![BCB framework baseline and SDK feature evidence]"
            f"({badge_url(BADGE_DIR + '/summary.json')})]"
            "(docs/compatibility_state.md)",
            "",
            "[Reviewed model/deployment compatibility table]"
            "(docs/compatibility_state.md). "
            "Framework baselines are not SDK feature verification; "
            "inspect timestamps and limitations.",
            END,
        ]
    )


def replace_readme(source: str, *, previous: bool) -> str:
    counts = (source.count(BEGIN), source.count(END))
    if counts == (0, 0):
        if previous:
            raise ValueError("Previously managed README section is missing")
        first, separator, rest = source.partition("\n")
        if not separator or not first.startswith("# "):
            raise ValueError("Expected README title for managed block insertion")
        return first + "\n\n" + readme_block() + "\n" + rest
    if counts != (1, 1):
        raise ValueError("Ambiguous managed README markers")
    start, end = source.index(BEGIN), source.index(END) + len(END)
    if end <= start or source[start:end] != readme_block() or not previous:
        raise ValueError("README managed block collision")
    return source[:start] + readme_block() + source[end:]


def render_files(data: dict[str, Any]) -> dict[str, str]:
    rows = data["rows"]
    baseline_count = sum(row["framework_baseline"] == "verified" for row in rows)
    sdk_counts = {
        state: sum(
            row["capabilities"][capability] == state
            for row in rows
            for capability in CAPABILITIES
        )
        for state in SDK_STATES
    }
    sdk_message = (
        "unknown"
        if sdk_counts["unknown"] == len(rows) * len(CAPABILITIES)
        else ", ".join(
            f"{sdk_counts[state]} {state}"
            for state in ("verified", "failed", "stale", "unknown")
        )
    )
    files = {SNAPSHOT_PATH: json_text(data)}
    files[BADGE_DIR + "/summary.json"] = json_text(
        {
            "schemaVersion": 1,
            "label": "BCB framework / SDK",
            "message": (
                f"{baseline_count}/{len(rows)} current baselines / "
                f"SDK {sdk_message}"
            ),
            "color": "green"
            if rows and all(sdk_green(row) for row in rows)
            else (
                "red"
                if any(badge(row)["color"] == "red" for row in rows)
                else (
                    "orange"
                    if any(badge(row)["color"] == "orange" for row in rows)
                    else "lightgrey"
                )
            ),
        }
    )
    lines = [
        "<!-- Generated by scripts/bcb_compatibility_publish.py; do not edit. -->",
        "# Reviewed BCB compatibility state",
        "",
        f"Snapshot generated: {data['generated_at']}. "
        f"Maximum observation age: {data['max_age_days']} days.",
        f"Source-owner role: {SOURCE_OWNER}.",
        "",
        "This static snapshot does not automatically refresh. "
        "Compare observation timestamps with the current date; "
        "evidence beyond the maximum age is stale even if a badge is cached.",
        "Framework baseline states describe framework-level evidence, "
        "not ChatNVIDIA or other SDK feature execution. "
        "A verified baseline cannot certify any feature or tested SDK version. "
        "Native probes, OpenAI-compatible wire checks and installed package "
        "inventories are not SDK proof.",
        "SDK features are classified only from explicitly "
        "version/model/target/result/deployment-bound SDK execution. "
        "Each feature is verified, failed, stale or unknown independently. "
        "Untested features remain unknown. A green aggregate badge requires "
        "every listed SDK feature and framework baseline to be current "
        "and verified; partial verification remains amber.",
        "Measured deployment is evidence-derived; hosted and downloadable "
        "results are not interchangeable. Unknown deployment does not "
        "inherit the selection's deployment type.",
        "The generator does not grant publication approval. Source-owner "
        "and security review of the exact public snapshot is required "
        "before dispatch; connector maintainer review is required before merge.",
        "Private identifiers, digests, fingerprints, links, freeform text, "
        "package inventories and runtime details are intentionally excluded.",
        "",
        "| Model | Measured deployment | Framework baseline | "
        "Framework observed (UTC) | Framework freshness | "
        "Tested SDK version | SDK observed (UTC) | SDK freshness | "
        + " | ".join(CAPABILITIES)
        + " | Badge | Limitations |",
        "| " + " | ".join(["---"] * (10 + len(CAPABILITIES))) + " |",
    ]
    for row in rows:
        path = BADGE_DIR + "/" + badge_name(row)
        files[path] = json_text(badge(row))
        cells = [
            f"`{row['model_id']}`",
            row["deployment"],
            row["framework_baseline"],
            row["observed_at"] or "unknown",
            row["freshness"],
            row["tested_sdk_version"] or "unknown",
            row["sdk_observed_at"] or "unknown",
            row["sdk_freshness"],
            *(row["capabilities"][capability] for capability in CAPABILITIES),
            f"![Framework baseline and SDK feature evidence]({badge_url(path)})",
            ", ".join(f"`{code}`" for code in row["limitation_codes"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    if not rows:
        lines += [
            "",
            "No reviewed model/deployment rows are supplied. No support claim is made.",
        ]
    lines += ["", "## Limitations", ""]
    codes = sorted(
        {code for row in rows for code in row["limitation_codes"]}
        | REQUIRED_LIMITATIONS
    )
    lines.extend(f"- `{code}`: {LIMITATIONS[code]}" for code in codes)
    files[DOC_PATH] = "\n".join(lines) + "\n"
    return files


def source_text(root: Path, relative: str) -> str | None:
    path = root / relative
    if path.is_symlink() or any(
        parent.is_symlink() for parent in path.parents if parent != root.parent
    ):
        raise ValueError("Generated source paths must not traverse symlinks")
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError("Generated source path collision")
    return path.read_text(encoding="utf-8")


def file_diff(relative: str, before: str | None, after: str | None) -> str:
    chunks = difflib.unified_diff(
        (before or "").splitlines(keepends=True),
        (after or "").splitlines(keepends=True),
        fromfile=f"a/{PACKAGE_PATH}/{relative}" if before is not None else "/dev/null",
        tofile=f"b/{PACKAGE_PATH}/{relative}" if after is not None else "/dev/null",
    )
    return "".join(
        chunk if chunk.endswith("\n") else chunk + "\n\\ No newline at end of file\n"
        for chunk in chunks
    )


def prepare_bundle(
    data: Any, ids: set[str], *, root: Path, now: datetime
) -> tuple[str, str]:
    validate_public(data, ids, now=now)
    current_text = source_text(root, SNAPSHOT_PATH)
    previous = None
    if current_text is not None:
        previous = load_json(root / SNAPSHOT_PATH)
        # Existing generated ownership is checked at its generation time. An expired
        # previous snapshot must remain removable/replacable by newly reviewed data.
        generated = (
            date(previous.get("generated_at")) if isinstance(previous, dict) else None
        )
        if generated is None:
            raise ValueError("Invalid existing public snapshot")
        validate_public(previous, ids, now=generated)
    old_files = render_files(previous) if previous is not None else {}
    new_files = render_files(data)
    if previous == data:
        new_files[SNAPSHOT_PATH] = current_text
    before_files: dict[str, str | None] = {}
    for path in old_files.keys() | new_files.keys():
        before = source_text(root, path)
        if path == SNAPSHOT_PATH:
            before_files[path] = before
            continue
        if path in old_files:
            if before != old_files[path]:
                raise ValueError("Previously generated file is missing or modified")
        elif before is not None:
            raise ValueError("Unmanaged generated path collision")
        before_files[path] = before
    readme = source_text(root, "README.md")
    if readme is None:
        raise ValueError("README source is missing")
    new_files["README.md"] = replace_readme(readme, previous=previous is not None)
    before_files["README.md"] = readme
    patch = "".join(
        file_diff(path, before_files[path], new_files.get(path))
        for path in sorted(before_files)
    )
    before_rows = (
        {(row["model_id"], row["deployment"]): row for row in previous["rows"]}
        if previous
        else {}
    )
    after_rows = {(row["model_id"], row["deployment"]): row for row in data["rows"]}
    lines = [
        "# Draft reviewed BCB compatibility documentation",
        "",
        "This paired patch is review-only. Its snapshot is not approved "
        "by export, validation, a commit SHA, a badge or this PR. "
        "Obtain source-owner and security review of this exact public snapshot "
        "before dispatch and connector maintainer acceptance before merge.",
        "",
        "## Public compatibility deltas (release-note input)",
        "",
    ]
    changes = 0
    for key in sorted(before_rows.keys() | after_rows.keys()):
        old, new = before_rows.get(key), after_rows.get(key)
        if old == new:
            continue
        changes += 1
        prefix = f"- `{key[0]}` / {key[1]}: "
        if old is None:
            lines.append(
                prefix + f"added; framework baseline {new['framework_baseline']}; "
                f"SDK {sdk_summary(new)}."
            )
        elif new is None:
            lines.append(
                prefix + "removed from this snapshot; "
                "removal is not a claim of incompatibility."
            )
        else:
            lines.append(
                prefix + f"framework baseline {old['framework_baseline']} → "
                f"{new['framework_baseline']}; "
                f"freshness {old['freshness']} → {new['freshness']}; "
                f"SDK {sdk_summary(old)} → {sdk_summary(new)}."
            )
        for row in (old, new):
            if row is not None:
                lines.append(
                    f"  - {'Before' if row is old else 'After'}: "
                    f"observed {row['observed_at'] or 'unknown'}; "
                    "limitations " + ", ".join(row["limitation_codes"]) + "."
                )
                lines.append(
                    f"  - SDK version {row['tested_sdk_version'] or 'unknown'}; "
                    f"observed {row['sdk_observed_at'] or 'unknown'}; "
                    f"freshness {row['sdk_freshness']}; "
                    + ", ".join(
                        f"{capability}: {row['capabilities'][capability]}"
                        for capability in CAPABILITIES
                    )
                    + "."
                )
    if not changes:
        lines.append("- No public model/deployment state changes.")
    lines += [
        "",
        f"Snapshot generation: {data['generated_at']}; "
        f"maximum age: {data['max_age_days']} days.",
        "",
        "## Provenance, test and acceptance gates",
        "",
        "- [ ] Source owner confirms measured deployment, target/result "
        "agreement, actual timestamps and framework baseline and SDK "
        "execution provenance/package version/check identities against "
        "private evidence; no private evidence is attached publicly.",
        "- [ ] Security owner approves the exact public snapshot "
        "and publication scope before manual dispatch.",
        "- [ ] Confirm no installed-package/native/wire result "
        "is represented as executed SDK support.",
        "- [ ] Apply this entire paired patch and check data, table, "
        "README links and Shields endpoint files together.",
        "- [ ] Run `pytest tests/unit_tests/test_bcb_compatibility_publish.py` "
        "from `libs/ai-endpoints`; regenerate to confirm an empty patch "
        "and inspect privacy/freshness failure cases.",
        "- [ ] Verify public badge URLs after merge; retain static freshness "
        "limitations and obtain connector maintainer review.",
        "- [ ] Use only the public state deltas above for release notes "
        "after acceptance; no automatic merge or release approval.",
        "",
    ]
    return patch, "\n".join(lines)


def write_bundle(
    outputs: list[tuple[Path, str]], *, inputs: list[Path], root: Path
) -> None:
    """Own a NEW invocation-private output directory, never a shared namespace.

    All outputs share one new parent; its parent must already exist. Inputs must
    be outside it. The helper exclusively creates the directory mode 0700 and
    output files mode 0600. Creation/rollback uses its pinned directory descriptor.
    On failure created files are removed, but the reserved directory is retained:
    pathname replacement cannot make us remove an unrelated parent directory.
    Callers must not share this private namespace with other writers.
    """
    if not outputs:
        raise ValueError("At least one output is required")
    if any(path.parent.exists() or path.parent.is_symlink() for path, _ in outputs):
        raise ValueError("Output directory must be new, not caller-created")
    directories = {
        path.parent.parent.resolve() / path.parent.name for path, _ in outputs
    }
    if len(directories) != 1:
        raise ValueError("Outputs must share one new private directory")
    directory = directories.pop()
    names = [path.name for path, _ in outputs]
    resolved_inputs = {path.resolve() for path in inputs}
    package = root.resolve()
    if (
        len(set(names)) != len(names)
        or directory == package
        or package in directory.parents
        or any(
            path == directory or directory in path.parents for path in resolved_inputs
        )
    ):
        raise ValueError("Private outputs must differ from inputs and package sources")
    # Pin the existing containing directory; mkdir is exclusive, including against
    # preexisting files/symlinks. No caller-created output directory is accepted.
    parent_fd = os.open(directory.parent, os.O_RDONLY | os.O_DIRECTORY)
    directory_fd = None
    created: list[str] = []
    try:
        os.mkdir(directory.name, mode=0o700, dir_fd=parent_fd)
        directory_fd = os.open(
            directory.name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        try:
            with ExitStack() as stack:
                for name, (_, text) in zip(names, outputs):
                    fd = os.open(
                        name,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600,
                        dir_fd=directory_fd,
                    )
                    created.append(name)
                    output = stack.enter_context(os.fdopen(fd, "w", encoding="utf-8"))
                    output.write(text)
        except BaseException:
            for name in created:
                try:
                    os.unlink(name, dir_fd=directory_fd)
                except FileNotFoundError:
                    pass
            raise
    finally:
        if directory_fd is not None:
            os.close(directory_fd)
        os.close(parent_fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser(
        "export-public", help="Write an UNAPPROVED allowlisted snapshot; never publish"
    )
    export.add_argument("--decisions", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser(
        "prepare-pr",
        help="Prepare an applicable paired patch/body without applying or pushing",
    )
    prepare.add_argument("--reviewed-snapshot", type=Path, required=True)
    prepare.add_argument("--patch-output", type=Path, required=True)
    prepare.add_argument("--pr-body-output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        ids = registry_ids()
        now = datetime.now(timezone.utc)
        if args.command == "export-public":
            data = export_public(load_json(args.decisions), ids, now=now)
            write_bundle(
                [(args.output, json_text(data))], inputs=[args.decisions], root=ROOT
            )
            print(
                "Exported UNAPPROVED public-safe snapshot; owner/security "
                "review required before dispatch"
            )
        else:
            patch, body = prepare_bundle(
                load_json(args.reviewed_snapshot), ids, root=ROOT, now=now
            )
            write_bundle(
                [(args.patch_output, patch), (args.pr_body_output, body)],
                inputs=[args.reviewed_snapshot],
                root=ROOT,
            )
            print(
                "Prepared paired compatibility patch/body; "
                "no sources applied and no approval granted"
            )
    except (OSError, ValueError, TypeError, KeyError, OverflowError):
        # Never reflect private data, arbitrary filenames or JSON decoder excerpts.
        print(
            "Compatibility publication input/output rejected; "
            "inspect locally against the public schema",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
