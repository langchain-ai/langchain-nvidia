# The review CLI deliberately reports status on stdout/stderr.
# ruff: noqa: T201
"""Review-only model registry renderer and hosted/NGC drift reporter.

This script uses only the standard library and never changes reviewed registry data
from a catalog response. Run from any directory; see docs/model_registry.md.
"""

import argparse
import json
import os
import pprint
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "langchain_nvidia_ai_endpoints/data/model_registry.json"
STATICS = ROOT / "langchain_nvidia_ai_endpoints/_statics.py"
HOSTED_URL = "https://integrate.api.nvidia.com/v1/models"
TABLES = (
    "CHAT_MODEL_TABLE",
    "QA_MODEL_TABLE",
    "VLM_MODEL_TABLE",
    "EMBEDDING_MODEL_TABLE",
    "RANKING_MODEL_TABLE",
    "RANKING_VLM_MODEL_TABLE",
    "COMPLETION_MODEL_TABLE",
    "OPENAI_MODEL_TABLE",
)
CAPABILITIES = ("supports_tools", "supports_structured_output", "supports_thinking")
STATES = {"unknown", "current", "stale", "unavailable"}
CAPABILITY_SOURCES = {
    "unknown",
    "legacy_static",
    "hosted_catalog",
    "ngc_catalog",
    "bcb",
}

HARNESS_UPSTREAM_NEEDED = re.compile(
    r"UPSTREAM NEEDED: add '([A-Za-z0-9][A-Za-z0-9_.:/-]*)' "
    r"to langchain-nvidia _statics\.py"
)

BEGIN = (
    "# fmt: off\n"
    "# BEGIN GENERATED MODEL TABLES (scripts/model_registry.py; do not edit)\n"
)
END = "# END GENERATED MODEL TABLES\n# fmt: on\n"


def deployment_type(sources: set[str]) -> str:
    if len(sources) == 2:
        return "both"
    return next(iter(sources)) if sources else "unknown"


def load_registry(path: Path = REGISTRY) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or not isinstance(data.get("models"), list):
        raise ValueError("Expected model registry schema_version 1 with models array")
    models = data["models"]
    ids: set[str] = set()
    for row in models:
        identifier = row["id"]
        model = row["model"]
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError(f"Duplicate or invalid model id: {identifier!r}")
        ids.add(identifier)
        if row["table"] not in TABLES or model.get("id") != identifier:
            raise ValueError(f"Invalid table or mismatched model id: {identifier}")
        if (
            not isinstance(row["served_names"], list)
            or not all(isinstance(name, str) and name for name in row["served_names"])
            or identifier not in row["served_names"]
        ):
            raise ValueError(f"Missing or invalid served name: {identifier}")
        if len(set(row["served_names"])) != len(row["served_names"]):
            raise ValueError(f"Duplicate served name: {identifier}")
        if not isinstance(row["stale"], bool):
            raise ValueError(f"Invalid stale value: {identifier}")
        if set(row["capabilities"]) != set(CAPABILITIES):
            raise ValueError(f"Incomplete capabilities: {identifier}")
        for field in CAPABILITIES:
            state = row["capabilities"][field]
            if state not in ("supported", "unsupported", "unknown"):
                raise ValueError(f"Invalid capability: {identifier} {field}")
            if state != "unknown" and model.get(field) is not (state == "supported"):
                raise ValueError(
                    f"Capability disagrees with runtime model: {identifier} {field}"
                )
        if set(row["capability_provenance"]) != set(CAPABILITIES):
            raise ValueError(f"Incomplete capability provenance: {identifier}")
        for field in CAPABILITIES:
            evidence = row["capability_provenance"][field]
            evidence_source = evidence["source"]
            url = evidence["url"]
            state = row["capabilities"][field]
            if (
                evidence_source not in CAPABILITY_SOURCES
                or (evidence_source == "unknown") != (state == "unknown")
                or (
                    evidence_source == "legacy_static"
                    and (field not in model or model[field] is None or url is not None)
                )
                or (
                    evidence_source in {"hosted_catalog", "ngc_catalog", "bcb"}
                    and (not isinstance(url, str) or not url.startswith("https://"))
                )
                or (evidence_source == "unknown" and url is not None)
            ):
                raise ValueError(f"Invalid capability provenance: {identifier} {field}")
        if set(row["provenance"]) != {"hosted", "downloadable"}:
            raise ValueError(f"Incomplete provenance: {identifier}")
        for source in ("hosted", "downloadable"):
            info = row["provenance"][source]
            if (
                info["state"] not in STATES
                or (info["state"] != "unknown" and not info["url"])
                or (
                    info["url"] is not None
                    and (
                        not isinstance(info["url"], str)
                        or not info["url"].startswith("https://")
                    )
                )
            ):
                raise ValueError(f"Invalid provenance: {identifier} {source}")
        current_sources = {
            source
            for source in ("hosted", "downloadable")
            if row["provenance"][source]["state"] == "current"
        }
        if row["deployment_type"] != deployment_type(current_sources):
            raise ValueError(f"Deployment type disagrees with provenance: {identifier}")
    return models


def render(models: list[dict[str, Any]]) -> str:
    lines = [BEGIN.rstrip("\n")]
    for table in TABLES:
        lines.append(f"{table} = {{")
        for row in models:
            if row["table"] != table:
                continue
            lines.append(f"    {row['id']!r}: Model(")
            for key, value in row["model"].items():
                literal = pprint.pformat(value, width=72, sort_dicts=False)
                lines.append(f"        {key}={literal},")
            lines.append("    ),")
        lines.extend(("}", ""))
    lines.append(END.rstrip("\n"))
    return "\n".join(lines) + "\n\n"


def replace_generated(source: str, generated: str) -> str:
    if source.count(BEGIN) != 1 or source.count(END) != 1:
        raise ValueError("Expected exactly one generated model table section")
    start = source.index(BEGIN)
    stop = source.index(END, start) + len(END)
    next_table = source.index("MODEL_TABLE = {\n", stop)
    if source[stop:next_table].strip():
        raise ValueError("Unexpected code between generated tables and MODEL_TABLE")
    stop = next_table
    return source[:start] + generated + source[stop:]


def generate(write: bool) -> int:
    expected = replace_generated(
        STATICS.read_text(encoding="utf-8"), render(load_registry())
    )
    current = STATICS.read_text(encoding="utf-8")
    if current == expected:
        print("Generated model tables are current")
        return 0
    if not write:
        print(
            "Generated model tables differ; run "
            "scripts/model_registry.py generate --write",
            file=sys.stderr,
        )
        return 1
    STATICS.write_text(expected, encoding="utf-8")
    print("Updated generated model tables (review the diff before merging)")
    return 0


def snapshot(
    path: Path | None, source: str
) -> tuple[dict[str, dict[str, Any]] | None, str]:
    if path is None and source == "hosted":
        try:
            headers = {"Accept": "application/json"}
            if token := os.getenv("NVIDIA_API_KEY"):
                headers["Authorization"] = f"Bearer {token}"
            req = urllib.request.Request(HOSTED_URL, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response:
                payload = json.load(response)
        except (OSError, ValueError) as exc:
            return None, f"{HOSTED_URL}: {exc}"
        source_url = HOSTED_URL
    elif path is not None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return None, f"{path}: {exc}"
        source_url = payload.get("source_url", "") if isinstance(payload, dict) else ""
    else:
        return None, "NGC snapshot not supplied (--ngc-file; do not infer removals)"
    key = "data" if source == "hosted" else "models"
    if not isinstance(payload, dict) or not isinstance(payload.get(key), list):
        return None, f"{source}: expected an object containing {key} array"
    if not isinstance(source_url, str) or not source_url.startswith("https://"):
        return None, f"{source}: missing HTTPS source_url evidence"
    if not payload[key]:
        return None, f"{source}: empty catalog; refusing to infer removals"
    entries: dict[str, dict[str, Any]] = {}
    for item in payload[key]:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("id"), str)
            or not item["id"]
        ):
            return None, f"{source}: record missing id"
        if item["id"] in entries:
            return None, f"{source}: duplicate id {item['id']}"
        if "url" in item and (
            not isinstance(item["url"], str) or not item["url"].startswith("https://")
        ):
            return None, f"{source}: invalid HTTPS evidence URL for {item['id']}"
        for field in ("aliases", "served_names"):
            if field in item and (
                not isinstance(item[field], list)
                or not all(isinstance(n, str) for n in item[field])
            ):
                return None, f"{source}: invalid {field} for {item['id']}"
        if "replaces" in item and not isinstance(item["replaces"], str):
            return None, f"{source}: invalid replaces for {item['id']}"
        if "deprecated" in item and not isinstance(item["deprecated"], bool):
            return None, f"{source}: invalid deprecation for {item['id']}"
        if "capabilities" in item and not isinstance(item["capabilities"], dict):
            return None, f"{source}: invalid capabilities for {item['id']}"
        entries[item["id"]] = {**item, "url": item.get("url", source_url)}
    return entries, source_url


def compare(
    models: list[dict[str, Any]],
    hosted: dict[str, dict[str, Any]] | None,
    ngc: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    known = {row["id"]: row for row in models if row["table"] != "OPENAI_MODEL_TABLE"}
    for source, observed in (("hosted", hosted), ("downloadable", ngc)):
        if observed is None:
            continue
        for identifier, item in sorted(observed.items()):
            url = item["url"]
            old = known.get(identifier)
            replaces = item.get("replaces")
            # Upstream replacements are rename proposals, not automatic aliases.
            # Each catalog is evidence for its own deployment type only.
            if old is None:
                kind = "rename" if replaces in known else "addition"
                changes.append(
                    {
                        "source": source,
                        "kind": kind,
                        "id": identifier,
                        "replaces": replaces if kind == "rename" else None,
                        "suggested_deployment_type": source,
                        "evidence": url,
                    }
                )
                continue
            prior = old["provenance"][source]["state"]
            if prior in {"stale", "unavailable"}:
                changes.append(
                    {
                        "source": source,
                        "kind": "availability",
                        "id": identifier,
                        "before": prior,
                        "after": "current",
                        "evidence": url,
                    }
                )
            if prior != "current":
                available = {
                    name
                    for name in ("hosted", "downloadable")
                    if old["provenance"][name]["state"] == "current"
                }
                changes.append(
                    {
                        "source": source,
                        "kind": "deployment_type",
                        "id": identifier,
                        "before": old["deployment_type"],
                        "after": deployment_type(available | {source}),
                        "provenance_before": prior,
                        "provenance_after": "current",
                        "evidence": url,
                    }
                )
            for field in ("aliases", "served_names"):
                if field in item:
                    before = (
                        old["model"].get("aliases", [])
                        if field == "aliases"
                        else old[field]
                    )
                    after = item[field]
                    if set(before or []) != set(after):
                        changes.append(
                            {
                                "source": source,
                                "kind": field,
                                "id": identifier,
                                "before": before,
                                "after": after,
                                "evidence": url,
                            }
                        )
            if "deprecated" in item and item["deprecated"] != old["model"].get(
                "deprecated", False
            ):
                changes.append(
                    {
                        "source": source,
                        "kind": "deprecation",
                        "id": identifier,
                        "before": old["model"].get("deprecated", False),
                        "after": item["deprecated"],
                        "evidence": url,
                    }
                )
            for field in CAPABILITIES:
                reported = item.get("capabilities", {}).get(field, "unknown")
                if reported not in ("supported", "unsupported", "unknown"):
                    raise ValueError(
                        f"Invalid upstream capability {field} for {identifier}"
                    )
                if reported == "unknown":
                    continue
                previous = old["capability_provenance"][field]
                new_evidence = {
                    "source": "hosted_catalog" if source == "hosted" else "ngc_catalog",
                    "url": url,
                }
                if old["capabilities"][field] != reported or previous["source"] in (
                    "unknown",
                    "legacy_static",
                ):
                    changes.append(
                        {
                            "source": source,
                            "kind": "capability"
                            if old["capabilities"][field] != reported
                            else "capability provenance",
                            "id": identifier,
                            "field": field,
                            "before": old["capabilities"][field],
                            "after": reported,
                            "provenance_before": previous,
                            "provenance_after": new_evidence,
                            "claim_only": True,
                            "evidence": url,
                        }
                    )
        for identifier, old in sorted(known.items()):
            # Absence alone is not proof a model was ever in that catalog.
            if (
                identifier not in observed
                and old["provenance"][source]["state"] == "current"
            ):
                remaining = {
                    name
                    for name in ("hosted", "downloadable")
                    if name != source and old["provenance"][name]["state"] == "current"
                }
                changes.append(
                    {
                        "source": source,
                        "kind": "removal/stale",
                        "id": identifier,
                        "before": old["deployment_type"],
                        "after": deployment_type(remaining),
                        "evidence": old["provenance"][source]["url"],
                    }
                )
    return changes


def harness_candidates(
    path: Path, evidence_url: str, known_ids: set[str]
) -> list[dict[str, Any]]:
    """Read the downloadable harness's explicit UPSTREAM NEEDED lines only."""
    text = path.read_text(encoding="utf-8")
    return [
        {
            "source": "downloadable_harness",
            "kind": "addition",
            "id": identifier,
            "suggested_deployment_type": "downloadable",
            "evidence": evidence_url,
            "review_note": (
                "Harness candidate only; verify NGC resource and actual served name."
            ),
        }
        for identifier in sorted(set(HARNESS_UPSTREAM_NEEDED.findall(text)) - known_ids)
    ]


def report(
    changes: list[dict[str, Any]], sources: dict[str, str], missing: list[str]
) -> str:
    lines = [
        "# Model registry drift — maintainer review required",
        "",
        "This is a proposal, **not** a published model update. "
        "Validate each linked source,",
        "probe any claimed capability with the compatibility harness, "
        "and review registry",
        "and generated Python changes in a pull request before merging.",
        "",
        "## Source evidence",
        "",
    ]
    lines += [f"- {name}: {url}" for name, url in sources.items()]
    lines += [f"- **Unavailable**: {problem}" for problem in missing]
    lines += ["", "## Changes to review", ""]
    if not changes:
        lines.append(
            "No differences detected in the available sources "
            "(not a compatibility claim)."
        )
    for change in changes:
        lines.append(
            f"- **{change['source']} / {change['kind']}** `{change['id']}` "
            f"— {change['evidence']}"
        )
        details = {
            k: v
            for k, v in change.items()
            if k not in ("source", "kind", "id", "evidence") and v is not None
        }
        if details:
            lines.append(f"  - Review: `{json.dumps(details, sort_keys=True)}`")
    lines += [
        "",
        "## Review checklist",
        "",
        "- [ ] Confirm source completeness and links; "
        "unavailable source never means removal.",
        "- [ ] Review each deployment type against its own hosted or NGC evidence.",
        "- [ ] Verify served IDs, rename/replacement and aliases "
        "with actual endpoint behavior.",
        "- [ ] Verify capability claims with behavioral evidence; "
        "catalog claims are not BCB proof.",
        "- [ ] Update reviewed JSON, regenerate Python, inspect diff, "
        "run unit tests and obtain maintainer approval.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate_cmd = commands.add_parser(
        "generate", help="Check or render reviewed registry into _statics.py"
    )
    generate_cmd.add_argument(
        "--write", action="store_true", help="Write generated Python"
    )
    drift_cmd = commands.add_parser("drift", help="Produce a review-only drift report")
    drift_cmd.add_argument(
        "--hosted-file", type=Path, help="Saved /v1/models response with source_url"
    )
    drift_cmd.add_argument(
        "--ngc-file", type=Path, help="Normalized NGC export with source_url"
    )
    drift_cmd.add_argument(
        "--harness-log", type=Path, help="Saved downloadable harness log"
    )
    drift_cmd.add_argument(
        "--harness-url", help="HTTPS pipeline/job URL for the harness log"
    )
    drift_cmd.add_argument(
        "--output", type=Path, required=True, help="PR-ready Markdown review artifact"
    )
    args = parser.parse_args()
    if args.command == "generate":
        return generate(args.write)
    if bool(args.harness_log) != bool(args.harness_url):
        parser.error("--harness-log and --harness-url must be supplied together")
    if args.harness_url and not args.harness_url.startswith("https://"):
        parser.error("--harness-url requires an HTTPS evidence link")
    models = load_registry()
    hosted, hosted_source = snapshot(args.hosted_file, "hosted")
    ngc, ngc_source = snapshot(args.ngc_file, "downloadable")
    missing = [
        text
        for records, text in ((hosted, hosted_source), (ngc, ngc_source))
        if records is None
    ]
    sources = {
        name: source
        for name, records, source in (
            ("hosted", hosted, hosted_source),
            ("NGC", ngc, ngc_source),
        )
        if records is not None
    }
    changes = compare(models, hosted, ngc)
    if args.harness_log:
        try:
            changes.extend(
                harness_candidates(
                    args.harness_log, args.harness_url, {row["id"] for row in models}
                )
            )
            sources["downloadable harness"] = args.harness_url
        except OSError:
            missing.append("Downloadable harness log could not be read")
    args.output.write_text(report(changes, sources, missing), encoding="utf-8")
    print(
        f"Wrote {args.output} ({len(changes)} changes, "
        f"{len(missing)} unavailable sources)"
    )
    return 2 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
