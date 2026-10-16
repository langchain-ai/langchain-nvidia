# The review CLI deliberately reports status on stdout/stderr.
# ruff: noqa: T201
"""Offline reviewed model registry validator and renderer.

This script uses only the standard library and never changes reviewed registry data
from a catalog response. Run from any directory; see docs/model_registry.md.
"""

import argparse
import json
import pprint
import re
import sys
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser(
        "generate", help="Check or render reviewed registry into _statics.py"
    )
    command.add_argument("--write", action="store_true", help="Write generated Python")
    args = parser.parse_args()
    return generate(args.write)


if __name__ == "__main__":
    sys.exit(main())
