#!/usr/bin/env python3
"""Render private, read-only maintainer views from locally reviewed artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlsplit

import model_registry

CLASSIFICATION = "internal_restricted"
SCHEMA_VERSION = 1
METRICS = (
    "downloads",
    "docs_traffic",
    "cookbook_runs",
    "github_issues",
    "github_prs",
    "model_usage_examples",
)
LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _./:@+()#-]{0,119}\Z")
CAPABILITY = re.compile(r"[a-z][a-z0-9_]*\Z")
SAFE_PATH = re.compile(r"/[A-Za-z0-9/._~%-]*\Z")
SECRET = re.compile(
    r"(?:bearer|token|secret|password|api[_-]?key|authorization|credential)", re.I
)
REPO = "/langchain-ai/langchain-nvidia"


def label(value: object) -> str:
    """Never display arbitrary artifact content, including credential-like strings."""
    if (
        not isinstance(value, str)
        or not LABEL.fullmatch(value)
        or SECRET.search(value)
        or "://" in value
    ):
        return "unavailable"
    return value


def positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def link(value: object, *, category: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 500
        or "\\" in value
        or SECRET.search(value)
    ):
        return ""
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or parsed.username
            or parsed.password
            or parsed.port
            or parsed.query
            or parsed.fragment
            or not SAFE_PATH.fullmatch(parsed.path)
            or any(segment in (".", "..") for segment in parsed.path.split("/"))
            or "%" in parsed.path
        ):
            return ""
        host = parsed.hostname
        if category == "issue":
            valid = host == "github.com" and re.fullmatch(
                REPO + r"/issues/[1-9][0-9]*", parsed.path
            )
        elif category == "pr":
            valid = host == "github.com" and re.fullmatch(
                REPO + r"/pull/[1-9][0-9]*", parsed.path
            )
        elif category == "evidence":
            valid = host == "gitlab-master.nvidia.com" and parsed.path != "/"
        elif category == "source":
            valid = (
                host
                in {
                    "github.com",
                    "gitlab-master.nvidia.com",
                    "docs.langchain.com",
                    "pypi.org",
                    "catalog.ngc.nvidia.com",
                    "integrate.api.nvidia.com",
                }
                and parsed.path != "/"
            )
        else:
            valid = False
        return value if valid else ""
    except ValueError:
        return ""


def stamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        date = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return date.astimezone(timezone.utc) if date.tzinfo else None
    except ValueError:
        return None


def fresh(value: object, now: datetime, days: int) -> bool:
    date = stamp(value)
    return date is not None and now - timedelta(days=days) <= date <= now


def cell(value: object) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace("`", "\\`")
        .replace("\n", " ")
    )


def linked(url: str) -> str:
    return url if url else "unavailable"


def read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("expected JSON object")
    return data


def review_input(data: dict) -> dict:
    if (
        data.get("schema_version") != SCHEMA_VERSION
        or data.get("data_classification") != CLASSIFICATION
    ):
        raise ValueError(
            "review input requires schema_version 1 and "
            "internal_restricted classification"
        )
    if not isinstance(data.get("issues"), list) or not isinstance(
        data.get("adoption"), dict
    ):
        raise ValueError("review input requires issues array and adoption object")
    if any(not isinstance(row, dict) for row in data["issues"]):
        raise ValueError("issues must contain objects")
    return data


def release_view(data: dict, owner: object, now: datetime, days: int) -> list[str]:
    gate = data.get("gate") if isinstance(data.get("gate"), dict) else {}
    targets = data.get("targets") if isinstance(data.get("targets"), list) else []
    recent = fresh(data.get("collected_at"), now, days) and fresh(
        gate.get("evaluated_at"), now, days
    )
    valid_targets = len(targets) >= 2 and all(
        isinstance(t, dict)
        and t.get("status") == "pass"
        and isinstance(t.get("required_capabilities"), list)
        and bool(t["required_capabilities"])
        and all(
            isinstance(c, str) and CAPABILITY.fullmatch(c)
            for c in t["required_capabilities"]
        )
        and isinstance(t.get("freshness"), dict)
        and t["freshness"].get("state") == "known"
        and t["freshness"].get("stale") is False
        and fresh(t.get("tested_at"), now, days)
        and isinstance(t.get("latest_result"), dict)
        and positive_int(t["latest_result"].get("result_id"))
        and t["latest_result"].get("status") in {"passed", "current", "compatible"}
        and isinstance(t.get("accepted_result"), dict)
        and positive_int(t["accepted_result"].get("result_id"))
        and t["accepted_result"].get("result_id") == t["latest_result"]["result_id"]
        and any(
            link(t["accepted_result"].get(field), category="evidence")
            for field in (
                "pipeline_url",
                "job_url",
                "artifact_url",
                "detailed_report_url",
            )
        )
        for t in targets
    )
    ready = (
        data.get("contract_version") == "bcb-public-v1alpha1"
        and data.get("promotion") == "owner_approval_required"
        and data.get("status") == "pass"
        and gate.get("status") == "pass"
        and recent
        and valid_targets
        and {t.get("deployment") for t in targets} >= {"hosted", "downloadable"}
    )
    assessment = (
        "review evidence (not approval)"
        if ready
        else "HOLD — stale, unknown or blocked evidence"
    )
    lines = [
        "## Release Readiness — Manager BCB internal-alpha",
        "",
        (
            f"Assessment: **{assessment}** · Owner: {cell(label(owner))} · "
            "Release decision: **owner/security approval required**"
        ),
        (
            f"Baseline: {cell(label(data.get('baseline_version')))} · "
            f"Gate: {cell(label(gate.get('status')))} · "
            f"Evaluated: {cell(label(gate.get('evaluated_at')))}"
        ),
        "",
        (
            "| Deployment / target | Required capabilities (scope, not per-capability "
            "proof) | Latest run | Freshness | Blockers / action | Evidence |"
        ),
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for target in targets:
        if not isinstance(target, dict):
            continue
        latest = (
            target.get("latest_result")
            if isinstance(target.get("latest_result"), dict)
            else {}
        )
        accepted = (
            target.get("accepted_result")
            if isinstance(target.get("accepted_result"), dict)
            else {}
        )
        state = (
            target.get("freshness") if isinstance(target.get("freshness"), dict) else {}
        )
        is_fresh = (
            state.get("state") == "known"
            and state.get("stale") is False
            and fresh(target.get("tested_at"), now, days)
            and recent
        )
        reasons = (
            target.get("reasons") if isinstance(target.get("reasons"), list) else []
        )
        blockers = (
            ", ".join(label(r) for r in reasons)
            if reasons
            else (
                "inspect per-capability BCB evidence"
                if is_fresh and target.get("status") == "pass"
                else "refresh/inspect target evidence"
            )
        )
        caps = (
            target.get("required_capabilities")
            if isinstance(target.get("required_capabilities"), list)
            else []
        )
        urls = [
            link(result.get(field), category="evidence")
            for result in (latest, accepted)
            for field in (
                "pipeline_url",
                "job_url",
                "artifact_url",
                "detailed_report_url",
            )
        ]
        evidence = (
            ", ".join(linked(url) for url in dict.fromkeys(url for url in urls if url))
            or "unavailable"
        )
        nim_id = target.get("nim_id")
        result_id = latest.get("result_id")
        fields = (
            (
                f"{label(target.get('deployment'))} #"
                f"{nim_id if positive_int(nim_id) else 'unknown'}"
            ),
            ", ".join(label(c) for c in caps) or "unavailable",
            (
                f"#{result_id if positive_int(result_id) else 'unknown'} "
                f"{label(latest.get('status'))}"
            ),
            "current" if is_fresh else "stale/unknown",
            blockers,
            evidence,
        )
        lines.append("| " + " | ".join(cell(field) for field in fields) + " |")
    if not targets:
        lines.append(
            "| unavailable | unavailable | unavailable | stale/unknown | "
            "obtain Manager evidence | unavailable |"
        )
    reasons = data.get("reasons") if isinstance(data.get("reasons"), list) else []
    gate_reasons = gate.get("reasons") if isinstance(gate.get("reasons"), list) else []
    lines += [
        "",
        "Gate blockers: "
        + (
            ", ".join(label(r) for r in reasons + gate_reasons)
            or "none reported; verify complete BCB workflow back-reference"
        ),
        "",
    ]
    return lines


def drift_view(
    changes: list[dict], sources: dict[str, str], missing: list[str], pr_url: object
) -> list[str]:
    groups = {
        "Additions": {"addition", "rename"},
        "Deprecations / removals": {"deprecation", "removal/stale"},
        "Aliases / served names": {"aliases", "served_names"},
        "Capability flags (catalog claim only)": {
            "capability",
            "capability provenance",
        },
        "Other provenance / deployment changes": {"availability", "deployment_type"},
    }
    lines = [
        "## Model Registry Drift — proposals only",
        "",
        (
            f"Generated review PR: {linked(link(pr_url, category='pr'))} "
            "(not created by this command)"
        ),
        "",
    ]
    for title, kinds in groups.items():
        selected = [c for c in changes if c.get("kind") in kinds]
        lines.append(f"### {title} ({len(selected)})")
        for change in selected:
            evidence = link(change.get("evidence"), category="source") or link(
                change.get("evidence"), category="evidence"
            )
            lines.append(
                f"- {cell(label(change.get('source')))} / "
                f"{cell(label(change.get('kind')))} / "
                f"`{cell(label(change.get('id')))}`: {linked(evidence)} — "
                "review against source; verify served behavior before registry edit"
            )
        if not selected:
            lines.append(
                "- None observed in available sources (not a compatibility claim)."
            )
    lines += ["", "Source evidence:"]
    for source, url in sources.items():
        lines.append(
            f"- {cell(label(source))}: "
            f"{linked(link(url, category='source') or link(url, category='evidence'))}"
        )
    for source in missing:
        lines.append(f"- **Unavailable**: {cell(source)}; no removal inferred.")
    lines += [
        "- Catalog capability flags are claims, never BCB compatibility proof.",
        "",
    ]
    return lines


def issues_view(issues: list[dict]) -> list[str]:
    lines = [
        "## Issue Hygiene",
        "",
        (
            "| Issue | Priority | Affected version | Repro | Owner | Tests | "
            "Planned release | Triage / evidence | Action |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for issue in issues:
        fields = [
            label(issue.get(k))
            for k in (
                "priority",
                "affected_version",
                "repro",
                "owner",
                "tests",
                "planned_release",
            )
        ]
        url = link(issue.get("issue_url"), category="issue")
        triage = link(issue.get("triage_url"), category="issue")
        evidence = link(issue.get("evidence_url"), category="evidence")
        gaps = [
            name
            for name, value in zip(
                (
                    "priority",
                    "affected version",
                    "repro",
                    "owner",
                    "tests",
                    "planned release",
                ),
                fields,
            )
            if value == "unavailable"
        ]
        if not url:
            gaps.append("issue link")
        if not triage and not evidence:
            gaps.append("triage/evidence")
        action = (
            "fill " + ", ".join(gaps) if gaps else "review triage and release scope"
        )
        row = [
            linked(url),
            *fields,
            " / ".join(linked(u) for u in (triage, evidence) if u) or "unavailable",
            action,
        ]
        lines.append("| " + " | ".join(cell(field) for field in row) + " |")
    if not issues:
        lines.append(
            "| No issue artifact supplied | unavailable | unavailable | unavailable | "
            "unavailable | unavailable | unavailable | unavailable | "
            "supply reviewed issues |"
        )
    lines.append("")
    return lines


def adoption_view(adoption: dict, now: datetime, days: int) -> list[str]:
    lines = [
        "## Adoption Signals — observed sources only",
        "",
        "| Signal | Observation | As of | Source | Action |",
        "| --- | --- | --- | --- | --- |",
    ]
    for metric in METRICS:
        record = adoption.get(metric)
        record = record if isinstance(record, dict) else {}
        available = (
            record.get("status") == "available"
            and isinstance(record.get("count"), int)
            and not isinstance(record["count"], bool)
            and record["count"] >= 0
        )
        url = link(record.get("source_url"), category="source") or link(
            record.get("source_url"), category="evidence"
        )
        current = (
            available and bool(url) and fresh(record.get("observed_at"), now, days)
        )
        lines.append(
            "| "
            + " | ".join(
                cell(v)
                for v in (
                    metric.replace("_", " "),
                    str(record["count"])
                    if current
                    else "unavailable"
                    if not available or not url
                    else "stale/unknown (count withheld)",
                    label(record.get("observed_at")) if current else "unavailable",
                    linked(url) if current else "unavailable",
                    "inspect observed scope"
                    if current
                    else "obtain fresh, attributed source",
                )
            )
            + " |"
        )
    lines += [
        "",
        (
            "Counts are supplied by explicitly reviewed inputs, not fetched or "
            "inferred here; missing/stale sources are unavailable. Model usage "
            "examples count examples, not usage events."
        ),
        "",
    ]
    return lines


def render(
    review: dict,
    release: dict,
    changes: list[dict],
    sources: dict[str, str],
    missing: list[str],
    now: datetime,
    days: int,
) -> str:
    return "\n".join(
        [
            "# Connector maintainer dashboard — PRIVATE REVIEW ONLY",
            "",
            (
                "**Internal restricted artifact.** No public/published surface until "
                "designated owner AND security approval. This view never promotes, "
                "opens a PR, or asserts per-capability compatibility."
            ),
            "",
            *release_view(release, review.get("release_owner"), now, days),
            *drift_view(changes, sources, missing, review.get("generated_pr_url")),
            *issues_view(review["issues"]),
            *adoption_view(review["adoption"], now, days),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-file",
        type=Path,
        required=True,
        help="local internal_restricted JSON artifact",
    )
    parser.add_argument(
        "--release-file",
        type=Path,
        required=True,
        help="local bcb_release_gate.py JSON artifact",
    )
    parser.add_argument(
        "--hosted-file", type=Path, help="local hosted snapshot; never fetches"
    )
    parser.add_argument(
        "--ngc-file", type=Path, help="local normalized NGC snapshot; never fetches"
    )
    parser.add_argument("--max-age-days", type=int, default=14)
    args = parser.parse_args(argv)
    if args.max_age_days < 1:
        parser.error("max age must be positive")
    try:
        review = review_input(read_json(args.review_file))
        release = read_json(args.release_file)
        sources: dict[str, str] = {}
        missing: list[str] = []
        snapshots = []
        complete_sources: set[str] = set()
        for name, path in (
            ("hosted", args.hosted_file),
            ("downloadable", args.ngc_file),
        ):
            if path is None:
                records, source, complete = None, "snapshot not supplied", False
            else:
                records, source, complete = model_registry.snapshot(path, name)
            if complete:
                complete_sources.add(name)
            snapshots.append(records)
            if records is None:
                missing.append(f"{name} snapshot unavailable")
            else:
                sources[name if complete else f"{name} (partial; no removals)"] = source
        changes = model_registry.compare(
            model_registry.load_registry(),
            *snapshots,
            complete_sources=frozenset(complete_sources),
        )
        sys.stdout.write(
            render(
                review,
                release,
                changes,
                sources,
                missing,
                datetime.now(timezone.utc),
                args.max_age_days,
            )
            + "\n"
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        # Never echo file contents, arbitrary exception messages or private paths.
        sys.stderr.write("Dashboard inputs unavailable or invalid; no view rendered.\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
