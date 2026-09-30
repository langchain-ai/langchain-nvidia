#!/usr/bin/env python3
"""Create private, review-only BCB failure triage and maintainer drafts."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

CONTRACT = "bcb-public-v1alpha1"
INPUT_CONTRACT = "bcb-failure-triage-input-v1"
OUTPUT_CONTRACT = "bcb-failure-triage-draft-v1"
LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/+:-]{0,127}\Z", re.ASCII)
LINK_PATH = re.compile(r"/[A-Za-z0-9/_\-.]*\Z", re.ASCII)
CATEGORIES = {
    "authentication_failed",
    "runner_unavailable",
    "job_timeout",
    "artifact_schema_invalid",
    "result_contract_missing",
    "test_fixture_defect",
    "assertion_failure",
    "behavioral_failure",
    "unknown",
}
ISSUE_ACTIONS = {
    "sdk_regression": (
        "Connector maintainers",
        "Compare the failing connector version with the last passing version; "
        "review a targeted fix against the linked BCB results.",
    ),
    "nim_behavior_delta": (
        "Manager/BCB owner",
        "Inspect the accepted and latest fingerprint/digest and linked results "
        "in Manager; use its existing remediation workflow for a reviewed fix.",
    ),
    "infrastructure": (
        "BCB runner owner",
        "Inspect runner health and retry only after the underlying "
        "infrastructure fault is resolved.",
    ),
    "credentials": (
        "Credential owner",
        "Verify scoped credential provisioning privately; do not paste tokens "
        "into the issue or drafts.",
    ),
    "test_quality": (
        "BCB test owner",
        "Review the test fixture or result contract and rerun before making "
        "a compatibility claim.",
    ),
    "unknown": (
        "Connector and BCB owners",
        "Inspect the linked Manager result and workflow; supply a sanitized "
        "diagnostic and a comparable passing result before attributing cause.",
    ),
}


def label(value: object) -> str:
    return value if isinstance(value, str) and LABEL.fullmatch(value) else ""


def positive(value: object) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
        else None
    )


def link(value: object) -> str:
    if not isinstance(value, str) or len(value) > 512 or "\\" in value:
        return ""
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme == "https"
            and parsed.netloc == "gitlab-master.nvidia.com"
            and LINK_PATH.fullmatch(parsed.path)
            and not parsed.query
            and not parsed.fragment
        ):
            return value
    except ValueError:
        pass
    return ""


def record(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def safe_result(value: object) -> dict:
    raw = record(value)
    return {
        "result_id": positive(raw.get("result_id")),
        "status": label(raw.get("status")),
        "links": [
            url
            for name in (
                "pipeline_url",
                "job_url",
                "artifact_url",
                "detailed_report_url",
            )
            if (url := link(raw.get(name)))
        ],
    }


def sanitize(data: object) -> dict:
    """Project allowlisted fields; exclude raw logs, messages, URLs and metadata."""
    wrapper = record(data)
    if wrapper.get("contract_version") == INPUT_CONTRACT:
        source = record(wrapper.get("release_report"))
        diagnostic = record(wrapper.get("failure"))
        comparison = record(wrapper.get("comparison"))
        if not source:
            raise ValueError("A sanitized release_report is required")
    else:
        source, diagnostic, comparison = wrapper, {}, {}
    if source.get("contract_version") != CONTRACT or not isinstance(
        source.get("targets"), list
    ):
        raise ValueError("Expected a bcb-public-v1alpha1 release evidence report")
    targets = []
    for value in source["targets"]:
        item = record(value)
        if item.get("deployment") not in ("hosted", "downloadable"):
            continue
        targets.append(
            {
                "nim_id": positive(item.get("nim_id")),
                "model": label(item.get("model_name")),
                "deployment": item["deployment"],
                "status": label(item.get("status")),
                "compatibility_status": label(item.get("compatibility_status")),
                "reasons": [
                    s for value in item.get("reasons", []) if (s := label(value))
                ]
                if isinstance(item.get("reasons"), list)
                else [],
                "accepted_fingerprint": label(item.get("accepted_fingerprint")),
                "observed_fingerprint": label(item.get("observed_fingerprint")),
                "accepted_digest": label(item.get("accepted_digest")),
                "observed_digest": label(item.get("observed_digest")),
                "accepted_result": safe_result(item.get("accepted_result")),
                "latest_result": safe_result(item.get("latest_result")),
            }
        )
    category = diagnostic.get("failure_category")
    failure = {
        "nim_id": positive(diagnostic.get("nim_id")),
        "result_id": positive(diagnostic.get("result_id")),
        "stage": diagnostic.get("stage")
        if diagnostic.get("stage") in ("setup", "behavioral")
        else "unknown",
        "failure_category": category
        if isinstance(category, str) and category in CATEGORIES
        else "unknown",
        "status": diagnostic.get("status")
        if diagnostic.get("status") in ("failed", "blocked", "warning")
        else "unknown",
        "evidence_url": link(diagnostic.get("evidence_url")),
    }
    prior = {
        "connector_version": label(comparison.get("prior_connector_version")),
        "result_id": positive(comparison.get("prior_result_id")),
        "status": comparison.get("prior_status")
        if comparison.get("prior_status") == "passed"
        else "unknown",
        "fingerprint": label(comparison.get("prior_fingerprint")),
        "evidence_url": link(comparison.get("prior_evidence_url")),
    }
    return {
        "connector_version": label(wrapper.get("connector_version")),
        "baseline_version": label(source.get("baseline_version")),
        "targets": targets,
        "failure": failure,
        "prior": prior,
    }


def assess(data: dict) -> dict:
    failure = data["failure"]
    selected = [t for t in data["targets"] if t["nim_id"] == failure["nim_id"]]
    target = selected[0] if len(selected) == 1 else None
    if (
        failure["result_id"]
        and target
        and target["latest_result"]["result_id"] != failure["result_id"]
    ):
        target = None  # Never attribute an unrelated Manager result to this failure.
    if not failure["result_id"] and len(data["targets"]) == 1:
        target = data["targets"][0]
    evidence = []
    if target:
        for kind in ("accepted_result", "latest_result"):
            result = target[kind]
            evidence.append(
                {
                    "kind": kind,
                    "result_id": result["result_id"],
                    "links": result["links"],
                }
            )
    if failure["result_id"] or failure["evidence_url"]:
        evidence.append(
            {
                "kind": "diagnostic_result",
                "result_id": failure["result_id"],
                "links": [failure["evidence_url"]] if failure["evidence_url"] else [],
            }
        )
    prior = data["prior"]
    if prior["result_id"] or prior["evidence_url"]:
        evidence.append(
            {
                "kind": "prior_connector_result",
                "result_id": prior["result_id"],
                "links": [prior["evidence_url"]] if prior["evidence_url"] else [],
            }
        )
    cause, confidence, reason = (
        "unknown",
        "insufficient",
        "No result-bound diagnostic establishes a failure cause.",
    )
    kind, stage = failure["failure_category"], failure["stage"]
    if (
        failure["status"] in ("failed", "blocked", "warning")
        and failure["result_id"]
        and target
        and target["latest_result"]["status"] in ("failed", "blocked", "warning")
    ):
        if stage == "setup" and kind == "authentication_failed":
            cause, confidence, reason = (
                "credentials",
                "medium",
                "The bound Manager result identifies a setup authentication failure.",
            )
        elif stage == "setup" and kind in ("runner_unavailable", "job_timeout"):
            cause, confidence, reason = (
                "infrastructure",
                "medium",
                "The bound Manager result identifies a runner/setup fault.",
            )
        elif kind in (
            "artifact_schema_invalid",
            "result_contract_missing",
            "test_fixture_defect",
        ):
            cause, confidence, reason = (
                "test_quality",
                "medium",
                "The bound result reports an evidence-contract or fixture defect, "
                "not a model regression.",
            )
        elif stage == "behavioral" and kind in (
            "behavioral_failure",
            "assertion_failure",
        ):
            changed = any(
                target[a] and target[b] and target[a] != target[b]
                for a, b in (
                    ("accepted_fingerprint", "observed_fingerprint"),
                    ("accepted_digest", "observed_digest"),
                )
            )
            if (
                changed
                and target["accepted_result"]["result_id"]
                and target["accepted_result"]["status"]
                in ("passed", "current", "compatible")
                and target["latest_result"]["links"]
                and target["accepted_result"]["links"]
            ):
                cause, confidence, reason = (
                    "nim_behavior_delta",
                    "high",
                    "Accepted and failing result evidence is linked, and the target "
                    "fingerprint/digest changed.",
                )
            elif (
                not changed
                and target["accepted_fingerprint"]
                and target["accepted_fingerprint"] == target["observed_fingerprint"]
                and prior["status"] == "passed"
                and prior["result_id"]
                and prior["result_id"] == target["accepted_result"]["result_id"]
                and target["accepted_result"]["status"]
                in ("passed", "current", "compatible")
                and prior["evidence_url"] in target["accepted_result"]["links"]
                and prior["fingerprint"] == target["observed_fingerprint"]
                and prior["connector_version"]
                and data["connector_version"]
                and prior["connector_version"] != data["connector_version"]
                and target["latest_result"]["links"]
            ):
                cause, confidence, reason = (
                    "sdk_regression",
                    "high",
                    "Prior connector version passed the same target fingerprint; "
                    "the current version failed with result-bound evidence.",
                )
    elif (
        target
        and target["status"] == "block"
        and "accepted_observation_changed" in target["reasons"]
    ):
        # A release-gate projection alone establishes drift, not its root cause.
        reason = (
            "Manager reports accepted-observation drift; attach a result-bound "
            "behavioral diagnostic to attribute cause."
        )
    owner, action = ISSUE_ACTIONS[cause]
    return {
        "classification": cause,
        "confidence": confidence,
        "basis": reason,
        "stage": stage,
        "owner": owner,
        "next_action": action,
        "target": {
            "nim_id": target["nim_id"],
            "model": target["model"],
            "deployment": target["deployment"],
        }
        if target
        else None,
        "evidence": evidence,
        "review_required": True,
    }


def drafts(data: dict, triage: dict) -> dict:
    model = triage["target"]["model"] if triage["target"] else ""
    version = data["connector_version"]
    display_model = model or "model not established"
    display_version = version or "connector version not supplied"
    cause = triage["classification"].replace("_", " ")
    evidence = list(
        dict.fromkeys(url for item in triage["evidence"] for url in item["links"])
    )
    reference = (
        ", ".join(evidence)
        if evidence
        else "No allowlisted evidence link supplied; inspect Manager result privately."
    )
    scope = f"{display_version} / {display_model}"
    verified = triage["classification"] != "unknown" and bool(
        model and version and evidence
    )
    return {
        "issue": {
            "title": f"[Draft] BCB {cause}: {scope}",
            "body": (
                f"Scope: {scope}. Baseline: "
                f"{data['baseline_version'] or 'not supplied'}.\n"
                f"Assessment: {cause} ({triage['confidence']}); {triage['basis']}\n"
                f"Evidence: {reference}\nOwner: {triage['owner']}.\n"
                f"Next: {triage['next_action']}\n"
                "Review Manager workflow and existing remediation "
                "before creating an issue."
            ),
        },
        "pr_patch_suggestion": {
            "title": f"[Draft suggestion] {cause}: {scope}",
            "body": (
                "Review a targeted connector fix against the prior passing and "
                "current failing results; use Manager's existing remediation workflow "
                "and require owner approval before any PR."
                if verified and triage["classification"] == "sdk_regression"
                else "Do not propose a connector patch until the failure cause "
                "and affected scope are confirmed."
            )
            + f"\nEvidence: {reference}",
        },
        "release_note_docs_freshness": {
            "title": f"[Draft review] BCB evidence for {scope}",
            "body": (
                f"No compatibility or release claim: {cause}; evidence and "
                "owner review pending. "
                "Check whether model docs or release notes need an update after "
                "a passing, accepted BCB result. "
                f"Evidence: {reference}"
            ),
        },
        "publishable": False,
        "owner_review_required": True,
    }


def markdown(report: dict) -> str:
    def safe(value: object) -> str:
        # All dynamic prose comes from constrained labels or static templates.
        return (
            str(value)
            .replace("\\", "\\\\")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("`", "\\`")
            .replace("[", "\\[")
            .replace("]", "\\]")
        )

    triage = report["triage"]
    lines = [
        "# BCB failure triage — review-only draft",
        "",
        "No AI model was called. This is deterministic, evidence-bound triage. "
        "No publishing, merging or release approval.",
        "",
        f"Classification: **{triage['classification']}** · "
        f"Confidence: **{triage['confidence']}** · Stage: **{triage['stage']}**",
        "",
        f"Basis: {triage['basis']}",
        f"Owner: {triage['owner']}",
        f"Next: {triage['next_action']}",
        "",
        "Evidence (sanitized Manager/GitLab references):",
    ]
    for item in triage["evidence"]:
        lines.append(
            f"- {item['kind']}: result #{item['result_id'] or 'unknown'}"
            + (
                f" — {', '.join(item['links'])}"
                if item["links"]
                else " (link unavailable)"
            )
        )
    if not triage["evidence"]:
        lines.append("- No result-bound evidence supplied.")
    for key, heading in (
        ("issue", "Issue draft"),
        ("pr_patch_suggestion", "PR/patch suggestion draft"),
        ("release_note_docs_freshness", "Release-note/docs-freshness draft"),
    ):
        part = report["drafts"][key]
        lines.extend(
            ["", f"## {heading}", "", safe(part["title"]), "", safe(part["body"]), ""]
        )
    lines.extend(
        [
            "**Explicit maintainer review required.** Use Manager Agent Pipeline / "
            "existing BCB remediation workflow for any state-changing action. "
            "These drafts do not authorize compatibility, PR publication, "
            "merge or release.",
            "",
        ]
    )
    return "\n".join(lines)


def private_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600
    )
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write(content)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        type=Path,
        required=True,
        help="Local sanitized JSON release report or triage-input wrapper",
    )
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        paths = [
            p.resolve() for p in (args.evidence, args.json_output, args.markdown_output)
        ]
        if len(set(paths)) != 3 or any(p.exists() for p in paths[1:]):
            raise ValueError(
                "Input and output paths must differ; outputs must not exist"
            )
        if args.evidence.stat().st_size > 1024 * 1024:
            raise ValueError("Evidence exceeds the 1 MiB limit")
        data = sanitize(json.loads(args.evidence.read_text(encoding="utf-8")))
        triage = assess(data)
        report = {
            "contract_version": OUTPUT_CONTRACT,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_contract_version": CONTRACT,
            "method": "deterministic_rules_no_model_call",
            "connector_version": data["connector_version"] or None,
            "baseline_version": data["baseline_version"] or None,
            "triage": triage,
            "drafts": drafts(data, triage),
        }
        private_write(
            args.json_output, json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        private_write(args.markdown_output, markdown(report))
    except (ValueError, OSError, UnicodeError, TypeError) as exc:
        # Never echo raw evidence, paths, parser exceptions or transport details.
        sys.stderr.write(
            f"BCB triage unavailable ({type(exc).__name__}); review local "
            "input/output permissions and evidence contract.\n"
        )
        return 2
    sys.stdout.write("BCB triage drafts written locally; maintainer review required.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
