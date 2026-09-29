#!/usr/bin/env python3
"""Nonblocking BCB release evidence report from Manager public-alpha read routes."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

API_PATH = "/api/bcb/public/v1"
CONTRACT = "bcb-public-v1alpha1"
CAPABILITY = re.compile(r"[a-z][a-z0-9_]*\Z")
EVIDENCE_HOST = "gitlab-master.nvidia.com"
LEVEL = {"pass": 0, "warn": 1, "block": 2}


def text(value: object) -> str:
    return value if isinstance(value, str) else ""


def obj(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def number(value: object) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else None
    )


def safe_link(value: object) -> str:
    candidate = text(value)
    try:
        parsed = urlsplit(candidate)
        if (
            parsed.scheme == "https"
            and parsed.hostname == EVIDENCE_HOST
            and not parsed.username
            and not parsed.password
            and not parsed.query
            and not parsed.fragment
            and not parsed.port
            and "\\" not in candidate
            and not any(c.isspace() for c in candidate)
        ):
            return candidate
    except ValueError:
        pass
    return ""


def safe_label(value: object) -> str:
    """Constrain strings retained in the report; never copy arbitrary API fields."""
    candidate = text(value)
    return (
        candidate
        if len(candidate) <= 256 and all(c.isprintable() for c in candidate)
        else ""
    )


def safe_evidence(value: object) -> dict:
    record = obj(value)
    return {
        "result_id": number(record.get("result_id")),
        "status": safe_label(record.get("status")),
        "completed_at": safe_label(record.get("completed_at")),
        **{
            key: safe_link(record.get(key))
            for key in (
                "pipeline_url",
                "job_url",
                "artifact_url",
                "detailed_report_url",
            )
        },
    }


def capabilities(value: str) -> list[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if (
        not result
        or len(set(result)) != len(result)
        or any(not CAPABILITY.fullmatch(item) for item in result)
    ):
        raise ValueError(
            "required capabilities must be distinct comma-separated identifiers"
        )
    return result


def target_config(kind: str, identifier: str, required: str) -> dict | None:
    if not identifier and not required:
        return None
    if not identifier or not required:
        raise ValueError(
            f"{kind} needs both a NIM ID and explicit required capabilities"
        )
    if not identifier.isdecimal() or int(identifier) < 1:
        raise ValueError(f"{kind} NIM ID must be a positive integer")
    return {
        "deployment": kind,
        "nim_id": int(identifier),
        "required_capabilities": capabilities(required),
    }


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(
        self,
        request: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def fetch(base_url: str, token: str, endpoint: str, query: dict[str, str]) -> dict:
    url = f"{base_url}{API_PATH}/{endpoint}"
    if query:
        url += f"?{urlencode(query)}"
    try:
        request = Request(
            url,
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        )
        # urllib's default redirect handler forwards Authorization; reject redirects
        # before any request to a second origin can be attempted.
        with build_opener(NoRedirect()).open(request, timeout=15) as response:
            payload = json.load(response)
    except HTTPError as exc:
        # Manager uses 404 for unknown baselines and absent targets; retain only
        # its projected public-alpha response, never the transport exception.
        try:
            payload = json.load(exc) if exc.code == 404 else None
        except (ValueError, OSError):
            payload = None
        if payload is None:
            raise RuntimeError(
                f"{endpoint} request unavailable (HTTP {exc.code})"
            ) from None
    except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        # Never include URL, exception body, headers, or bearer token in artifacts/logs.
        raise RuntimeError(
            f"{endpoint} request unavailable ({type(exc).__name__})"
        ) from None
    if (
        not isinstance(payload, dict)
        or payload.get("contract_version") != CONTRACT
        or payload.get("readonly") is not True
        or payload.get("endpoint") != endpoint
        or payload.get("api_scope") != "bcb-public-alpha"
        or payload.get("api_visibility") != "internal-alpha"
        or payload.get("public_api") is not False
    ):
        raise RuntimeError(
            f"{endpoint} response has an unexpected public-alpha contract"
        )
    return payload


def target_report(config: dict, payload: dict) -> dict:
    accepted = obj(payload.get("accepted_baseline"))
    latest = obj(payload.get("latest_observation"))
    evidence = obj(payload.get("evidence"))
    freshness = obj(payload.get("freshness"))
    manager_target = obj(payload.get("nim_target"))
    required = config["required_capabilities"]
    status = safe_label(payload.get("compatibility_status"))
    accepted_digest = safe_label(accepted.get("digest"))
    observed_digest = safe_label(latest.get("digest"))
    accepted_hash = safe_label(accepted.get("fingerprint_hash"))
    observed_hash = safe_label(latest.get("fingerprint_hash"))
    accepted_result = safe_evidence(evidence.get("accepted_result"))
    latest_result = safe_evidence(evidence.get("latest_result"))
    reasons = []
    if (
        payload.get("ok") is not True
        or number(manager_target.get("id")) != config["nim_id"]
    ):
        reasons.append("target_not_found_or_mismatched")
    if status in {"changed", "failed", "blocked"}:
        reasons.append(f"required_target_{status}")
    elif status != "current" or payload.get("status") != "compatible":
        reasons.append("baseline_not_current")
    if (accepted_digest and observed_digest and accepted_digest != observed_digest) or (
        accepted_hash and observed_hash and accepted_hash != observed_hash
    ):
        reasons.append("accepted_observation_changed")
    if not all((accepted_hash, observed_hash)) or (
        config["deployment"] == "downloadable"
        and not all((accepted_digest, observed_digest))
    ):
        reasons.append("fingerprint_or_digest_missing")
    if (
        freshness.get("state") != "known"
        or freshness.get("stale") is not False
        or not text(payload.get("tested_at"))
    ):
        reasons.append("freshness_unknown_or_stale")
    if (
        accepted_result["status"] not in {"passed", "current", "compatible"}
        or not accepted_result["result_id"]
    ):
        reasons.append("accepted_check_not_passing")
    if number(accepted.get("result_id")) != accepted_result["result_id"]:
        reasons.append("accepted_result_pointer_mismatch")
    if (
        not latest_result["result_id"]
        or number(latest.get("result_id")) != latest_result["result_id"]
        or latest_result["status"] not in {"passed", "current", "compatible"}
    ):
        reasons.append("latest_check_unknown")
    if latest_result["status"] in {"failed", "blocked"} or safe_label(
        latest.get("status")
    ) in {"failed", "blocked"}:
        reasons.append("latest_check_failed")
    if not required:
        reasons.append("required_capabilities_unconfigured")
    level = (
        "block"
        if required
        and number(manager_target.get("id")) == config["nim_id"]
        and any(
            reason in reasons
            for reason in (
                "required_target_changed",
                "required_target_failed",
                "required_target_blocked",
                "accepted_observation_changed",
                "latest_check_failed",
            )
        )
        else "warn"
        if reasons
        else "pass"
    )
    return {
        "deployment": config["deployment"],
        "nim_id": config["nim_id"],
        "target_name": safe_label(manager_target.get("name")),
        "model_name": safe_label(manager_target.get("model_name")),
        "required_capabilities": required,
        "status": level,
        "reasons": reasons,
        "compatibility_status": status or "unknown",
        "accepted_digest": accepted_digest,
        "observed_digest": observed_digest,
        "accepted_fingerprint": accepted_hash,
        "observed_fingerprint": observed_hash,
        "fingerprint_family": safe_label(accepted.get("fingerprint_family")),
        "tested_at": safe_label(payload.get("tested_at")),
        "freshness": {
            "state": safe_label(freshness.get("state")) or "unknown",
            "stale": freshness.get("stale") is True,
        },
        "accepted_result": accepted_result,
        "latest_result": latest_result,
    }


def gate_report(payload: dict, *, requested_version: str = "") -> dict:
    summary = obj(payload.get("summary"))
    evidence = obj(payload.get("evidence"))
    status = safe_label(payload.get("status"))
    scope = obj(payload.get("scope"))
    scoped_targets = (
        [
            {
                "id": number(item.get("id")),
                "name": safe_label(item.get("name")),
                "model_name": safe_label(item.get("model_name")),
            }
            for item in scope.get("targets", [])
            if isinstance(item, dict)
        ]
        if isinstance(scope.get("targets"), list)
        else []
    )
    projects = (
        [
            safe_label(item)
            for item in scope.get("projects", [])
            if isinstance(item, str)
        ]
        if isinstance(scope.get("projects"), list)
        else []
    )
    guards = (
        [
            obj(item)
            for item in payload.get("release_guards", [])
            if isinstance(item, dict)
        ]
        if isinstance(payload.get("release_guards"), list)
        else []
    )
    counts = {
        key: number(summary.get(key))
        for key in (
            "numerator",
            "denominator",
            "decision_ready_pairs",
            "blocked_pairs",
            "stale_pairs",
        )
    }
    reasons = []
    baseline_version = safe_label(payload.get("baseline_version"))
    if (
        payload.get("ok") is not True
        or not baseline_version
        or (requested_version and baseline_version != requested_version)
    ):
        reasons.append("baseline_unavailable")
    if (
        status == "blocked"
        or (counts["blocked_pairs"] is not None and counts["blocked_pairs"] > 0)
        or any(item.get("status") == "blocked" for item in guards)
    ):
        reasons.append("release_gate_blocked")
    elif (
        status not in {"ready", "decision_ready"}
        or payload.get("release_ready") is not True
    ):
        reasons.append("release_gate_not_ready")
    if (
        counts["denominator"] is None
        or counts["denominator"] == 0
        or counts["numerator"] is None
        or counts["numerator"] != counts["denominator"]
        or counts["blocked_pairs"] is None
    ):
        reasons.append("summary_incomplete_or_failed")
    if counts["stale_pairs"] is None or counts["stale_pairs"] > 0:
        reasons.append("stale_or_unknown_summary")
    if (
        number(evidence.get("checks_count")) in (None, 0)
        or number(evidence.get("checks_count")) != counts["denominator"]
        or not text(evidence.get("evaluated_at"))
    ):
        reasons.append("gate_evidence_missing")
    level = (
        "block" if "release_gate_blocked" in reasons else "warn" if reasons else "pass"
    )
    return {
        "status": level,
        "manager_status": status or "unknown",
        "reasons": reasons,
        "baseline_version": baseline_version,
        "summary": counts,
        "checks_count": number(evidence.get("checks_count")),
        "evaluated_at": safe_label(evidence.get("evaluated_at")),
        "scope": {"projects": projects, "targets": scoped_targets},
    }


def assemble(gate: dict, targets: list[dict], *, collected_at: str) -> dict:
    selected = {target["deployment"] for target in targets}
    reasons = (
        []
        if selected == {"hosted", "downloadable"}
        else ["hosted_and_downloadable_targets_required"]
    )
    if not any(
        project.casefold() in {"langchain", "langchain-nvidia"}
        for project in gate["scope"]["projects"]
    ):
        reasons.append("connector_not_in_gate_scope")
    for target in targets:
        if not any(
            (item["id"] is not None and item["id"] == target["nim_id"])
            or (item["name"] and item["name"] == target["target_name"])
            for item in gate["scope"]["targets"]
        ):
            reasons.append(f"{target['deployment']}_not_in_gate_scope")
    severity = max(
        (
            LEVEL[gate["status"]],
            *(LEVEL[target["status"]] for target in targets),
            LEVEL["warn"] if reasons else LEVEL["pass"],
        )
    )
    return {
        "contract_version": CONTRACT,
        "baseline_version": gate["baseline_version"],
        "collected_at": collected_at,
        "status": next(key for key, value in LEVEL.items() if value == severity),
        "reasons": reasons,
        "gate": gate,
        "targets": targets,
        "promotion": "owner_approval_required",
    }


def markdown(report: dict) -> str:
    def cell(value: object) -> str:
        value = (
            (str(value) if value is not None else "—")
            .replace("\n", " ")
            .replace("\r", " ")
        )
        for character in ("\\", "|", "<", ">", "`", "[", "]", "*", "_"):
            value = value.replace(
                character,
                "&lt;"
                if character == "<"
                else "&gt;"
                if character == ">"
                else "\\" + character,
            )
        return value

    gate = report["gate"]
    lines = [
        "# BCB release evidence (nonblocking)",
        "",
        (
            f"**Assessment: {report['status'].upper()}** — "
            "owner approval required before promotion."
        ),
        (
            f"Baseline: {cell(report['baseline_version'])} · "
            f"Manager gate: {cell(gate['manager_status'])} · "
            f"evaluated: {cell(gate['evaluated_at'])}"
        ),
        "",
        "This reads Manager evidence; it does not publish or promote.",
        (
            "Required capabilities declare release scope, not per-capability proof: "
            "public-alpha exposes only aggregate and target-level results."
        ),
        "",
        (
            "| Deployment | NIM ID | Required capabilities | Assessment | Baseline | "
            "Digest (accepted / observed) | Fingerprint (accepted / observed) | "
            "Tested at | Freshness | Evidence | Reasons |"
        ),
        "| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for target in report["targets"]:
        pointers = []
        for label in ("accepted_result", "latest_result"):
            item = target[label]
            links = ", ".join(
                cell(item[key])
                for key in (
                    "pipeline_url",
                    "job_url",
                    "artifact_url",
                    "detailed_report_url",
                )
                if item[key]
            )
            pointers.append(
                f"{label}: #{cell(item['result_id'])}"
                + (f" ({links})" if links else "")
            )
        fields = [
            target["deployment"],
            target["nim_id"],
            ", ".join(target["required_capabilities"]),
            target["status"],
            target["compatibility_status"],
            f"{target['accepted_digest']} / {target['observed_digest']}",
            f"{target['accepted_fingerprint']} / {target['observed_fingerprint']}",
            target["tested_at"],
            f"{target['freshness']['state']}, stale={target['freshness']['stale']}",
            "; ".join(pointers),
            ", ".join(target["reasons"]) or "—",
        ]
        lines.append("| " + " | ".join(cell(field) for field in fields) + " |")
    lines.extend(
        [
            "",
            (
                f"Gate summary: `{json.dumps(gate['summary'], sort_keys=True)}`; "
                f"checks: {gate['checks_count']}."
            ),
            "Reasons: "
            + ", ".join(
                report["reasons"]
                + gate["reasons"]
                + [reason for item in report["targets"] for reason in item["reasons"]]
            )
            + ".",
            "",
            (
                "This report and CI results do not authorize publication; "
                "the release owner reviews evidence and approves promotion."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manager-url",
        default=os.getenv("BCB_MANAGER_URL", ""),
        help="HTTPS Manager origin",
    )
    parser.add_argument(
        "--baseline-version",
        default=os.getenv("BCB_BASELINE_VERSION", ""),
        help="Optional Manager release baseline override; omitted uses Manager default",
    )
    parser.add_argument("--hosted-nim-id", default=os.getenv("BCB_HOSTED_NIM_ID", ""))
    parser.add_argument(
        "--hosted-required", default=os.getenv("BCB_HOSTED_REQUIRED_CAPABILITIES", "")
    )
    parser.add_argument(
        "--downloadable-nim-id", default=os.getenv("BCB_DOWNLOADABLE_NIM_ID", "")
    )
    parser.add_argument(
        "--downloadable-required",
        default=os.getenv("BCB_DOWNLOADABLE_REQUIRED_CAPABILITIES", ""),
    )
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args(argv)
    now = datetime.now(timezone.utc).isoformat()
    try:
        configs: list[dict] = [
            config
            for config in (
                target_config("hosted", args.hosted_nim_id, args.hosted_required),
                target_config(
                    "downloadable", args.downloadable_nim_id, args.downloadable_required
                ),
            )
            if config is not None
        ]
        parsed = urlsplit(args.manager_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Manager origin must be HTTPS without credentials, path or query"
            )
        token = os.getenv("BCB_MANAGER_TOKEN", "")
        if not token:
            raise ValueError("Manager bearer token is not configured")
        base = args.manager_url.rstrip("/")
        gate = gate_report(
            fetch(
                base,
                token,
                "release-gate",
                {"baseline_version": args.baseline_version}
                if args.baseline_version
                else {},
            ),
            requested_version=args.baseline_version,
        )
        targets = [
            target_report(
                config,
                fetch(base, token, "recommend", {"nim_id": str(config["nim_id"])}),
            )
            for config in configs
        ]
        report = assemble(gate, targets, collected_at=now)
    except (ValueError, RuntimeError) as exc:
        # Only messages generated by this script are printed, never transport details.
        report = assemble(
            {
                "status": "warn",
                "manager_status": "unknown",
                "reasons": ["manager_evidence_unavailable"],
                "baseline_version": safe_label(args.baseline_version),
                "summary": {},
                "checks_count": None,
                "evaluated_at": "",
                "scope": {"projects": [], "targets": []},
            },
            [],
            collected_at=now,
        )
        report["reasons"].append("configuration_or_request_unavailable")
        sys.stderr.write(f"BCB assessment unavailable: {exc}\n")
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown_output.write_text(markdown(report), encoding="utf-8")
    sys.stdout.write(f"BCB assessment: {report['status']} (nonblocking)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
