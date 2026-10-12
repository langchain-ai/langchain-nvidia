"""Render a review-only NIM compatibility view from Manager's authenticated alpha API.

Only Manager owns compatibility. The registry and maintainer selections supply
identity, not behavioral claims. This script does not read inference or catalog flags.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
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
CONTRACT = "bcb-public-v1alpha1"
FRAMEWORK = "langchain-nvidia"
SAFE_TEXT = re.compile(r"^[\w./:@+ -]{1,160}$", re.ASCII)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(
        self, request: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        return None


def text(value: Any) -> str:
    """Allow only short, non-URL display values; no arbitrary Manager text."""
    if not isinstance(value, str) or not SAFE_TEXT.fullmatch(value) or "://" in value:
        return "—"
    return value


def markdown(value: Any) -> str:
    return escape(str(value), quote=False).replace("|", "&#124;")


def selection(value: Any, registry: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("Selection must be a JSON array")
    chosen = []
    seen: set[int] = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("Each selection must be an object")
        model = item.get("model")
        target_id = item.get("nim_target_id")
        deployment = item.get("deployment_type")
        if (
            not isinstance(model, str)
            or model not in registry
            or not isinstance(target_id, int)
            or isinstance(target_id, bool)
            or target_id <= 0
            or target_id in seen
            or deployment not in ("hosted", "downloadable")
        ):
            raise ValueError(
                "Select unique positive target IDs, reviewed models, "
                "and hosted/downloadable type"
            )
        seen.add(target_id)
        chosen.append(
            {"model": model, "nim_target_id": target_id, "deployment_type": deployment}
        )
    return sorted(chosen, key=lambda item: (item["model"], item["deployment_type"]))


def get(base: str, token: str, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    url = f"{base}/api/bcb/public/v1/{endpoint}?{urlencode(params)}"
    request = Request(
        url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"}
    )
    with build_opener(NoRedirect).open(request, timeout=15) as response:
        if response.status != 200:
            raise ValueError("Manager response unavailable")
        data = json.load(response)
    if (
        not isinstance(data, dict)
        or data.get("contract_version") != CONTRACT
        or data.get("endpoint") != endpoint
        or data.get("api_visibility") != "internal-alpha"
        or data.get("ok") is not True
    ):
        raise ValueError("Manager alpha evidence unavailable")
    return data


def _date(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else None
    except ValueError:
        return None


def classify(
    item: dict[str, Any],
    compatible: dict[str, Any],
    recommend: dict[str, Any],
    badge: dict[str, Any],
    *,
    now: datetime,
    max_age_days: int,
) -> dict[str, Any]:
    """A verified row requires agreeing, recent, target-bound Manager projections."""
    ident = item["nim_target_id"]
    targets = [
        projection.get("nim_target") for projection in (compatible, recommend, badge)
    ]
    identity = all(
        isinstance(target, dict)
        and target.get("id") == ident
        and target.get("model_name") == item["model"]
        for target in targets
    )
    states = [
        compatible.get("status"),
        recommend.get("compatibility_status"),
        badge.get("baseline_state"),
    ]
    freshness = [
        projection.get("freshness") for projection in (compatible, recommend, badge)
    ]
    tested_at = (
        compatible.get("freshness", {}).get("observed_at")
        if isinstance(compatible.get("freshness"), dict)
        else None
    )
    observed = _date(tested_at)
    recent = (
        observed is not None and now - timedelta(days=max_age_days) <= observed <= now
    )
    fresh = all(
        isinstance(f, dict)
        and f.get("state") == "known"
        and f.get("stale") is False
        and (stamp := _date(f.get("observed_at"))) is not None
        and now - timedelta(days=max_age_days) <= stamp <= now
        for f in freshness
    )
    checks = compatible.get("framework_results")
    passing = (
        isinstance(checks, list)
        and bool(checks)
        and all(
            isinstance(check, dict)
            and check.get("framework") == FRAMEWORK
            and check.get("active") is True
            and check.get("compatible") is True
            and check.get("status")
            not in ("failed", "blocked", "stale", "changed", "incompatible")
            for check in checks
        )
    )
    baseline = recommend.get("accepted_baseline")
    latest = recommend.get("latest_observation")
    fingerprint = recommend.get("fingerprint")
    accepted = baseline.get("digest") if isinstance(baseline, dict) else None
    digest = recommend.get("recommended_digest")
    latest_agrees = (
        isinstance(latest, dict)
        and (item["deployment_type"] == "hosted" or bool(accepted))
        and isinstance(baseline, dict)
        and latest.get("digest") == accepted
        and latest.get("fingerprint_hash") == baseline.get("fingerprint_hash")
        and latest.get("status")
        not in ("changed", "failed", "blocked", "stale", "incompatible")
    )
    verified = bool(
        identity
        and recent
        and fresh
        and passing
        and states == ["compatible", "current", "current"]
        and compatible.get("compatible") is True
        and badge.get("compatible") is True
        and badge.get("status") == "compatible"
        and recommend.get("status") == "compatible"
        and isinstance(baseline, dict)
        and baseline.get("digest") == digest
        and isinstance(fingerprint, dict)
        and isinstance(targets[0], dict)
        and fingerprint.get("family") == targets[0].get("family")
        and latest_agrees
    )
    reported = [str(s) for s in states if isinstance(s, str)]
    if not identity:
        status = "unknown"
    elif any(s in ("changed", "failed", "blocked", "incompatible") for s in reported):
        status = next(
            s for s in ("changed", "failed", "blocked", "incompatible") if s in reported
        )
    elif not fresh or not recent or "stale" in reported:
        status = "stale" if observed is not None else "unknown"
    elif verified:
        status = "verified"
    else:
        status = "unknown"
    action = recommend.get("recommended_action")
    if status == "verified":
        action = "no_action"
    elif status == "stale":
        action = "refresh_baseline"
    elif status == "changed":
        action = "review_delta"
    elif status in ("failed", "blocked", "incompatible"):
        action = "inspect_failure"
    else:
        action = "review"
    # Exclude raw evidence URLs, target names, images, and freeform messages.
    return {
        "model": text(item["model"]),
        "nim_target_id": ident,
        "deployment_type": item["deployment_type"],
        "status": status,
        "digest": text(digest) if identity else "—",
        "fingerprint_family": text(fingerprint.get("family"))
        if identity and isinstance(fingerprint, dict)
        else "—",
        "tested_at": observed.isoformat().replace("+00:00", "Z")
        if observed and identity
        else "—",
        "freshness": "current"
        if verified
        else ("stale" if status == "stale" else "unknown"),
        "recommended_action": action,
        "capabilities": {key: "unknown" for key in CAPABILITIES},
        "limitations": (
            "Capability-level checks and GPU/runtime details unavailable "
            "in public-alpha projections"
        ),
        "gpu_runtime": "unavailable/redacted",
        "source_owner": (
            "NIM OSS Manager BCB (framework); connector maintainers (selection)"
        ),
    }


def unavailable(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": item["model"],
        "nim_target_id": item["nim_target_id"],
        "deployment_type": item["deployment_type"],
        "status": "unknown",
        "digest": "—",
        "fingerprint_family": "—",
        "tested_at": "—",
        "freshness": "unknown",
        "recommended_action": "review",
        "capabilities": {key: "unknown" for key in CAPABILITIES},
        "limitations": "Manager evidence unavailable; capability checks not exposed",
        "gpu_runtime": "unavailable/redacted",
        "source_owner": (
            "NIM OSS Manager BCB (framework); connector maintainers (selection)"
        ),
    }


def render(
    rows: list[dict[str, Any]],
    summary: dict[str, Any] | None,
    version: str,
    now: datetime,
    max_age_days: int,
) -> str:
    counts = {
        s: sum(row["status"] == s for row in rows)
        for s in (
            "verified",
            "unknown",
            "stale",
            "changed",
            "failed",
            "blocked",
            "incompatible",
        )
    }
    lines = [
        "# NIM compatibility evidence (review artifact)",
        "",
        (
            f"Generated: {now.isoformat().replace('+00:00', 'Z')} · "
            f"Connector package: {version} · Max evidence age: {max_age_days} days."
        ),
        "",
        (
            "**Not release approval.** Manager owns BCB evidence; connector "
            "maintainers own target selection. Review before publication."
        ),
        "",
        (
            f"Triage: {counts['verified']} framework-verified; "
            f"{counts['unknown']} unknown; {counts['stale']} stale; "
            f"{counts['changed']} changed; {counts['failed']} failed; "
            f"{counts['blocked']} blocked; {counts['incompatible']} incompatible."
        ),
    ]
    if summary is not None:
        data = summary.get("summary")
        if isinstance(data, dict):
            lines.append(
                "Manager matrix (all returned pairs, not a per-model verdict): "
                + ", ".join(
                    f"{key.replace('_', ' ')} {data[key]}"
                    for key in (
                        "total_pairs",
                        "blocked_pairs",
                        "incompatible_pairs",
                        "stale_baselines",
                        "changed_baselines",
                        "failed_checks",
                    )
                    if isinstance(data.get(key), int)
                    and not isinstance(data[key], bool)
                    and data[key] >= 0
                )
                + "."
            )
    else:
        lines.append(
            "Manager matrix summary unavailable; no matrix-wide claim is made."
        )
    lines += [
        "",
        (
            "| Model / target | Deployment (selection) | Package | Framework status | "
            "Chat | Streaming | Tools | Structured output | Reasoning | VLM | "
            "Embedding | Rerank | Digest | Fingerprint family | Tested (UTC) | "
            "Freshness | GPU / runtime | Limitations | Action | Source ownership |"
        ),
        (
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | "
            "--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
        ),
    ]
    for row in rows:
        cells = [
            f"{row['model']} (#{row['nim_target_id']})",
            row["deployment_type"],
            version,
            row["status"],
        ]
        cells.extend(row["capabilities"][key] for key in CAPABILITIES)
        cells.extend(
            row[key]
            for key in (
                "digest",
                "fingerprint_family",
                "tested_at",
                "freshness",
                "gpu_runtime",
                "limitations",
                "recommended_action",
                "source_owner",
            )
        )
        lines.append("| " + " | ".join(markdown(cell) for cell in cells) + " |")
    if not rows:
        lines.append(
            "| No targets selected; no compatibility assertions. | "
            + " | ".join("—" for _ in range(19))
            + " |"
        )
    lines += [
        "",
        (
            "**Interpretation:** Verified means a recent, agreeing Manager "
            "framework baseline and passing linked check, not proof of individual "
            "capabilities. All eight capability states remain unknown because "
            "v1alpha1 does not project capability-specific results. Unknown, "
            "missing, stale, failed, changed or mismatched evidence never becomes "
            "verified. Hosted/downloadable is a maintainer selection, not a "
            "Manager-verified fact. GPU/runtime measurements are not exposed."
        ),
        "",
        (
            "**Freshness:** Manager's known/not-stale flag and a timestamp no "
            "older than the displayed limit are required. A later refresh may "
            "demote a previous verdict; this snapshot is not live status."
        ),
        "",
        (
            "**Provenance:** Authenticated non-mutating "
            "`/api/bcb/public/v1/{compatible,recommend,badge,matrix-summary}` "
            "GETs; no raw logs, private links or telemetry. Badge JSON, if "
            "generated, needs review and must not be published automatically."
        ),
        "",
    ]
    return "\n".join(lines)


def badge_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        f"{row['model']}:{row['nim_target_id']}": {
            "schemaVersion": 1,
            "label": "BCB framework",
            "message": row["status"],
            "color": "brightgreen"
            if row["status"] == "verified"
            else "red"
            if row["status"] in ("failed", "incompatible")
            else "orange"
            if row["status"] in ("changed", "stale", "blocked")
            else "lightgrey",
        }
        for row in rows
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manager-url", required=True, help="Authenticated Manager HTTPS origin"
    )
    parser.add_argument(
        "--selection",
        type=Path,
        help="JSON array of reviewed model, nim_target_id, deployment_type selectors",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--badges", type=Path, help="Optional review-only Shields JSON")
    parser.add_argument("--max-age-days", type=int, default=30)
    args = parser.parse_args(argv)
    base = args.manager_url.rstrip("/")
    url = urlsplit(base)
    if (
        url.scheme != "https"
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path not in ("", "/")
        or args.max_age_days <= 0
    ):
        parser.error(
            "Manager must be a credential-free HTTPS origin; max age must be positive"
        )
    token = os.environ.get("NIM_OSS_MANAGER_TOKEN", "")
    if not token:
        parser.error(
            "NIM_OSS_MANAGER_TOKEN is required (never supply it on the command line)"
        )
    registry = json.loads(
        (ROOT / "langchain_nvidia_ai_endpoints/data/model_registry.json").read_text()
    )
    models = {model["id"]: model for model in registry["models"]}
    try:
        selected = selection(
            json.loads(
                args.selection.read_text()
                if args.selection
                else os.environ.get("BCB_COMPATIBILITY_SELECTION_JSON", "[]")
            ),
            models,
        )
    except (ValueError, OSError) as exc:
        parser.error(f"Invalid selection: {type(exc).__name__}")
    version_config = (ROOT / "pyproject.toml").read_text()
    version = re.search(
        r'^version = "([0-9][0-9A-Za-z.+-]*)"$', version_config, re.MULTILINE
    )
    if not version:
        parser.error("Package version missing")
    now = datetime.now(timezone.utc)
    rows = []
    for item in selected:
        try:
            params = {"nim_target_id": item["nim_target_id"]}
            compatible = get(
                base, token, "compatible", {**params, "framework": FRAMEWORK}
            )
            recommend = get(base, token, "recommend", params)
            badge = get(base, token, "badge", {**params, "framework": FRAMEWORK})
            rows.append(
                classify(
                    item,
                    compatible,
                    recommend,
                    badge,
                    now=now,
                    max_age_days=args.max_age_days,
                )
            )
        except (HTTPError, URLError, ValueError, OSError):
            rows.append(unavailable(item))
    try:
        summary = get(base, token, "matrix-summary", {})
    except (HTTPError, URLError, ValueError, OSError):
        summary = None
    args.output.write_text(
        render(rows, summary, version.group(1), now, args.max_age_days),
        encoding="utf-8",
    )
    if args.badges:
        args.badges.write_text(
            json.dumps(badge_payload(rows), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
