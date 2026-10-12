# The preflight CLI reports its checks to stdout by design.
# ruff: noqa: T201
"""Read-only endpoint preflight. Never instantiate an inference client here."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any
from urllib.parse import urlparse

import requests

from langchain_nvidia_ai_endpoints._common import _NVIDIABaseClient
from langchain_nvidia_ai_endpoints._statics import determine_model

_DEFAULT_URL = "https://integrate.api.nvidia.com/v1"
_HOSTED = {"integrate.api.nvidia.com", "ai.api.nvidia.com"}
_CAPABILITIES = {
    "chat": {"chat", "vlm", "nv-vlm", "qa"},
    "embeddings": {"embedding"},
    "completions": {"completions"},
    "ranking": {"ranking", "ranking-vlm"},
}


def _models_check(
    base_url: str,
    key: str | None,
    model: str | None,
    capability: str | None,
    timeout: float,
) -> tuple[bool, list[str]]:
    headers = {"Accept": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        response = requests.get(
            f"{base_url}/models",
            headers=headers,
            timeout=timeout,
            allow_redirects=False,
        )
    except requests.Timeout:
        return False, [
            f"FAIL /models timed out after {timeout:g}s. "
            "Check connectivity or increase --timeout."
        ]
    except requests.RequestException:
        return False, [
            "FAIL Cannot connect to /models. Check host, port, TLS and network access."
        ]

    if response.status_code in (401, 403):
        return False, [
            "FAIL /models rejected authentication (401/403). "
            "Check NVIDIA_API_KEY or deployment access policy."
        ]
    if response.status_code in (404, 405):
        return False, [
            "FAIL /models unavailable. Use the deployment root or /v1 base URL; "
            "ensure the service implements GET /v1/models."
        ]
    if response.status_code != 200:
        return False, [
            f"FAIL /models returned HTTP {response.status_code}. "
            "Check deployment health and endpoint URL."
        ]
    try:
        payload = response.json()
        entries = payload["data"]
        if not isinstance(entries, list) or not all(
            isinstance(item, dict) and isinstance(item.get("id"), str)
            for item in entries
        ):
            raise ValueError("invalid model list")
    except (ValueError, TypeError, KeyError):
        return False, [
            "FAIL /models returned an invalid model list. "
            "Check that this is an OpenAI-compatible NIM endpoint."
        ]

    messages = ["OK /models reachable (read-only; no inference sent)."]
    if not model:
        return True, messages + [
            "INFO Supply --model and --capability to check a specific model."
        ]
    matches = [item for item in entries if item["id"] == model]
    if not matches:
        return False, messages + [
            "FAIL Requested model is absent from /models. Check its ID and deployment; "
            "custom hosted endpoints may need a different base URL."
        ]
    messages.append("OK Requested model appears in /models.")
    if capability:
        declared = matches[0].get("model_type")
        if not isinstance(declared, str):
            declared = matches[0].get("type")
        # Hosted registry is advisory for known catalog models only; never use it
        # to assert a self-hosted deployment's capabilities.
        if not declared and urlparse(base_url).hostname in _HOSTED:
            known = determine_model(model)
            declared = known.model_type if known else None
        if isinstance(declared, str) and any(
            declared in kinds for kinds in _CAPABILITIES.values()
        ):
            if declared not in _CAPABILITIES[capability]:
                return False, messages + [
                    f"FAIL Model does not support {capability} according to "
                    "model metadata. Choose a matching model or capability."
                ]
            messages.append(
                f"OK Model metadata indicates {capability} capability; "
                "inference was not tested."
            )
        else:
            messages.append(
                f"UNKNOWN {capability} capability: /models provides no recognized "
                "type. Verify against deployment documentation; no inference was sent."
            )
    return True, messages


def _evidence(manager_url: str, model: str, timeout: float) -> str:
    """Optional authenticated alpha API; unavailable evidence never blocks preflight."""
    try:
        parsed = urlparse(manager_url)
        hostname = parsed.hostname
    except ValueError:
        return "EVIDENCE unavailable: invalid --manager-url; use an HTTPS Manager root."
    if (
        parsed.scheme not in ("http", "https")
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.params
        or (
            parsed.scheme != "https"
            and hostname not in {"localhost", "127.0.0.1", "::1"}
        )
    ):
        return (
            "EVIDENCE unavailable: --manager-url must be HTTPS "
            "(HTTP only for localhost), without credentials or query."
        )
    token = os.getenv("NIM_OSS_MANAGER_TOKEN")
    if not token:
        return (
            "EVIDENCE unavailable: set NIM_OSS_MANAGER_TOKEN for Manager's "
            "authenticated internal-alpha API."
        )
    try:
        response = requests.get(
            f"{manager_url.rstrip('/')}/api/bcb/public/v1/compatible",
            params={"framework": "langchain-nvidia", "model": model},
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            timeout=timeout,
            allow_redirects=False,
        )
        if response.status_code != 200:
            return (
                f"EVIDENCE unavailable: Manager returned HTTP {response.status_code}; "
                "verify auth and alpha API access."
            )
        data: Any = response.json()
    except requests.Timeout:
        return f"EVIDENCE unavailable: Manager timed out after {timeout:g}s."
    except (requests.RequestException, ValueError):
        return "EVIDENCE unavailable: Manager request failed or returned invalid JSON."
    if (
        not isinstance(data, dict)
        or data.get("contract_version") != "bcb-public-v1alpha1"
        or data.get("ok") is not True
    ):
        return (
            "EVIDENCE unknown: Manager did not return a recognized successful "
            "v1alpha1 compatibility record."
        )
    freshness = data.get("freshness")
    if (
        not isinstance(freshness, dict)
        or freshness.get("stale") is not False
        or freshness.get("state") != "known"
    ):
        return (
            "EVIDENCE stale/unknown: no current observed BCB result; "
            "do not infer compatibility."
        )
    if data.get("status") == "compatible" and data.get("compatible") is True:
        return (
            "EVIDENCE compatible: Manager reports current BCB evidence for "
            "langchain-nvidia and this model (internal-alpha)."
        )
    if data.get("compatible") is False:
        return (
            "EVIDENCE not confirmed: Manager has no accepted passing BCB result "
            "for this framework/model; review linked checks before calling "
            "it incompatible."
        )
    return (
        "EVIDENCE unknown: Manager has no definitive framework/model "
        "compatibility result."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m langchain_nvidia_ai_endpoints doctor",
        description=(
            "Check NVIDIA endpoint configuration without making inference calls."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Deployment root or /v1 URL (default: NVIDIA_BASE_URL or hosted catalog).",
    )
    parser.add_argument("--model", help="Exact model ID to look up in GET /v1/models.")
    parser.add_argument(
        "--capability",
        choices=sorted(_CAPABILITIES),
        help="Check available model type metadata; never invokes inference.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Per-request timeout in seconds (default: 5).",
    )
    parser.add_argument(
        "--manager-url",
        help="Opt in to authenticated internal-alpha BCB evidence lookup "
        "(requires NIM_OSS_MANAGER_TOKEN).",
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0 or args.timeout > 120:
        parser.error("--timeout must be greater than 0 and at most 120 seconds")
    if args.capability and not args.model:
        parser.error("--capability requires --model")
    if args.manager_url and not args.model:
        parser.error("--manager-url requires --model")
    try:
        base_url = _NVIDIABaseClient._validate_base_url(
            args.base_url or os.getenv("NVIDIA_BASE_URL") or _DEFAULT_URL
        )
    except (ValueError, TypeError):
        print(
            "FAIL Invalid base URL: use an http(s) deployment root or /v1, "
            "not a /models or inference URL; do not include credentials."
        )
        return 1
    hosted = urlparse(base_url).hostname in _HOSTED
    print(
        "MODE hosted NVIDIA API Catalog"
        if hosted
        else "MODE self-hosted/custom endpoint"
    )
    key = os.getenv("NVIDIA_API_KEY")
    if hosted and not key:
        print(
            "FAIL NVIDIA_API_KEY is missing. Generate a key on build.nvidia.com "
            "and set NVIDIA_API_KEY in your environment."
        )
        return 1
    if not hosted and not key:
        print(
            "INFO No NVIDIA_API_KEY set; self-hosted NIM may not require "
            "authentication."
        )
    success, messages = _models_check(
        base_url, key, args.model, args.capability, args.timeout
    )
    for message in messages:
        print(message)
    if args.manager_url:
        print(_evidence(args.manager_url, args.model, args.timeout))
    print(
        "RESULT preflight passed (no inference executed)."
        if success
        else "RESULT preflight failed."
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
