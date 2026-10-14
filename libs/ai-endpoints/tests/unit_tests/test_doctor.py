"""Deterministic network-only Doctor checks; no inference or telemetry."""

import json
import re
from pathlib import Path

import pytest
import requests
from requests_mock import Mocker

from langchain_nvidia_ai_endpoints._doctor import _SELF_HOST_NIMS, main

_DOCS = Path(__file__).resolve().parents[2] / "docs"


def test_hosted_success_checks_model_and_capability_without_inference(
    monkeypatch: pytest.MonkeyPatch,
    requests_mock: Mocker,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("NVIDIA_BASE_URL", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "private-value")
    monkeypatch.delenv("NIM_OSS_MANAGER_TOKEN", raising=False)
    requests_mock.get(
        "https://integrate.api.nvidia.com/v1/models",
        json={"data": [{"id": "sample-chat", "model_type": "chat"}]},
    )
    assert main(["--model", "sample-chat", "--capability", "chat"]) == 0
    text = capsys.readouterr().out
    assert "preflight passed" in text
    assert "private-value" not in text
    assert len(requests_mock.request_history) == 1
    assert requests_mock.last_request is not None
    assert requests_mock.last_request.method == "GET"
    assert requests_mock.last_request.headers["Authorization"] == "Bearer private-value"


@pytest.mark.parametrize(
    "status,reason",
    [
        (401, "authentication"),
        (403, "authentication"),
        (404, "unavailable"),
    ],
)
def test_bad_key_or_models_route(
    monkeypatch: pytest.MonkeyPatch,
    requests_mock: Mocker,
    capsys: pytest.CaptureFixture[str],
    status: int,
    reason: str,
) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "super-secret")
    requests_mock.get(
        "https://integrate.api.nvidia.com/v1/models",
        status_code=status,
        text="super-secret: internal server error",
    )
    assert main(["--base-url", "https://integrate.api.nvidia.com/v1"]) == 1
    text = capsys.readouterr().out
    assert reason in text
    assert "super-secret" not in text
    assert "internal server error" not in text


def test_missing_hosted_auth(
    monkeypatch: pytest.MonkeyPatch,
    requests_mock: Mocker,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    assert main(["--base-url", "https://integrate.api.nvidia.com/v1"]) == 1
    assert "NVIDIA_API_KEY is missing" in capsys.readouterr().out
    assert not requests_mock.called


def test_local_models_and_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    requests_mock: Mocker,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    requests_mock.get(
        "http://localhost:8000/v1/models",
        json={"data": [{"id": "local-embed", "model_type": "embedding"}]},
    )
    assert (
        main(
            [
                "--base-url",
                "http://localhost:8000",
                "--model",
                "local-embed",
                "--capability",
                "embeddings",
            ]
        )
        == 0
    )
    assert "self-hosted" in capsys.readouterr().out
    assert (
        main(
            [
                "--base-url",
                "http://localhost:8000/v1",
                "--model",
                "local-embed",
                "--capability",
                "chat",
            ]
        )
        == 1
    )
    assert "does not support chat" in capsys.readouterr().out
    assert main(["--base-url", "http://localhost:8000/v1", "--model", "absent"]) == 1
    assert "absent from /models" in capsys.readouterr().out
    assert all(req.method == "GET" for req in requests_mock.request_history)


def test_timeout_and_invalid_url(
    monkeypatch: pytest.MonkeyPatch,
    requests_mock: Mocker,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    requests_mock.get("http://localhost:8000/v1/models", exc=requests.Timeout)
    assert main(["--base-url", "http://localhost:8000/v1", "--timeout", "0.5"]) == 1
    assert "0.5s" in capsys.readouterr().out
    assert main(["--base-url", "http://user:secret@localhost:8000/v1/models"]) == 1
    assert "secret" not in capsys.readouterr().out


def test_evidence_is_opt_in_and_requires_fresh_verified_contract(
    monkeypatch: pytest.MonkeyPatch,
    requests_mock: Mocker,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("NIM_OSS_MANAGER_TOKEN", "manager-secret")
    requests_mock.get("http://localhost:8000/v1/models", json={"data": [{"id": "m"}]})
    manager = "https://manager.example/api/bcb/public/v1/compatible"
    requests_mock.get(
        manager,
        json={
            "ok": True,
            "contract_version": "bcb-public-v1alpha1",
            "status": "compatible",
            "compatible": True,
            "freshness": {"state": "known", "stale": False},
        },
    )
    args = ["--base-url", "http://localhost:8000", "--model", "m"]
    assert main(args) == 0
    assert len(requests_mock.request_history) == 1
    capsys.readouterr()
    assert main(args + ["--manager-url", "https://manager.example"]) == 0
    text = capsys.readouterr().out
    assert "EVIDENCE compatible" in text
    assert "manager-secret" not in text
    assert requests_mock.last_request is not None
    assert requests_mock.last_request.qs == {
        "framework": ["langchain-nvidia"],
        "model": ["m"],
    }
    requests_mock.get(
        manager,
        json={
            "ok": True,
            "contract_version": "bcb-public-v1alpha1",
            "status": "compatible",
            "compatible": True,
            "freshness": {"state": "known", "stale": True},
        },
    )
    assert main(args + ["--manager-url", "https://manager.example"]) == 0
    assert "EVIDENCE stale/unknown" in capsys.readouterr().out
    requests_mock.get(manager, status_code=401, text="manager-secret")
    assert main(args + ["--manager-url", "https://manager.example"]) == 0
    text = capsys.readouterr().out
    assert "EVIDENCE unavailable" in text
    assert "manager-secret" not in text
    requests_mock.get(
        manager,
        json={
            "ok": True,
            "contract_version": "bcb-public-v1alpha1",
            "status": "current",
            "compatible": False,
            "confidence": "needs_review",
            "freshness": {"state": "known", "stale": False},
            "affected_checks": [{"status": "warning"}],
        },
    )
    assert main(args + ["--manager-url", "https://manager.example"]) == 0
    text = capsys.readouterr().out
    assert "EVIDENCE not confirmed" in text
    assert "EVIDENCE not compatible" not in text


def test_unknown_capability_and_invalid_models_payload(
    monkeypatch: pytest.MonkeyPatch,
    requests_mock: Mocker,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    url = "http://localhost:8000/v1/models"
    requests_mock.get(url, json={"data": [{"id": "m", "type": "model"}]})
    args = [
        "--base-url",
        "http://localhost:8000",
        "--model",
        "m",
        "--capability",
        "ranking",
    ]
    assert main(args) == 0
    assert "UNKNOWN ranking capability" in capsys.readouterr().out
    requests_mock.get(url, json={"data": [{"object": "model"}]})
    assert main(args) == 1
    assert "invalid model list" in capsys.readouterr().out


def test_manager_without_token_does_not_request_evidence(
    monkeypatch: pytest.MonkeyPatch,
    requests_mock: Mocker,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NIM_OSS_MANAGER_TOKEN", raising=False)
    requests_mock.get("http://localhost:8000/v1/models", json={"data": [{"id": "m"}]})
    assert (
        main(
            [
                "--base-url",
                "http://localhost:8000",
                "--model",
                "m",
                "--manager-url",
                "https://manager.example",
            ]
        )
        == 0
    )
    assert "EVIDENCE unavailable" in capsys.readouterr().out
    assert len(requests_mock.request_history) == 1


@pytest.mark.parametrize(
    "model,container,port",
    [
        ("nvidia/nemotron-3-super-120b-a12b", "nemotron-3-super-120b-a12b", 8000),
        ("nvidia/llama-nemotron-embed-1b-v2", "llama-nemotron-embed-1b-v2", 8080),
    ],
)
def test_hosted_success_prints_self_host_hint(
    monkeypatch: pytest.MonkeyPatch,
    requests_mock: Mocker,
    capsys: pytest.CaptureFixture[str],
    model: str,
    container: str,
    port: int,
) -> None:
    monkeypatch.delenv("NVIDIA_BASE_URL", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "private-value")
    requests_mock.get(
        "https://integrate.api.nvidia.com/v1/models", json={"data": [{"id": model}]}
    )
    assert main(["--model", model]) == 0
    text = capsys.readouterr().out
    assert text.index("RESULT preflight passed") < text.index("HINT Self-host")
    assert f"nvcr.io/nim/nvidia/{container}" in text
    assert f"catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/{container}" in (
        text
    )
    assert f'base_url="http://localhost:{port}/v1"' in text
    assert "docs/self_hosting.md" in text
    assert "private-value" not in text
    assert len(requests_mock.request_history) == 1


def test_hosted_unknown_model_gets_generic_self_host_hint(
    monkeypatch: pytest.MonkeyPatch,
    requests_mock: Mocker,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("NVIDIA_BASE_URL", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "private-value")
    requests_mock.get(
        "https://integrate.api.nvidia.com/v1/models", json={"data": [{"id": "x/y"}]}
    )
    assert main(["--model", "x/y"]) == 0
    hints = [line for line in capsys.readouterr().out.splitlines() if "HINT" in line]
    assert len(hints) == 1
    assert "docs/self_hosting.md" in hints[0]
    assert "nvcr.io" not in hints[0]


def test_no_self_host_hint_for_failures_or_self_hosted(
    monkeypatch: pytest.MonkeyPatch,
    requests_mock: Mocker,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("NVIDIA_BASE_URL", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "private-value")
    model = "nvidia/nemotron-3-super-120b-a12b"
    requests_mock.get(
        "https://integrate.api.nvidia.com/v1/models", json={"data": [{"id": "other"}]}
    )
    assert main(["--model", model]) == 1
    assert "HINT" not in capsys.readouterr().out
    requests_mock.get("http://localhost:8000/v1/models", json={"data": [{"id": model}]})
    assert main(["--base-url", "http://localhost:8000", "--model", model]) == 0
    assert "HINT" not in capsys.readouterr().out


def test_self_host_mapping_matches_registry_and_compose() -> None:
    registry = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "langchain_nvidia_ai_endpoints"
            / "data"
            / "model_registry.json"
        ).read_text()
    )
    current = {
        entry["model"]["id"]
        for entry in registry["models"]
        if not entry["model"].get("deprecated")
    }
    assert set(_SELF_HOST_NIMS) <= current
    compose = (_DOCS / "self_hosting" / "docker-compose.yml").read_text()
    guide = (_DOCS / "self_hosting.md").read_text()
    published = {int(port) for port in re.findall(r'"(\d+):8000"', compose)}
    for model, (container, port) in _SELF_HOST_NIMS.items():
        assert port in published
        assert f"nvcr.io/nim/nvidia/{container}:" in guide
        assert model in guide
