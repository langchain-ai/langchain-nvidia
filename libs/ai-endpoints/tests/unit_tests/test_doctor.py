"""Deterministic network-only Doctor checks; no inference or telemetry."""

import pytest
import requests
from requests_mock import Mocker

from langchain_nvidia_ai_endpoints._doctor import main


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
