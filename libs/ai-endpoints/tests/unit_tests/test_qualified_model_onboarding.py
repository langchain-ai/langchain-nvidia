"""Offline routing and telemetry regressions; live evidence is separate."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from requests_mock import Mocker

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_nvidia_ai_endpoints import _telemetry as telemetry
from langchain_nvidia_ai_endpoints._statics import MODEL_TABLE


@pytest.mark.parametrize("operation", ["query", "documents"])
@pytest.mark.parametrize(
    "identifier",
    ["nvidia/nemotron-3-embed-1b", "nvidia/llama-nemotron-embed-vl-1b-v2"],
)
def test_qualified_embedding_request_and_usage_identity(
    operation: str, identifier: str, requests_mock: Mocker
) -> None:
    payloads = []
    collector = telemetry._UsageTelemetry(
        endpoint="https://example.invalid/offline-only",
        post=lambda endpoint, payload: payloads.append(payload),
        now=lambda: datetime(2026, 10, 29, 12, tzinfo=timezone.utc),
        start_worker=False,
    )
    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/embeddings",
        json={
            "data": [{"index": 0, "embedding": [0.1, 0.2]}],
            "usage": {"prompt_tokens": 3},
        },
    )
    try:
        with patch.object(telemetry, "_get_state", return_value=collector):
            embedder = NVIDIAEmbeddings(model=identifier, api_key="offline-test")
            if operation == "query":
                assert embedder.embed_query("synthetic") == [0.1, 0.2]
            else:
                assert embedder.embed_documents(["synthetic"]) == [[0.1, 0.2]]
            assert requests_mock.last_request is not None
            request = requests_mock.last_request.json()
            assert request["model"] == identifier
            assert request["input_type"] == (
                "query" if operation == "query" else "passage"
            )
            assert collector.flush(include_current=True) == 1
        parameters = payloads[0]["events"][0]["parameters"]
        assert parameters["nimId"] == identifier
        assert parameters["requestCount"] == parameters["successCount"] == 1
        assert parameters["failureCount"] == 0
        assert parameters["missingTokenCount"] == 0
        assert parameters["inputTokenSum"] == 3
        assert parameters["outputTokenSum"] == 0
    finally:
        collector.close()


@pytest.mark.parametrize(
    "identifier",
    [
        "nvidia/llama-3.1-nemotron-safety-guard-8b-v3",
        "nvidia/riva-translate-4b-instruct-v2",
        "nvidia/ising-calibration-1.5-31b",
    ],
)
def test_qualified_chat_request_and_usage_identity(
    identifier: str, requests_mock: Mocker
) -> None:
    payloads = []
    collector = telemetry._UsageTelemetry(
        endpoint="https://example.invalid/offline-only",
        post=lambda endpoint, payload: payloads.append(payload),
        now=lambda: datetime(2026, 10, 29, 12, tzinfo=timezone.utc),
        start_worker=False,
    )
    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        json={
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "synthetic"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        },
    )
    try:
        with patch.object(telemetry, "_get_state", return_value=collector):
            client = ChatNVIDIA(model=identifier, api_key="offline-test")
            assert client.invoke("synthetic").content == "synthetic"
            assert requests_mock.last_request is not None
            assert requests_mock.last_request.json()["model"] == identifier
            assert collector.flush(include_current=True) == 1
        parameters = payloads[0]["events"][0]["parameters"]
        assert parameters["nimId"] == identifier
        assert parameters["requestCount"] == parameters["successCount"] == 1
        assert parameters["failureCount"] == parameters["missingTokenCount"] == 0
        assert parameters["inputTokenSum"] == 3
        assert parameters["outputTokenSum"] == 1
    finally:
        collector.close()


def test_live_qualification_matches_reviewed_registry_entries() -> None:
    """Prevent live evidence being reused after a model's defaults change."""
    root = Path(__file__).resolve().parents[2]
    evidence = json.loads(
        (root / "tests/data/qualified_hosted_models_e2e_20261029.json").read_text()
    )
    registry = {
        row["id"]: row
        for row in json.loads(
            (
                root / "langchain_nvidia_ai_endpoints/data/model_registry.json"
            ).read_text()
        )["models"]
    }
    expected_modes = {
        "chat": {"invoke", "ainvoke", "stream", "astream"},
        "embedding": {"query", "documents", "aquery", "adocuments"},
        "vlm": {
            "invoke",
            "ainvoke",
            "stream",
            "astream",
            "image_invoke",
            "image_ainvoke",
            "image_stream",
            "image_astream",
        },
    }
    assert len(evidence["models"]) == 5
    for result in evidence["models"]:
        entry = registry[result["id"]]
        assert result["qualified"]
        assert (
            hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
            == (result["registry_entry_sha256"])
        )
        assert {check["mode"] for check in result["checks"]} == expected_modes[
            result["type"]
        ]
        assert all(value == "unknown" for value in entry["capabilities"].values())
        for check in result["checks"]:
            assert check["status"] == "passed"
            assert check["http_status"] == 200
            assert check["request_model_matches"] and check["request_route_matches"]
            assert check["telemetry_matches_response"]
            assert check["token_measurements_present"]


def test_held_candidates_are_not_onboarded() -> None:
    evidence = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "data/held_hosted_models_20261029.json"
        ).read_text()
    )
    assert len(evidence["models"]) == 7
    for result in evidence["models"]:
        assert not result["qualified"]
        assert result["id"] not in MODEL_TABLE
        assert result["failed_check"]["status"] == "failed"
