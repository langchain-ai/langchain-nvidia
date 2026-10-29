"""Offline routing and telemetry regressions; live evidence is separate."""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from requests_mock import Mocker

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_nvidia_ai_endpoints import _telemetry as telemetry


@pytest.mark.parametrize("operation", ["query", "documents"])
def test_qualified_embedding_request_and_usage_identity(
    operation: str, requests_mock: Mocker
) -> None:
    identifier = "nvidia/nemotron-3-embed-1b"
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
