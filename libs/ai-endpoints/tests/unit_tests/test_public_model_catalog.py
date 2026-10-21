"""Offline request/identity checks for the reviewed public catalog additions."""

import json
import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from requests_mock import Mocker

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank
from langchain_nvidia_ai_endpoints import _telemetry as telemetry
from langchain_nvidia_ai_endpoints._statics import MODEL_TABLE

DATA = Path(__file__).resolve().parents[1] / "data"
REVIEWED = json.loads((DATA / "public_model_catalog_20261021.json").read_text())


@pytest.mark.parametrize("entry", REVIEWED["models"], ids=lambda row: row["id"])
def test_documented_request_preserves_model_identity(
    entry: dict[str, Any], requests_mock: Mocker
) -> None:
    """Exercise public clients and telemetry with mocked endpoint responses."""
    captured: list[dict[str, Any]] = []
    collector = telemetry._UsageTelemetry(
        endpoint="https://example.invalid/offline-only",
        post=lambda endpoint, payload: captured.append(payload),
        now=lambda: datetime(2026, 10, 21, 12, tzinfo=timezone.utc),
        start_worker=False,
    )
    identifier = entry["id"]
    classes = {
        "ChatNVIDIA": ChatNVIDIA,
        "NVIDIAEmbeddings": NVIDIAEmbeddings,
        "NVIDIARerank": NVIDIARerank,
    }
    expected_path = {
        "ChatNVIDIA": "https://integrate.api.nvidia.com/v1/chat/completions",
        "NVIDIAEmbeddings": "https://integrate.api.nvidia.com/v1/embeddings",
        "NVIDIARerank": entry["endpoint"],
    }[entry["client"]]
    requests_mock.post(
        expected_path,
        json={
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "data": [{"index": 0, "embedding": [0.1, 0.2]}],
            "rankings": [{"index": 0, "logit": 1.0}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )
    with (
        warnings.catch_warnings(),
        patch.object(telemetry, "_get_state", return_value=collector),
    ):
        warnings.simplefilter("ignore", UserWarning)
        client = classes[entry["client"]](
            model=identifier, api_key="offline-test", usage_telemetry_enabled=True
        )
        if isinstance(client, ChatNVIDIA):
            assert client.invoke("synthetic input").content == "ok"
        elif isinstance(client, NVIDIAEmbeddings):
            assert client.embed_query("synthetic input") == [0.1, 0.2]
        else:
            result = client.compress_documents(
                [Document(page_content="synthetic")], "query"
            )
            assert len(result) == 1
        assert requests_mock.last_request is not None
        assert requests_mock.last_request.json()["model"] == identifier
        assert collector.flush(include_current=True) == 1
    collector.close()
    parameters = captured[0]["events"][0]["parameters"]
    assert parameters["nimId"] == identifier
    assert parameters["requestCount"] == parameters["successCount"] == 1
    assert parameters["failureCount"] == 0


def test_reviewed_ids_fit_existing_telemetry_model_contract() -> None:
    schema = json.loads((DATA / "nim_client_usage_schema.json").read_text())
    properties = schema["definitions"]["events"]["nim_client_usage"]["properties"]
    rule = properties["nimId"]
    family_enum = schema["definitions"]["types"]["NimModelFamilyEnum"]["enum"]
    for entry in REVIEWED["models"]:
        identifier = entry["id"]
        assert len(identifier) <= rule["maxLength"]
        assert re.fullmatch(rule["pattern"], identifier)
        canonical, family = telemetry.canonical_model_identity(identifier)
        assert canonical == identifier
        assert family in family_enum


def test_model_id_and_family_recognition_are_distinct() -> None:
    for identifier in ("openai/gpt-oss-120b", "nvidia/nv-embed-v1"):
        assert telemetry.canonical_model_identity(identifier) == (identifier, "unknown")
    assert MODEL_TABLE["nvidia/nv-embed-v1"].model_type == "embedding"
    assert telemetry.operation_for_client("NVIDIAEmbeddings") == "embedding"
    assert telemetry.canonical_model_identity("private/not-reviewed") == (
        "unknown",
        "unknown",
    )
