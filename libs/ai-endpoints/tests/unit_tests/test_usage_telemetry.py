from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Generator, cast
from unittest.mock import AsyncMock, Mock, patch

import pytest
import requests

from langchain_nvidia_ai_endpoints._common import (
    _NVIDIAAsyncClient,
    _NVIDIASyncClient,
)
from langchain_nvidia_ai_endpoints._telemetry import (
    _post_envelope,
    _TokenUsage,
    _UsageTelemetry,
    canonical_model_identity,
    classify_error,
    flush_usage_telemetry,
    merge_token_usage,
    operation_for_client,
    record_usage,
    token_usage,
    usage_telemetry_enabled,
)
from langchain_nvidia_ai_endpoints.chat_models import ChatNVIDIA
from langchain_nvidia_ai_endpoints.embeddings import NVIDIAEmbeddings
from langchain_nvidia_ai_endpoints.llm import NVIDIA
from langchain_nvidia_ai_endpoints.reranking import NVIDIARerank


@pytest.fixture(autouse=True)
def reset_global_telemetry() -> Generator[None, None, None]:
    from langchain_nvidia_ai_endpoints import _telemetry

    _telemetry._reset_usage_telemetry_for_tests()
    yield
    _telemetry._reset_usage_telemetry_for_tests()


def _sync_client(**overrides: Any) -> _NVIDIASyncClient:
    values: dict[str, Any] = {
        "default_hosted_model_name": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "mdl_name": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "model": None,
        "is_hosted": True,
        "cls": "ChatNVIDIA",
        "infer_path": "{base_url}/chat/completions",
        "usage_telemetry_enabled": True,
    }
    values.update(overrides)
    return _NVIDIASyncClient(**values)


def _async_client(**overrides: Any) -> _NVIDIAAsyncClient:
    values: dict[str, Any] = {
        "default_hosted_model_name": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "mdl_name": "nvidia/nemotron-3.5-lightning-30b-a3b",
        "model": None,
        "is_hosted": True,
        "cls": "ChatNVIDIA",
        "infer_path": "{base_url}/chat/completions",
        "usage_telemetry_enabled": True,
    }
    values.update(overrides)
    return _NVIDIAAsyncClient(**values)


def _response(status: int = 200, payload: dict | None = None) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response._content = json.dumps(payload or {}).encode("utf-8")
    return response


def test_usage_telemetry_is_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NVIDIA_USAGE_TELEMETRY_ENABLED", raising=False)
    assert usage_telemetry_enabled() is False
    assert usage_telemetry_enabled(False) is False
    assert usage_telemetry_enabled(True) is True

    monkeypatch.setenv("NVIDIA_USAGE_TELEMETRY_ENABLED", "true")
    assert usage_telemetry_enabled() is True


def test_canonical_model_identity_uses_versioned_static_allowlist() -> None:
    assert canonical_model_identity("nvidia/nemotron-3.5-lightning-30b-a3b") == (
        "nvidia/nemotron-3.5-lightning-30b-a3b",
        "nemotron",
    )
    assert canonical_model_identity("ai-mixtral-8x7b-instruct") == (
        "mistralai/mixtral-8x7b-instruct-v0.1",
        "mixtral",
    )
    assert canonical_model_identity("private/raw-model-name") == ("unknown", "unknown")


def test_operation_mapping_is_closed() -> None:
    assert operation_for_client("ChatNVIDIA") == "chat"
    assert operation_for_client("ChatNVIDIACustom") == "chat"
    assert operation_for_client("NVIDIA") == "completion"
    assert operation_for_client("NVIDIAEmbeddings") == "embedding"
    assert operation_for_client("NVIDIARerank") == "rerank"
    assert operation_for_client("UnrecognizedClient") == "other"


def test_token_usage_reads_only_known_aggregate_fields() -> None:
    usage = token_usage(
        {
            "choices": [{"message": {"content": "private response"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 7},
        }
    )
    assert usage == _TokenUsage(input_tokens=12, output_tokens=7, found=True)
    assert merge_token_usage(_TokenUsage(), usage) == usage
    assert token_usage({"usage": {}}) == _TokenUsage()


def test_error_classification_uses_status_and_exception_type() -> None:
    assert (
        classify_error(SimpleNamespace(status_code=401), Exception("secret"))
        == "authentication"
    )
    assert (
        classify_error(SimpleNamespace(status_code=403), Exception("secret"))
        == "authorization"
    )
    assert (
        classify_error(SimpleNamespace(status_code=429), Exception("secret"))
        == "rate_limited"
    )
    assert (
        classify_error(SimpleNamespace(status_code=503), Exception("secret"))
        == "provider_5xx"
    )
    assert classify_error(None, TimeoutError("secret")) == "timeout"
    assert classify_error(None, requests.ConnectionError("secret")) == "network"
    assert classify_error(None, ValueError("private prompt")) == "client_validation"


def test_aggregate_flush_matches_schema_and_has_no_identity_values() -> None:
    captured: list[dict] = []
    fixed_now = datetime(2026, 8, 28, 14, 30, tzinfo=timezone.utc)
    state = _UsageTelemetry(
        endpoint="https://events.example.test/v1.1/events/json",
        post=lambda endpoint, payload: captured.append(payload),
        now=lambda: fixed_now,
        start_worker=False,
    )
    state.record(
        client_name="ChatNVIDIA",
        model_name="nvidia/nemotron-3.5-lightning-30b-a3b",
        success=True,
        usage=_TokenUsage(100, 50, True),
    )
    state.record(
        client_name="ChatNVIDIA",
        model_name="nvidia/nemotron-3.5-lightning-30b-a3b",
        success=True,
        usage=_TokenUsage(20, 10, True),
    )
    state.record(
        client_name="ChatNVIDIA",
        model_name="private/model",
        success=False,
        usage=_TokenUsage(),
        error_category="authorization",
    )

    assert state.flush(include_current=True) == 2
    assert len(captured) == 1
    envelope = captured[0]
    assert envelope["clientId"] == "623344073512268"
    assert envelope["eventSchemaVer"] == "0.1"
    for field in ("userId", "externalUserId", "idpId", "sessionId", "deviceId"):
        assert envelope[field] == "undefined"

    events = envelope["events"]
    assert {event["name"] for event in events} == {"nim_client_usage"}
    assert len({event["parameters"]["batchId"] for event in events}) == 1
    success = next(
        event["parameters"]
        for event in events
        if event["parameters"]["errorCategory"] == "none"
    )
    assert success["requestCount"] == 2
    assert success["successCount"] == 2
    assert success["failureCount"] == 0
    assert success["inputTokenSum"] == 120
    assert success["outputTokenSum"] == 60
    failure = next(
        event["parameters"]
        for event in events
        if event["parameters"]["errorCategory"] == "authorization"
    )
    assert failure["nimId"] == "unknown"
    assert failure["modelFamily"] == "unknown"
    assert failure["successCount"] == 0
    assert failure["failureCount"] == 1
    assert failure["missingTokenCount"] == 1

    schema = json.loads(
        (
            Path(__file__).parents[1] / "data" / "nim_client_usage_schema.json"
        ).read_text()
    )
    required = set(schema["definitions"]["events"]["nim_client_usage"]["required"])
    for event in events:
        assert set(event["parameters"]) == required
    serialized = json.dumps(envelope)
    for forbidden in (
        "private prompt",
        "private response",
        "endpoint_url",
        "hostname",
        "exception",
        "credential",
    ):
        assert forbidden not in serialized


def test_aggregate_bucket_count_is_bounded() -> None:
    captured: list[dict] = []
    state = _UsageTelemetry(
        endpoint="https://events.example.test/v1.1/events/json",
        post=lambda endpoint, payload: captured.append(payload),
        now=lambda: datetime(2026, 8, 28, 14, 30, tzinfo=timezone.utc),
        start_worker=False,
        max_buckets=1,
    )
    state.record(
        client_name="ChatNVIDIA",
        model_name="nvidia/nemotron-3.5-lightning-30b-a3b",
        success=True,
        usage=_TokenUsage(found=False),
    )
    state.record(
        client_name="NVIDIAEmbeddings",
        model_name="nvidia/nv-embedqa-e5-v5",
        success=True,
        usage=_TokenUsage(found=False),
    )

    assert state.flush(include_current=True) == 1
    assert len(captured[0]["events"]) == 1


def test_disabled_and_self_hosted_records_do_not_create_global_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langchain_nvidia_ai_endpoints import _telemetry

    record_usage(
        enabled=False,
        is_hosted=True,
        client_name="ChatNVIDIA",
        model_name="nvidia/nemotron-3.5-lightning-30b-a3b",
        success=True,
        usage=_TokenUsage(found=False),
    )
    record_usage(
        enabled=True,
        is_hosted=False,
        client_name="ChatNVIDIA",
        model_name="nvidia/nemotron-3.5-lightning-30b-a3b",
        success=True,
        usage=_TokenUsage(found=False),
    )
    monkeypatch.setenv(
        "NVIDIA_USAGE_TELEMETRY_ENDPOINT", "https://unapproved.example.test/events"
    )
    record_usage(
        enabled=True,
        is_hosted=True,
        client_name="ChatNVIDIA",
        model_name="nvidia/nemotron-3.5-lightning-30b-a3b",
        success=True,
        usage=_TokenUsage(found=False),
    )
    assert _telemetry._STATE is None
    assert flush_usage_telemetry() == 0


def test_sync_request_records_success_and_failure_without_content() -> None:
    client = _sync_client()
    success_response = _response(
        payload={
            "choices": [{"message": {"content": "private response"}}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 4},
        }
    )
    with (
        patch.object(client, "_post", return_value=(success_response, Mock())),
        patch.object(client, "_wait", return_value=success_response),
        patch("langchain_nvidia_ai_endpoints._common.record_usage") as record,
    ):
        assert (
            client.get_req({"messages": [{"content": "private prompt"}]})
            is success_response
        )
    assert record.call_count == 1
    kwargs = record.call_args.kwargs
    assert kwargs["success"] is True
    assert kwargs["usage"] == _TokenUsage(9, 4, True)
    assert "private" not in repr(kwargs)

    failure_response = _response(status=429)
    client.last_response = failure_response
    with (
        patch.object(client, "_post", side_effect=RuntimeError("private exception")),
        patch("langchain_nvidia_ai_endpoints._common.record_usage") as record,
        pytest.raises(RuntimeError, match="private exception"),
    ):
        client.get_req({"messages": [{"content": "private prompt"}]})
    kwargs = record.call_args.kwargs
    assert kwargs["success"] is False
    assert kwargs["error_category"] == "rate_limited"
    assert "private" not in repr(kwargs)


def test_telemetry_failures_do_not_change_user_request_result() -> None:
    client = _sync_client()
    response = _response(payload={"usage": {"prompt_tokens": 1}})
    with (
        patch.object(client, "_post", return_value=(response, Mock())),
        patch.object(client, "_wait", return_value=response),
        patch(
            "langchain_nvidia_ai_endpoints._common.record_usage",
            side_effect=RuntimeError("telemetry failed"),
        ),
    ):
        assert client.get_req({"messages": []}) is response

    with (
        patch.object(client, "_post", side_effect=ValueError("request failed")),
        patch(
            "langchain_nvidia_ai_endpoints._common.record_usage",
            side_effect=RuntimeError("telemetry failed"),
        ),
        pytest.raises(ValueError, match="request failed"),
    ):
        client.get_req({"messages": []})


def test_sync_stream_records_usage_and_cancellation() -> None:
    client = _sync_client()
    response = SimpleNamespace(
        status_code=200,
        raise_for_status=lambda: None,
        iter_lines=lambda: iter([b"data: chunk", b"data: [DONE]"]),
    )
    session = Mock()
    session.post.return_value = response
    client.get_session_fn = lambda: session
    with (
        patch.object(
            _NVIDIASyncClient,
            "postprocess",
            return_value=(
                {"usage": {"prompt_tokens": 5, "completion_tokens": 3}},
                False,
            ),
        ),
        patch("langchain_nvidia_ai_endpoints._common.record_usage") as record,
    ):
        assert list(client.get_req_stream({"messages": []}))
    assert record.call_args.kwargs["success"] is True
    assert record.call_args.kwargs["usage"] == _TokenUsage(5, 3, True)

    response = SimpleNamespace(
        status_code=200,
        raise_for_status=lambda: None,
        iter_lines=lambda: iter([b"data: first", b"data: second"]),
    )
    session.post.return_value = response
    with (
        patch.object(_NVIDIASyncClient, "postprocess", return_value=({}, False)),
        patch("langchain_nvidia_ai_endpoints._common.record_usage") as record,
    ):
        stream = cast(
            Generator[dict, Any, Any],
            client.get_req_stream({"messages": []}),
        )
        next(stream)
        stream.close()
    assert record.call_args.kwargs["success"] is False
    assert record.call_args.kwargs["error_category"] == "cancelled"


@pytest.mark.asyncio
async def test_async_request_records_success() -> None:
    client = _async_client()
    response = SimpleNamespace(
        status=200,
        text=AsyncMock(
            return_value=json.dumps(
                {"usage": {"prompt_tokens": 6, "completion_tokens": 2}}
            )
        ),
    )
    session = SimpleNamespace(close=AsyncMock())
    with (
        patch.object(
            client, "_post_async", AsyncMock(return_value=(response, session))
        ),
        patch.object(client, "_wait_async", AsyncMock(return_value=response)),
        patch("langchain_nvidia_ai_endpoints._common.record_usage") as record,
    ):
        await client.aget_req({"messages": []})
    assert record.call_args.kwargs["success"] is True
    assert record.call_args.kwargs["usage"] == _TokenUsage(6, 2, True)
    session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_stream_records_usage_and_cancellation() -> None:
    class Reader:
        def __init__(self, lines: list[bytes]) -> None:
            self.lines = iter(lines)

        async def readline(self) -> bytes:
            return next(self.lines, b"")

    client = _async_client()
    response = SimpleNamespace(
        status=200,
        content=Reader([b"data: chunk", b"data: [DONE]"]),
    )
    session = SimpleNamespace(post=AsyncMock(return_value=response), close=AsyncMock())
    client.get_async_session_fn = lambda: session
    with (
        patch.object(
            _NVIDIAAsyncClient,
            "postprocess",
            return_value=(
                {"usage": {"prompt_tokens": 8, "completion_tokens": 3}},
                False,
            ),
        ),
        patch("langchain_nvidia_ai_endpoints._common.record_usage") as record,
    ):
        assert [item async for item in client.aget_req_stream({"messages": []})]
    assert record.call_args.kwargs["success"] is True
    assert record.call_args.kwargs["usage"] == _TokenUsage(8, 3, True)

    response = SimpleNamespace(
        status=200,
        content=Reader([b"data: first", b"data: second"]),
    )
    session.post = AsyncMock(return_value=response)
    with (
        patch.object(_NVIDIAAsyncClient, "postprocess", return_value=({}, False)),
        patch("langchain_nvidia_ai_endpoints._common.record_usage") as record,
    ):
        stream = cast(
            AsyncGenerator[dict, None],
            client.aget_req_stream({"messages": []}),
        )
        await anext(stream)
        await stream.aclose()
    assert record.call_args.kwargs["success"] is False
    assert record.call_args.kwargs["error_category"] == "cancelled"


def test_transport_failures_never_escape() -> None:
    with patch(
        "langchain_nvidia_ai_endpoints._telemetry.requests.post",
        side_effect=requests.ConnectionError("private network detail"),
    ):
        _post_envelope(
            "https://events.example.test/v1.1/events/json",
            {"events": []},
        )


@pytest.mark.parametrize(
    ("client", "expected"),
    [
        (
            ChatNVIDIA(
                model="nvidia/nemotron-3.5-lightning-30b-a3b",
                api_key="test",
                usage_telemetry_enabled=True,
            ),
            True,
        ),
        (
            NVIDIAEmbeddings(api_key="test", usage_telemetry_enabled=True),
            True,
        ),
        (
            NVIDIA(api_key="test", usage_telemetry_enabled=True),
            True,
        ),
        (
            NVIDIARerank(api_key="test", usage_telemetry_enabled=True),
            True,
        ),
        (
            ChatNVIDIA(
                model="nvidia/nemotron-3.5-lightning-30b-a3b",
                api_key="test",
                usage_telemetry_enabled=False,
            ),
            False,
        ),
    ],
)
def test_public_clients_propagate_explicit_opt_in(client: Any, expected: bool) -> None:
    assert client._client.usage_telemetry_enabled is expected
    assert client._async_client.usage_telemetry_enabled is expected
