"""Content-free aggregate usage telemetry with process and client opt-out."""

from __future__ import annotations

import asyncio
import atexit
import importlib.metadata
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping, Optional
from uuid import uuid4

import aiohttp
import requests

from langchain_nvidia_ai_endpoints._statics import MODEL_TABLE
from langchain_nvidia_ai_endpoints._version import __version__

_CLIENT_ID = "623344073512268"
_EVENT_SCHEMA_VERSION = "1.0"
_REGISTRY_SCHEMA_VERSION = "0.2"
_EVENT_PROTOCOL = "1.6"
_EVENT_NAME = "nim_client_usage"
_CONNECTOR = "langchain-nvidia-ai-endpoints"
_TELEMETRY_CLIENT_VERSION = "1.0"
_DEFAULT_ENDPOINT = "https://events.telemetry.data.nvidia.com/v1.1/events/json"
_UAT_ENDPOINT = "https://events.telemetry.data-uat.nvidia.com/v1.1/events/json"
_ALLOWED_ENDPOINTS = frozenset({_DEFAULT_ENDPOINT, _UAT_ENDPOINT})
_ENABLED_ENV = "NVIDIA_USAGE_TELEMETRY_ENABLED"
_ENDPOINT_ENV = "NVIDIA_USAGE_TELEMETRY_ENDPOINT"
_DISABLED_VALUES = {"0", "false", "no", "off"}
_MAX_COUNT = 2**31 - 1
_MAX_TOKEN_SUM = 2**63 - 1
_MAX_BUCKETS = 256

_OPERATION_BY_CLIENT = {
    "ChatNVIDIA": "chat",
    "NVIDIAEmbeddings": "embedding",
    "NVIDIARerank": "rerank",
    "NVIDIA": "completion",
}
_ERROR_CATEGORIES = (
    "authentication",
    "authorization",
    "rate_limited",
    "timeout",
    "network",
    "provider_4xx",
    "provider_5xx",
    "client_validation",
    "cancelled",
    "unknown",
)
_HTTP_STATUS_CLASSES = ("2xx", "4xx", "5xx", "none", "unknown")
_LATENCY_BUCKETS = (
    ("le_100_ms", 0.1),
    ("le_250_ms", 0.25),
    ("le_500_ms", 0.5),
    ("le_1000_ms", 1.0),
    ("le_2500_ms", 2.5),
    ("le_5000_ms", 5.0),
    ("le_10000_ms", 10.0),
)
_OVERFLOW_BUCKET = "gt_10000_ms"

_APPROVED_MODEL_LOOKUP: dict[str, str] = {}
for _approved_model in MODEL_TABLE.values():
    _APPROVED_MODEL_LOOKUP[_approved_model.id] = _approved_model.id
    for _alias in _approved_model.aliases or ():
        _APPROVED_MODEL_LOOKUP[_alias] = _approved_model.id


@dataclass(frozen=True)
class _UsageKey:
    window_start: str
    connector_version: str
    execution_mode: str
    framework_version: str
    operation: str
    model_family: str
    nim_id: str
    error_category: str


@dataclass
class _UsageAggregate:
    success_count: int = 0
    failure_count: int = 0
    cancelled_count: int = 0
    partial_count: int = 0
    request_count: int = 0
    attempt_count: int = 0
    retried_request_count: int = 0
    first_attempt_success_count: int = 0
    input_token_sum: int = 0
    output_token_sum: int = 0
    missing_token_count: int = 0
    missing_token_attempt_count: int = 0
    error_category_counts: dict[str, int] = field(
        default_factory=lambda: _zero_counts(_ERROR_CATEGORIES)
    )
    http_status_class_counts: dict[str, int] = field(
        default_factory=lambda: _zero_counts(_HTTP_STATUS_CLASSES)
    )
    latency_bucket_counts: dict[str, int] = field(
        default_factory=lambda: _zero_latency_buckets()
    )
    ttft_bucket_counts: dict[str, int] = field(
        default_factory=lambda: _zero_latency_buckets()
    )
    missing_ttft_count: int = 0
    client_drop_count: int = 0

    def record(
        self,
        *,
        success: bool,
        input_tokens: int,
        output_tokens: int,
        tokens_missing: bool,
        error_category: str,
        http_status_class: str,
        execution_mode: str,
        latency_seconds: float,
        ttft_seconds: Optional[float],
        partial: bool,
    ) -> None:
        self.request_count = min(_MAX_COUNT, self.request_count + 1)
        self.attempt_count = min(_MAX_COUNT, self.attempt_count + 1)
        if success:
            self.success_count = min(_MAX_COUNT, self.success_count + 1)
            self.first_attempt_success_count = min(
                _MAX_COUNT, self.first_attempt_success_count + 1
            )
        else:
            self.failure_count = min(_MAX_COUNT, self.failure_count + 1)
            if error_category == "cancelled":
                self.cancelled_count = min(_MAX_COUNT, self.cancelled_count + 1)
            if partial:
                self.partial_count = min(_MAX_COUNT, self.partial_count + 1)
            if error_category in self.error_category_counts:
                _increment_count(self.error_category_counts, error_category)
        self.input_token_sum = min(
            _MAX_TOKEN_SUM, self.input_token_sum + max(0, input_tokens)
        )
        self.output_token_sum = min(
            _MAX_TOKEN_SUM, self.output_token_sum + max(0, output_tokens)
        )
        if tokens_missing:
            self.missing_token_count = min(_MAX_COUNT, self.missing_token_count + 1)
            self.missing_token_attempt_count = min(
                _MAX_COUNT, self.missing_token_attempt_count + 1
            )
        _increment_count(self.http_status_class_counts, http_status_class)
        _increment_bucket(self.latency_bucket_counts, latency_seconds)
        if execution_mode == "streaming":
            if ttft_seconds is None:
                self.missing_ttft_count = min(_MAX_COUNT, self.missing_ttft_count + 1)
            else:
                _increment_bucket(self.ttft_bucket_counts, ttft_seconds)


@dataclass(frozen=True)
class _TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    found: bool = False


@dataclass(frozen=True)
class _TransportDelivery:
    retry_count: int = 0
    terminal_failure: bool = False


def usage_telemetry_enabled(value: Optional[bool] = None) -> bool:
    """Return an explicit constructor setting or the default-on environment setting."""

    if value is not None:
        return value
    return os.getenv(_ENABLED_ENV, "").strip().casefold() not in _DISABLED_VALUES


def operation_for_client(client_name: str) -> str:
    """Map a public client class to the approved coarse operation category."""

    for prefix, operation in _OPERATION_BY_CLIENT.items():
        if client_name.startswith(prefix):
            return operation
    return "other"


def canonical_model_identity(model_name: Optional[str]) -> tuple[str, str]:
    """Return a canonical allowlisted NIM ID and approved model-family bucket."""

    canonical = _APPROVED_MODEL_LOOKUP.get(str(model_name or ""), "unknown")
    if canonical == "unknown":
        return "unknown", "unknown"
    lowered = canonical.casefold()
    if "nemotron" in lowered:
        family = "nemotron"
    elif "mixtral" in lowered:
        family = "mixtral"
    elif "mistral" in lowered:
        family = "mistral"
    elif "llama" in lowered:
        family = "llama"
    elif "gemma" in lowered:
        family = "gemma"
    elif "qwen" in lowered:
        family = "qwen"
    else:
        family = "unknown"
    return canonical, family


def token_usage(value: Any) -> _TokenUsage:
    """Extract only aggregate token counters from a known usage object."""

    payload: Any = value
    if isinstance(value, requests.Response):
        try:
            payload = value.json()
        except (requests.JSONDecodeError, ValueError):
            return _TokenUsage()
    elif isinstance(value, str):
        try:
            import json

            payload = json.loads(value)
        except (TypeError, ValueError):
            return _TokenUsage()
    if not isinstance(payload, Mapping):
        return _TokenUsage()
    usage = payload.get("usage") or payload.get("usage_metadata")
    if not isinstance(usage, Mapping):
        return _TokenUsage()
    input_tokens = _nonnegative_int(
        usage.get("prompt_tokens")
        if "prompt_tokens" in usage
        else usage.get("input_tokens", usage.get("input_token_count"))
    )
    output_tokens = _nonnegative_int(
        usage.get("completion_tokens")
        if "completion_tokens" in usage
        else usage.get("output_tokens", usage.get("output_token_count"))
    )
    found = any(
        key in usage
        for key in (
            "prompt_tokens",
            "input_tokens",
            "input_token_count",
            "completion_tokens",
            "output_tokens",
            "output_token_count",
        )
    )
    return _TokenUsage(input_tokens, output_tokens, found)


def merge_token_usage(current: _TokenUsage, candidate: _TokenUsage) -> _TokenUsage:
    """Keep the latest cumulative usage counters observed in a stream."""

    return candidate if candidate.found else current


def classify_error(response: Any, error: BaseException) -> str:
    """Map an error to an approved coarse category without retaining its text."""

    status = getattr(response, "status_code", None)
    if status is None:
        status = getattr(response, "status", None)
    if status == 401:
        return "authentication"
    if status == 403:
        return "authorization"
    if status == 408:
        return "timeout"
    if status == 429:
        return "rate_limited"
    if isinstance(status, int) and 400 <= status < 500:
        return "provider_4xx"
    if isinstance(status, int) and status >= 500:
        return "provider_5xx"
    if isinstance(error, (asyncio.CancelledError, GeneratorExit, KeyboardInterrupt)):
        return "cancelled"
    if isinstance(
        error,
        (
            TimeoutError,
            requests.exceptions.Timeout,
            asyncio.TimeoutError,
        ),
    ):
        return "timeout"
    if isinstance(
        error,
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.RequestException,
            aiohttp.ClientError,
        ),
    ):
        return "network"
    if isinstance(error, (TypeError, ValueError)):
        return "client_validation"
    return "unknown"


def http_status_class(response: Any) -> str:
    """Map a response status to the approved coarse status-class bucket."""

    status = getattr(response, "status_code", None)
    if status is None:
        status = getattr(response, "status", None)
    if not isinstance(status, int):
        return "none"
    if 200 <= status < 300:
        return "2xx"
    if 400 <= status < 500:
        return "4xx"
    if status >= 500:
        return "5xx"
    return "unknown"


class _UsageTelemetry:
    def __init__(
        self,
        *,
        endpoint: str,
        post: Callable[[str, dict[str, Any]], Any],
        now: Callable[[], datetime],
        start_worker: bool = True,
        max_buckets: int = _MAX_BUCKETS,
    ) -> None:
        self._endpoint = endpoint
        self._post = post
        self._now = now
        self._start_worker = start_worker
        self._max_buckets = max(1, max_buckets)
        self._aggregates: dict[_UsageKey, _UsageAggregate] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._wake = threading.Event()
        self._closed = False
        self._pending_client_drop_count = 0
        self._pending_transport_retry_count = 0
        self._pending_terminal_send_failure_count = 0

    def record(
        self,
        *,
        client_name: str,
        model_name: Optional[str],
        success: bool,
        usage: _TokenUsage,
        error_category: str = "none",
        execution_mode: str = "sync",
        http_status: str = "none",
        latency_seconds: float = 0.0,
        ttft_seconds: Optional[float] = None,
        partial: bool = False,
    ) -> None:
        now = self._now().astimezone(timezone.utc)
        canonical_id, family = canonical_model_identity(model_name)
        category = "none" if success else _error_category(error_category)
        status_class = _http_status_class(http_status)
        key = _UsageKey(
            window_start=_hour_start(now),
            connector_version=_version_bucket(__version__),
            execution_mode=_execution_mode(execution_mode),
            framework_version=_framework_version_bucket(),
            operation=operation_for_client(client_name),
            model_family=family,
            nim_id=canonical_id,
            error_category=category,
        )
        with self._lock:
            if self._closed:
                return
            aggregate = self._aggregates.get(key)
            if aggregate is None:
                if len(self._aggregates) >= self._max_buckets:
                    self._pending_client_drop_count = min(
                        _MAX_COUNT, self._pending_client_drop_count + 1
                    )
                    return
                aggregate = self._aggregates[key] = _UsageAggregate()
            if self._pending_client_drop_count:
                aggregate.client_drop_count = min(
                    _MAX_COUNT,
                    aggregate.client_drop_count + self._pending_client_drop_count,
                )
                self._pending_client_drop_count = 0
            aggregate.record(
                success=success,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                tokens_missing=not usage.found,
                error_category=category,
                http_status_class=status_class,
                execution_mode=key.execution_mode,
                latency_seconds=latency_seconds,
                ttft_seconds=ttft_seconds,
                partial=partial,
            )
            if self._start_worker and self._thread is None:
                self._thread = threading.Thread(
                    target=self._run,
                    name="nvidia-usage-telemetry",
                    daemon=True,
                )
                self._thread.start()
        if self._start_worker:
            self._wake.set()

    def flush(self, *, include_current: bool) -> int:
        now = self._now().astimezone(timezone.utc)
        current_hour = _hour_start(now)
        with self._lock:
            ready = [
                (key, aggregate)
                for key, aggregate in self._aggregates.items()
                if include_current or key.window_start < current_hour
            ]
            for key, _ in ready:
                del self._aggregates[key]
        if not ready:
            return 0
        batch_id = str(uuid4())
        pending_retry_count = self._pending_transport_retry_count
        pending_failure_count = self._pending_terminal_send_failure_count
        events = []
        for index, (key, aggregate) in enumerate(ready):
            events.append(
                _event_payload(
                    key,
                    aggregate,
                    batch_id=batch_id,
                    transport_retry_count=pending_retry_count if index == 0 else 0,
                    prior_terminal_send_failure_count=(
                        pending_failure_count if index == 0 else 0
                    ),
                )
            )
        try:
            delivery = self._post(self._endpoint, _transport_envelope(events, now=now))
        except Exception:
            delivery = _TransportDelivery(terminal_failure=True)
        if isinstance(delivery, _TransportDelivery):
            with self._lock:
                if delivery.terminal_failure:
                    self._pending_transport_retry_count = min(
                        _MAX_COUNT,
                        self._pending_transport_retry_count + delivery.retry_count,
                    )
                    self._pending_terminal_send_failure_count = min(
                        _MAX_COUNT,
                        self._pending_terminal_send_failure_count + 1,
                    )
                else:
                    self._pending_transport_retry_count = min(
                        _MAX_COUNT, delivery.retry_count
                    )
                    self._pending_terminal_send_failure_count = 0
        else:
            with self._lock:
                self._pending_transport_retry_count = 0
                self._pending_terminal_send_failure_count = 0
        return len(events)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._stop.set()
        self._wake.set()
        self.flush(include_current=True)
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.1)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(_seconds_until_next_hour(self._now()))
            self._wake.clear()
            if not self._stop.is_set():
                self.flush(include_current=False)


_STATE: Optional[_UsageTelemetry] = None
_STATE_LOCK = threading.Lock()


def record_usage(
    *,
    enabled: bool,
    is_hosted: bool,
    client_name: str,
    model_name: Optional[str],
    success: bool,
    usage: _TokenUsage,
    error_category: str = "none",
    execution_mode: str = "sync",
    http_status: str = "none",
    latency_seconds: float = 0.0,
    ttft_seconds: Optional[float] = None,
    partial: bool = False,
) -> None:
    """Record one logical request without retaining request or response content."""

    if not enabled or not is_hosted:
        return
    endpoint = _configured_endpoint()
    if endpoint is None:
        return
    _get_state(endpoint).record(
        client_name=client_name,
        model_name=model_name,
        success=success,
        usage=usage,
        error_category=error_category,
        execution_mode=execution_mode,
        http_status=http_status,
        latency_seconds=latency_seconds,
        ttft_seconds=ttft_seconds,
        partial=partial,
    )


def flush_usage_telemetry() -> int:
    """Flush the current in-memory aggregate, primarily for controlled shutdown."""

    state = _STATE
    return state.flush(include_current=True) if state is not None else 0


def _get_state(endpoint: str) -> _UsageTelemetry:
    global _STATE
    if _STATE is None:
        with _STATE_LOCK:
            if _STATE is None:
                _STATE = _UsageTelemetry(
                    endpoint=endpoint,
                    post=_post_envelope,
                    now=lambda: datetime.now(timezone.utc),
                )
    return _STATE


def _configured_endpoint() -> Optional[str]:
    endpoint = os.getenv(_ENDPOINT_ENV, _DEFAULT_ENDPOINT).strip()
    return endpoint if endpoint in _ALLOWED_ENDPOINTS else None


def _close_state() -> None:
    state = _STATE
    if state is not None:
        state.close()


def _reset_usage_telemetry_for_tests() -> None:
    global _STATE
    state = _STATE
    _STATE = None
    if state is not None:
        state.close()


def _post_envelope(endpoint: str, payload: dict[str, Any]) -> _TransportDelivery:
    timeout = _bounded_float(os.getenv("NVIDIA_USAGE_TELEMETRY_TIMEOUT_SECONDS"), 2.0)
    retry_count = 0
    for attempt in range(3):
        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json;charset=utf-8",
                    "X-Event-Protocol": _EVENT_PROTOCOL,
                },
                timeout=timeout,
            )
        except requests.exceptions.RequestException:
            if attempt < 2:
                retry_count += 1
                time.sleep(0.1 * (2**attempt))
            continue
        if response.ok:
            return _TransportDelivery(retry_count=retry_count)
        if response.status_code not in {408, 429} and response.status_code < 500:
            return _TransportDelivery(retry_count=retry_count, terminal_failure=True)
        if attempt < 2:
            retry_count += 1
            time.sleep(0.1 * (2**attempt))
    return _TransportDelivery(retry_count=retry_count, terminal_failure=True)


def _event_payload(
    key: _UsageKey,
    aggregate: _UsageAggregate,
    *,
    batch_id: str,
    transport_retry_count: int = 0,
    prior_terminal_send_failure_count: int = 0,
) -> dict[str, Any]:
    return {
        "ts": _iso_timestamp(datetime.now(timezone.utc)),
        "name": _EVENT_NAME,
        "parameters": {
            "eventSchemaVersion": _EVENT_SCHEMA_VERSION,
            "windowStart": key.window_start,
            "telemetryClientVersion": _TELEMETRY_CLIENT_VERSION,
            "connector": _CONNECTOR,
            "connectorVersion": key.connector_version,
            "framework": "langchain",
            "frameworkVersion": key.framework_version,
            "operation": key.operation,
            "executionMode": key.execution_mode,
            "modelFamily": key.model_family,
            "nimId": key.nim_id,
            "successCount": aggregate.success_count,
            "failureCount": aggregate.failure_count,
            "cancelledCount": aggregate.cancelled_count,
            "partialCount": aggregate.partial_count,
            "requestCount": aggregate.request_count,
            "attemptCount": aggregate.attempt_count,
            "retriedRequestCount": aggregate.retried_request_count,
            "firstAttemptSuccessCount": aggregate.first_attempt_success_count,
            "inputTokenSum": aggregate.input_token_sum,
            "outputTokenSum": aggregate.output_token_sum,
            "missingTokenCount": aggregate.missing_token_count,
            "missingTokenAttemptCount": aggregate.missing_token_attempt_count,
            "errorCategory": key.error_category,
            "errorCategoryCounts": dict(aggregate.error_category_counts),
            "httpStatusClassCounts": dict(aggregate.http_status_class_counts),
            "latencyBucketCounts": dict(aggregate.latency_bucket_counts),
            "ttftBucketCounts": dict(aggregate.ttft_bucket_counts),
            "missingTtftCount": aggregate.missing_ttft_count,
            "clientDropCount": aggregate.client_drop_count,
            "transportRetryCount": transport_retry_count,
            "priorTerminalSendFailureCount": prior_terminal_send_failure_count,
            "batchId": batch_id,
        },
    }


def _transport_envelope(
    events: list[dict[str, Any]], *, now: datetime
) -> dict[str, Any]:
    return {
        "clientId": _CLIENT_ID,
        "eventSchemaVer": _REGISTRY_SCHEMA_VERSION,
        "eventProtocol": _EVENT_PROTOCOL,
        "sentTs": _iso_timestamp(now),
        "clientVer": "1.0.0.0",
        "integrationId": "undefined",
        "clientType": "Native",
        "clientVariant": "Release",
        "browserType": "undefined",
        "userId": "undefined",
        "externalUserId": "undefined",
        "idpId": "undefined",
        "sessionId": "undefined",
        "eventSysVer": "1.0.0.0",
        "deviceId": "undefined",
        "cpuArchitecture": "undefined",
        "deviceOS": "undefined",
        "deviceOSVersion": "undefined",
        "deviceType": "undefined",
        "deviceMake": "undefined",
        "deviceModel": "undefined",
        "productName": "undefined",
        "productVersion": "undefined",
        "gdprTechOptIn": "None",
        "gdprBehOptIn": "None",
        "gdprFuncOptIn": "Full",
        "deviceGdprTechOptIn": "None",
        "deviceGdprBehOptIn": "None",
        "deviceGdprFuncOptIn": "None",
        "events": events,
    }


def _framework_version_bucket() -> str:
    try:
        return _version_bucket(importlib.metadata.version("langchain-core"))
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _version_bucket(value: str) -> str:
    parts = value.strip().split(".")
    if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        return "unknown"
    return f"{int(parts[0])}.{int(parts[1])}"


def _execution_mode(value: str) -> str:
    return (
        value
        if value in {"sync", "async", "streaming", "batch", "other"}
        else "unknown"
    )


def _error_category(value: str) -> str:
    return value if value in _ERROR_CATEGORIES else "unknown"


def _http_status_class(value: str) -> str:
    return value if value in _HTTP_STATUS_CLASSES else "unknown"


def _hour_start(value: datetime) -> str:
    value = value.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return value.strftime("%Y-%m-%dT%H:00:00Z")


def _seconds_until_next_hour(value: datetime) -> float:
    value = value.astimezone(timezone.utc)
    next_hour = value.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return max(0.1, (next_hour - value).total_seconds())


def _iso_timestamp(value: datetime) -> str:
    value = value.astimezone(timezone.utc)
    return value.strftime("%Y-%m-%dT%H:%M:%S.") + f"{value.microsecond // 1000:03d}Z"


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _zero_counts(keys: tuple[str, ...]) -> dict[str, int]:
    return {key: 0 for key in keys}


def _zero_latency_buckets() -> dict[str, int]:
    return {key: 0 for key, _ in _LATENCY_BUCKETS} | {_OVERFLOW_BUCKET: 0}


def _increment_count(values: dict[str, int], key: str) -> None:
    values[key] = min(_MAX_COUNT, values.get(key, 0) + 1)


def _increment_bucket(values: dict[str, int], duration_seconds: float) -> None:
    bucket = _OVERFLOW_BUCKET
    duration_seconds = max(0.0, duration_seconds)
    for candidate, upper_bound in _LATENCY_BUCKETS:
        if duration_seconds <= upper_bound:
            bucket = candidate
            break
    _increment_count(values, bucket)


def _bounded_float(value: Optional[str], default: float) -> float:
    try:
        parsed = float(value) if value is not None else default
    except ValueError:
        parsed = default
    return min(10.0, max(0.1, parsed))


atexit.register(_close_state)
