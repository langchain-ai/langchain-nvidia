"""Integration-style tests for ChatNVIDIADynamo using a local dummy server.

Spins up a lightweight HTTP server that validates the full request body
structure, including nvext.agent_hints, and returns well-formed
OpenAI-compatible responses.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List

import pytest

from langchain_nvidia_ai_endpoints.chat_models_dynamo import ChatNVIDIADynamo


@pytest.fixture(autouse=True)
def mock_v1_models(requests_mock):
    """Override the conftest autouse fixture to allow real HTTP to local server."""
    requests_mock.real_http = True


# --- Validation helpers ---

REQUIRED_TOP_LEVEL_KEYS = {"model", "messages"}
REQUIRED_AGENT_HINTS_KEYS = {
    "prefix_id",
    "osl",
    "iat",
    "latency_sensitivity",
    "priority",
}


def validate_request_body(body: Dict[str, Any]) -> List[str]:
    """Return a list of validation errors (empty == valid)."""
    errors: List[str] = []

    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in body:
            errors.append(f"missing top-level key: {key}")

    messages = body.get("messages", [])
    if not isinstance(messages, list) or len(messages) == 0:
        errors.append("messages must be a non-empty list")
    else:
        for i, msg in enumerate(messages):
            if "role" not in msg:
                errors.append(f"messages[{i}] missing 'role'")
            if "content" not in msg and msg.get("role") != "assistant":
                errors.append(f"messages[{i}] missing 'content'")

    nvext = body.get("nvext")
    if nvext is None:
        errors.append("missing 'nvext' in request body")
    elif not isinstance(nvext, dict):
        errors.append("'nvext' must be a dict")
    else:
        hints = nvext.get("agent_hints")
        if hints is None:
            errors.append("missing 'nvext.agent_hints'")
        elif not isinstance(hints, dict):
            errors.append("'nvext.agent_hints' must be a dict")
        else:
            for key in REQUIRED_AGENT_HINTS_KEYS:
                if key not in hints:
                    errors.append(f"missing 'nvext.agent_hints.{key}'")

            if "prefix_id" in hints:
                if not isinstance(hints["prefix_id"], str):
                    errors.append("prefix_id must be a string")
                elif not hints["prefix_id"].startswith("langchain-dynamo-"):
                    errors.append(
                        f"prefix_id should start with 'langchain-dynamo-', "
                        f"got: {hints['prefix_id']}"
                    )

            if "latency_sensitivity" in hints:
                if not isinstance(hints["latency_sensitivity"], (int, float)):
                    errors.append("latency_sensitivity must be a number")

            if "priority" in hints:
                if not isinstance(hints["priority"], int):
                    errors.append("priority must be an int")

            if "osl" in hints:
                if not isinstance(hints["osl"], int):
                    errors.append("osl must be an int")

            if "iat" in hints:
                if not isinstance(hints["iat"], int):
                    errors.append("iat must be an int")

    return errors


# --- Dummy server ---

_CHAT_RESPONSE = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "mock-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from dummy server!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

_STREAM_CHUNKS = [
    '{"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,'
    '"model":"mock-model","choices":[{"index":0,"delta":{"role":"assistant","content":null},'
    '"finish_reason":null}]}',
    '{"id":"chatcmpl-test","object":"chat.completion.chunk","created":1234567890,'
    '"model":"mock-model","choices":[{"index":0,"delta":{"role":null,"content":"Hello"},'
    '"finish_reason":null}]}',
    '{"id":"chatcmpl-test","object":"chat.completion.chunk",'
    '"created":1234567890,"model":"mock-model","choices":'
    '[{"index":0,"delta":{"role":null,"content":" world"},'
    '"finish_reason":"stop"}]}',
    "[DONE]",
]


class _Handler(BaseHTTPRequestHandler):
    """HTTP handler that validates requests and returns canned responses."""

    # Shared across requests; set by the test fixture
    validation_errors: List[List[str]] = []
    received_bodies: List[Dict[str, Any]] = []

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._json_response({"data": [{"id": "mock-model", "object": "model"}]})
        else:
            self._json_response({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        body = json.loads(raw)

        errors = validate_request_body(body)
        _Handler.validation_errors.append(errors)
        _Handler.received_bodies.append(body)

        if errors:
            self._json_response(
                {"error": {"message": f"Validation failed: {errors}"}},
                status=400,
            )
            return

        is_stream = body.get("stream", False)
        if is_stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in _STREAM_CHUNKS:
                self.wfile.write(f"data: {chunk}\n\n".encode())
                self.wfile.flush()
        else:
            self._json_response(_CHAT_RESPONSE)

    def _json_response(self, data: dict, status: int = 200) -> None:
        payload = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        pass  # suppress request logging


@pytest.fixture()
def dummy_server():
    """Start a local HTTP server on an ephemeral port, yield the base URL."""
    _Handler.validation_errors = []
    _Handler.received_bodies = []

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}/v1"
    server.shutdown()


# --- Tests ---


def test_invoke_sends_valid_body(dummy_server: str) -> None:
    llm = ChatNVIDIADynamo(
        model="mock-model",
        base_url=dummy_server,
        api_key="fake-key",
    )
    result = llm.invoke("What is Dynamo?")

    assert len(_Handler.validation_errors) == 1, "expected exactly one request"
    assert _Handler.validation_errors[0] == [], (
        f"request body validation failed: {_Handler.validation_errors[0]}"
    )
    assert result.content == "Hello from dummy server!"

    body = _Handler.received_bodies[0]
    hints = body["nvext"]["agent_hints"]
    assert hints["osl"] == 512
    assert hints["iat"] == 250
    assert hints["latency_sensitivity"] == 1.0
    assert hints["priority"] == 1
    assert hints["prefix_id"].startswith("langchain-dynamo-")


def test_invoke_with_overrides(dummy_server: str) -> None:
    llm = ChatNVIDIADynamo(
        model="mock-model",
        base_url=dummy_server,
        api_key="fake-key",
        osl=1024,
        iat=100,
    )
    llm.invoke("test", osl=2048, latency_sensitivity=0.5, priority=3)

    assert _Handler.validation_errors[0] == []
    hints = _Handler.received_bodies[0]["nvext"]["agent_hints"]
    assert hints["osl"] == 2048  # per-invocation override
    assert hints["iat"] == 100  # constructor value
    assert hints["latency_sensitivity"] == 0.5
    assert hints["priority"] == 3


def test_stream_sends_valid_body(dummy_server: str) -> None:
    llm = ChatNVIDIADynamo(
        model="mock-model",
        base_url=dummy_server,
        api_key="fake-key",
    )
    chunks = list(llm.stream("Stream test"))

    assert len(_Handler.validation_errors) == 1
    assert _Handler.validation_errors[0] == [], (
        f"request body validation failed: {_Handler.validation_errors[0]}"
    )
    assert len(chunks) > 0

    body = _Handler.received_bodies[0]
    assert body["stream"] is True
    hints = body["nvext"]["agent_hints"]
    assert hints["prefix_id"].startswith("langchain-dynamo-")
    assert hints["osl"] == 512


@pytest.mark.asyncio
async def test_ainvoke_sends_valid_body(dummy_server: str) -> None:
    llm = ChatNVIDIADynamo(
        model="mock-model",
        base_url=dummy_server,
        api_key="fake-key",
    )
    result = await llm.ainvoke("Async test")

    assert len(_Handler.validation_errors) == 1
    assert _Handler.validation_errors[0] == []
    assert result.content == "Hello from dummy server!"

    hints = _Handler.received_bodies[0]["nvext"]["agent_hints"]
    assert hints["prefix_id"].startswith("langchain-dynamo-")


def test_unique_prefix_id_per_request(dummy_server: str) -> None:
    llm = ChatNVIDIADynamo(
        model="mock-model",
        base_url=dummy_server,
        api_key="fake-key",
    )
    llm.invoke("request 1")
    llm.invoke("request 2")

    assert len(_Handler.received_bodies) == 2
    id1 = _Handler.received_bodies[0]["nvext"]["agent_hints"]["prefix_id"]
    id2 = _Handler.received_bodies[1]["nvext"]["agent_hints"]["prefix_id"]
    assert id1 != id2, "each request should get a unique prefix_id"
