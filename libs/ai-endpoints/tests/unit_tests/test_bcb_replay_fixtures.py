"""Replay public-safe BCB-like response shapes through connector parsing paths."""

from __future__ import annotations

import json
import re
from functools import reduce
from operator import add
from pathlib import Path
from typing import Any

import pytest
import requests_mock
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings, NVIDIARerank

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "data" / "bcb_replay_fixtures"
CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
EMBEDDINGS_URL = "https://integrate.api.nvidia.com/v1/embeddings"
RANKING_URL = "https://integrate.api.nvidia.com/v1/ranking"
FORBIDDEN_KEY_RE = re.compile(
    r"(^|_)(authorization|cookie|raw_prompt|request_?id|trace_?id)($|_)",
    re.IGNORECASE,
)
FORBIDDEN_VALUE_RE = re.compile(
    r"(bearer\s+|nvapi-|api[_-]?key|NVCF-REQID|data:image|https?://(?!integrate\.api\.nvidia\.com))",
    re.IGNORECASE,
)

pytestmark = [
    pytest.mark.filterwarnings("ignore:Found mock-model in available_models.*"),
    pytest.mark.filterwarnings("ignore:.*not known to support tools.*"),
    pytest.mark.filterwarnings("ignore:.*not known to support structured output.*"),
]


@tool
def lookup_policy(name: str) -> str:
    """Look up a policy by name."""
    return f"policy:{name}"


@tool
def record_compatibility_status(status: str, surface: str = "") -> str:
    """Record a compatibility status."""
    return f"{status}:{surface}"


@tool
def record_compatibility_surface(surface: str) -> str:
    """Record a compatibility surface."""
    return surface


class PolicyAnswer(BaseModel):
    """Small structured-output model for replay parsing."""

    answer: str = Field(description="Resolved policy answer")
    confidence: float = Field(description="Confidence score")


class CompatibilityStatus(BaseModel):
    """Structured output model used by promoted BCB captures."""

    status: str
    detail: str


def load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def manifest_entries() -> list[dict[str, Any]]:
    manifest = load_fixture("manifest.json")
    entries = manifest.get("fixtures", [])
    return [entry for entry in entries if isinstance(entry, dict)]


def approved_entries(category: str) -> list[dict[str, Any]]:
    return [
        entry
        for entry in manifest_entries()
        if entry.get("category") == category
        and entry.get("approved_real_bcb_capture") is True
        and entry.get("review_status") == "approved"
    ]


def iter_json_items(value: Any) -> Any:
    if isinstance(value, dict):
        for key, item in value.items():
            yield key, item
            yield from iter_json_items(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_json_items(item)


def assert_public_safe(value: dict[str, Any]) -> None:
    for key, item in iter_json_items(value):
        assert not FORBIDDEN_KEY_RE.search(str(key)), f"unsafe fixture key: {key}"
        if isinstance(item, str):
            assert not FORBIDDEN_VALUE_RE.search(item), f"unsafe fixture value: {item}"


def sse_text(events: list[dict[str, Any]]) -> str:
    lines = [f"data: {json.dumps(event, separators=(',', ':'))}" for event in events]
    lines.append("data: [DONE]")
    return "\n\n".join(lines)


def test_replay_manifest_documents_current_fixture_boundary() -> None:
    manifest = load_fixture("manifest.json")
    assert manifest["schema_version"] == 1
    assert manifest["status"] in {
        "synthetic_contract_scaffold",
        "bcb_replay_fixtures",
    }
    categories = {fixture["category"] for fixture in manifest["fixtures"]}
    assert {
        "tool_call",
        "streaming",
        "structured_output_reasoning",
        "message_roundtrip",
        "embedding",
        "rerank",
        "error",
    } <= categories
    required_provenance = {
        "bcb_profile",
        "bcb_status",
        "bcb_fingerprint_hash",
        "model_family",
        "source_artifact_folder",
        "review_status",
        "approved_real_bcb_capture",
    }
    for fixture in manifest["fixtures"]:
        assert required_provenance <= fixture.keys()
        if fixture["approved_real_bcb_capture"]:
            assert fixture["source"] == "sanitized_bcb_wire_contract"
            assert fixture["review_status"] == "approved"
            assert fixture["bcb_status"] == "passed"
    assert_public_safe(manifest)


@pytest.mark.parametrize(
    "fixture_name",
    sorted({str(entry["path"]) for entry in manifest_entries() if "path" in entry}),
)
def test_replay_fixtures_are_public_safe(fixture_name: str) -> None:
    assert_public_safe(load_fixture(fixture_name))


@pytest.mark.parametrize(
    "entry",
    approved_entries("tool_call"),
    ids=lambda entry: str(entry.get("id")),
)
def test_replays_approved_bcb_tool_call_fixture(
    entry: dict[str, Any],
    requests_mock: requests_mock.Mocker,
) -> None:
    fixture = load_fixture(str(entry["path"]))
    requests_mock.post(CHAT_URL, json=fixture["body"])

    response = (
        ChatNVIDIA(model="mock-model", api_key="BOGUS")
        .bind_tools([record_compatibility_status, record_compatibility_surface])
        .invoke("ignored")
    )

    assert isinstance(response, AIMessage)
    assert response.tool_calls
    assert {tool_call["name"] for tool_call in response.tool_calls} <= {
        "record_compatibility_status",
        "record_compatibility_surface",
    }


@pytest.mark.parametrize(
    "entry",
    approved_entries("structured_output"),
    ids=lambda entry: str(entry.get("id")),
)
def test_replays_approved_bcb_structured_output_fixture(
    entry: dict[str, Any],
    requests_mock: requests_mock.Mocker,
) -> None:
    fixture = load_fixture(str(entry["path"]))
    requests_mock.post(CHAT_URL, json=fixture["body"])

    response = (
        ChatNVIDIA(model="mock-model", api_key="BOGUS")
        .with_structured_output(CompatibilityStatus)
        .invoke("ignored")
    )

    assert isinstance(response, CompatibilityStatus)
    assert response.status
    assert response.detail


@pytest.mark.parametrize(
    "entry",
    approved_entries("streaming"),
    ids=lambda entry: str(entry.get("id")),
)
def test_replays_approved_bcb_streaming_fixture(
    entry: dict[str, Any],
    requests_mock: requests_mock.Mocker,
) -> None:
    fixture = load_fixture(str(entry["path"]))
    requests_mock.post(CHAT_URL, text=sse_text(fixture["sse_events"]))

    response = reduce(
        add,
        ChatNVIDIA(model="mock-model", api_key="BOGUS").stream("ignored"),
    )

    assert response.content or response.tool_calls


@pytest.mark.parametrize(
    "entry",
    approved_entries("embedding"),
    ids=lambda entry: str(entry.get("id")),
)
def test_replays_approved_bcb_embedding_fixture(
    entry: dict[str, Any],
    requests_mock: requests_mock.Mocker,
) -> None:
    fixture = load_fixture(str(entry["path"]))
    requests_mock.post(EMBEDDINGS_URL, json=fixture["body"])

    response = NVIDIAEmbeddings(model="mock-model", api_key="BOGUS").embed_query(
        "ignored"
    )

    assert response
    assert all(isinstance(value, float) for value in response)


@pytest.mark.parametrize(
    "entry",
    approved_entries("rerank"),
    ids=lambda entry: str(entry.get("id")),
)
def test_replays_approved_bcb_rerank_fixture(
    entry: dict[str, Any],
    requests_mock: requests_mock.Mocker,
) -> None:
    fixture = load_fixture(str(entry["path"]))
    requests_mock.post(RANKING_URL, json=fixture["body"])
    documents = [
        Document(page_content=f"passage {index}", metadata={"id": str(index)})
        for index in range(4)
    ]

    response = list(
        NVIDIARerank(model="mock-model", api_key="BOGUS", top_n=4).compress_documents(
            documents=documents,
            query="ignored query",
        )
    )

    assert response
    assert all("relevance_score" in doc.metadata for doc in response)


@pytest.mark.parametrize(
    "entry",
    approved_entries("error"),
    ids=lambda entry: str(entry.get("id")),
)
def test_replays_approved_bcb_error_fixture(
    entry: dict[str, Any],
    requests_mock: requests_mock.Mocker,
) -> None:
    fixture = load_fixture(str(entry["path"]))
    requests_mock.post(
        CHAT_URL,
        status_code=int(fixture["http_status"]),
        json=fixture["body"],
    )

    with pytest.raises(Exception):
        ChatNVIDIA(model="mock-model", api_key="BOGUS").invoke("ignored")


def test_replays_bind_tools_completion_response(
    requests_mock: requests_mock.Mocker,
) -> None:
    fixture = load_fixture("chat_tool_call_completion.json")
    requests_mock.post(CHAT_URL, json=fixture["body"])

    response = (
        ChatNVIDIA(model="mock-model", api_key="BOGUS")
        .bind_tools([lookup_policy], tool_choice="lookup_policy")
        .invoke("ignored")
    )

    assert isinstance(response, AIMessage)
    assert response.content == ""
    assert response.tool_calls == [
        {
            "name": "lookup_policy",
            "args": {"name": "retention"},
            "id": "call_lookup_policy",
            "type": "tool_call",
        }
    ]
    assert response.usage_metadata == {
        "input_tokens": 12,
        "output_tokens": 8,
        "total_tokens": 20,
    }


def test_replays_streaming_bind_tools_message_chunks(
    requests_mock: requests_mock.Mocker,
) -> None:
    fixture = load_fixture("chat_streaming_tool_call.json")
    requests_mock.post(CHAT_URL, text=sse_text(fixture["sse_events"]))

    response = reduce(
        add,
        ChatNVIDIA(model="mock-model", api_key="BOGUS")
        .bind_tools([lookup_policy])
        .stream("ignored"),
    )

    assert response.tool_calls == [
        {
            "name": "lookup_policy",
            "args": {"name": "retention"},
            "id": "call_lookup_policy",
            "type": "tool_call",
        }
    ]
    assert response.usage_metadata == {
        "input_tokens": 15,
        "output_tokens": 9,
        "total_tokens": 24,
    }


def test_replays_structured_output_response(
    requests_mock: requests_mock.Mocker,
) -> None:
    fixture = load_fixture("chat_structured_reasoning.json")
    requests_mock.post(CHAT_URL, json=fixture["body"])

    response = (
        ChatNVIDIA(model="mock-model", api_key="BOGUS")
        .with_structured_output(PolicyAnswer)
        .invoke("ignored")
    )

    assert response == PolicyAnswer(answer="retention", confidence=0.98)


def test_replays_reasoning_message_fields_roundtrip(
    requests_mock: requests_mock.Mocker,
) -> None:
    turn1 = load_fixture("chat_structured_reasoning.json")
    turn2 = load_fixture("chat_followup_answer.json")
    requests_mock.post(
        CHAT_URL,
        [{"json": turn1["body"]}, {"json": turn2["body"]}],
    )

    llm = ChatNVIDIA(model="mock-model", api_key="BOGUS")
    response = llm.invoke([HumanMessage(content="ignored")])

    assert response.additional_kwargs["reasoning_content"].startswith(
        "Checked the approved policy options"
    )
    assert response.additional_kwargs["reasoning"].startswith(
        "Checked the approved policy options"
    )
    assert response.additional_kwargs["_reasoning_api_fields"] == [
        "reasoning_content",
        "reasoning",
    ]

    llm.invoke(
        [
            HumanMessage(content="ignored"),
            response,
            HumanMessage(content="ignored follow-up"),
        ]
    )

    chat_posts = [
        request
        for request in requests_mock.request_history
        if request.method == "POST" and request.url == CHAT_URL
    ]
    followup_payload = chat_posts[1].json()
    assistant_message = followup_payload["messages"][1]
    assert assistant_message["role"] == "assistant"
    assert assistant_message["content"] == '{"answer":"retention","confidence":0.98}'
    assert assistant_message["reasoning_content"].startswith(
        "Checked the approved policy options"
    )
    assert assistant_message["reasoning"].startswith(
        "Checked the approved policy options"
    )
    assert "_reasoning_api_fields" not in assistant_message


def test_replays_embedding_response_through_nvidia_embeddings(
    requests_mock: requests_mock.Mocker,
) -> None:
    fixture = load_fixture("embedding_basic.json")
    requests_mock.post(EMBEDDINGS_URL, json=fixture["body"])

    response = NVIDIAEmbeddings(model="mock-model", api_key="BOGUS").embed_query(
        "ignored"
    )

    assert response == [0.125, -0.25, 0.5]
    assert requests_mock.last_request is not None
    request_payload = requests_mock.last_request.json()
    assert request_payload["input"] == ["ignored"]
    assert request_payload["input_type"] == "query"


def test_replays_rerank_response_through_nvidia_rerank(
    requests_mock: requests_mock.Mocker,
) -> None:
    fixture = load_fixture("rerank_basic.json")
    requests_mock.post(RANKING_URL, json=fixture["body"])
    documents = [
        Document(page_content="first passage", metadata={"id": "first"}),
        Document(page_content="second passage", metadata={"id": "second"}),
    ]

    response = list(
        NVIDIARerank(model="mock-model", api_key="BOGUS", top_n=2).compress_documents(
            documents=documents, query="ignored query"
        )
    )

    assert [doc.metadata["id"] for doc in response] == ["second", "first"]
    assert [doc.metadata["relevance_score"] for doc in response] == [8.75, 1.25]
    assert all("relevance_score" not in doc.metadata for doc in documents)
    assert requests_mock.last_request is not None
    request_payload = requests_mock.last_request.json()
    assert request_payload["query"] == {"text": "ignored query"}
    assert request_payload["passages"] == [
        {"text": "first passage"},
        {"text": "second passage"},
    ]


def test_replays_provider_error_envelope(
    requests_mock: requests_mock.Mocker,
) -> None:
    fixture = load_fixture("negative-fixtures/provider_error_envelope.json")
    requests_mock.post(
        CHAT_URL,
        status_code=fixture["http_status"],
        json=fixture["body"],
    )

    with pytest.raises(Exception) as exc_info:
        ChatNVIDIA(model="mock-model", api_key="BOGUS").invoke("ignored")

    message = str(exc_info.value)
    assert "[400] Bad Request" in message
    assert "structured-output schema was rejected" in message
