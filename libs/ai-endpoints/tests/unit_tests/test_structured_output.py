import enum
import warnings
from typing import Callable, List, Optional, Type

import pytest
import requests_mock
from pydantic import BaseModel as pydanticV2BaseModel  # ignore: check_pydantic
from pydantic import Field
from pydantic.v1 import BaseModel as pydanticV1BaseModel  # ignore: check_pydantic

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_ai_endpoints.chat_models import _is_structured_output


class Joke(pydanticV2BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(description="How funny the joke is, from 1 to 10")


def test_method() -> None:
    with pytest.warns(UserWarning) as record:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                message=".*not known to support structured output.*",
            )
            ChatNVIDIA(api_key="BOGUS").with_structured_output(Joke, method="json_mode")
        assert len(record) == 1
        assert "unnecessary" in str(record[0].message)


def test_include_raw() -> None:
    with pytest.raises(NotImplementedError):
        ChatNVIDIA(api_key="BOGUS").with_structured_output(Joke, include_raw=True)

    with pytest.raises(NotImplementedError):
        ChatNVIDIA(api_key="BOGUS").with_structured_output(
            Joke.model_json_schema(), include_raw=True
        )


def test_known_does_not_warn(empty_v1_models: None) -> None:
    structured_model = [
        model
        for model in ChatNVIDIA.get_available_models(api_key="BOGUS")
        if model.supports_structured_output and not model.deprecated
    ]
    assert structured_model, "No models support structured output"

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ChatNVIDIA(
            api_key="BOGUS", model=structured_model[0].id
        ).with_structured_output(Joke)


def test_unknown_warns(empty_v1_models: None) -> None:
    unstructured_model = [
        model
        for model in ChatNVIDIA.get_available_models(api_key="BOGUS")
        if not model.supports_structured_output and not model.deprecated
    ]
    assert unstructured_model, "All models support structured output"

    with pytest.warns(UserWarning) as record:
        ChatNVIDIA(
            api_key="BOGUS", model=unstructured_model[0].id
        ).with_structured_output(Joke)
    assert len(record) == 1
    assert "not known to support structured output" in str(record[0].message)


def test_enum_negative() -> None:
    class Choices(enum.Enum):
        A = "A"
        B = "2"
        C = 3

    llm = ChatNVIDIA(api_key="BOGUS")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=".*not known to support structured output.*",
        )
        with pytest.raises(ValueError) as e:
            llm.with_structured_output(Choices)
    assert "only contain string choices" in str(e.value)


class Choices(enum.Enum):
    YES = "Yes it is"
    NO = "No it is not"


@pytest.mark.parametrize(
    "chunks",
    [
        ["Y", "es", " it", " is"],
        ["N", "o", " it", " is", " not"],
    ],
    ids=["YES", "NO"],
)
def test_stream_enum(
    mock_streaming_response: Callable,
    chunks: List[str],
) -> None:
    mock_streaming_response(chunks)

    warnings.filterwarnings("ignore", r".*not known to support structured output.*")
    structured_llm = ChatNVIDIA(api_key="BOGUS").with_structured_output(Choices)
    # chunks are progressively more complete, so we only consider the last
    for chunk in structured_llm.stream("This is ignored."):
        response = chunk
    assert isinstance(response, Choices)
    assert response in Choices


@pytest.mark.parametrize(
    "chunks",
    [
        ["Y", "es", " it"],
        ["N", "o", " it", " is"],
    ],
    ids=["YES", "NO"],
)
def test_stream_enum_incomplete(
    mock_streaming_response: Callable,
    chunks: List[str],
) -> None:
    mock_streaming_response(chunks)

    warnings.filterwarnings("ignore", r".*not known to support structured output.*")
    structured_llm = ChatNVIDIA(api_key="BOGUS").with_structured_output(Choices)
    # chunks are progressively more complete, so we only consider the last
    for chunk in structured_llm.stream("This is ignored."):
        response = chunk
    assert response is None


@pytest.mark.parametrize(
    "pydanticBaseModel",
    [
        pydanticV1BaseModel,
        pydanticV2BaseModel,
    ],
    ids=["pydantic-v1", "pydantic-v2"],
)
def test_pydantic_version(
    requests_mock: requests_mock.Mocker,
    pydanticBaseModel: Type,
) -> None:
    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        json={
            "id": "chatcmpl-ID",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "BOGUS",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": '{"name": "Sam Doe"}',
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 22,
                "completion_tokens": 20,
                "total_tokens": 42,
            },
            "system_fingerprint": None,
        },
    )

    class Person(pydanticBaseModel):  # type: ignore
        name: str

    warnings.filterwarnings("ignore", r".*not known to support structured output.*")
    llm = ChatNVIDIA(api_key="BOGUS").with_structured_output(Person)
    response = llm.invoke("This is ignored.")
    assert isinstance(response, Person)
    assert response.name == "Sam Doe"


@pytest.mark.parametrize(
    "strict",
    [False, None, "BOGUS"],
)
def test_strict_warns(strict: Optional[bool]) -> None:
    warnings.filterwarnings("error")  # no warnings should be raised

    # acceptable warnings
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=".*not known to support.*"
    )

    # warnings under test
    strict_warning = ".*`strict` is ignored.*"
    warnings.filterwarnings("default", category=UserWarning, message=strict_warning)

    with pytest.warns(UserWarning, match=strict_warning):
        ChatNVIDIA(api_key="BOGUS").with_structured_output(
            Joke,
            strict=strict,
        )


@pytest.mark.parametrize(
    "strict",
    [True, None],
    ids=["strict-True", "no-strict"],
)
def test_strict_no_warns(strict: Optional[bool]) -> None:
    warnings.filterwarnings("error")  # no warnings should be raised

    # acceptable warnings
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=".*not known to support.*"
    )

    ChatNVIDIA(api_key="BOGUS").with_structured_output(
        Joke,
        **({"strict": strict} if strict is not None else {}),
    )


# Tests for _is_structured_output detection
def test_is_structured_output_direct() -> None:
    """Test _is_structured_output detects direct format."""
    payload = {
        "messages": [{"role": "user", "content": "test"}],
        "guided_json": {"type": "object"},
    }
    assert _is_structured_output(payload) is True


def test_is_structured_output_nvext() -> None:
    """Test _is_structured_output detects nvext format."""
    payload = {
        "messages": [{"role": "user", "content": "test"}],
        "nvext": {"guided_json": {"type": "object"}},
    }
    assert _is_structured_output(payload) is True


def test_is_structured_output_guided_choice_direct() -> None:
    """Test _is_structured_output detects direct format with guided_choice."""
    payload = {
        "messages": [{"role": "user", "content": "test"}],
        "guided_choice": ["A", "B", "C"],
    }
    assert _is_structured_output(payload) is True


def test_is_structured_output_guided_choice_nvext() -> None:
    """Test _is_structured_output detects nvext format with guided_choice."""
    payload = {
        "messages": [{"role": "user", "content": "test"}],
        "nvext": {"guided_choice": ["A", "B", "C"]},
    }
    assert _is_structured_output(payload) is True


def test_is_structured_output_none() -> None:
    """Test _is_structured_output returns False when no structured output."""
    payload = {
        "messages": [{"role": "user", "content": "test"}],
    }
    assert _is_structured_output(payload) is False


def test_is_structured_output_openai_json_schema() -> None:
    """Test _is_structured_output detects OpenAI response_format with json_schema."""
    payload = {
        "messages": [{"role": "user", "content": "test"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": {"type": "object"}},
        },
    }
    assert _is_structured_output(payload) is True


_JOKE_JSON = (
    '{"setup": "Why did the chicken cross the road?",'
    ' "punchline": "To get to the other side", "rating": 5}'
)


def _success_response(content: str) -> dict:
    return {
        "id": "chatcmpl-ID",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "BOGUS",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
    }


# --- Hosted API endpoint tests (OpenAI -> direct -> nvext) ---


def test_hosted_api_endpoint_openai_format_succeeds(
    requests_mock: requests_mock.Mocker,
    empty_v1_models: None,
) -> None:
    """Test hosted: OpenAI format succeeds on first try."""
    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        json=_success_response(_JOKE_JSON),
    )

    warnings.filterwarnings("ignore", r".*not known to support structured output.*")
    llm = ChatNVIDIA(api_key="BOGUS").with_structured_output(Joke)
    result = llm.invoke("test")

    # Should only make 1 POST call (OpenAI format succeeds)
    post_calls = [req for req in requests_mock.request_history if req.method == "POST"]
    assert len(post_calls) == 1

    # First call: OpenAI format (succeeds)
    request_body = post_calls[0].json()
    assert "response_format" in request_body
    assert request_body["response_format"]["type"] == "json_schema"

    assert isinstance(result, Joke)
    assert result.setup == "Why did the chicken cross the road?"
    assert result.punchline == "To get to the other side"
    assert result.rating == 5


def test_hosted_api_endpoint_openai_fails_falls_back_to_direct(
    requests_mock: requests_mock.Mocker,
    empty_v1_models: None,
) -> None:
    """Test hosted: OpenAI format fails, falls back to direct format."""
    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        [
            {"status_code": 400, "json": {"error": "openai format failed"}},
            {"json": _success_response(_JOKE_JSON)},
        ],
    )

    warnings.filterwarnings("ignore", r".*not known to support structured output.*")
    llm = ChatNVIDIA(api_key="BOGUS").with_structured_output(Joke)
    result = llm.invoke("test")

    post_calls = [req for req in requests_mock.request_history if req.method == "POST"]
    assert len(post_calls) == 2, f"Expected 2 POST calls, got {len(post_calls)}"

    # First call: OpenAI format (fails)
    assert "response_format" in post_calls[0].json()
    # Second call: direct format (succeeds)
    assert "guided_json" in post_calls[1].json()
    assert "nvext" not in post_calls[1].json()

    assert isinstance(result, Joke)
    assert result.setup == "Why did the chicken cross the road?"
    assert result.punchline == "To get to the other side"
    assert result.rating == 5


def test_hosted_api_endpoint_openai_and_direct_fail_falls_back_to_nvext(
    requests_mock: requests_mock.Mocker,
    empty_v1_models: None,
) -> None:
    """Test hosted: OpenAI format and direct format fail, falls back to nvext format."""
    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        [
            {"status_code": 400, "json": {"error": "openai format failed"}},
            {"status_code": 400, "json": {"error": "direct format failed"}},
            {"json": _success_response(_JOKE_JSON)},
        ],
    )

    warnings.filterwarnings("ignore", r".*not known to support structured output.*")
    llm = ChatNVIDIA(api_key="BOGUS").with_structured_output(Joke)
    result = llm.invoke("test")

    post_calls = [req for req in requests_mock.request_history if req.method == "POST"]
    assert len(post_calls) == 3, f"Expected 3 POST calls, got {len(post_calls)}"

    # First call: OpenAI format (fails)
    assert "response_format" in post_calls[0].json()
    # Second call: direct format (fails)
    assert "guided_json" in post_calls[1].json()
    assert "nvext" not in post_calls[1].json()
    # Third call: nvext format (succeeds)
    assert "nvext" in post_calls[2].json()
    assert "guided_json" in post_calls[2].json()["nvext"]

    assert isinstance(result, Joke)
    assert result.setup == "Why did the chicken cross the road?"
    assert result.punchline == "To get to the other side"
    assert result.rating == 5


# --- Self-hosted NIM tests (direct -> nvext -> OpenAI) ---


def test_self_hosted_direct_format_succeeds(
    requests_mock: requests_mock.Mocker,
) -> None:
    """Test self-hosted: direct format succeeds on first try."""
    requests_mock.get(
        "http://my-nim:8000/v1/models",
        json={"data": [{"id": "my-model"}]},
    )
    requests_mock.post(
        "http://my-nim:8000/v1/chat/completions",
        json=_success_response(_JOKE_JSON),
    )

    warnings.filterwarnings("ignore", r".*not known to support structured output.*")
    llm = ChatNVIDIA(
        base_url="http://my-nim:8000/v1", api_key="BOGUS"
    ).with_structured_output(Joke)
    result = llm.invoke("test")

    # Should only make 1 POST call (direct format succeeds)
    post_calls = [req for req in requests_mock.request_history if req.method == "POST"]
    assert len(post_calls) == 1

    # First call: direct format (succeeds)
    body = post_calls[0].json()
    assert "guided_json" in body
    assert "nvext" not in body

    assert isinstance(result, Joke)
    assert result.setup == "Why did the chicken cross the road?"
    assert result.punchline == "To get to the other side"
    assert result.rating == 5


def test_self_hosted_direct_fails_falls_back_to_nvext(
    requests_mock: requests_mock.Mocker,
) -> None:
    """Test self-hosted: direct format fails, falls back to nvext format."""
    requests_mock.get(
        "http://my-nim:8000/v1/models",
        json={"data": [{"id": "my-model"}]},
    )
    # First call fails (direct), second succeeds (nvext)
    requests_mock.post(
        "http://my-nim:8000/v1/chat/completions",
        [
            {"status_code": 400, "json": {"error": "direct failed"}},
            {"json": _success_response(_JOKE_JSON)},
        ],
    )

    warnings.filterwarnings("ignore", r".*not known to support structured output.*")
    llm = ChatNVIDIA(
        base_url="http://my-nim:8000/v1", api_key="BOGUS"
    ).with_structured_output(Joke)
    result = llm.invoke("test")

    # Should make 2 POST calls: direct format (fails) + nvext format (succeeds)
    post_calls = [req for req in requests_mock.request_history if req.method == "POST"]
    assert len(post_calls) == 2, f"Expected 2 POST calls, got {len(post_calls)}"

    # First call: direct format (fails)
    assert "guided_json" in post_calls[0].json()
    assert "nvext" not in post_calls[0].json()
    # Second call: nvext format (succeeds)
    assert "nvext" in post_calls[1].json()
    assert "guided_json" in post_calls[1].json()["nvext"]

    assert isinstance(result, Joke)
    assert result.setup == "Why did the chicken cross the road?"
    assert result.punchline == "To get to the other side"
    assert result.rating == 5


def test_self_hosted_fallback_to_openai_format(
    requests_mock: requests_mock.Mocker,
) -> None:
    """Test self-hosted: direct and nvext fail, falls back to OpenAI."""
    requests_mock.get(
        "http://my-nim:8000/v1/models",
        json={"data": [{"id": "my-model"}]},
    )
    # First two calls fail (direct, nvext), third succeeds (OpenAI)
    requests_mock.post(
        "http://my-nim:8000/v1/chat/completions",
        [
            {"status_code": 400, "json": {"error": "direct failed"}},
            {"status_code": 400, "json": {"error": "nvext failed"}},
            {"json": _success_response(_JOKE_JSON)},
        ],
    )

    warnings.filterwarnings("ignore", r".*not known to support structured output.*")
    llm = ChatNVIDIA(
        base_url="http://my-nim:8000/v1", api_key="BOGUS"
    ).with_structured_output(Joke)
    result = llm.invoke("test")

    post_calls = [req for req in requests_mock.request_history if req.method == "POST"]
    assert len(post_calls) == 3, f"Expected 3 POST calls, got {len(post_calls)}"

    # First call: direct format (fails)
    assert "guided_json" in post_calls[0].json()
    assert "nvext" not in post_calls[0].json()
    # Second call: nvext format (fails)
    assert "nvext" in post_calls[1].json()
    # Third call: OpenAI format (succeeds)
    body = post_calls[2].json()
    assert "response_format" in body
    assert body["response_format"]["type"] == "json_schema"

    assert isinstance(result, Joke)
    assert result.setup == "Why did the chicken cross the road?"
    assert result.punchline == "To get to the other side"
    assert result.rating == 5


# --- Enum tests ---


def test_enum_openai_format_invoke(
    requests_mock: requests_mock.Mocker,
    empty_v1_models: None,
) -> None:
    """Test enum structured output via OpenAI response_format."""
    # Hosted endpoint -> OpenAI format first -> enum returned as JSON
    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        json=_success_response('{"choice": "Yes it is"}'),
    )

    warnings.filterwarnings("ignore", r".*not known to support structured output.*")
    llm = ChatNVIDIA(api_key="BOGUS").with_structured_output(Choices)
    with pytest.warns(UserWarning, match="Enum structured output is not guaranteed"):
        result = llm.invoke("test")

    assert isinstance(result, Choices)
    assert result == Choices.YES

    # Verify OpenAI format was used with strict enum schema
    post_calls = [req for req in requests_mock.request_history if req.method == "POST"]
    body = post_calls[0].json()
    assert "response_format" in body
    rf = body["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert "choice" in rf["json_schema"]["schema"]["properties"]
    assert rf["json_schema"]["schema"]["additionalProperties"] is False


def test_self_hosted_enum_guided_choice_succeeds(
    requests_mock: requests_mock.Mocker,
) -> None:
    """Test self-hosted: enum uses guided_choice first and succeeds."""
    requests_mock.get(
        "http://my-nim:8000/v1/models",
        json={"data": [{"id": "my-model"}]},
    )
    requests_mock.post(
        "http://my-nim:8000/v1/chat/completions",
        json=_success_response("Yes it is"),
    )

    warnings.filterwarnings("ignore", r".*not known to support structured output.*")
    llm = ChatNVIDIA(
        base_url="http://my-nim:8000/v1", api_key="BOGUS"
    ).with_structured_output(Choices)
    result = llm.invoke("test")

    # Should only make 1 POST call (guided_choice succeeds)
    post_calls = [req for req in requests_mock.request_history if req.method == "POST"]
    assert len(post_calls) == 1

    # Verify guided_choice was used (not OpenAI format)
    body = post_calls[0].json()
    assert "guided_choice" in body
    assert "response_format" not in body

    assert isinstance(result, Choices)
    assert result == Choices.YES


# --- Error handling test ---


def test_all_formats_fail_raises_last_exception(
    requests_mock: requests_mock.Mocker,
    empty_v1_models: None,
) -> None:
    """Test that when all format attempts fail, the last exception is raised."""
    # All three calls fail (OpenAI -> direct -> nvext)
    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        [
            {"status_code": 400, "json": {"error": "openai format failed"}},
            {"status_code": 400, "json": {"error": "direct format failed"}},
            {"status_code": 400, "json": {"error": "nvext format failed"}},
        ],
    )

    warnings.filterwarnings("ignore", r".*not known to support structured output.*")
    llm = ChatNVIDIA(api_key="BOGUS").with_structured_output(Joke)
    with pytest.raises(Exception):
        llm.invoke("test")


# --- None result triggers fallback ---


def test_none_result_triggers_next_chain(
    requests_mock: requests_mock.Mocker,
    empty_v1_models: None,
) -> None:
    """Test that a None parser result triggers the next fallback chain."""
    # First call returns malformed JSON (parser returns None),
    # second call returns valid JSON.
    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        [
            {"json": _success_response("not valid json")},
            {"json": _success_response(_JOKE_JSON)},
        ],
    )

    warnings.filterwarnings("ignore", r".*not known to support structured output.*")
    llm = ChatNVIDIA(api_key="BOGUS").with_structured_output(Joke)
    result = llm.invoke("test")

    # First call: OpenAI format returns unparseable content (parser returns None)
    # second call: direct format succeeds
    post_calls = [req for req in requests_mock.request_history if req.method == "POST"]
    assert len(post_calls) == 2, f"Expected 2 POST calls, got {len(post_calls)}"

    assert isinstance(result, Joke)
    assert result.setup == "Why did the chicken cross the road?"
    assert result.punchline == "To get to the other side"
    assert result.rating == 5
