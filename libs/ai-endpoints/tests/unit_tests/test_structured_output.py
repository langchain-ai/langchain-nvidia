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
        if model.supports_structured_output
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
        if not model.supports_structured_output
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


# Tests for fallback mechanism for structured output parameters
def test_fallback_direct_succeeds(
    requests_mock: requests_mock.Mocker,
    empty_v1_models: None,
) -> None:
    """Test that when direct format succeeds, nvext is not tried."""
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
                        "content": '{"name": "John", "age": 30}',
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
        },
    )

    class Person(pydanticV2BaseModel):
        name: str
        age: int

    warnings.filterwarnings("ignore", r".*not known to support structured output.*")
    llm = ChatNVIDIA(api_key="BOGUS").with_structured_output(Person)
    result = llm.invoke("test")

    # Should only make 1 POST call (direct format succeeded)
    post_calls = [req for req in requests_mock.request_history if req.method == "POST"]
    assert len(post_calls) == 1

    # Verify the call used direct format (guided_json at root level)
    request_body = post_calls[0].json()
    assert "guided_json" in request_body, "Should use direct format"
    assert "nvext" not in request_body, "Should not have nvext wrapper"

    assert isinstance(result, Person)
    assert result.name == "John"
    assert result.age == 30


def test_fallback_direct_fails_nvext_succeeds(
    requests_mock: requests_mock.Mocker,
    empty_v1_models: None,
) -> None:
    """Test that when direct format fails, it retries with nvext."""
    # First call fails (direct format), second succeeds (nvext format)
    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        [
            {
                "status_code": 400,
                "json": {
                    "error": "guided_json is unsupported at the root level",
                },
            },
            {
                "json": {
                    "id": "chatcmpl-ID",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "BOGUS",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": '{"name": "Jane", "age": 25}',
                            },
                            "logprobs": None,
                            "finish_reason": "stop",
                        }
                    ],
                }
            },
        ],
    )

    class Person(pydanticV2BaseModel):
        name: str
        age: int

    warnings.filterwarnings("ignore", r".*not known to support structured output.*")
    llm = ChatNVIDIA(api_key="BOGUS").with_structured_output(Person)
    result = llm.invoke("test")

    # Should make 2 POST calls: direct format (fails) + nvext format (succeeds)
    post_calls = [req for req in requests_mock.request_history if req.method == "POST"]
    assert len(post_calls) == 2, f"Expected 2 POST calls, got {len(post_calls)}"

    # Verify first call used direct format (guided_json at root level)
    first_body = post_calls[0].json()
    assert "guided_json" in first_body, "First call should use direct format"
    assert "nvext" not in first_body, "First call should not have nvext"

    # Verify second call used nvext format (guided_json wrapped in nvext)
    second_body = post_calls[1].json()
    assert "nvext" in second_body, "Second call should use nvext format"
    assert "guided_json" in second_body["nvext"], "nvext should contain guided_json"

    assert isinstance(result, Person)
    assert result.name == "Jane"
    assert result.age == 25
