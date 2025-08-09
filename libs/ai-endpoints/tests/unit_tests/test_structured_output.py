import enum
import warnings
from typing import Callable, List, Optional, Type

import pytest
import requests_mock
from pydantic import BaseModel as pydanticV2BaseModel  # ignore: check_pydantic
from pydantic import Field
from pydantic.v1 import BaseModel as pydanticV1BaseModel  # ignore: check_pydantic

from langchain_nvidia_ai_endpoints import ChatNVIDIA


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


# Test cases for thinking mode + structured output


# Test Pydantic schema
@pytest.mark.parametrize(
    "response_content,expected_setup,expected_punchline",
    [
        # With thinking tags
        (
            "<think>Let me think of a joke.</think>"
            '{"setup": "Setup A", "punchline": "Punchline A", "rating": 8}',
            "Setup A",
            "Punchline A",
        ),
        # Without thinking tags
        (
            '{"setup": "Setup B", "punchline": "Punchline B", "rating": null}',
            "Setup B",
            "Punchline B",
        ),
        # Multiple thinking blocks
        (
            "<think>First idea</think>Text<think>Second idea</think>"
            '{"setup": "Setup C", "punchline": "Punchline C", "rating": 7}',
            "Setup C",
            "Punchline C",
        ),
    ],
    ids=["with-thinking", "without-thinking", "multiple-thinking"],
)
def test_structured_output_thinking_mode_pydantic(
    requests_mock: requests_mock.Mocker,
    response_content: str,
    expected_setup: str,
    expected_punchline: str,
) -> None:
    """Test that structured output works with thinking mode for Pydantic schemas."""
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
                        "content": response_content,
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

    warnings.filterwarnings("ignore", r".*not known to support structured output.*")
    llm = ChatNVIDIA(api_key="BOGUS").with_structured_output(Joke)
    response = llm.invoke("Tell me a joke.")

    assert isinstance(response, Joke)
    assert response.setup == expected_setup
    assert response.punchline == expected_punchline


# Test dictionary schema
@pytest.mark.parametrize(
    "response_content,expected_result",
    [
        # With thinking tags
        (
            '<think>Thinking about response.</think>{"name": "Alice", "age": 30}',
            {"name": "Alice", "age": 30},
        ),
        # Without thinking tags
        ('{"name": "Bob", "age": 25}', {"name": "Bob", "age": 25}),
        # Multiple thinking blocks
        (
            "<think>First</think>Text<think>Second</think>"
            '{"name": "Charlie", "age": 35}',
            {"name": "Charlie", "age": 35},
        ),
    ],
    ids=["with-thinking", "without-thinking", "multiple-thinking"],
)
def test_structured_output_thinking_mode_dict(
    requests_mock: requests_mock.Mocker,
    response_content: str,
    expected_result: dict,
) -> None:
    """Test that structured output works with thinking mode for dictionary schemas."""
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
                        "content": response_content,
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

    json_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }

    warnings.filterwarnings("ignore", r".*not known to support structured output.*")
    llm = ChatNVIDIA(api_key="BOGUS").with_structured_output(json_schema)
    response = llm.invoke("Tell me about a person.")
    assert response == expected_result


# Test enum schema
class ChoiceEnum(enum.Enum):
    OPTION_A = "A"
    OPTION_B = "B"
    OPTION_C = "C"


@pytest.mark.parametrize(
    "response_content,expected_result",
    [
        # With thinking tags
        ("<think>Let me decide on option A.</think>A", ChoiceEnum.OPTION_A),
        # Without thinking tags
        ("B", ChoiceEnum.OPTION_B),
        # Multiple thinking blocks
        (
            "<think>First thought</think>Maybe<think>Final decision</think>C",
            ChoiceEnum.OPTION_C,
        ),
    ],
    ids=["with-thinking", "without-thinking", "multiple-thinking"],
)
def test_structured_output_thinking_mode_enum(
    requests_mock: requests_mock.Mocker,
    response_content: str,
    expected_result: ChoiceEnum,
) -> None:
    """Test that structured output works with thinking mode for enum schemas."""
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
                        "content": response_content,
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

    warnings.filterwarnings("ignore", r".*not known to support structured output.*")
    llm = ChatNVIDIA(api_key="BOGUS").with_structured_output(ChoiceEnum)
    response = llm.invoke("Make a choice.")

    assert isinstance(response, ChoiceEnum)
    assert response == expected_result


# Test streaming with thinking mode
def test_stream_thinking_mode(
    mock_streaming_response: Callable,
) -> None:
    mock_streaming_response(["<think>Thinking</think>A"])

    warnings.filterwarnings("ignore", r".*not known to support structured output.*")
    structured_llm = ChatNVIDIA(api_key="BOGUS").with_structured_output(ChoiceEnum)
    # chunks are progressively more complete, so we only consider the last
    for chunk in structured_llm.stream("Make a choice."):
        response = chunk
    assert isinstance(response, ChoiceEnum)
    assert response == ChoiceEnum.OPTION_A
