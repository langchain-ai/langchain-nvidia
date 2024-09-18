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
