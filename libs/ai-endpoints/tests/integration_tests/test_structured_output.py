import enum
from typing import Any, Callable, Optional, Union

import pytest
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from langchain_nvidia_ai_endpoints import ChatNVIDIA


def do_invoke(llm: ChatNVIDIA, message: str) -> Any:
    return llm.invoke(message)


def do_stream(llm: ChatNVIDIA, message: str) -> Any:
    # the way streaming works is to progressively grow the response
    # so we just return the last chunk. this is different from other
    # streaming results, which are *Chunks that can be concatenated.
    result = [chunk for chunk in llm.stream(message)]
    return result[-1] if result else None


@pytest.mark.xfail(reason="Accuracy is not guaranteed")
def test_accuracy(structured_model: str, mode: dict) -> None:
    class Person(BaseModel):
        name: str = Field(description="The name of the person")
        age: Optional[int] = Field(description="The age of the person")
        birthdate: Optional[str] = Field(description="The birthdate of the person")
        occupation: Optional[str] = Field(description="The occupation of the person")
        birthplace: Optional[str] = Field(description="The birthplace of the person")

    messages = [
        HumanMessage(
            """
        Jen-Hsun Huang was born in Tainan, Taiwan, on February 17, 1963. His family
        moved to Thailand when he was five; when he was nine, he and his brother were
        sent to the United States to live with an uncle in Tacoma, Washington. When he
        was ten, he lived in the boys' dormitory with his brother at Oneida Baptist
        Institute while attending Oneida Elementary school in Oneida, Kentucky—his
        uncle had mistaken what was actually a religious reform academy for a
        prestigious boarding school. Several years later, their parents also moved to
        the United States and settled in Oregon, where Huang graduated from Aloha
        High School in Aloha, Oregon. He skipped two years and graduated at sixteen.
        While growing up in Oregon in the 1980s, Huang got his first job at a local
        Denny's restaurant, where he worked as a busboy and waiter.
        Huang received his undergraduate degree in electrical engineering from Oregon
        State University in 1984, and his master's degree in electrical engineering
        from Stanford University in 1992.

        The current date is July 2034.
        """
        ),
        HumanMessage("Who is Jensen?"),
    ]

    llm = ChatNVIDIA(model=structured_model, **mode)
    structured_llm = llm.with_structured_output(Person)
    person = structured_llm.invoke(messages)
    assert isinstance(person, Person)
    assert person.name in ["Jen-Hsun Huang", "Jensen"]
    # assert person.age == 71  # this is too hard
    assert person.birthdate == "February 17, 1963"
    assert person.occupation and (
        "founder" in person.occupation.lower() or "CEO" in person.occupation.upper()
    )
    assert person.birthplace == "Tainan, Taiwan"


class Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline to the joke")
    rating: Optional[int] = Field(description="How funny the joke is, from 1 to 10")


@pytest.mark.parametrize("func", [do_invoke, do_stream], ids=["invoke", "stream"])
def test_pydantic(structured_model: str, mode: dict, func: Callable) -> None:
    llm = ChatNVIDIA(model=structured_model, temperature=0, **mode)
    structured_llm = llm.with_structured_output(Joke)
    result = func(structured_llm, "Tell me a joke about cats")
    assert isinstance(result, Joke)


@pytest.mark.parametrize("func", [do_invoke, do_stream], ids=["invoke", "stream"])
def test_dict(structured_model: str, mode: dict, func: Callable) -> None:
    json_schema = {
        "title": "joke",
        "description": "Joke to tell user.",
        "type": "object",
        "properties": {
            "setup": {
                "type": "string",
                "description": "The setup of the joke",
            },
            "punchline": {
                "type": "string",
                "description": "The punchline to the joke",
            },
            "rating": {
                "type": "integer",
                "description": "How funny the joke is, from 1 to 10",
            },
        },
        "required": ["setup", "punchline"],
    }

    llm = ChatNVIDIA(model=structured_model, temperature=0, **mode)
    structured_llm = llm.with_structured_output(json_schema)
    result = func(structured_llm, "Tell me a joke about cats")
    assert isinstance(result, dict)
    assert "setup" in result
    assert "punchline" in result


@pytest.mark.parametrize("func", [do_invoke, do_stream], ids=["invoke", "stream"])
def test_enum(structured_model: str, mode: dict, func: Callable) -> None:
    class Choices(enum.Enum):
        A = "A is an option"
        B = "B is an option"
        C = "C is an option"

    llm = ChatNVIDIA(model=structured_model, temperature=0, **mode)
    structured_llm = llm.with_structured_output(Choices)
    result = func(
        structured_llm,
        """
        What does 1+1 equal?
            A. -100
            B. 2
            C. doorstop
        """,
    )
    assert isinstance(result, Choices)
    assert result in Choices


@pytest.mark.parametrize("func", [do_invoke, do_stream], ids=["invoke", "stream"])
def test_enum_incomplete(structured_model: str, mode: dict, func: Callable) -> None:
    class Choices(enum.Enum):
        A = "A is an option you can pick"
        B = "B is an option you can pick"
        C = "C is an option you can pick"

    llm = ChatNVIDIA(model=structured_model, temperature=0, max_tokens=3, **mode)
    structured_llm = llm.with_structured_output(Choices)
    result = func(
        structured_llm,
        """
        What does 1+1 equal?
            A. -100
            B. 2
            C. doorstop
        """,
    )
    assert result is None


@pytest.mark.parametrize("func", [do_invoke, do_stream], ids=["invoke", "stream"])
def test_multiple_schema(structured_model: str, mode: dict, func: Callable) -> None:
    class ConversationalResponse(BaseModel):
        """Respond in a conversational manner. Be kind and helpful."""

        response: str = Field(
            description="A conversational response to the user's query"
        )

    class Response(BaseModel):
        output: Union[Joke, ConversationalResponse]

    llm = ChatNVIDIA(model=structured_model, temperature=0, **mode)
    structured_llm = llm.with_structured_output(Response)
    response = func(structured_llm, "Tell me a joke about cats")
    assert isinstance(response, Response)
    assert isinstance(response.output, Joke) or isinstance(
        response.output, ConversationalResponse
    )


@pytest.mark.parametrize("func", [do_invoke, do_stream], ids=["invoke", "stream"])
def test_pydantic_incomplete(structured_model: str, mode: dict, func: Callable) -> None:
    # 3 tokens is not enough to construct a Joke
    llm = ChatNVIDIA(model=structured_model, temperature=0, max_tokens=3, **mode)
    structured_llm = llm.with_structured_output(Joke)
    result = func(structured_llm, "Tell me a joke about cats")
    assert result is None
