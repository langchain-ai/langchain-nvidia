import json
import warnings
from typing import Any, List, Literal, Optional, Union

import pytest
from langchain_core.messages import AIMessage, ChatMessage
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import tool

from langchain_nvidia_ai_endpoints import ChatNVIDIA

#
# ways to specify tools:
#   0. bind_tools
# ways to specify tool_choice:
#   1. invoke
#   2. bind_tools
#   3. stream
# tool_choice levels:
#   4. "none"
#   5. "auto" (accuracy only)
#   6. None (accuracy only)
#   7. "required"
#   8. {"function": {"name": tool name}} (partial function)
#   9. {"type": "function", "function": {"name": tool name}}
#  10. "any" (bind_tools only)
#  11. tool name (bind_tools only)
#  12. True (bind_tools only)
#  13. False (bind_tools only)
# tools levels:
#  14. no tools
#  15. one tool
#  16. multiple tools (accuracy only)
# test types:
#  17. deterministic (minimial accuracy tests; relies on basic tool calling skills)
#  18. accuracy (proper tool; proper arguments)
# negative tests:
#  19. require unknown named tool (invoke/stream only)
#  20. partial function (invoke/stream only)
#

# todo: streaming
# todo: test tool with no arguments
# todo: parallel_tool_calls


@tool
def xxyyzz(
    a: int = Field(..., description="First number"),
    b: int = Field(..., description="Second number"),
) -> int:
    """xxyyzz two numbers"""
    return (a**b) % (b - a)


@tool
def zzyyxx(
    a: int = Field(..., description="First number"),
    b: int = Field(..., description="Second number"),
) -> int:
    """zzyyxx two numbers"""
    return (b**a) % (a - b)


def check_response_structure(response: AIMessage) -> None:
    assert not response.content  # should be `response.content is None` but
    # AIMessage.content: Union[str, List[Union[str, Dict]]] cannot be None.
    for tool_call in response.tool_calls:
        assert tool_call["id"] is not None
    assert response.response_metadata is not None
    assert isinstance(response.response_metadata, dict)
    assert "finish_reason" in response.response_metadata
    assert response.response_metadata["finish_reason"] in [
        "tool_calls",
        "stop",
    ]  # todo: remove "stop"
    assert len(response.tool_calls) > 0


# users can also get at the tool calls from the response.additional_kwargs
@pytest.mark.xfail(reason="Accuracy test")
def test_accuracy_default_invoke_additional_kwargs(tool_model: str, mode: dict) -> None:
    llm = ChatNVIDIA(temperature=0, model=tool_model, **mode).bind_tools([xxyyzz])
    response = llm.invoke("What is 11 xxyyzz 3?")
    assert not response.content  # should be `response.content is None` but
    # AIMessage.content: Union[str, List[Union[str, Dict]]] cannot be None.
    assert response.additional_kwargs is not None
    assert "tool_calls" in response.additional_kwargs
    assert isinstance(response.additional_kwargs["tool_calls"], list)
    assert response.additional_kwargs["tool_calls"]
    for tool_call in response.additional_kwargs["tool_calls"]:
        assert "id" in tool_call
        assert tool_call["id"] is not None
        assert "type" in tool_call
        assert tool_call["type"] == "function"
        assert "function" in tool_call
    assert response.response_metadata is not None
    assert isinstance(response.response_metadata, dict)
    assert "content" in response.response_metadata
    assert response.response_metadata["content"] is None
    assert "finish_reason" in response.response_metadata
    assert response.response_metadata["finish_reason"] in [
        "tool_calls",
        "stop",
    ]  # todo: remove "stop"
    assert len(response.additional_kwargs["tool_calls"]) > 0
    tool_call = response.additional_kwargs["tool_calls"][0]
    assert tool_call["function"]["name"] == "xxyyzz"
    assert json.loads(tool_call["function"]["arguments"]) == {"a": 11, "b": 3}


@pytest.mark.parametrize(
    "tool_choice",
    [
        "none",
        "required",
        {"function": {"name": "xxyyzz"}},
        {"type": "function", "function": {"name": "xxyyzz"}},
    ],
    ids=["none", "required", "partial", "function"],
)
def test_invoke_tool_choice_with_no_tool(
    tool_model: str, mode: dict, tool_choice: Any
) -> None:
    llm = ChatNVIDIA(model=tool_model, **mode)
    with pytest.raises(Exception) as e:
        llm.invoke("What is 11 xxyyzz 3?", tool_choice=tool_choice)
    assert "400" in str(e.value) or "###" in str(
        e.value
    )  # todo: stop transforming 400 -> ###
    assert (
        "Value error, When using `tool_choice`, `tools` must be set." in str(e.value)
        or (
            "Value error, Invalid value for `tool_choice`: `tool_choice` is only "
            "allowed when `tools` are specified."
        )
        in str(e.value)
        or "invalid_request_error" in str(e.value)
    )


def test_invoke_tool_choice_none(tool_model: str, mode: dict) -> None:
    llm = ChatNVIDIA(model=tool_model, **mode).bind_tools(tools=[xxyyzz])
    response = llm.invoke("What is 11 xxyyzz 3?", tool_choice="none")  # type: ignore
    assert isinstance(response, ChatMessage)
    assert "tool_calls" not in response.additional_kwargs


@pytest.mark.parametrize(
    "tool_choice",
    [
        {"function": {"name": "xxyyzz"}},
    ],
    ids=["partial"],
)
def test_invoke_tool_choice_negative(
    tool_model: str,
    mode: dict,
    tool_choice: Optional[
        Union[dict, str, Literal["auto", "none", "any", "required"], bool]
    ],
) -> None:
    llm = ChatNVIDIA(model=tool_model, **mode).bind_tools([xxyyzz])
    with pytest.raises(Exception) as e:
        llm.invoke("What is 11 xxyyzz 3?", tool_choice=tool_choice)  # type: ignore
    assert "400" in str(e.value) or "###" in str(
        e.value
    )  # todo: stop transforming 400 -> ###
    assert "invalid_request_error" in str(e.value) or "value_error" in str(e.value)


@pytest.mark.parametrize(
    "tool_choice",
    [
        "required",
        {"type": "function", "function": {"name": "xxyyzz"}},
    ],
    ids=["required", "function"],
)
def test_invoke_tool_choice(
    tool_model: str,
    mode: dict,
    tool_choice: Optional[
        Union[dict, str, Literal["auto", "none", "any", "required"], bool]
    ],
) -> None:
    llm = ChatNVIDIA(model=tool_model, **mode).bind_tools([xxyyzz])
    response = llm.invoke("What is 11 xxyyzz 3?", tool_choice=tool_choice)  # type: ignore
    assert isinstance(response, AIMessage)
    check_response_structure(response)


@pytest.mark.parametrize(
    "tool_choice",
    [
        "auto",
        None,
        "required",
        {"type": "function", "function": {"name": "xxyyzz"}},
    ],
    ids=["auto", "absent", "required", "function"],
)
@pytest.mark.parametrize(
    "tools",
    [[xxyyzz], [xxyyzz, zzyyxx], [zzyyxx, xxyyzz]],
    ids=["xxyyzz", "xxyyzz_and_zzyyxx", "zzyyxx_and_xxyyzz"],
)
@pytest.mark.xfail(reason="Accuracy test")
def test_accuracy_invoke_tool_choice(
    tool_model: str,
    mode: dict,
    tools: List,
    tool_choice: Any,
) -> None:
    llm = ChatNVIDIA(temperature=0, model=tool_model, **mode).bind_tools(tools)
    response = llm.invoke("What is 11 xxyyzz 3?", tool_choice=tool_choice)  # type: ignore
    assert isinstance(response, AIMessage)
    check_response_structure(response)
    tool_call = response.tool_calls[0]
    assert tool_call["name"] == "xxyyzz"
    assert tool_call["args"] == {"b": 3, "a": 11}


def test_invoke_tool_choice_with_unknown_tool(tool_model: str, mode: dict) -> None:
    llm = ChatNVIDIA(model=tool_model, **mode).bind_tools(tools=[xxyyzz])
    with pytest.raises(Exception) as e:
        llm.invoke(
            "What is 11 xxyyzz 3?",
            tool_choice={"type": "function", "function": {"name": "zzyyxx"}},
        )  # type: ignore
    assert (
        "not found in the tools list" in str(e.value)
        or "no function named" in str(e.value)
        or "does not match any of the specified" in str(e.value)
    )


@pytest.mark.parametrize(
    "tool_choice",
    [
        {"function": {"name": "xxyyzz"}},
        {"type": "function", "function": {"name": "xxyyzz"}},
        "xxyyzz",
    ],
    ids=["partial", "function", "name"],
)
def test_bind_tool_tool_choice_with_no_tool_client(
    tool_model: str,
    mode: dict,
    tool_choice: Optional[
        Union[dict, str, Literal["auto", "none", "any", "required"], bool]
    ],
) -> None:
    with pytest.raises(ValueError) as e:
        ChatNVIDIA(model=tool_model, **mode).bind_tools(
            tools=[], tool_choice=tool_choice
        )
    assert "not found in the tools list" in str(e.value)


@pytest.mark.parametrize(
    "tool_choice",
    [
        "none",
        "required",
        "any",
        True,
        False,
    ],
    ids=["none", "required", "any", "True", "False"],
)
def test_bind_tool_tool_choice_with_no_tool_server(
    tool_model: str, mode: dict, tool_choice: Any
) -> None:
    llm = ChatNVIDIA(model=tool_model, **mode).bind_tools([], tool_choice=tool_choice)
    with pytest.raises(Exception) as e:
        llm.invoke("What is 11 xxyyzz 3?")
    assert "400" in str(e.value) or "###" in str(
        e.value
    )  # todo: stop transforming 400 -> ###
    assert (
        "Value error, When using `tool_choice`, `tools` must be set." in str(e.value)
        or (
            "Value error, Invalid value for `tool_choice`: `tool_choice` is only "
            "allowed when `tools` are specified."
        )
        in str(e.value)
        or "Expected an array with minimum length" in str(e.value)
        or "should be non-empty" in str(e.value)
    )


@pytest.mark.parametrize(
    "tool_choice",
    ["none", False],
)
def test_bind_tool_tool_choice_none(
    tool_model: str, mode: dict, tool_choice: Any
) -> None:
    llm = ChatNVIDIA(model=tool_model, **mode).bind_tools(
        tools=[xxyyzz], tool_choice=tool_choice
    )
    response = llm.invoke("What is 11 xxyyzz 3?")
    assert isinstance(response, ChatMessage)
    assert "tool_calls" not in response.additional_kwargs


@pytest.mark.parametrize(
    "tool_choice",
    [
        "required",
        {"function": {"name": "xxyyzz"}},
        {"type": "function", "function": {"name": "xxyyzz"}},
        "any",
        "xxyyzz",
        True,
    ],
    ids=["required", "partial", "function", "any", "name", "True"],
)
def test_bind_tool_tool_choice(
    tool_model: str,
    mode: dict,
    tool_choice: Optional[
        Union[dict, str, Literal["auto", "none", "any", "required"], bool]
    ],
) -> None:
    llm = ChatNVIDIA(model=tool_model, **mode).bind_tools(
        [xxyyzz], tool_choice=tool_choice
    )
    response = llm.invoke("What is 11 xxyyzz 3?")
    assert isinstance(response, AIMessage)
    check_response_structure(response)


@pytest.mark.parametrize(
    "tool_choice",
    [
        "auto",
        None,
        "required",
        {"function": {"name": "xxyyzz"}},
        {"type": "function", "function": {"name": "xxyyzz"}},
        "any",
        "xxyyzz",
        True,
    ],
    ids=["auto", "absent", "required", "partial", "function", "any", "name", "True"],
)
@pytest.mark.parametrize(
    "tools",
    [[xxyyzz], [xxyyzz, zzyyxx], [zzyyxx, xxyyzz]],
    ids=["xxyyzz", "xxyyzz_and_zzyyxx", "zzyyxx_and_xxyyzz"],
)
@pytest.mark.xfail(reason="Accuracy test")
def test_accuracy_bind_tool_tool_choice(
    tool_model: str,
    mode: dict,
    tools: List,
    tool_choice: Any,
) -> None:
    llm = ChatNVIDIA(temperature=0, model=tool_model, **mode).bind_tools(
        tools=tools, tool_choice=tool_choice
    )
    response = llm.invoke("What is 11 xxyyzz 3?")
    assert isinstance(response, AIMessage)
    check_response_structure(response)
    tool_call = response.tool_calls[0]
    assert tool_call["name"] == "xxyyzz"
    assert tool_call["args"] == {"b": 3, "a": 11}


def test_known_does_not_warn(tool_model: str, mode: dict) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ChatNVIDIA(model=tool_model, **mode).bind_tools([xxyyzz])


def test_unknown_warns(mode: dict) -> None:
    with pytest.warns(UserWarning) as record:
        ChatNVIDIA(model="mock-model", **mode).bind_tools([xxyyzz])
    assert len(record) == 1
    assert "not known to support tools" in str(record[0].message)
