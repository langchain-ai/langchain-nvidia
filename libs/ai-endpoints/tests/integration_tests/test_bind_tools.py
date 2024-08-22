import json
import warnings
from typing import Any, Callable, List, Literal, Optional, Union

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
)
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
# edge/negative tests:
#  19. require unknown named tool (invoke/stream only)
#  20. partial function (invoke/stream only)
#  21. not enough tokens to generate tool call
#  22. tool with no arguments
#  23. duplicate tool names
#  24. unknown tool (invoke/stream only)
# ways to specify parallel_tool_calls: (accuracy only)
#  25. invoke
#  26. stream

# todo: parallel_tool_calls w/ bind_tools
# todo: parallel_tool_calls w/ tool_choice = function
# todo: async methods
# todo: too many tools


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


@tool
def tool_no_args() -> str:
    """8-ball"""
    return "lookin' good"


@tool
def get_current_weather(
    location: str = Field(..., description="The location to get the weather for"),
    scale: Optional[str] = Field(
        default="Fahrenheit",
        description="The temperature scale (e.g., Celsius or Fahrenheit)",
    ),
) -> str:
    """Get the current weather for a location"""
    return f"The current weather in {location} is sunny."


def eval_stream(
    llm: ChatNVIDIA,
    msg: str,
    tool_choice: Any = None,
    parallel_tool_calls: bool = False,
) -> BaseMessageChunk:
    params = {}
    if tool_choice:
        params["tool_choice"] = tool_choice
    if parallel_tool_calls:
        params["parallel_tool_calls"] = True

    generator = llm.stream(msg, **params)  # type: ignore
    response = next(generator)
    for chunk in generator:
        assert isinstance(chunk, AIMessageChunk)
        response += chunk
    return response


def eval_invoke(
    llm: ChatNVIDIA,
    msg: str,
    tool_choice: Any = None,
    parallel_tool_calls: bool = False,
) -> BaseMessage:
    params = {}
    if tool_choice:
        params["tool_choice"] = tool_choice
    if parallel_tool_calls:
        params["parallel_tool_calls"] = True

    return llm.invoke(msg, **params)  # type: ignore


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


@pytest.mark.parametrize(
    "func",
    [eval_invoke, eval_stream],
    ids=["invoke", "stream"],
)
@pytest.mark.xfail(reason="Accuracy test")
def test_accuracy_extra(tool_model: str, mode: dict, func: Callable) -> None:
    llm = ChatNVIDIA(temperature=0, model=tool_model, **mode).bind_tools([xxyyzz])
    response = func(llm, "What is 11 xxyyzz 3?")
    assert not response.content  # should be `response.content is None` but
    # AIMessage.content: Union[str, List[Union[str, Dict]]] cannot be None.
    assert response.additional_kwargs is not None
    # todo: this is not good, should not care about the param
    if func == eval_invoke:
        assert isinstance(response, AIMessage)
        assert "tool_calls" in response.additional_kwargs
        assert isinstance(response.additional_kwargs["tool_calls"], list)
        assert response.additional_kwargs["tool_calls"]
        assert response.tool_calls
        for tool_call in response.additional_kwargs["tool_calls"]:
            assert "id" in tool_call
            assert tool_call["id"] is not None
            assert "type" in tool_call
            assert tool_call["type"] == "function"
            assert "function" in tool_call
        assert len(response.additional_kwargs["tool_calls"]) > 0
        tool_call = response.additional_kwargs["tool_calls"][0]
        assert tool_call["function"]["name"] == "xxyyzz"
        assert json.loads(tool_call["function"]["arguments"]) == {"a": 11, "b": 3}
    else:
        assert isinstance(response, AIMessageChunk)
        assert response.tool_call_chunks
    assert response.response_metadata is not None
    assert isinstance(response.response_metadata, dict)
    if "content" in response.response_metadata:
        assert response.response_metadata["content"] is None
    assert "model_name" in response.response_metadata
    assert response.response_metadata["model_name"] == tool_model
    assert "finish_reason" in response.response_metadata
    assert response.response_metadata["finish_reason"] in [
        "tool_calls",
        "stop",
    ]  # todo: remove "stop"


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
@pytest.mark.parametrize(
    "func",
    [eval_invoke, eval_stream],
    ids=["invoke", "stream"],
)
def test_tool_choice_with_no_tool(
    tool_model: str, mode: dict, tool_choice: Any, func: Callable
) -> None:
    llm = ChatNVIDIA(model=tool_model, **mode)
    with pytest.raises(Exception) as e:
        func(llm, "What is 11 xxyyzz 3?", tool_choice=tool_choice)
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
        or "Bad Request" in str(e.value)
    )


@pytest.mark.parametrize(
    "func",
    [eval_invoke, eval_stream],
    ids=["invoke", "stream"],
)
def test_tool_choice_none(tool_model: str, mode: dict, func: Callable) -> None:
    llm = ChatNVIDIA(model=tool_model, **mode).bind_tools(tools=[xxyyzz])
    response = func(llm, "What is 11 xxyyzz 3?", tool_choice="none")
    assert "tool_calls" not in response.additional_kwargs


@pytest.mark.parametrize(
    "tool_choice",
    [
        {"function": {"name": "xxyyzz"}},
    ],
    ids=["partial"],
)
@pytest.mark.parametrize(
    "func",
    [eval_invoke, eval_stream],
    ids=["invoke", "stream"],
)
def test_tool_choice_negative(
    tool_model: str,
    mode: dict,
    tool_choice: Optional[
        Union[dict, str, Literal["auto", "none", "any", "required"], bool]
    ],
    func: Callable,
) -> None:
    llm = ChatNVIDIA(model=tool_model, **mode).bind_tools([xxyyzz])
    with pytest.raises(Exception) as e:
        func(llm, "What is 11 xxyyzz 3?", tool_choice=tool_choice)
    assert "400" in str(e.value) or "###" in str(
        e.value
    )  # todo: stop transforming 400 -> ###
    assert (
        "invalid_request_error" in str(e.value)
        or "value_error" in str(e.value)
        or "Bad Request" in str(e.value)
        or "Incorrectly formatted `tool_choice`" in str(e.value)
    )


@pytest.mark.parametrize(
    "func",
    [eval_invoke, eval_stream],
    ids=["invoke", "stream"],
)
@pytest.mark.xfail(reason="Server side is broken")
def test_tool_choice_negative_max_tokens_required(
    tool_model: str,
    mode: dict,
    func: Callable,
) -> None:
    llm = ChatNVIDIA(max_tokens=5, model=tool_model, **mode).bind_tools([xxyyzz])
    with pytest.raises(Exception) as e:
        func(llm, "What is 11 xxyyzz 3?", tool_choice="required")
    assert "400" in str(e.value) or "###" in str(
        e.value
    )  # todo: stop transforming 400 -> ###
    assert "invalid_request_error" in str(e.value)
    assert (
        "Could not finish the message because max_tokens was reached. "
        "Please try again with higher max_tokens."
    ) in str(e.value)


@pytest.mark.parametrize(
    "func",
    [eval_invoke, eval_stream],
    ids=["invoke", "stream"],
)
@pytest.mark.xfail(reason="Server side is broken")
def test_tool_choice_negative_max_tokens_function(
    tool_model: str,
    mode: dict,
    func: Callable,
) -> None:
    llm = ChatNVIDIA(max_tokens=5, model=tool_model, **mode).bind_tools([xxyyzz])
    response = func(
        llm,
        "What is 11 xxyyzz 3?",
        tool_choice={"type": "function", "function": {"name": "xxyyzz"}},
    )
    # todo: this is not good, should not care about the param
    if func == eval_invoke:
        assert isinstance(response, AIMessage)
        assert "tool_calls" in response.additional_kwargs
        assert response.invalid_tool_calls
    else:
        assert isinstance(response, AIMessageChunk)
        assert response.tool_call_chunks
    assert "finish_reason" in response.response_metadata
    assert response.response_metadata["finish_reason"] == "length"


@pytest.mark.parametrize(
    "tool_choice",
    [
        "required",
        {"type": "function", "function": {"name": "tool_no_args"}},
    ],
    ids=["required", "function"],
)
@pytest.mark.parametrize(
    "func",
    [eval_invoke, eval_stream],
    ids=["invoke", "stream"],
)
def test_tool_choice_negative_no_args(
    tool_model: str,
    mode: dict,
    tool_choice: Optional[
        Union[dict, str, Literal["auto", "none", "any", "required"], bool]
    ],
    func: Callable,
) -> None:
    llm = ChatNVIDIA(model=tool_model, **mode).bind_tools([tool_no_args])
    response = func(llm, "What does the 8-ball say?", tool_choice=tool_choice)
    # todo: this is not good, should not care about the param
    if func == eval_invoke:
        assert isinstance(response, AIMessage)
        assert response.tool_calls
    else:
        assert isinstance(response, AIMessageChunk)
        assert response.tool_call_chunks
    # assert "tool_calls" in response.additional_kwargs


@pytest.mark.parametrize(
    "tool_choice",
    [
        "required",
        {"type": "function", "function": {"name": "tool_no_args"}},
    ],
    ids=["required", "function"],
)
@pytest.mark.parametrize(
    "func",
    [eval_invoke, eval_stream],
    ids=["invoke", "stream"],
)
@pytest.mark.xfail(reason="Accuracy test")
def test_accuracy_tool_choice_negative_no_args(
    tool_model: str,
    mode: dict,
    tool_choice: Optional[
        Union[dict, str, Literal["auto", "none", "any", "required"], bool]
    ],
    func: Callable,
) -> None:
    llm = ChatNVIDIA(model=tool_model, **mode).bind_tools([tool_no_args])
    response = func(llm, "What does the 8-ball say?", tool_choice=tool_choice)
    assert isinstance(response, AIMessage)
    # assert "tool_calls" in response.additional_kwargs
    assert response.tool_calls
    assert response.tool_calls[0]["name"] == "tool_no_args"
    assert response.tool_calls[0]["args"] == {}


@pytest.mark.parametrize(
    "tool_choice",
    [
        "required",
        {"type": "function", "function": {"name": "xxyyzz"}},
    ],
    ids=["required", "function"],
)
@pytest.mark.parametrize(
    "func",
    [eval_invoke, eval_stream],
    ids=["invoke", "stream"],
)
def test_tool_choice_negative_duplicate_tool(
    tool_model: str,
    mode: dict,
    tool_choice: Optional[
        Union[dict, str, Literal["auto", "none", "any", "required"], bool]
    ],
    func: Callable,
) -> None:
    llm = ChatNVIDIA(model=tool_model, **mode).bind_tools([xxyyzz, xxyyzz])
    response = func(llm, "What is 11 xxyyzz 3?", tool_choice=tool_choice)
    assert isinstance(response, AIMessage)
    assert response.tool_calls
    # assert "tool_calls" in response.additional_kwargs


@pytest.mark.parametrize(
    "tool_choice",
    [
        "required",
        {"type": "function", "function": {"name": "xxyyzz"}},
    ],
    ids=["required", "function"],
)
@pytest.mark.parametrize(
    "func",
    [eval_invoke, eval_stream],
    ids=["invoke", "stream"],
)
def test_tool_choice(
    tool_model: str,
    mode: dict,
    tool_choice: Optional[
        Union[dict, str, Literal["auto", "none", "any", "required"], bool]
    ],
    func: Callable,
) -> None:
    llm = ChatNVIDIA(model=tool_model, **mode).bind_tools([xxyyzz])
    response = func(llm, "What is 11 xxyyzz 3?", tool_choice=tool_choice)
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
@pytest.mark.parametrize(
    "func",
    [eval_invoke, eval_stream],
    ids=["invoke", "stream"],
)
@pytest.mark.xfail(reason="Accuracy test")
def test_accuracy_tool_choice(
    tool_model: str,
    mode: dict,
    tools: List,
    tool_choice: Any,
    func: Callable,
) -> None:
    llm = ChatNVIDIA(temperature=0, model=tool_model, **mode).bind_tools(tools)
    response = func(llm, "What is 11 xxyyzz 3?", tool_choice=tool_choice)
    assert isinstance(response, AIMessage)
    check_response_structure(response)
    tool_call = response.tool_calls[0]
    assert tool_call["name"] == "xxyyzz"
    assert tool_call["args"] == {"b": 3, "a": 11}


@pytest.mark.parametrize(
    "func",
    [eval_invoke, eval_stream],
    ids=["invoke", "stream"],
)
def test_tool_choice_negative_unknown_tool(
    tool_model: str,
    mode: dict,
    func: Callable,
) -> None:
    llm = ChatNVIDIA(model=tool_model, **mode).bind_tools(tools=[xxyyzz])
    with pytest.raises(Exception) as e:
        func(
            llm,
            "What is 11 xxyyzz 3?",
            tool_choice={"type": "function", "function": {"name": "zzyyxx"}},
        )
    assert (
        "not found in the tools list" in str(e.value)
        or "no function named" in str(e.value)
        or "does not match any of the specified" in str(e.value)
        or "Bad Request" in str(e.value)
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
        or "Bad Request" in str(e.value)
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
    assert isinstance(response, AIMessage)
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
    candidates = [
        model for model in ChatNVIDIA.get_available_models() if not model.supports_tools
    ]
    assert candidates, "All models support tools"
    with pytest.warns(UserWarning) as record:
        ChatNVIDIA(model=candidates[0].id, **mode).bind_tools([xxyyzz])
    assert len(record) == 1
    assert "not known to support tools" in str(record[0].message)


@pytest.mark.parametrize(
    "tool_choice",
    [
        "auto",
        None,
        "required",
    ],
    ids=["auto", "absent", "required"],
)
@pytest.mark.parametrize(
    "tools",
    [[xxyyzz, zzyyxx], [zzyyxx, xxyyzz]],
    ids=["xxyyzz_and_zzyyxx", "zzyyxx_and_xxyyzz"],
)
@pytest.mark.parametrize(
    "func",
    [eval_invoke, eval_stream],
    ids=["invoke", "stream"],
)
@pytest.mark.xfail(reason="Accuracy test")
def test_accuracy_parallel_tool_calls_hard(
    tool_model: str,
    mode: dict,
    tools: List,
    tool_choice: Any,
    func: Callable,
) -> None:
    llm = ChatNVIDIA(seed=42, temperature=1, model=tool_model, **mode).bind_tools(tools)
    response = func(
        llm,
        "What is 11 xxyyzz 3 zzyyxx 5?",
        tool_choice=tool_choice,
        parallel_tool_calls=True,
    )
    assert isinstance(response, AIMessage)
    check_response_structure(response)
    assert len(response.tool_calls) == 2
    valid_tool_names = ["xxyyzz", "zzyyxx"]
    tool_call0 = response.tool_calls[0]
    assert tool_call0["name"] in valid_tool_names
    valid_tool_names.remove(tool_call0["name"])
    tool_call1 = response.tool_calls[1]
    assert tool_call1["name"] in valid_tool_names


@pytest.mark.parametrize(
    "tool_choice",
    [
        "auto",
        None,
        "required",
    ],
    ids=["auto", "absent", "required"],
)
@pytest.mark.parametrize(
    "func",
    [eval_invoke, eval_stream],
    ids=["invoke", "stream"],
)
@pytest.mark.xfail(reason="Accuracy test")
def test_accuracy_parallel_tool_calls_easy(
    tool_model: str,
    mode: dict,
    tool_choice: Any,
    func: Callable,
) -> None:
    llm = ChatNVIDIA(seed=42, temperature=1, model=tool_model, **mode).bind_tools(
        tools=[get_current_weather],
    )
    response = func(
        llm,
        "What is the weather in Boston, and what is the weather in Dublin?",
        tool_choice=tool_choice,
        parallel_tool_calls=True,
    )
    assert isinstance(response, AIMessage)
    check_response_structure(response)
    assert len(response.tool_calls) == 2
    valid_args = ["Boston", "Dublin"]
    tool_call0 = response.tool_calls[0]
    assert tool_call0["name"] == "get_current_weather"
    assert tool_call0["args"]["location"] in valid_args
    valid_args.remove(tool_call0["args"]["location"])
    tool_call1 = response.tool_calls[1]
    assert tool_call1["name"] == "get_current_weather"
    assert tool_call1["args"]["location"] in valid_args
