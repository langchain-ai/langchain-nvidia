"""Test ChatNVIDIA chat model."""

from typing import Callable, List, Union

import aiohttp
import pytest
import requests
from langchain_core.load.dump import dumps
from langchain_core.load.load import loads
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from langchain_nvidia_ai_endpoints.chat_models import ChatNVIDIA

#
# we setup an --all-models flag in conftest.py, when passed it configures chat_model
# and image_in_model to be all available models of type chat or image_in
#
# note: currently --all-models only works with the default mode because different
#       modes may have different available models
#


def assert_response_error(
    response: Union[requests.Response, aiohttp.ClientResponse],
    expected_status: List[int],
    validate_content: Callable[[str], bool],
) -> None:
    """Assert error response for both sync and async.

    Args:
        response: HTTP response object (requests.Response for sync,
            aiohttp.ClientResponse for async)
        expected_status: List of acceptable HTTP error status codes
        validate_content: Function that takes response content string and
            returns True if valid
    """
    if isinstance(response, aiohttp.ClientResponse):
        assert response.status in expected_status
        # Content is cached in _cached_content after being read in _try_raise_async
        content_str = str(response._cached_content)  # type: ignore[attr-defined]
    else:
        assert response.status_code in expected_status
        content_str = str(response.content)

    assert validate_content(content_str)


@pytest.mark.parametrize(
    "func",
    ["invoke", "ainvoke"],
)
async def test_chat_ai_endpoints(chat_model: str, mode: dict, func: str) -> None:
    """Test ChatNVIDIA wrapper."""
    chat = ChatNVIDIA(model=chat_model, temperature=0.7, **mode)
    message = HumanMessage(content="Hello")

    if func == "invoke":
        response = chat.invoke([message])
    else:  # ainvoke
        response = await chat.ainvoke([message])

    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)
    # compatibility test for ChatMessage (pre 0.2)
    # assert isinstance(response, ChatMessage)
    assert hasattr(response, "role")
    assert response.role == "assistant"


def test_unknown_model() -> None:
    with pytest.warns(UserWarning, match="Model unknown_model is unknown"):
        ChatNVIDIA(model="unknown_model")


@pytest.mark.parametrize(
    "func",
    ["invoke", "ainvoke"],
)
async def test_chat_ai_endpoints_system_message(
    chat_model: str, mode: dict, func: str
) -> None:
    """Test wrapper with system message."""
    # mamba_chat only supports 'user' or 'assistant' messages -
    #  Exception: [422] Unprocessable Entity
    #  body -> messages -> 0 -> role
    #    Input should be 'user' or 'assistant'
    #     (type=literal_error; expected='user' or 'assistant')
    if chat_model == "mamba_chat":
        pytest.skip(f"{chat_model} does not support system messages")

    chat = ChatNVIDIA(model=chat_model, max_tokens=36, **mode)
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")

    if func == "invoke":
        response = chat.invoke([system_message, human_message])
    else:  # ainvoke
        response = await chat.ainvoke([system_message, human_message])

    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


@pytest.mark.parametrize(
    "exchange",
    [
        pytest.param([], id="no_message"),
        pytest.param([HumanMessage(content="Hello")], id="single_human_message"),
        pytest.param([AIMessage(content="Hi")], id="single_ai_message"),
        pytest.param(
            [HumanMessage(content="Hello"), HumanMessage(content="Hello")],
            id="double_human_message",
            marks=pytest.mark.xfail(
                reason="Known issue, messages types must alternate"
            ),
        ),
        pytest.param(
            [AIMessage(content="Hi"), AIMessage(content="Hi")],
            id="double_ai_message",
            marks=pytest.mark.xfail(reason="Known issue, message types must alternate"),
        ),
        pytest.param(
            [HumanMessage(content="Hello"), AIMessage(content="Hi")],
            id="human_ai_message",
        ),
        pytest.param(
            [AIMessage(content="Hi"), HumanMessage(content="Hello")],
            id="ai_human_message",
        ),
        pytest.param(
            [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi"),
                HumanMessage(content="There"),
            ],
            id="human_ai_human_message",
        ),
        pytest.param(
            [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi"),
                HumanMessage(content="There"),
                AIMessage(content="Ok"),
            ],
            id="human_ai_human_ai_message",
        ),
        pytest.param(
            [
                HumanMessage(content="Hello"),
                AIMessage(content="Hi"),
                HumanMessage(content="There"),
                AIMessage(content="Ok"),
                HumanMessage(content="Now what?"),
            ],
            id="human_ai_human_ai_human_message",
        ),
    ],
)
@pytest.mark.parametrize(
    "system",
    [
        pytest.param([], id="no_system_message"),  # no system message
        pytest.param(
            [SystemMessage(content="You are to chat with the user.")],
            id="single_system_message",
        ),
    ],
)
@pytest.mark.xfail(
    reason=(
        "not all endpoints support system messages, "
        "repeated message types or ending with an ai message"
    )
)
def test_messages(
    chat_model: str, mode: dict, system: List, exchange: List[BaseMessage]
) -> None:
    if not system and not exchange:
        pytest.skip("No messages to test")
    chat = ChatNVIDIA(model=chat_model, max_tokens=36, **mode)
    response = chat.invoke(system + exchange)
    assert isinstance(response, BaseMessage)
    assert response.response_metadata["role"] == "assistant"
    assert isinstance(response.content, str)


## TODO: Not sure if we want to support the n syntax. Trash or keep test


def test_ai_endpoints_streaming(chat_model: str, mode: dict) -> None:
    """Test streaming tokens from ai endpoints."""
    llm = ChatNVIDIA(model=chat_model, max_tokens=36, **mode)

    generator = llm.stream("Count to 100, e.g. 1 2 3 4")
    response = next(generator)
    cnt = 0
    for chunk in generator:
        assert isinstance(chunk.content, str)
        response += chunk
        cnt += 1
    assert cnt > 1, response
    # compatibility test for ChatMessageChunk (pre 0.2)
    # assert hasattr(response, "role")
    # assert response.role == "assistant"  # does not work, role not passed through


async def test_ai_endpoints_astream(chat_model: str, mode: dict) -> None:
    """Test streaming tokens from ai endpoints."""
    llm = ChatNVIDIA(model=chat_model, max_tokens=35, **mode)

    generator = llm.astream("Count to 100, e.g. 1 2 3 4")
    response = (
        await generator.__anext__()
    )  # todo: use anext(generator) when py3.8 is dropped
    cnt = 0
    async for chunk in generator:
        assert isinstance(chunk.content, str)
        response += chunk
        cnt += 1
    assert cnt > 1, response


async def test_ai_endpoints_abatch(chat_model: str, mode: dict) -> None:
    """Test streaming tokens."""
    llm = ChatNVIDIA(model=chat_model, max_tokens=36, **mode)

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_ai_endpoints_abatch_tags(chat_model: str, mode: dict) -> None:
    """Test batch tokens."""
    llm = ChatNVIDIA(model=chat_model, max_tokens=55, **mode)

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


def test_ai_endpoints_batch(chat_model: str, mode: dict) -> None:
    """Test batch tokens."""
    llm = ChatNVIDIA(model=chat_model, max_tokens=60, **mode)

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_ai_endpoints_ainvoke(chat_model: str, mode: dict) -> None:
    """Test invoke tokens."""
    llm = ChatNVIDIA(model=chat_model, max_tokens=60, **mode)

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


def test_ai_endpoints_invoke(chat_model: str, mode: dict) -> None:
    """Test invoke tokens."""
    llm = ChatNVIDIA(model=chat_model, max_tokens=60, **mode)

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)


# todo: max_tokens test for batch, abatch, stream, astream


@pytest.mark.parametrize("max_tokens", [-100, 0])
@pytest.mark.parametrize(
    "func",
    ["invoke", "ainvoke"],
)
async def test_ai_endpoints_invoke_max_tokens_negative_a(
    chat_model: str,
    mode: dict,
    max_tokens: int,
    func: str,
) -> None:
    """Test invoke's max_tokens' negative bounds (caught by client-side validation)."""
    with pytest.raises(Exception):
        llm = ChatNVIDIA(model=chat_model, max_tokens=max_tokens, **mode)
        if func == "invoke":
            llm.invoke("Show me the tokens")
        else:
            await llm.ainvoke("Show me the tokens")


@pytest.mark.parametrize("max_tokens", [2**31 - 1])
@pytest.mark.parametrize(
    "func",
    ["invoke", "ainvoke"],
)
async def test_ai_endpoints_invoke_max_tokens_negative_b(
    chat_model: str,
    mode: dict,
    max_tokens: int,
    func: str,
) -> None:
    """Test invoke's max_tokens' positive bounds."""
    with pytest.raises(Exception):
        llm = ChatNVIDIA(model=chat_model, max_tokens=max_tokens, **mode)
        if func == "invoke":
            llm.invoke("Show me the tokens")
        else:
            await llm.ainvoke("Show me the tokens")
    assert llm._client.last_response is not None
    # custom error string -
    #    model inference failed -- ValueError: A requested length of the model output
    #    is too big. Maximum allowed output length is X, whereas requested output
    #    length is Y.
    #  or
    #    body -> max_tokens
    #    Input should be less than or equal to 2048 (type=less_than_equal; le=2048)
    assert_response_error(
        llm._client.last_response,
        [400, 422],
        lambda content: "length" in content
        or ("max_tokens" in content and "less_than_equal" in content),
    )


@pytest.mark.parametrize(
    "func",
    ["invoke", "ainvoke"],
)
async def test_ai_endpoints_invoke_max_tokens_positive(
    chat_model: str, mode: dict, func: str, max_tokens: int = 21
) -> None:
    """Test invoke's max_tokens."""

    llm = ChatNVIDIA(model=chat_model, max_tokens=max_tokens, **mode)

    if func == "invoke":
        result = llm.invoke("Show me the tokens")
    else:  # ainvoke
        result = await llm.ainvoke("Show me the tokens")

    assert isinstance(result.content, str)
    assert "token_usage" in result.response_metadata
    assert "completion_tokens" in result.response_metadata["token_usage"]
    assert result.response_metadata["token_usage"]["completion_tokens"] <= max_tokens


# todo: seed test for ainvoke, batch, abatch, stream, astream


@pytest.mark.xfail(reason="seed does not consistently control determinism")
@pytest.mark.parametrize(
    "func",
    ["invoke", "ainvoke"],
)
async def test_ai_endpoints_invoke_seed_default(
    chat_model: str, mode: dict, func: str
) -> None:
    """Test invoke's seed (default)."""
    llm0 = ChatNVIDIA(model=chat_model, **mode)  # default seed should not repeat
    if func == "ainvoke":
        result0 = await llm0.ainvoke("What's in a seed?")
    else:
        result0 = llm0.invoke("What's in a seed?")
    assert isinstance(result0.content, str)
    llm1 = ChatNVIDIA(model=chat_model, **mode)  # default seed should not repeat
    if func == "ainvoke":
        result1 = await llm1.ainvoke("What's in a seed?")
    else:
        result1 = llm1.invoke("What's in a seed?")
    assert isinstance(result1.content, str)
    # if this fails, consider setting a high temperature to avoid deterministic results
    assert result0.content != result1.content


@pytest.mark.parametrize(
    "seed",
    [
        pytest.param(
            -1000,
            marks=pytest.mark.xfail(reason="Known issue, negative seed not supported"),
        ),
        0,
        1000,
    ],
)
@pytest.mark.parametrize(
    "func",
    ["invoke", "ainvoke"],
)
async def test_ai_endpoints_invoke_seed_range(
    chat_model: str, mode: dict, seed: int, func: str
) -> None:
    llm = ChatNVIDIA(model=chat_model, seed=seed, **mode)
    if func == "invoke":
        llm.invoke("What's in a seed?")
    else:
        await llm.ainvoke("What's in a seed?")
    assert llm._client.last_response is not None
    if func == "ainvoke":
        assert llm._client.last_response.status == 200  # type: ignore[union-attr]
    else:
        assert llm._client.last_response.status_code == 200  # type: ignore[union-attr]


@pytest.mark.xfail(reason="seed does not consistently control determinism")
@pytest.mark.parametrize(
    "func",
    ["invoke", "ainvoke"],
)
async def test_ai_endpoints_invoke_seed_functional(
    chat_model: str, mode: dict, func: str, seed: int = 413
) -> None:
    llm = ChatNVIDIA(model=chat_model, seed=seed, **mode)
    if func == "ainvoke":
        result0 = await llm.ainvoke("What's in a seed?")
    else:
        result0 = llm.invoke("What's in a seed?")
    assert isinstance(result0.content, str)
    if func == "ainvoke":
        result1 = await llm.ainvoke("What's in a seed?")
    else:
        result1 = llm.invoke("What's in a seed?")
    assert isinstance(result1.content, str)
    assert result0.content == result1.content


# todo: temperature test for batch, abatch, stream, astream
@pytest.mark.parametrize("temperature", [-0.1])
@pytest.mark.parametrize(
    "func",
    ["invoke", "ainvoke"],
)
async def test_ai_endpoints_invoke_temperature_negative(
    chat_model: str, mode: dict, temperature: int, func: str
) -> None:
    """Test temperature validation (negative rejected by client-side)."""
    with pytest.raises(Exception):
        llm = ChatNVIDIA(model=chat_model, temperature=temperature, **mode)
        if func == "invoke":
            llm.invoke("What's in a temperature?")
        else:
            await llm.ainvoke("What's in a temperature?")


@pytest.mark.xfail(reason="temperature not consistently implemented")
@pytest.mark.parametrize(
    "func",
    ["invoke", "ainvoke"],
)
async def test_ai_endpoints_invoke_temperature_positive(
    chat_model: str, mode: dict, func: str
) -> None:
    """Test invoke's temperature (positive)."""
    # idea is to have a fixed seed and vary temperature to get different results
    llm0 = ChatNVIDIA(model=chat_model, seed=608, temperature=0, **mode)
    if func == "ainvoke":
        result0 = await llm0.ainvoke("What's in a temperature?")
    else:
        result0 = llm0.invoke("What's in a temperature?")
    assert isinstance(result0.content, str)
    llm1 = ChatNVIDIA(model=chat_model, seed=608, temperature=1, **mode)
    if func == "ainvoke":
        result1 = await llm1.ainvoke("What's in a temperature?")
    else:
        result1 = llm1.invoke("What's in a temperature?")
    assert isinstance(result1.content, str)
    assert result0.content != result1.content


# todo: top_p test for ainvoke, batch, abatch, stream, astream


@pytest.mark.parametrize("top_p", [-10, 10])
@pytest.mark.parametrize(
    "func",
    ["invoke", "ainvoke"],
)
async def test_ai_endpoints_invoke_top_p_negative(
    chat_model: str, mode: dict, top_p: int, func: str
) -> None:
    """Test invoke's top_p (negative, caught by client-side validation)."""

    with pytest.raises(Exception):
        llm = ChatNVIDIA(model=chat_model, top_p=top_p, **mode)
        if func == "invoke":
            llm.invoke("What's in a top_p?")
        else:
            await llm.ainvoke("What's in a top_p?")


@pytest.mark.xfail(reason="seed does not consistently control determinism")
@pytest.mark.parametrize(
    "func",
    ["invoke", "ainvoke"],
)
async def test_ai_endpoints_invoke_top_p_positive(
    chat_model: str, mode: dict, func: str
) -> None:
    """Test invoke's top_p (positive)."""
    # idea is to have a fixed seed and vary top_p to get different results
    llm0 = ChatNVIDIA(model=chat_model, seed=608, top_p=0.1, **mode)
    if func == "ainvoke":
        result0 = await llm0.ainvoke("What's in a top_p?")
    else:
        result0 = llm0.invoke("What's in a top_p?")
    assert isinstance(result0.content, str)
    llm1 = ChatNVIDIA(model=chat_model, seed=608, top_p=1, **mode)
    if func == "ainvoke":
        result1 = await llm1.ainvoke("What's in a top_p?")
    else:
        result1 = llm1.invoke("What's in a top_p?")
    assert isinstance(result1.content, str)
    assert result0.content != result1.content


@pytest.mark.skip("serialization support is broken, needs attention")
def test_serialize_chatnvidia(chat_model: str, mode: dict) -> None:
    llm = ChatNVIDIA(model=chat_model, **mode)
    model = loads(dumps(llm), valid_namespaces=["langchain_nvidia_ai_endpoints"])
    result = model.invoke("What is there if there is nothing?")
    assert isinstance(result.content, str)


# todo: test that stop is cased and works with multiple words


@pytest.mark.parametrize(
    "prop",
    [
        False,
        True,
    ],
    ids=["no_prop", "prop"],
)
@pytest.mark.parametrize(
    "param",
    [
        False,
        True,
    ],
    ids=["no_param", "param"],
)
@pytest.mark.parametrize(
    "targets",
    [["5"], ["6", "100"], ["100", "7"]],
    ids=["5", "6,100", "100,7"],
)
@pytest.mark.parametrize(
    "func",
    [
        "invoke",
        "stream",
        "ainvoke",
        "astream",
    ],
)
@pytest.mark.xfail(reason="stop is not consistently implemented")
async def test_stop(
    chat_model: str, mode: dict, func: str, prop: bool, param: bool, targets: List[str]
) -> None:
    if not prop and not param:
        pytest.skip("Skipping test, no stop parameter")
    llm = ChatNVIDIA(
        model=chat_model, stop=targets if prop else None, max_tokens=512, **mode
    )
    result = ""
    if func == "invoke":
        response = llm.invoke(
            "please count to 20 by 1s, e.g. 1 2 3 4",
            stop=targets if param else None,
        )  # invoke returns Union[str, List[Union[str, Dict[Any, Any]]]]
        assert isinstance(response.content, str)
        result = response.content
    elif func == "stream":
        for token in llm.stream(
            "please count to 20 by 1s, e.g. 1 2 3 4",
            stop=targets if param else None,
        ):
            assert isinstance(token.content, str)
            result += f"{token.content}|"
    elif func == "ainvoke":
        response = await llm.ainvoke(
            "please count to 20 by 1s, e.g. 1 2 3 4",
            stop=targets if param else None,
        )
        assert isinstance(response.content, str)
        result = response.content
    elif func == "astream":
        async for token in llm.astream(
            "please count to 20 by 1s, e.g. 1 2 3 4",
            stop=targets if param else None,
        ):
            assert isinstance(token.content, str)
            result += f"{token.content}|"
    assert all(target not in result for target in targets)
