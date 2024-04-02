"""Test ChatNVIDIA chat model."""

import warnings

import pytest
from langchain_core.load.dump import dumps
from langchain_core.load.load import loads
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from langchain_nvidia_ai_endpoints.chat_models import ChatNVIDIA

#
# we setup an --all-models flag in conftest.py, when passed it configures chat_model
# and image_in_model to be all available models of type chat or image_in
#
# note: currently --all-models only works with the default mode because different
#       modes may have different available models
#


def test_chat_ai_endpoints(chat_model: str, mode: dict) -> None:
    """Test ChatNVIDIA wrapper."""
    chat = ChatNVIDIA(
        model=chat_model,
        temperature=0.7,
    ).mode(**mode)
    message = HumanMessage(content="Hello")
    response = chat.invoke([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_ai_endpoints_model() -> None:
    """Test wrapper handles model."""
    chat = ChatNVIDIA(model="mistral")
    assert chat.model == "mistral"


def test_chat_ai_endpoints_system_message(chat_model: str, mode: dict) -> None:
    """Test wrapper with system message."""
    # mamba_chat only supports 'user' or 'assistant' messages -
    #  Exception: [422] Unprocessable Entity
    #  body -> messages -> 0 -> role
    #    Input should be 'user' or 'assistant'
    #     (type=literal_error; expected='user' or 'assistant')
    if chat_model == "mamba_chat":
        pytest.skip(f"{chat_model} does not support system messages")

    chat = ChatNVIDIA(model=chat_model, max_tokens=36).mode(**mode)
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat.invoke([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


## TODO: Not sure if we want to support the n syntax. Trash or keep test


def test_ai_endpoints_streaming(chat_model: str, mode: dict) -> None:
    """Test streaming tokens from ai endpoints."""
    llm = ChatNVIDIA(model=chat_model, max_tokens=36).mode(**mode)

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_ai_endpoints_astream(chat_model: str, mode: dict) -> None:
    """Test streaming tokens from ai endpoints."""
    llm = ChatNVIDIA(model=chat_model, max_tokens=35).mode(**mode)

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


async def test_ai_endpoints_abatch(chat_model: str, mode: dict) -> None:
    """Test streaming tokens."""
    llm = ChatNVIDIA(model=chat_model, max_tokens=36).mode(**mode)

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_ai_endpoints_abatch_tags(chat_model: str, mode: dict) -> None:
    """Test batch tokens."""
    llm = ChatNVIDIA(model=chat_model, max_tokens=55).mode(**mode)

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


def test_ai_endpoints_batch(chat_model: str, mode: dict) -> None:
    """Test batch tokens."""
    llm = ChatNVIDIA(model=chat_model, max_tokens=60).mode(**mode)

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


async def test_ai_endpoints_ainvoke(chat_model: str, mode: dict) -> None:
    """Test invoke tokens."""
    llm = ChatNVIDIA(model=chat_model, max_tokens=60).mode(**mode)

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


def test_ai_endpoints_invoke(chat_model: str, mode: dict) -> None:
    """Test invoke tokens."""
    llm = ChatNVIDIA(model=chat_model, max_tokens=60).mode(**mode)

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)


def test_chat_ai_endpoints_context_message(qa_model: str, mode: dict) -> None:
    """Test wrapper with context message."""
    chat = ChatNVIDIA(model=qa_model, max_tokens=36).mode(**mode)
    context_message = BaseMessage(
        content="Once upon a time there was a little langchainer", type="context"
    )
    human_message = HumanMessage(content="What was there once upon a time?")
    response = chat.invoke([context_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_image_in_models(image_in_model: str, mode: dict) -> None:
    try:
        chat = ChatNVIDIA(model=image_in_model).mode(**mode)
        response = chat.invoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Describe this image:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "tests/data/nvidia-picasso.jpg"},
                        },
                    ]
                )
            ]
        )
        assert isinstance(response, BaseMessage)
        assert isinstance(response.content, str)
    except TimeoutError as e:
        message = f"TimeoutError: {image_in_model} {e}"
        warnings.warn(message)
        pytest.skip(message)


# todo: tests for parameters -
#        bad: str - Bad words to avoid (cased)
#        stop: str - Stop words (cased)


# todo: max_tokens test for ainvoke, batch, abatch, stream, astream


@pytest.mark.parametrize("max_tokens", [-100, 0, 2**31 - 1])
def test_ai_endpoints_invoke_max_tokens_negative(
    chat_model: str,
    mode: dict,
    max_tokens: int,
) -> None:
    """Test invoke's max_tokens' bounds."""
    with pytest.raises(Exception):
        llm = ChatNVIDIA(model=chat_model, max_tokens=max_tokens).mode(**mode)
        llm.invoke("Show me the tokens")
        assert llm.client.last_response.status_code == 422


def test_ai_endpoints_invoke_max_tokens_positive(
    chat_model: str, mode: dict, max_tokens: int = 21
) -> None:
    """Test invoke's max_tokens."""
    llm = ChatNVIDIA(model=chat_model, max_tokens=max_tokens).mode(**mode)
    result = llm.invoke("Show me the tokens")
    assert isinstance(result.content, str)
    assert "token_usage" in result.response_metadata
    assert "completion_tokens" in result.response_metadata["token_usage"]
    assert result.response_metadata["token_usage"]["completion_tokens"] <= max_tokens


# todo: seed test for ainvoke, batch, abatch, stream, astream


@pytest.mark.skip("seed does not consistently control determinism")
def test_ai_endpoints_invoke_seed_default(chat_model: str, mode: dict) -> None:
    """Test invoke's seed (default)."""
    llm0 = ChatNVIDIA(model=chat_model).mode(**mode)  # default seed should not repeat
    result0 = llm0.invoke("What's in a seed?")
    assert isinstance(result0.content, str)
    llm1 = ChatNVIDIA(model=chat_model).mode(**mode)  # default seed should not repeat
    result1 = llm1.invoke("What's in a seed?")
    assert isinstance(result1.content, str)
    # if this fails, consider setting a high temperature to avoid deterministic results
    assert result0.content != result1.content


def test_ai_endpoints_invoke_seed_negative(chat_model: str, mode: dict) -> None:
    """Test invoke's seed (negative)."""
    with pytest.raises(Exception):
        llm = ChatNVIDIA(model=chat_model, seed=-1000).mode(**mode)
        llm.invoke("What's in a seed?")
        assert llm.client.last_response.status_code == 422


@pytest.mark.skip("seed does not consistently control determinism")
def test_ai_endpoints_invoke_seed_positive(
    chat_model: str, mode: dict, seed: int = 413
) -> None:
    """Test invoke's seed (positive)."""
    llm = ChatNVIDIA(model=chat_model, seed=seed).mode(**mode)
    result0 = llm.invoke("What's in a seed?")
    assert isinstance(result0.content, str)
    result1 = llm.invoke("What's in a seed?")
    assert isinstance(result1.content, str)
    assert result0.content == result1.content


# todo: temperature test for ainvoke, batch, abatch, stream, astream


@pytest.mark.parametrize("temperature", [-0.1, 1.1])
def test_ai_endpoints_invoke_temperature_negative(
    chat_model: str, mode: dict, temperature: int
) -> None:
    """Test invoke's temperature (negative)."""
    with pytest.raises(Exception):
        llm = ChatNVIDIA(model=chat_model, temperature=temperature).mode(**mode)
        llm.invoke("What's in a temperature?")
        assert llm.client.last_response.status_code == 422


@pytest.mark.skip("seed does not consistently control determinism")
def test_ai_endpoints_invoke_temperature_positive(chat_model: str, mode: dict) -> None:
    """Test invoke's temperature (positive)."""
    # idea is to have a fixed seed and vary temperature to get different results
    llm0 = ChatNVIDIA(model=chat_model, seed=608, templerature=0).mode(**mode)
    result0 = llm0.invoke("What's in a temperature?")
    assert isinstance(result0.content, str)
    llm1 = ChatNVIDIA(model=chat_model, seed=608, templerature=1).mode(**mode)
    result1 = llm1.invoke("What's in a temperature?")
    assert isinstance(result1.content, str)
    assert result0.content != result1.content


# todo: top_p test for ainvoke, batch, abatch, stream, astream


@pytest.mark.parametrize("top_p", [-10, 0])
def test_ai_endpoints_invoke_top_p_negative(
    chat_model: str, mode: dict, top_p: int
) -> None:
    """Test invoke's top_p (negative)."""
    with pytest.raises(Exception):
        llm = ChatNVIDIA(model=chat_model, top_p=top_p).mode(**mode)
        llm.invoke("What's in a top_p?")
        assert llm.client.last_response.status_code == 422


@pytest.mark.skip("seed does not consistently control determinism")
def test_ai_endpoints_invoke_top_p_positive(chat_model: str, mode: dict) -> None:
    """Test invoke's top_p (positive)."""
    # idea is to have a fixed seed and vary top_p to get different results
    llm0 = ChatNVIDIA(model=chat_model, seed=608, top_p=1).mode(**mode)
    result0 = llm0.invoke("What's in a top_p?")
    assert isinstance(result0.content, str)
    llm1 = ChatNVIDIA(model=chat_model, seed=608, top_p=100).mode(**mode)
    result1 = llm1.invoke("What's in a top_p?")
    assert isinstance(result1.content, str)
    assert result0.content != result1.content


def test_serialize_chatnvidia() -> None:
    model = loads(
        dumps(ChatNVIDIA()), valid_namespaces=["langchain_nvidia_ai_endpoints"]
    )
    result = model.invoke("What is there if there is nothing?")
    assert isinstance(result.content, str)
