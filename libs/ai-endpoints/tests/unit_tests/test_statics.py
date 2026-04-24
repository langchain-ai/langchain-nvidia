from typing import Any

import pytest

from langchain_nvidia_ai_endpoints._statics import MODEL_TABLE, Model, determine_model


@pytest.fixture(params=MODEL_TABLE.keys())
def entry(request: Any) -> str:
    return request.param


@pytest.fixture(
    params=[
        alias
        for ls in [model.aliases for model in MODEL_TABLE.values() if model.aliases]
        for alias in ls
    ]
)
def alias(request: Any) -> str:
    return request.param


def test_model_table_integrity_name_id(entry: str) -> None:
    model = MODEL_TABLE[entry]
    assert model.id == entry


def test_determine_model_deprecated_alternative_warns(alias: str) -> None:
    with pytest.warns(UserWarning) as record:
        determine_model(alias)
    assert len(record) == 1
    assert f"Model {alias} is deprecated" in str(record[0].message)


def test_model_warns_on_both_thinking_modes() -> None:
    """Test that a warning is issued when both param-based and tag-based
    thinking are configured."""
    with pytest.warns(UserWarning) as record:
        Model(
            id="test-model",
            model_type="chat",
            client="ChatNVIDIA",
            thinking_param_enable={"chat_template_kwargs": {"enable_thinking": True}},
            thinking_prefix="detailed thinking on",
        )
    assert len(record) == 1
    assert "both param-based thinking" in str(record[0].message).lower()
    assert "param-based thinking will take precedence" in str(record[0].message).lower()


def test_model_no_warning_param_based_only() -> None:
    """Test that no warning is issued when only param-based thinking is configured."""
    Model(
        id="test-model",
        model_type="chat",
        client="ChatNVIDIA",
        thinking_param_enable={"chat_template_kwargs": {"enable_thinking": True}},
    )


def test_model_no_warning_tag_based_only() -> None:
    """Test that no warning is issued when only tag-based thinking is configured."""
    Model(
        id="test-model",
        model_type="chat",
        client="ChatNVIDIA",
        thinking_prefix="detailed thinking on",
    )
