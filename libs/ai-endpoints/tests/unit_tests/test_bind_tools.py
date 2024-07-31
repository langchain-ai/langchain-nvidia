import warnings
from typing import Any

import pytest
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.tools import tool

from langchain_nvidia_ai_endpoints import ChatNVIDIA


def xxyyzz_func(a: int, b: int) -> int:
    """xxyyzz two numbers"""
    return 42


class xxyyzz_cls(BaseModel):
    """xxyyzz two numbers"""

    a: int = Field(..., description="First number")
    b: int = Field(..., description="Second number")


@tool
def xxyyzz_tool(
    a: int = Field(..., description="First number"),
    b: int = Field(..., description="Second number"),
) -> int:
    """xxyyzz two numbers"""
    return 42


@pytest.mark.parametrize(
    "tools, choice",
    [
        ([xxyyzz_func], "xxyyzz_func"),
        ([xxyyzz_cls], "xxyyzz_cls"),
        ([xxyyzz_tool], "xxyyzz_tool"),
    ],
    ids=["func", "cls", "tool"],
)
def test_bind_tool_and_select(tools: Any, choice: str) -> None:
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=".*not known to support tools.*"
    )
    ChatNVIDIA(api_key="BOGUS").bind_tools(tools=tools, tool_choice=choice)


@pytest.mark.parametrize(
    "tools, choice",
    [
        ([], "wrong"),
        ([xxyyzz_func], "wrong_xxyyzz_func"),
        ([xxyyzz_cls], "wrong_xxyyzz_cls"),
        ([xxyyzz_tool], "wrong_xxyyzz_tool"),
    ],
    ids=["empty", "func", "cls", "tool"],
)
def test_bind_tool_and_select_negative(tools: Any, choice: str) -> None:
    warnings.filterwarnings(
        "ignore", category=UserWarning, message=".*not known to support tools.*"
    )
    with pytest.raises(ValueError) as e:
        ChatNVIDIA(api_key="BOGUS").bind_tools(tools=tools, tool_choice=choice)
    assert "not found in the tools list" in str(e.value)
