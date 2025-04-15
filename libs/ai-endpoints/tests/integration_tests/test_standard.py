"""Standard LangChain interface tests"""

from typing import Any, Coroutine, Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_nvidia_ai_endpoints import ChatNVIDIA


class TestNVIDIAStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatNVIDIA

    @property
    def chat_model_params(self) -> dict:
        return {"model": "meta/llama-3.3-70b-instruct", "temperature": 0}

    @pytest.mark.xfail(reason="anthropic-style list content not supported")
    def test_tool_message_histories_list_content(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        return super().test_tool_message_histories_list_content(model, my_adder_tool)

    @pytest.mark.xfail(reason="Empty AIMessage content not supported")
    def test_tool_message_error_status(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        return super().test_tool_message_error_status(model, my_adder_tool)

    @pytest.mark.xfail(reason="Empty AIMessage content not supported")
    def test_tool_message_histories_string_content(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        return super().test_tool_message_histories_string_content(model, my_adder_tool)

    @pytest.mark.xfail(
        reason="Only one chunk should set input_tokens, the rest should be 0 or None"
    )
    def test_usage_metadata_streaming(self, model: BaseChatModel) -> None:
        return super().test_usage_metadata_streaming(model)

    @pytest.mark.parametrize("schema_type", ["typeddict"])
    @pytest.mark.xfail(reason="TypedDict schema type not supported")
    def test_structured_output(self, model: BaseChatModel, schema_type: str) -> None:
        return super().test_structured_output(model, schema_type)

    @pytest.mark.parametrize("schema_type", ["typeddict"])
    @pytest.mark.xfail(reason="TypedDict schema type not supported")
    async def test_structured_output_async(
        self, model: BaseChatModel, schema_type: str
    ) -> Coroutine[Any, Any, None]:
        # Return the coroutine directly without awaiting it
        return super().test_structured_output_async(model, schema_type)

    @pytest.mark.xfail(reason="TypedDict schema type not supported")
    def test_structured_output_optional_param(self, model: BaseChatModel) -> None:
        # Don't return anything since the return type is None
        super().test_structured_output_optional_param(model)
