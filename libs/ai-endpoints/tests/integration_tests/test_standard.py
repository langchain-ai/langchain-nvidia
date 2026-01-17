"""Standard LangChain interface tests"""

from typing import Any, Coroutine, Literal, Type

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

    @pytest.mark.parametrize("model", [{}, {"output_version": "v1"}], indirect=True)
    @pytest.mark.xfail(
        reason="Backend returns tool arguments as strings for some models, "
        "ints for others",
        strict=False,
    )
    def test_tool_calling(self, model: BaseChatModel) -> None:
        """Override to accept both string and int types in tool arguments.

        Backend behavior varies - some models return '3', others return 3.
        Both are semantically correct, so we accept either.
        """
        from langchain_core.messages import AIMessage
        from langchain_tests.integration_tests.chat_models import magic_function

        tool_choice_value = None if not self.has_tool_choice else "any"
        model_with_tools = model.bind_tools(
            [magic_function], tool_choice=tool_choice_value
        )

        query = "What is the value of magic_function(3)? Use the tool."
        result = model_with_tools.invoke(query)

        # Validate but accept both string and int types (backend varies by model)
        assert isinstance(result, AIMessage)
        assert len(result.tool_calls) == 1
        tool_call = result.tool_calls[0]
        assert tool_call["name"] == "magic_function"
        # Accept both int and string - backend behavior varies by model
        assert tool_call["args"]["input"] in (
            3,
            "3",
        ), f"Expected 3 or '3', got {tool_call['args']['input']!r}"

    @pytest.mark.xfail(
        reason="Backend returns tool arguments as strings for some models, "
        "ints for others",
        strict=False,
    )
    async def test_tool_calling_async(self, model: BaseChatModel) -> None:
        """Override to accept both string and int types in tool arguments.

        Backend behavior varies - some models return '3', others return 3.
        Both are semantically correct, so we accept either.
        """
        from langchain_core.messages import AIMessage
        from langchain_tests.integration_tests.chat_models import magic_function

        tool_choice_value = None if not self.has_tool_choice else "any"
        model_with_tools = model.bind_tools(
            [magic_function], tool_choice=tool_choice_value
        )

        query = "What is the value of magic_function(3)? Use the tool."
        result = await model_with_tools.ainvoke(query)

        # Validate but accept both string and int (backend varies by model)
        assert isinstance(result, AIMessage)
        assert len(result.tool_calls) == 1
        tool_call = result.tool_calls[0]
        assert tool_call["name"] == "magic_function"
        # Accept both int and string - backend behavior varies by model
        assert tool_call["args"]["input"] in (
            3,
            "3",
        ), f"Expected 3 or '3', got {tool_call['args']['input']!r}"

    @pytest.mark.xfail(
        reason="Pydantic v1 BaseModel structured output not fully supported"
    )
    def test_structured_output_pydantic_2_v1(self, model: BaseChatModel) -> None:
        return super().test_structured_output_pydantic_2_v1(model)

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
    def test_structured_output(
        self,
        model: BaseChatModel,
        schema_type: Literal["pydantic", "typeddict", "json_schema"],
    ) -> None:
        return super().test_structured_output(model, schema_type)

    @pytest.mark.parametrize("schema_type", ["typeddict"])
    @pytest.mark.xfail(reason="TypedDict schema type not supported")
    async def test_structured_output_async(
        self,
        model: BaseChatModel,
        schema_type: Literal["pydantic", "typeddict", "json_schema"],
    ) -> Coroutine[Any, Any, None]:
        # Return the coroutine directly without awaiting it
        return super().test_structured_output_async(model, schema_type)

    @pytest.mark.xfail(reason="TypedDict schema type not supported")
    def test_structured_output_optional_param(self, model: BaseChatModel) -> None:
        # Don't return anything since the return type is None
        super().test_structured_output_optional_param(model)

    @pytest.mark.xfail(reason="Some models return double-escapes Unicode in tool calls")
    def test_unicode_tool_call_integration(
        self,
        model: BaseChatModel,
        *,
        tool_choice: str | None = None,
        force_tool_call: bool = True,
    ) -> None:
        return super().test_unicode_tool_call_integration(
            model, tool_choice=tool_choice, force_tool_call=force_tool_call
        )
