"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests

from langchain_nvidia_ai_endpoints import ChatNVIDIA


class TestNVIDIAStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatNVIDIA

    @property
    def chat_model_params(self) -> dict:
        return {"model": "meta/llama-3.1-8b-instruct"}

    @pytest.mark.xfail(reason="anthropic-style list content not supported")
    def test_tool_message_histories_list_content(self, model: BaseChatModel) -> None:
        return super().test_tool_message_histories_list_content(model)
