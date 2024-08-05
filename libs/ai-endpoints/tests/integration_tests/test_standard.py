"""Standard LangChain interface tests"""

from typing import Type

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
