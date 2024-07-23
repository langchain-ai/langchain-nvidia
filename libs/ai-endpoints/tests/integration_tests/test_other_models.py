"""Test ChatNVIDIA chat model."""

from langchain_core.messages import BaseMessage, HumanMessage

from langchain_nvidia_ai_endpoints.chat_models import ChatNVIDIA


def test_chat_ai_endpoints_context_message(qa_model: str, mode: dict) -> None:
    """Test wrapper with context message."""
    chat = ChatNVIDIA(model=qa_model, max_tokens=36, **mode)
    context_message = BaseMessage(
        content="Once upon a time there was a little langchainer", type="context"
    )
    human_message = HumanMessage(content="What was there once upon a time?")
    response = chat.invoke([context_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)
