import requests_mock
from langchain_core.messages import AIMessage

from langchain_nvidia_ai_endpoints import ChatNVIDIA


def test_invoke_aimessage_content_none(requests_mock: requests_mock.Mocker) -> None:
    requests_mock.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        json={
            "id": "mock-id",
            "created": 1234567890,
            "object": "chat.completion",
            "model": "mock-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "WORKED"},
                }
            ],
        },
    )

    empty_aimessage = AIMessage(content="EMPTY")
    empty_aimessage.content = None  # type: ignore

    llm = ChatNVIDIA()
    response = llm.invoke([empty_aimessage])
    request = requests_mock.request_history[0]
    assert request.method == "POST"
    assert request.url == "https://integrate.api.nvidia.com/v1/chat/completions"
    message = request.json()["messages"][0]
    assert "content" in message and message["content"] != "EMPTY"
    assert "content" in message and message["content"] is None
    assert isinstance(response, AIMessage)
    assert response.content == "WORKED"
