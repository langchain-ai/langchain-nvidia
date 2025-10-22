"""Test default_headers functionality for all model types."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.parametrize("func_name", ["invoke", "stream"])
@patch("langchain_nvidia_ai_endpoints.chat_models._NVIDIAClient")
def test_chat_default_headers_passed_to_request(
    mock_client_class: MagicMock, func_name: str
) -> None:
    """Test that default_headers are included in invoke and stream requests."""
    from langchain_core.messages import HumanMessage

    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    # Setup mock
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    mock_client.model = None
    mock_client.mdl_name = "test-model"
    mock_client.base_url = "http://test"

    # Setup mock responses
    mock_client.get_req_stream.return_value = iter(
        [{"role": "assistant", "content": "chunk", "finish_reason": "stop"}]
    )
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "test"},
                "finish_reason": "stop",
            }
        ]
    }
    mock_client.get_req.return_value = mock_response
    mock_client.postprocess.return_value = (
        {"role": "assistant", "content": "test", "finish_reason": "stop"},
        True,
    )

    # Create model with default_headers
    custom_headers = {"X-Custom-Header": "custom-value"}
    llm = ChatNVIDIA(
        model="test-model", api_key="test-key", default_headers=custom_headers
    )

    # Make request using the parametrized function name
    response = getattr(llm, func_name)([HumanMessage(content="test")])
    if func_name == "stream":
        list(response)  # Consume the generator
        call_kwargs = mock_client.get_req_stream.call_args[1]
    else:
        call_kwargs = mock_client.get_req.call_args[1]

    # Verify headers were passed
    assert call_kwargs["extra_headers"] == custom_headers


@patch("langchain_nvidia_ai_endpoints.embeddings._NVIDIAClient")
def test_embeddings_default_headers_passed_to_request(
    mock_client_class: MagicMock,
) -> None:
    """Test that default_headers are included in embedding requests."""
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings  # noqa: E402

    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    mock_client.mdl_name = "test-model"
    mock_client.base_url = "http://test"

    mock_response = MagicMock()
    mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2], "index": 0}]}
    mock_response.raise_for_status = MagicMock()
    mock_client.get_req.return_value = mock_response

    custom_headers = {"X-Custom-Header": "custom-value"}
    embeddings = NVIDIAEmbeddings(
        model="test-model", api_key="test-key", default_headers=custom_headers
    )

    embeddings.embed_query("test")

    # Verify headers were passed
    call_kwargs = mock_client.get_req.call_args[1]
    assert call_kwargs["extra_headers"] == custom_headers


@patch("langchain_nvidia_ai_endpoints.reranking._NVIDIAClient")
def test_rerank_default_headers_passed_to_request(
    mock_client_class: MagicMock,
) -> None:
    """Test that default_headers are included in reranking requests."""
    from langchain_core.documents import Document

    from langchain_nvidia_ai_endpoints import NVIDIARerank

    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    mock_client.mdl_name = "test-model"
    mock_client.base_url = "http://test"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"rankings": [{"index": 0, "logit": 0.9}]}
    mock_client.get_req.return_value = mock_response

    custom_headers = {"X-Custom-Header": "custom-value"}
    reranker = NVIDIARerank(
        model="test-model", api_key="test-key", default_headers=custom_headers
    )

    reranker.compress_documents([Document(page_content="doc1")], "query")

    # Verify headers were passed
    call_kwargs = mock_client.get_req.call_args[1]
    assert call_kwargs["extra_headers"] == custom_headers


@patch("langchain_nvidia_ai_endpoints.reranking._NVIDIAClient")
def test_rerank_extra_headers_backwards_compatibility(
    mock_client_class: MagicMock,
) -> None:
    """Test that old extra_headers parameter still works."""
    from langchain_nvidia_ai_endpoints import NVIDIARerank  # noqa: E402

    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    mock_client.mdl_name = "test-model"
    mock_client.base_url = "http://test"

    custom_headers = {"X-Custom-Header": "custom-value"}
    reranker = NVIDIARerank(
        model="test-model", api_key="test-key", extra_headers=custom_headers
    )

    # Verify extra_headers was copied to default_headers
    assert reranker.default_headers == custom_headers
