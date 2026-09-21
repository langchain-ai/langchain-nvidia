from langchain_core.outputs import LLMResult

from langchain_nvidia_ai_endpoints.callbacks import UsageCallbackHandler


def _llm_result(total_tokens: int = 10) -> LLMResult:
    return LLMResult(
        generations=[],
        llm_output={
            "token_usage": {
                "total_tokens": total_tokens,
                "prompt_tokens": 4,
                "completion_tokens": total_tokens - 4,
            },
            "model_name": "mock-model",
        },
    )


def test_usage_callback_handler_state_is_per_instance() -> None:
    callback = UsageCallbackHandler()
    callback.on_llm_end(_llm_result())

    assert callback.total_tokens == 10
    assert callback.successful_requests == 1

    fresh_callback = UsageCallbackHandler()
    assert fresh_callback.total_tokens == 0
    assert fresh_callback.prompt_tokens == 0
    assert fresh_callback.completion_tokens == 0
    assert fresh_callback.successful_requests == 0


def test_usage_callback_handler_mutable_fields_are_per_instance() -> None:
    callback = UsageCallbackHandler()
    callback.price_map["mock-model"] = 1.0
    callback.llm_output["model_name"] = "mock-model"

    fresh_callback = UsageCallbackHandler()
    assert "mock-model" not in fresh_callback.price_map
    assert fresh_callback.llm_output == {}
