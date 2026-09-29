from langchain_core.outputs import LLMResult

from langchain_nvidia_ai_endpoints.callbacks import (
    UsageCallbackHandler,
    get_usage_callback,
)


def _usage_result(model_name: str = "meta/llama-3.1-8b-instruct") -> LLMResult:
    return LLMResult(
        generations=[],
        llm_output={
            "token_usage": {
                "total_tokens": 10,
                "prompt_tokens": 4,
                "completion_tokens": 6,
            },
            "model_name": model_name,
        },
    )


def test_handlers_track_usage_independently() -> None:
    first = UsageCallbackHandler()
    first.on_llm_end(_usage_result())
    assert first.total_tokens == 10

    second = UsageCallbackHandler()
    assert second.total_tokens == 0
    assert second._model_usage is not first._model_usage


def test_price_map_is_not_shared_between_contexts() -> None:
    with get_usage_callback(price_map={"some-model": 1.0}) as first:
        assert "some-model" in first.price_map
    with get_usage_callback() as second:
        assert "some-model" not in second.price_map
