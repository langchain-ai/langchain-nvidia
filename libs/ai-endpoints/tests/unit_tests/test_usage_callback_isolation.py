from __future__ import annotations

import pytest
from langchain_core.outputs import LLMResult

from langchain_nvidia_ai_endpoints.callbacks import (
    UsageCallbackHandler,
    get_usage_callback,
    usage_callback_var,
)


def _result(model: str, prompt_tokens: int, completion_tokens: int) -> LLMResult:
    return LLMResult(
        generations=[],
        llm_output={
            "model_name": model,
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        },
    )


def test_independent_callback_sessions_do_not_share_cost_or_price_map() -> None:
    with get_usage_callback(price_map={"tenant-a-model": 1.0}) as first:
        first.on_llm_end(_result("tenant-a-model", 4, 6))
        assert first.total_tokens == 10
        assert first.total_cost == 0.01

    with get_usage_callback() as second:
        assert second.total_tokens == 0
        assert second.total_cost == 0
        assert "tenant-a-model" not in second.price_map
        second.on_llm_end(_result("tenant-b-model", 2, 1))
        assert second.total_tokens == 3
        assert second.total_cost == 0

    assert first.total_tokens == 10
    assert first.total_cost == 0.01


def test_nested_callback_scope_restores_outer_handler_even_after_exception() -> None:
    assert usage_callback_var.get() is None
    outer = UsageCallbackHandler()
    with get_usage_callback(callback=outer):
        assert usage_callback_var.get() is outer
        with pytest.raises(RuntimeError, match="request failed"):
            with get_usage_callback() as inner:
                assert usage_callback_var.get() is inner
                inner.on_llm_end(_result("inner", 2, 1))
                raise RuntimeError("request failed")
        assert usage_callback_var.get() is outer
        outer.on_llm_end(_result("outer", 1, 1))
        assert outer.total_tokens == 2
    assert usage_callback_var.get() is None
