"""Unit tests for LangChainLLMDetector public interfaces."""

from __future__ import annotations

import pytest
from langchain_core.language_models import BaseLanguageModel
from langchain_core.outputs import GenerationChunk, LLMResult

from langchain_nvidia_langgraph.adapter.llm_detector import LangChainLLMDetector

# ---------------------------------------------------------------------------
# Minimal BaseLanguageModel for isinstance tests
# ---------------------------------------------------------------------------


class MinimalLLM(BaseLanguageModel):
    """Minimal BaseLanguageModel implementation for testing."""

    @property
    def _llm_type(self) -> str:
        return "minimal"

    def _generate(self, *args: object, **kwargs: object) -> GenerationChunk:
        return GenerationChunk(text="")

    def generate_prompt(
        self,
        prompts: list,
        stop: list[str] | None = None,
        callbacks: object = None,
        **kwargs: object,
    ) -> LLMResult:
        return LLMResult(generations=[[self._generate()]] * len(prompts))

    async def agenerate_prompt(
        self,
        prompts: list,
        stop: list[str] | None = None,
        callbacks: object = None,
        **kwargs: object,
    ) -> LLMResult:
        return self.generate_prompt(prompts, stop, callbacks, **kwargs)

    def invoke(self, *args: object, **kwargs: object) -> str:
        return ""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def detector() -> LangChainLLMDetector:
    """LangChainLLMDetector instance."""
    return LangChainLLMDetector()


# ---------------------------------------------------------------------------
# invocation_methods property
# ---------------------------------------------------------------------------


def test_invocation_methods_returns_frozenset(detector: LangChainLLMDetector) -> None:
    """invocation_methods returns a frozenset of method names."""
    methods = detector.invocation_methods
    assert isinstance(methods, frozenset)


def test_invocation_methods_contains_expected_names(
    detector: LangChainLLMDetector,
) -> None:
    """invocation_methods contains invoke, ainvoke, stream, astream, etc."""
    methods = detector.invocation_methods
    expected = {
        "invoke",
        "ainvoke",
        "stream",
        "astream",
        "generate",
        "agenerate",
        "batch",
        "abatch",
    }
    assert methods == expected


def test_invocation_methods_returns_consistent_value(
    detector: LangChainLLMDetector,
) -> None:
    """invocation_methods returns equal frozenset on each access."""
    m1 = detector.invocation_methods
    m2 = detector.invocation_methods
    assert m1 == m2


# ---------------------------------------------------------------------------
# is_llm method
# ---------------------------------------------------------------------------


def test_is_llm_true_for_base_language_model_subclass(
    detector: LangChainLLMDetector,
) -> None:
    """is_llm returns True for BaseLanguageModel instances."""
    llm = MinimalLLM()
    assert detector.is_llm(llm) is True


def test_is_llm_true_for_duck_typed_object(detector: LangChainLLMDetector) -> None:
    """is_llm returns True for objects with invoke, ainvoke, and bind_tools."""
    duck = type(
        "DuckLLM",
        (),
        {
            "invoke": lambda s: None,
            "ainvoke": lambda s: None,
            "bind_tools": lambda s: None,
        },
    )()
    assert detector.is_llm(duck) is True


def test_is_llm_false_when_missing_invoke(detector: LangChainLLMDetector) -> None:
    """is_llm returns False when object lacks invoke."""
    obj = type(
        "Partial", (), {"ainvoke": lambda s: None, "bind_tools": lambda s: None}
    )()
    assert detector.is_llm(obj) is False


def test_is_llm_false_when_missing_ainvoke(detector: LangChainLLMDetector) -> None:
    """is_llm returns False when object lacks ainvoke."""
    obj = type(
        "Partial", (), {"invoke": lambda s: None, "bind_tools": lambda s: None}
    )()
    assert detector.is_llm(obj) is False


def test_is_llm_false_when_missing_bind_tools(detector: LangChainLLMDetector) -> None:
    """is_llm returns False when object lacks bind_tools."""
    obj = type("Partial", (), {"invoke": lambda s: None, "ainvoke": lambda s: None})()
    assert detector.is_llm(obj) is False


def test_is_llm_false_for_plain_object(detector: LangChainLLMDetector) -> None:
    """is_llm returns False for plain objects."""
    assert detector.is_llm(object()) is False
    assert detector.is_llm("string") is False
    assert detector.is_llm(42) is False
    assert detector.is_llm(None) is False
