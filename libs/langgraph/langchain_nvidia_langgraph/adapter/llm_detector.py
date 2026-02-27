"""LangChain and LangGraph LLM detector for the priority pipeline.

Identifies ``BaseLanguageModel`` instances for use in the compiler's
priority analysis.
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseLanguageModel


class LangChainLLMDetector:
    """Identify LangChain ``BaseLanguageModel`` instances and their call methods.

    Uses isinstance when available, with duck-typing fallback for compatible
    objects that implement ``invoke``, ``ainvoke``, and ``bind_tools``.
    """

    @property
    def invocation_methods(self) -> frozenset[str]:
        """Method names used to invoke an LLM (invoke, ainvoke, stream, etc.).

        Returns:
            Frozen set of invocation method names.
        """
        return frozenset(
            {
                "invoke",
                "ainvoke",
                "stream",
                "astream",
                "generate",
                "agenerate",
                "batch",
                "abatch",
            }
        )

    def is_llm(self, obj: Any) -> bool:
        """Check whether the object is an LLM or LLM-compatible.

        Args:
            obj: Object to check. May be a BaseLanguageModel subclass or a
                duck-typed object with invoke, ainvoke, and bind_tools.

        Returns:
            True if obj is an LLM or LLM-compatible, False otherwise.
        """
        if isinstance(obj, BaseLanguageModel):
            return True
        return self._duck_type_check(obj)

    @staticmethod
    def _duck_type_check(obj: Any) -> bool:
        """Check for LLM-like interface via duck typing.

        Args:
            obj: Object to check for invoke, ainvoke, and bind_tools.

        Returns:
            True if obj has all required attributes, False otherwise.
        """
        return (
            hasattr(obj, "invoke")
            and hasattr(obj, "ainvoke")
            and hasattr(obj, "bind_tools")
        )
