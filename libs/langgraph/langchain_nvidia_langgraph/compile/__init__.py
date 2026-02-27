"""LangGraph optimization: one-call API.

Provides ``compile_langgraph`` and ``transform_graph`` for analyzing and
building optimized LangGraph instances with parallel and speculative execution.
"""

from .compiler import compile_langgraph, transform_graph

__all__ = [
    "compile_langgraph",
    "transform_graph",
]
