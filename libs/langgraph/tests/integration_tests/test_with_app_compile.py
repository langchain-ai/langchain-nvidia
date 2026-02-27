"""Integration tests: with_app_compile wrapper flow."""

from __future__ import annotations

import pytest
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from langchain_nvidia_langgraph.graph.constraints import OptimizationConfig
from langchain_nvidia_langgraph.graph.state_graph import with_app_compile


@pytest.mark.integration
class TestWithAppCompile:
    """with_app_compile wraps existing graphs for optimization."""

    def test_wrap_compile_invoke_vanilla(
        self,
        simple_linear_graph: StateGraph,
    ) -> None:
        """Wrap StateGraph, compile vanilla, invoke."""
        compilable = with_app_compile(simple_linear_graph)
        compiled = compilable.compile()
        assert isinstance(compiled, CompiledStateGraph)
        result = compiled.invoke({"value": "hello"})
        assert result["value"] == "hello"

    def test_wrap_compile_invoke_parallel(
        self,
        simple_linear_graph: StateGraph,
    ) -> None:
        """Wrap StateGraph, compile with parallel optimization, invoke."""
        compilable = with_app_compile(simple_linear_graph)
        compiled = compilable.compile(  # type: ignore[call-arg]
            optimization=OptimizationConfig(enable_parallel=True),
        )
        assert compiled is not None
        result = compiled.invoke({"value": "hello"})
        assert result["value"] == "hello"

    def test_wrapped_equivalent_to_subclass(
        self,
        linear_graph: StateGraph,
        langgraph_linear_graph: StateGraph,
    ) -> None:
        """Wrapped graph produces same result as StateGraph subclass."""
        compiled_subclass = linear_graph.compile(  # type: ignore[call-arg]
            optimization=OptimizationConfig(enable_parallel=True),
        )
        compilable = with_app_compile(langgraph_linear_graph)
        compiled_wrapped = compilable.compile(  # type: ignore[call-arg]
            optimization=OptimizationConfig(enable_parallel=True),
        )

        initial = {"value": "x", "path": "", "visited": ""}
        result_subclass = compiled_subclass.invoke(initial)
        result_wrapped = compiled_wrapped.invoke(initial)
        assert set(result_subclass["visited"]) == {"a", "b", "c"}
        assert set(result_wrapped["visited"]) == {"a", "b", "c"}
