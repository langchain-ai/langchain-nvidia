"""Integration tests: compile modes (vanilla, parallel, speculative)."""

from __future__ import annotations

import warnings

import pytest
from langgraph.graph.state import CompiledStateGraph

from langchain_nvidia_langgraph.graph.constraints import OptimizationConfig
from langchain_nvidia_langgraph.graph.state_graph import StateGraph


@pytest.mark.integration
class TestCompileModes:
    """End-to-end compile and invoke for each optimization mode."""

    def test_vanilla_compile_invoke(self, linear_graph: StateGraph) -> None:
        """Vanilla compile: invoke returns correct state, all nodes run."""
        compiled = linear_graph.compile()
        assert isinstance(compiled, CompiledStateGraph)
        result = compiled.invoke({"value": "x", "path": "", "visited": ""})
        assert set(result["visited"]) == {"a", "b", "c"}
        assert result["value"] == "x"

    def test_parallel_compile_invoke(self, linear_graph: StateGraph) -> None:
        """Parallel compile: invoke returns correct state."""
        compiled = linear_graph.compile(
            optimization=OptimizationConfig(enable_parallel=True),
        )
        assert compiled is not None
        result = compiled.invoke({"value": "x", "path": "", "visited": ""})
        assert set(result["visited"]) == {"a", "b", "c"}
        assert result["value"] == "x"

    def test_speculative_compile_invoke(self, conditional_graph: StateGraph) -> None:
        """Speculative compile: invoke returns correct state for conditional graph."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore", UserWarning)
            compiled = conditional_graph.compile(
                optimization=OptimizationConfig(
                    enable_parallel=True,
                    enable_speculation=True,
                ),
            )
        assert compiled is not None
        assert hasattr(compiled, "invoke")
        # Route to b
        result_b = compiled.invoke({"value": "to_b", "path": "", "visited": ""})
        assert result_b["path"] == "b"
        assert "a" in result_b["visited"]
        # Route to c
        result_c = compiled.invoke({"value": "x", "path": "", "visited": ""})
        assert result_c["path"] == "c"
        assert "a" in result_c["visited"]
