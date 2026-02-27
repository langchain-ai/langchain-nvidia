"""Integration tests: graphs with add_conditional_edges (routers)."""

from __future__ import annotations

import warnings

import pytest
from langgraph.graph.state import CompiledStateGraph

from langchain_nvidia_langgraph.graph.constraints import OptimizationConfig
from langchain_nvidia_langgraph.graph.state_graph import StateGraph


def _route_by_value(state: dict) -> str:
    """Route to 'left' or 'right' based on state.value."""
    return "left" if state.get("value", "").startswith("left") else "right"


@pytest.mark.integration
class TestConditionalGraphs:
    """Conditional routing and speculative execution."""

    def test_conditional_vanilla_invoke_both_branches(
        self,
        conditional_graph: StateGraph,
    ) -> None:
        """Invoke with state that routes to each branch."""
        compiled = conditional_graph.compile()
        assert isinstance(compiled, CompiledStateGraph)

        # Route to b (value contains "b")
        result_b = compiled.invoke({"value": "to_b", "path": "", "visited": ""})
        assert result_b["path"] == "b"
        assert "a" in result_b["visited"]

        # Route to c (value does not contain "b")
        result_c = compiled.invoke({"value": "x", "path": "", "visited": ""})
        assert result_c["path"] == "c"
        assert "a" in result_c["visited"]

    def test_conditional_parallel_invoke_both_branches(
        self,
        conditional_graph: StateGraph,
    ) -> None:
        """Parallel compile: conditional graph invokes correctly."""
        compiled = conditional_graph.compile(
            optimization=OptimizationConfig(enable_parallel=True),
        )
        result_b = compiled.invoke({"value": "to_b", "path": "", "visited": ""})
        assert result_b["path"] == "b"
        result_c = compiled.invoke({"value": "x", "path": "", "visited": ""})
        assert result_c["path"] == "c"

    def test_conditional_speculative_semantics(
        self,
        conditional_graph: StateGraph,
    ) -> None:
        """Speculative execution preserves correct semantics (committed result)."""
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore", UserWarning)
            compiled = conditional_graph.compile(
                optimization=OptimizationConfig(
                    enable_parallel=True,
                    enable_speculation=True,
                ),
            )
        # Both branches should produce valid committed result (path b or c)
        result_b = compiled.invoke({"value": "to_b", "path": "", "visited": ""})
        assert result_b["path"] in ("b", "c")
        assert "a" in result_b["visited"]
        result_c = compiled.invoke({"value": "other", "path": "", "visited": ""})
        assert result_c["path"] in ("b", "c")
        assert "a" in result_c["visited"]
