"""Integration tests: snapshot testing of invoke output structure."""

from __future__ import annotations

import pytest

from langchain_nvidia_langgraph.graph.constraints import OptimizationConfig
from langchain_nvidia_langgraph.graph.state_graph import StateGraph


@pytest.mark.integration
class TestSnapshots:
    """Snapshot tests for invoke output structure (deterministic parts)."""

    def test_linear_vanilla_output_structure(
        self,
        linear_graph: StateGraph,
    ) -> None:
        """Vanilla compile invoke returns expected keys and all nodes ran."""
        compiled = linear_graph.compile()
        result = compiled.invoke({"value": "test", "path": "", "visited": ""})
        assert "value" in result
        assert "path" in result
        assert "visited" in result
        assert result["value"] == "test"
        assert set(result["visited"]) == {"a", "b", "c"}

    def test_linear_parallel_output_structure(
        self,
        linear_graph: StateGraph,
    ) -> None:
        """Parallel compile invoke returns expected keys and all nodes ran."""
        compiled = linear_graph.compile(
            optimization=OptimizationConfig(enable_parallel=True),
        )
        result = compiled.invoke({"value": "test", "path": "", "visited": ""})
        assert "value" in result
        assert "path" in result
        assert "visited" in result
        assert result["value"] == "test"
        assert set(result["visited"]) == {"a", "b", "c"}
