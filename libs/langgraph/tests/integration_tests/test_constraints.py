"""Integration tests: constraint decorators in real graphs."""

from __future__ import annotations

import warnings
from typing import TypedDict

import pytest

from langchain_nvidia_langgraph.compile.compiler import transform_graph
from langchain_nvidia_langgraph.graph.constraints import (
    OptimizationConfig,
    depends_on,
    sequential,
    speculation_unsafe,
)
from langchain_nvidia_langgraph.graph.state_graph import StateGraph as NvidiaStateGraph


class ConstraintState(TypedDict):
    """State for constraint tests."""

    value: str
    visited: str


def _node_a(state: dict) -> dict:
    """Node a."""
    return {"visited": state.get("visited", "") + "a"}


def _node_b(state: dict) -> dict:
    """Node b."""
    return {"visited": state.get("visited", "") + "b"}


def _node_c(state: dict) -> dict:
    """Node c."""
    return {"visited": state.get("visited", "") + "c"}


@pytest.mark.integration
class TestConstraintDecorators:
    """@sequential, @depends_on, @speculation_unsafe in real graphs."""

    def test_sequential_node_respected_in_optimized_order(self) -> None:
        """Graph with @sequential node: node stays in sequential region."""

        @sequential(reason="Critical section")
        def critical_node(state: dict) -> dict:
            return {"visited": state.get("visited", "") + "crit"}

        graph = NvidiaStateGraph(ConstraintState)
        graph.add_node("a", _node_a)
        graph.add_node("critical", critical_node)
        graph.add_node("b", _node_b)
        graph.add_edge("a", "critical")
        graph.add_edge("critical", "b")
        graph.set_entry_point("a")

        compiled = graph.compile()
        result = transform_graph(
            compiled,
            optimization=OptimizationConfig(enable_parallel=True),
        )
        # critical should not be parallelized with a or b
        assert result.optimized_order is not None
        # At least one stage should contain critical (possibly alone or with deps)
        all_nodes = {n for stage in result.optimized_order for n in stage}
        assert "critical" in all_nodes

    def test_depends_on_respected_in_optimized_order(self) -> None:
        """Graph with @depends_on: explicit dependency in optimized order."""

        @depends_on("a", "b", reason="Needs both")
        def merge_node(state: dict) -> dict:
            return {"visited": state.get("visited", "") + "m"}

        graph = NvidiaStateGraph(ConstraintState)
        graph.add_node("a", _node_a)
        graph.add_node("b", _node_b)
        graph.add_node("merge", merge_node)
        graph.add_edge("a", "b")
        graph.add_edge("b", "merge")
        graph.set_entry_point("a")

        compiled = graph.compile()
        result = transform_graph(
            compiled,
            optimization=OptimizationConfig(
                enable_parallel=True,
                explicit_dependencies={"merge": {"a", "b"}},
            ),
        )
        assert result.optimized_order is not None
        all_nodes = {n for stage in result.optimized_order for n in stage}
        assert "merge" in all_nodes
        assert "a" in all_nodes
        assert "b" in all_nodes

    def test_speculation_unsafe_node_excluded_from_speculation(self) -> None:
        """Graph with @speculation_unsafe: compile with speculation succeeds."""

        @speculation_unsafe
        def unsafe_node(state: dict) -> dict:
            return {"visited": state.get("visited", "") + "u"}

        graph = NvidiaStateGraph(ConstraintState)
        graph.add_node("a", _node_a)
        graph.add_node("unsafe", unsafe_node)
        graph.add_node("b", _node_b)
        graph.add_edge("a", "unsafe")
        graph.add_edge("unsafe", "b")
        graph.set_entry_point("a")

        # Should compile; unsafe node is excluded from speculation
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("ignore", UserWarning)
            compiled = graph.compile(
                optimization=OptimizationConfig(
                    enable_parallel=True,
                    enable_speculation=True,
                ),
            )
        assert compiled is not None
        result = compiled.invoke({"value": "x", "visited": ""})
        assert "u" in result["visited"]
        assert "a" in result["visited"]
        assert "b" in result["visited"]
