"""Integration tests: nested subgraphs and max_subgraph_depth."""

from __future__ import annotations

from typing import TypedDict

import pytest
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from langchain_nvidia_langgraph.compile.compiler import (
    transform_graph,
)
from langchain_nvidia_langgraph.graph.constraints import OptimizationConfig
from langchain_nvidia_langgraph.graph.state_graph import StateGraph as NvidiaStateGraph


class SubgraphState(TypedDict):
    """Shared state for parent and subgraph."""

    value: str
    visited: str


def _node_pre(state: dict) -> dict:
    """Pre-subgraph node."""
    return {"visited": state.get("visited", "") + "pre"}


def _node_post(state: dict) -> dict:
    """Post-subgraph node."""
    return {"visited": state.get("visited", "") + "post"}


def _sub_node_a(state: dict) -> dict:
    """Subgraph node a."""
    return {"visited": state.get("visited", "") + "s_a"}


def _sub_node_b(state: dict) -> dict:
    """Subgraph node b."""
    return {"visited": state.get("visited", "") + "s_b"}


@pytest.fixture
def subgraph_compiled() -> CompiledStateGraph:
    """Compiled subgraph: s_a -> s_b -> END."""
    graph = StateGraph(SubgraphState)
    graph.add_node("s_a", _sub_node_a)
    graph.add_node("s_b", _sub_node_b)
    graph.add_edge("s_a", "s_b")
    graph.set_entry_point("s_a")
    return graph.compile()


@pytest.fixture
def parent_graph_with_subgraph(
    subgraph_compiled: CompiledStateGraph,
) -> NvidiaStateGraph:
    """Parent graph: pre -> subgraph -> post. Subgraph is a CompiledStateGraph node."""
    graph = NvidiaStateGraph(SubgraphState)
    graph.add_node("pre", _node_pre)
    graph.add_node("sub", subgraph_compiled)  # Subgraph as node
    graph.add_node("post", _node_post)
    graph.add_edge("pre", "sub")
    graph.add_edge("sub", "post")
    graph.set_entry_point("pre")
    return graph


@pytest.mark.integration
class TestSubgraphs:
    """Nested subgraph optimization."""

    def test_parent_with_subgraph_compiles_and_invokes(
        self,
        parent_graph_with_subgraph: NvidiaStateGraph,
    ) -> None:
        """Graph with subgraph node compiles and invoke runs full pipeline."""
        compiled = parent_graph_with_subgraph.compile()
        assert isinstance(compiled, CompiledStateGraph)
        result = compiled.invoke({"value": "x", "visited": ""})
        assert "pre" in result["visited"]
        assert "s_a" in result["visited"]
        assert "s_b" in result["visited"]
        assert "post" in result["visited"]

    def test_parent_with_subgraph_parallel_optimization(
        self,
        parent_graph_with_subgraph: NvidiaStateGraph,
    ) -> None:
        """Compile with parallel optimization and max_subgraph_depth."""
        compiled = parent_graph_with_subgraph.compile(
            optimization=OptimizationConfig(enable_parallel=True),
            max_subgraph_depth=2,
        )
        result = compiled.invoke({"value": "x", "visited": ""})
        assert "pre" in result["visited"]
        assert "s_a" in result["visited"]
        assert "s_b" in result["visited"]
        assert "post" in result["visited"]

    def test_transform_graph_with_subgraph(
        self,
        parent_graph_with_subgraph: NvidiaStateGraph,
    ) -> None:
        """transform_graph handles graph with subgraph nodes."""
        compiled = parent_graph_with_subgraph.compile()
        result = transform_graph(
            compiled,
            optimization=OptimizationConfig(enable_parallel=True),
            max_depth=2,
        )
        assert result.optimized_order is not None
        all_nodes = {n for stage in result.optimized_order for n in stage}
        assert "pre" in all_nodes
        assert "sub" in all_nodes
        assert "post" in all_nodes
