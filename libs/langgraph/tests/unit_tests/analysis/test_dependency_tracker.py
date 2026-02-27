"""Unit tests for dependency_tracker public interfaces."""

from __future__ import annotations

from typing import TypedDict

import pytest
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from langchain_nvidia_langgraph.analysis.analyzer import GraphAnalysis, RouterInfo
from langchain_nvidia_langgraph.analysis.dependency_tracker import (
    DependencyTracker,
    NodeDependencies,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class SimpleState(TypedDict):
    """Minimal state schema."""

    value: str


def _node_passthrough(state: dict) -> dict:
    """Simple node."""
    return {}


@pytest.fixture
def acyclic_compiled_graph() -> CompiledStateGraph:
    """Minimal acyclic graph: a -> b -> END."""
    graph = StateGraph(SimpleState)
    graph.add_node("a", _node_passthrough)
    graph.add_node("b", _node_passthrough)
    graph.add_edge("a", "b")
    graph.set_entry_point("a")
    return graph.compile()


@pytest.fixture
def acyclic_analysis() -> GraphAnalysis:
    """GraphAnalysis for acyclic graph (no routers, no cycles)."""
    return GraphAnalysis(
        routers=[],
        entry_point="a",
        has_cycles=False,
        back_edges=[],
    )


@pytest.fixture
def router_compiled_graph() -> CompiledStateGraph:
    """Graph with router: start -> router -+-> a -> end
    +-> b -> end"""
    graph = StateGraph(SimpleState)
    graph.add_node("router", _node_passthrough)
    graph.add_node("a", _node_passthrough)
    graph.add_node("b", _node_passthrough)
    graph.set_entry_point("router")
    graph.add_conditional_edges("router", lambda s: "a", {"a": "a", "b": "b"})
    return graph.compile()


@pytest.fixture
def router_analysis() -> GraphAnalysis:
    """GraphAnalysis with one router."""
    return GraphAnalysis(
        routers=[
            RouterInfo(
                name="router",
                possible_targets=["a", "b"],
                conditional_edge_fn=None,
                path_mapping={"a": "a", "b": "b"},
            ),
        ],
        entry_point="router",
        has_cycles=False,
        back_edges=[],
    )


@pytest.fixture
def cyclic_compiled_graph() -> CompiledStateGraph:
    """Graph with cycle: a -> b -> a."""
    graph = StateGraph(SimpleState)
    graph.add_node("a", _node_passthrough)
    graph.add_node("b", _node_passthrough)
    graph.add_edge("a", "b")
    graph.add_edge("b", "a")
    graph.set_entry_point("a")
    return graph.compile()


@pytest.fixture
def cyclic_analysis() -> GraphAnalysis:
    """GraphAnalysis with cycle (back edge b->a)."""
    return GraphAnalysis(
        routers=[],
        entry_point="a",
        has_cycles=True,
        back_edges=[("b", "a")],
    )


# ---------------------------------------------------------------------------
# NodeDependencies
# ---------------------------------------------------------------------------


def test_node_dependencies_construction() -> None:
    """NodeDependencies stores upstream_routers, upstream_nodes, is_on_path."""
    deps = NodeDependencies(
        upstream_routers=["r1"],
        upstream_nodes=["a", "b"],
        is_on_path={"r1": "a"},
    )
    assert deps.upstream_routers == ["r1"]
    assert deps.upstream_nodes == ["a", "b"]
    assert deps.is_on_path == {"r1": "a"}


def test_node_dependencies_defaults() -> None:
    """NodeDependencies has empty list/dict defaults."""
    deps = NodeDependencies()
    assert deps.upstream_routers == []
    assert deps.upstream_nodes == []
    assert deps.is_on_path == {}


# ---------------------------------------------------------------------------
# DependencyTracker - construction
# ---------------------------------------------------------------------------


def test_dependency_tracker_builds_dependencies(
    acyclic_compiled_graph: CompiledStateGraph,
    acyclic_analysis: GraphAnalysis,
) -> None:
    """DependencyTracker builds dependencies for each node."""
    tracker = DependencyTracker(acyclic_compiled_graph, acyclic_analysis)
    assert "a" in tracker.dependencies
    assert "b" in tracker.dependencies
    assert tracker.dependencies["a"].upstream_nodes == []
    assert tracker.dependencies["b"].upstream_nodes == ["a"]


def test_dependency_tracker_accepts_plans(
    acyclic_compiled_graph: CompiledStateGraph,
    acyclic_analysis: GraphAnalysis,
) -> None:
    """DependencyTracker accepts optional plans parameter."""
    tracker = DependencyTracker(
        acyclic_compiled_graph,
        acyclic_analysis,
        plans={},
    )
    assert tracker.dependencies


def test_dependency_tracker_router_dependencies(
    router_compiled_graph: CompiledStateGraph,
    router_analysis: GraphAnalysis,
) -> None:
    """DependencyTracker identifies upstream routers for branch nodes."""
    tracker = DependencyTracker(router_compiled_graph, router_analysis)
    assert tracker.dependencies["a"].upstream_routers == ["router"]
    assert tracker.dependencies["a"].is_on_path == {"router": "a"}
    assert tracker.dependencies["b"].upstream_routers == ["router"]
    assert tracker.dependencies["b"].is_on_path == {"router": "b"}


# ---------------------------------------------------------------------------
# DependencyTracker.is_ready
# ---------------------------------------------------------------------------


def test_is_ready_returns_false_for_cancelled_node(
    acyclic_compiled_graph: CompiledStateGraph,
    acyclic_analysis: GraphAnalysis,
) -> None:
    """is_ready returns False when node is in cancelled_nodes."""
    tracker = DependencyTracker(acyclic_compiled_graph, acyclic_analysis)
    assert tracker.is_ready("a", {}, {}, set()) is True
    assert tracker.is_ready("a", {}, {}, {"a"}) is False


def test_is_ready_returns_false_when_completed_and_no_reexecution(
    acyclic_compiled_graph: CompiledStateGraph,
    acyclic_analysis: GraphAnalysis,
) -> None:
    """is_ready returns False when node completed and allow_reexecution=False."""
    tracker = DependencyTracker(acyclic_compiled_graph, acyclic_analysis)
    assert tracker.is_ready("a", {"a": 1}, {}, set()) is True
    assert tracker.is_ready("a", {"a": 1}, {}, set(), allow_reexecution=False) is False


def test_is_ready_returns_true_for_unknown_node(
    acyclic_compiled_graph: CompiledStateGraph,
    acyclic_analysis: GraphAnalysis,
) -> None:
    """is_ready returns True when node not in dependencies (unknown node)."""
    tracker = DependencyTracker(acyclic_compiled_graph, acyclic_analysis)
    assert tracker.is_ready("nonexistent", {}, {}, set()) is True


def test_is_ready_acyclic_entry_node_ready(
    acyclic_compiled_graph: CompiledStateGraph,
    acyclic_analysis: GraphAnalysis,
) -> None:
    """In acyclic graph, entry node is ready with no completions."""
    tracker = DependencyTracker(acyclic_compiled_graph, acyclic_analysis)
    assert tracker.is_ready("a", {}, {}, set()) is True


def test_is_ready_acyclic_downstream_requires_predecessor(
    acyclic_compiled_graph: CompiledStateGraph,
    acyclic_analysis: GraphAnalysis,
) -> None:
    """In acyclic graph, node requires upstream nodes completed."""
    tracker = DependencyTracker(acyclic_compiled_graph, acyclic_analysis)
    assert tracker.is_ready("b", {}, {}, set()) is False
    assert tracker.is_ready("b", {"a": None}, {}, set()) is True


def test_is_ready_router_chose_branch(
    router_compiled_graph: CompiledStateGraph,
    router_analysis: GraphAnalysis,
) -> None:
    """With router, node is ready only if router chose path to that node."""
    tracker = DependencyTracker(router_compiled_graph, router_analysis)
    assert tracker.is_ready("a", {}, {"router": "a"}, set()) is True
    assert tracker.is_ready("b", {}, {"router": "a"}, set()) is False
    assert tracker.is_ready("b", {}, {"router": "b"}, set()) is True


def test_is_ready_cyclic_requires_forward_predecessor(
    cyclic_compiled_graph: CompiledStateGraph,
    cyclic_analysis: GraphAnalysis,
) -> None:
    """In cyclic graph, node requires forward predecessors completed."""
    tracker = DependencyTracker(cyclic_compiled_graph, cyclic_analysis)
    assert tracker.is_ready("a", {}, {}, set()) is True
    assert tracker.is_ready("b", {}, {}, set()) is False
    assert tracker.is_ready("b", {"a": None}, {}, set()) is True
