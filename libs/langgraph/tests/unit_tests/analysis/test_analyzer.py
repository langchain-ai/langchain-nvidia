"""Unit tests for analyzer public interfaces."""

from __future__ import annotations

from typing import TypedDict

import pytest
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from nat_app.graph.scheduling import CompilationResult
from nat_app.graph.topology import CycleInfo, GraphTopology
from nat_app.graph.topology import RouterInfo as TopoRouterInfo
from nat_app.graph.types import Graph

from langchain_nvidia_langgraph.analysis.analyzer import GraphAnalysis, RouterInfo

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
def simple_compiled_graph() -> CompiledStateGraph:
    """Minimal compiled LangGraph: a -> b -> END."""
    graph = StateGraph(SimpleState)
    graph.add_node("a", _node_passthrough)
    graph.add_node("b", _node_passthrough)
    graph.add_edge("a", "b")
    graph.set_entry_point("a")
    return graph.compile()


@pytest.fixture
def minimal_graph() -> Graph:
    """Minimal nat_app Graph."""
    g = Graph()
    g.add_node("a")
    g.add_node("b")
    g.add_edge("a", "b")
    g.entry_point = "a"
    return g


@pytest.fixture
def compilation_result_no_topology(minimal_graph: Graph) -> CompilationResult:
    """CompilationResult with topology=None."""
    return CompilationResult(
        graph=minimal_graph,
        node_analyses={},
        necessary_edges=set(),
        unnecessary_edges=set(),
        optimized_order=[{"a"}, {"b"}],
        topology=None,
    )


@pytest.fixture
def topology_with_router_and_cycle() -> GraphTopology:
    """GraphTopology with one router and one cycle."""
    routers = [
        TopoRouterInfo(node="router1", branches={"x": ["a"], "y": ["b"]}),
    ]
    cycles = [
        CycleInfo(
            nodes=frozenset({"a", "b"}),
            entry_node="a",
            exit_node="b",
            back_edge=("b", "a"),
        ),
    ]
    return GraphTopology(
        nodes={"a", "b"},
        edges={("a", "b"), ("b", "a")},
        node_types={},
        routers=routers,
        cycles=cycles,
        parallelizable_regions=[],
        sequential_regions=[],
    )


@pytest.fixture
def compilation_result_with_topology(
    minimal_graph: Graph,
    topology_with_router_and_cycle: GraphTopology,
) -> CompilationResult:
    """CompilationResult with topology."""
    return CompilationResult(
        graph=minimal_graph,
        node_analyses={},
        necessary_edges=set(),
        unnecessary_edges=set(),
        optimized_order=[{"a"}, {"b"}],
        topology=topology_with_router_and_cycle,
    )


# ---------------------------------------------------------------------------
# RouterInfo
# ---------------------------------------------------------------------------


def test_router_info_construction() -> None:
    """RouterInfo stores name, possible_targets, conditional_edge_fn, path_mapping."""
    ri = RouterInfo(
        name="r1",
        possible_targets=["a", "b"],
        conditional_edge_fn=None,
        path_mapping={"x": "a", "y": "b"},
    )
    assert ri.name == "r1"
    assert ri.possible_targets == ["a", "b"]
    assert ri.conditional_edge_fn is None
    assert ri.path_mapping == {"x": "a", "y": "b"}


def test_router_info_default_path_mapping() -> None:
    """RouterInfo path_mapping defaults to empty dict."""
    ri = RouterInfo(name="r1", possible_targets=[], conditional_edge_fn=None)
    assert ri.path_mapping == {}


# ---------------------------------------------------------------------------
# GraphAnalysis
# ---------------------------------------------------------------------------


def test_graph_analysis_default_construction() -> None:
    """GraphAnalysis has sensible defaults when constructed directly."""
    analysis = GraphAnalysis()
    assert analysis.routers == []
    assert analysis.entry_point == ""
    assert analysis.has_cycles is False
    assert analysis.back_edges == []


def test_graph_analysis_from_compilation_result_returns_empty_when_no_topology(
    compilation_result_no_topology: CompilationResult,
    simple_compiled_graph: CompiledStateGraph,
) -> None:
    """from_compilation_result returns empty GraphAnalysis when topology None."""
    analysis = GraphAnalysis.from_compilation_result(
        compilation_result_no_topology,
        simple_compiled_graph,
    )
    assert analysis.routers == []
    assert analysis.entry_point == ""
    assert analysis.has_cycles is False
    assert analysis.back_edges == []


def test_graph_analysis_from_compilation_result_empty_when_no_optimized_topology(
    compilation_result_with_topology: CompilationResult,
    simple_compiled_graph: CompiledStateGraph,
) -> None:
    """from_compilation_result returns empty when optimized_topology None."""
    analysis = GraphAnalysis.from_compilation_result(
        compilation_result_with_topology,
        simple_compiled_graph,
        optimized_topology=None,
    )
    assert analysis.routers == []
    assert analysis.entry_point == ""
    assert analysis.has_cycles is False
    assert analysis.back_edges == []


def test_graph_analysis_from_compilation_result_populates_from_topology(
    compilation_result_with_topology: CompilationResult,
    simple_compiled_graph: CompiledStateGraph,
    topology_with_router_and_cycle: GraphTopology,
) -> None:
    """from_compilation_result populates routers, entry_point, has_cycles."""
    analysis = GraphAnalysis.from_compilation_result(
        compilation_result_with_topology,
        simple_compiled_graph,
        optimized_topology=topology_with_router_and_cycle,
        optimized_entry_point="a",
    )
    assert len(analysis.routers) == 1
    assert analysis.routers[0].name == "router1"
    assert set(analysis.routers[0].possible_targets) == {"a", "b"}
    assert analysis.entry_point == "a"
    assert analysis.has_cycles is True
    assert analysis.back_edges == [("b", "a")]


def test_graph_analysis_from_compilation_result_uses_find_entry_point_when_not_provided(
    compilation_result_with_topology: CompilationResult,
    simple_compiled_graph: CompiledStateGraph,
    topology_with_router_and_cycle: GraphTopology,
) -> None:
    """from_compilation_result derives entry_point when optimized_entry_point empty."""
    analysis = GraphAnalysis.from_compilation_result(
        compilation_result_with_topology,
        simple_compiled_graph,
        optimized_topology=topology_with_router_and_cycle,
        optimized_entry_point="",
    )
    assert analysis.entry_point == "a"


def test_graph_analysis_from_compilation_result_no_cycles(
    minimal_graph: Graph,
    simple_compiled_graph: CompiledStateGraph,
) -> None:
    """from_compilation_result sets has_cycles=False when topology has no cycles."""
    topo = GraphTopology(
        nodes={"a", "b"},
        edges={("a", "b")},
        node_types={},
        routers=[],
        cycles=[],
        parallelizable_regions=[],
        sequential_regions=[],
    )
    cr = CompilationResult(
        graph=minimal_graph,
        node_analyses={},
        necessary_edges=set(),
        unnecessary_edges=set(),
        optimized_order=[{"a"}, {"b"}],
        topology=topo,
    )
    analysis = GraphAnalysis.from_compilation_result(
        cr,
        simple_compiled_graph,
        optimized_topology=topo,
        optimized_entry_point="a",
    )
    assert analysis.has_cycles is False
    assert analysis.back_edges == []
    assert analysis.routers == []
