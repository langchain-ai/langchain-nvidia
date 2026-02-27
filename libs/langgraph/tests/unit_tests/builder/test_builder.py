"""Unit tests for builder public interfaces."""

from __future__ import annotations

import dataclasses
from typing import TypedDict

import pytest
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from nat_app.graph.scheduling import CompilationResult
from nat_app.graph.types import Graph

from langchain_nvidia_langgraph.builder.builder import (
    CycleContext,
    OptimizedGraph,
    OptimizedGraphBuilder,
    build_optimized_langgraph,
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


def _make_transformation_graph(node_names: list[str]) -> Graph:
    """Create nat_app Graph with nodes that have func."""
    g = Graph()
    for name in node_names:
        g.add_node(name, func=_node_passthrough)
    g.entry_point = node_names[0] if node_names else ""
    return g


def _add_edges_linear(g: Graph, nodes: list[str]) -> None:
    """Add linear edges a->b->c."""
    for i in range(len(nodes) - 1):
        g.add_edge(nodes[i], nodes[i + 1])


@pytest.fixture
def simple_compiled_graph() -> CompiledStateGraph:
    """Minimal compiled graph: a -> b -> END."""
    graph = StateGraph(SimpleState)
    graph.add_node("a", _node_passthrough)
    graph.add_node("b", _node_passthrough)
    graph.add_edge("a", "b")
    graph.set_entry_point("a")
    return graph.compile()


@pytest.fixture
def compilation_result_linear() -> CompilationResult:
    """CompilationResult for linear a->b (staged build)."""
    g = _make_transformation_graph(["a", "b"])
    _add_edges_linear(g, ["a", "b"])
    return CompilationResult(
        graph=g,
        node_analyses={},
        necessary_edges=set(),
        unnecessary_edges=set(),
        optimized_order=[{"a"}, {"b"}],
        topology=None,
    )


@pytest.fixture
def compilation_result_parallel() -> CompilationResult:
    """CompilationResult for parallel [a,b] -> c (fan-out/fan-in)."""
    g = _make_transformation_graph(["a", "b", "c"])
    g.add_edge("a", "c")
    g.add_edge("b", "c")
    return CompilationResult(
        graph=g,
        node_analyses={},
        necessary_edges=set(),
        unnecessary_edges=set(),
        optimized_order=[{"a", "b"}, {"c"}],
        topology=None,
    )


@pytest.fixture
def compilation_result_empty_stages() -> CompilationResult:
    """CompilationResult with no stages."""
    g = _make_transformation_graph(["a"])
    return CompilationResult(
        graph=g,
        node_analyses={},
        necessary_edges=set(),
        unnecessary_edges=set(),
        optimized_order=[],
        topology=None,
    )


# ---------------------------------------------------------------------------
# CycleContext
# ---------------------------------------------------------------------------


def test_cycle_context_construction() -> None:
    """CycleContext stores all_cycle_nodes, cycle_back_edges, absorbed_entries."""
    ctx = CycleContext(
        all_cycle_nodes=frozenset({"a", "b"}),
        cycle_back_edges=frozenset({("b", "a")}),
        absorbed_entries={"a": "__cycle_a_entry__"},
    )
    assert ctx.all_cycle_nodes == frozenset({"a", "b"})
    assert ctx.cycle_back_edges == frozenset({("b", "a")})
    assert ctx.absorbed_entries == {"a": "__cycle_a_entry__"}


def test_cycle_context_default_absorbed_entries() -> None:
    """CycleContext absorbed_entries defaults to empty dict."""
    ctx = CycleContext(
        all_cycle_nodes=frozenset(),
        cycle_back_edges=frozenset(),
    )
    assert ctx.absorbed_entries == {}


def test_cycle_context_is_frozen() -> None:
    """CycleContext is immutable (frozen dataclass)."""
    ctx = CycleContext(
        all_cycle_nodes=frozenset({"a"}),
        cycle_back_edges=frozenset(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.all_cycle_nodes = frozenset({"b"})  # type: ignore[misc]


# ---------------------------------------------------------------------------
# OptimizedGraph
# ---------------------------------------------------------------------------


def test_optimized_graph_construction(
    simple_compiled_graph: CompiledStateGraph,
    compilation_result_linear: CompilationResult,
) -> None:
    """OptimizedGraph stores original, optimized, transformation, stages."""
    result = build_optimized_langgraph(simple_compiled_graph, compilation_result_linear)
    assert result.original_graph is simple_compiled_graph
    assert result.optimized_graph is not None
    assert result.transformation is compilation_result_linear
    assert result.stages == [{"a"}, {"b"}]
    assert result.speedup_estimate >= 1.0


def test_optimized_graph_repr(
    simple_compiled_graph: CompiledStateGraph,
    compilation_result_linear: CompilationResult,
) -> None:
    """OptimizedGraph __repr__ includes stages count and speedup."""
    result = build_optimized_langgraph(simple_compiled_graph, compilation_result_linear)
    repr_str = repr(result)
    assert "OptimizedGraph" in repr_str
    assert "stages" in repr_str
    assert "speedup" in repr_str.lower() or "x" in repr_str


# ---------------------------------------------------------------------------
# OptimizedGraphBuilder
# ---------------------------------------------------------------------------


def test_optimized_graph_builder_init(
    simple_compiled_graph: CompiledStateGraph,
    compilation_result_linear: CompilationResult,
) -> None:
    """OptimizedGraphBuilder stores original_graph and transformation."""
    builder = OptimizedGraphBuilder(simple_compiled_graph, compilation_result_linear)
    assert builder.original_graph is simple_compiled_graph
    assert builder.transformation is compilation_result_linear


def test_optimized_graph_builder_accepts_compile_kwargs(
    simple_compiled_graph: CompiledStateGraph,
    compilation_result_linear: CompilationResult,
) -> None:
    """OptimizedGraphBuilder accepts optional compile_kwargs."""
    builder = OptimizedGraphBuilder(
        simple_compiled_graph,
        compilation_result_linear,
        compile_kwargs={"debug": True},
    )
    result = builder.build()
    assert result.optimized_graph is not None


def test_optimized_graph_builder_build_returns_optimized_graph(
    simple_compiled_graph: CompiledStateGraph,
    compilation_result_linear: CompilationResult,
) -> None:
    """OptimizedGraphBuilder.build returns OptimizedGraph."""
    builder = OptimizedGraphBuilder(simple_compiled_graph, compilation_result_linear)
    result = builder.build()
    assert isinstance(result, OptimizedGraph)
    assert result.optimized_graph is not None
    assert "a" in result.optimized_graph.nodes
    assert "b" in result.optimized_graph.nodes


def test_optimized_graph_builder_build_idempotent(
    simple_compiled_graph: CompiledStateGraph,
    compilation_result_linear: CompilationResult,
) -> None:
    """OptimizedGraphBuilder.build returns cached result on second call."""
    builder = OptimizedGraphBuilder(simple_compiled_graph, compilation_result_linear)
    r1 = builder.build()
    r2 = builder.build()
    assert r1 is r2


def test_optimized_graph_builder_empty_stages_returns_original(
    simple_compiled_graph: CompiledStateGraph,
    compilation_result_empty_stages: CompilationResult,
) -> None:
    """When optimized_order is empty, build returns original graph as optimized."""
    result = build_optimized_langgraph(
        simple_compiled_graph,
        compilation_result_empty_stages,
    )
    assert result.stages == []
    assert result.speedup_estimate == 1.0
    assert result.optimized_graph is simple_compiled_graph


# ---------------------------------------------------------------------------
# build_optimized_langgraph
# ---------------------------------------------------------------------------


def test_build_optimized_langgraph_returns_optimized_graph(
    simple_compiled_graph: CompiledStateGraph,
    compilation_result_linear: CompilationResult,
) -> None:
    """build_optimized_langgraph returns OptimizedGraph."""
    result = build_optimized_langgraph(simple_compiled_graph, compilation_result_linear)
    assert isinstance(result, OptimizedGraph)
    assert result.original_graph is simple_compiled_graph
    assert result.optimized_graph is not None


def test_build_optimized_langgraph_linear_stages(
    simple_compiled_graph: CompiledStateGraph,
    compilation_result_linear: CompilationResult,
) -> None:
    """build_optimized_langgraph produces graph for linear stages."""
    result = build_optimized_langgraph(simple_compiled_graph, compilation_result_linear)
    assert result.stages == [{"a"}, {"b"}]
    assert len(result.stages) == 2


def test_build_optimized_langgraph_parallel_stages(
    simple_compiled_graph: CompiledStateGraph,
    compilation_result_parallel: CompilationResult,
) -> None:
    """build_optimized_langgraph produces fan-out/fan-in for parallel stages."""
    # Need compiled graph with a,b,c for schema
    graph = StateGraph(SimpleState)
    graph.add_node("a", _node_passthrough)
    graph.add_node("b", _node_passthrough)
    graph.add_node("c", _node_passthrough)
    graph.add_edge("a", "c")
    graph.add_edge("b", "c")
    graph.set_entry_point("a")
    compiled = graph.compile()

    result = build_optimized_langgraph(compiled, compilation_result_parallel)
    assert result.stages == [{"a", "b"}, {"c"}]
    assert result.speedup_estimate > 1.0
    # Optimized graph should have a, b, c (and possibly fanout/collector nodes)
    opt_nodes = [n for n in result.optimized_graph.nodes if not n.startswith("__")]
    assert "a" in opt_nodes
    assert "b" in opt_nodes
    assert "c" in opt_nodes


def test_build_optimized_langgraph_accepts_compile_kwargs(
    simple_compiled_graph: CompiledStateGraph,
    compilation_result_linear: CompilationResult,
) -> None:
    """build_optimized_langgraph accepts optional compile_kwargs."""
    result = build_optimized_langgraph(
        simple_compiled_graph,
        compilation_result_linear,
        compile_kwargs={},
    )
    assert result.optimized_graph is not None
