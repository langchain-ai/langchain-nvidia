"""Unit tests for compiler public interfaces."""

from __future__ import annotations

from typing import TypedDict

import pytest
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from langchain_nvidia_langgraph.compile.compiler import (
    compile_langgraph,
    transform_graph,
)
from langchain_nvidia_langgraph.graph.constraints import OptimizationConfig

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
def simple_state_graph() -> StateGraph:
    """Minimal StateGraph: a -> b -> END."""
    graph = StateGraph(SimpleState)
    graph.add_node("a", _node_passthrough)
    graph.add_node("b", _node_passthrough)
    graph.add_edge("a", "b")
    graph.set_entry_point("a")
    return graph


@pytest.fixture
def simple_compiled_graph(simple_state_graph: StateGraph) -> CompiledStateGraph:
    """Compiled version of simple_state_graph."""
    return simple_state_graph.compile()


# ---------------------------------------------------------------------------
# compile_langgraph
# ---------------------------------------------------------------------------


def test_compile_langgraph_accepts_state_graph(
    simple_state_graph: StateGraph,
) -> None:
    """compile_langgraph accepts StateGraph and compiles it first."""
    result = compile_langgraph(simple_state_graph)
    assert isinstance(result, CompiledStateGraph)
    assert "a" in result.nodes
    assert "b" in result.nodes


def test_compile_langgraph_accepts_compiled_state_graph(
    simple_compiled_graph: CompiledStateGraph,
) -> None:
    """compile_langgraph accepts CompiledStateGraph directly."""
    result = compile_langgraph(simple_compiled_graph)
    assert isinstance(result, CompiledStateGraph)
    assert result is not None


def test_compile_langgraph_returns_compiled_state_graph(
    simple_compiled_graph: CompiledStateGraph,
) -> None:
    """compile_langgraph returns CompiledStateGraph."""
    result = compile_langgraph(simple_compiled_graph)
    assert isinstance(result, CompiledStateGraph)


def test_compile_langgraph_with_default_optimization(
    simple_compiled_graph: CompiledStateGraph,
) -> None:
    """compile_langgraph with no optimization returns vanilla-style graph."""
    result = compile_langgraph(simple_compiled_graph)
    assert "a" in result.nodes
    assert "b" in result.nodes


def test_compile_langgraph_with_optimization_config(
    simple_compiled_graph: CompiledStateGraph,
) -> None:
    """compile_langgraph accepts OptimizationConfig."""
    result = compile_langgraph(
        simple_compiled_graph,
        optimization=OptimizationConfig(enable_parallel=True),
    )
    assert isinstance(result, CompiledStateGraph)
    assert "a" in result.nodes
    assert "b" in result.nodes


def test_compile_langgraph_with_interrupt_nodes(
    simple_compiled_graph: CompiledStateGraph,
) -> None:
    """compile_langgraph accepts interrupt_before and interrupt_after."""
    result = compile_langgraph(
        simple_compiled_graph,
        interrupt_before=["a"],
        interrupt_after=["b"],
    )
    assert isinstance(result, CompiledStateGraph)


def test_compile_langgraph_with_compile_kwargs(
    simple_compiled_graph: CompiledStateGraph,
) -> None:
    """compile_langgraph passes through compile kwargs (name, debug, etc.)."""
    result = compile_langgraph(
        simple_compiled_graph,
        name="test_graph",
        debug=False,
    )
    assert isinstance(result, CompiledStateGraph)


def test_compile_langgraph_with_max_subgraph_depth(
    simple_compiled_graph: CompiledStateGraph,
) -> None:
    """compile_langgraph accepts max_subgraph_depth."""
    result = compile_langgraph(
        simple_compiled_graph,
        max_subgraph_depth=5,
    )
    assert isinstance(result, CompiledStateGraph)


def test_compile_langgraph_with_speculation_returns_wrapper(
    simple_compiled_graph: CompiledStateGraph,
) -> None:
    """compile_langgraph with enable_speculation returns speculative wrapper."""
    import warnings

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore", UserWarning)
        result = compile_langgraph(
            simple_compiled_graph,
            optimization=OptimizationConfig(
                enable_parallel=True,
                enable_speculation=True,
            ),
        )
    # With speculation, returns SpeculativeGraphWrapper, not raw CompiledStateGraph
    assert result is not None
    # Wrapper typically has .graph or similar - just verify we got something
    assert hasattr(result, "invoke") or hasattr(result, "graph")


# ---------------------------------------------------------------------------
# transform_graph
# ---------------------------------------------------------------------------


def test_transform_graph_returns_transformation_result(
    simple_compiled_graph: CompiledStateGraph,
) -> None:
    """transform_graph returns TransformationResult."""
    result = transform_graph(simple_compiled_graph)
    assert result is not None
    assert hasattr(result, "optimized_order")
    assert hasattr(result, "graph")
    assert hasattr(result, "node_analyses")


def test_transform_graph_populates_optimized_order(
    simple_compiled_graph: CompiledStateGraph,
) -> None:
    """transform_graph produces optimized_order from graph structure."""
    result = transform_graph(simple_compiled_graph)
    assert result.optimized_order is not None
    assert isinstance(result.optimized_order, list)
    assert len(result.optimized_order) >= 1
    assert all(isinstance(stage, set) for stage in result.optimized_order)


def test_transform_graph_accepts_optimization(
    simple_compiled_graph: CompiledStateGraph,
) -> None:
    """transform_graph accepts OptimizationConfig."""
    result = transform_graph(
        simple_compiled_graph,
        optimization=OptimizationConfig(enable_parallel=True),
    )
    assert result.optimized_order is not None
    assert "a" in result.optimized_order[0] or "b" in result.optimized_order[0]


def test_transform_graph_accepts_max_depth(
    simple_compiled_graph: CompiledStateGraph,
) -> None:
    """transform_graph accepts max_depth parameter."""
    result = transform_graph(simple_compiled_graph, max_depth=3)
    assert result.optimized_order is not None


def test_transform_graph_with_none_optimization(
    simple_compiled_graph: CompiledStateGraph,
) -> None:
    """transform_graph accepts optimization=None (uses defaults)."""
    result = transform_graph(
        simple_compiled_graph,
        optimization=None,
    )
    assert result.optimized_order is not None
