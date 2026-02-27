"""Unit tests for state_graph public interfaces."""

from __future__ import annotations

from typing import TypedDict

import pytest
from langgraph.graph import StateGraph as LangGraphStateGraph
from langgraph.graph.state import CompiledStateGraph

from langchain_nvidia_langgraph.graph.constraints import OptimizationConfig
from langchain_nvidia_langgraph.graph.state_graph import (
    CompilableGraph,
    StateGraph,
    with_app_compile,
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
def state_graph() -> StateGraph:
    """Minimal StateGraph: a -> b -> END."""
    graph = StateGraph(SimpleState)
    graph.add_node("a", _node_passthrough)
    graph.add_node("b", _node_passthrough)
    graph.add_edge("a", "b")
    graph.set_entry_point("a")
    return graph


@pytest.fixture
def langgraph_state_graph() -> LangGraphStateGraph:
    """Minimal LangGraph StateGraph (for wrapping)."""
    graph = LangGraphStateGraph(SimpleState)
    graph.add_node("a", _node_passthrough)
    graph.add_node("b", _node_passthrough)
    graph.add_edge("a", "b")
    graph.set_entry_point("a")
    return graph


# ---------------------------------------------------------------------------
# StateGraph
# ---------------------------------------------------------------------------


def test_state_graph_is_subclass_of_langgraph_state_graph() -> None:
    """StateGraph inherits from LangGraph StateGraph."""
    assert issubclass(StateGraph, LangGraphStateGraph)


def test_state_graph_builder_api(state_graph: StateGraph) -> None:
    """StateGraph supports standard builder API (add_node, add_edge, etc.)."""
    assert "a" in state_graph.nodes
    assert "b" in state_graph.nodes


def test_state_graph_compile_default_returns_compiled(state_graph: StateGraph) -> None:
    """StateGraph.compile() with no optimization returns CompiledStateGraph."""
    compiled = state_graph.compile()
    assert isinstance(compiled, CompiledStateGraph)
    assert "a" in compiled.nodes
    assert "b" in compiled.nodes


def test_state_graph_compile_with_optimization(state_graph: StateGraph) -> None:
    """StateGraph.compile(optimization=...) accepts OptimizationConfig."""
    compiled = state_graph.compile(
        optimization=OptimizationConfig(enable_parallel=True),
    )
    assert compiled is not None
    assert hasattr(compiled, "invoke")


def test_state_graph_compile_with_optimization_none(state_graph: StateGraph) -> None:
    """StateGraph.compile(optimization=None) gives vanilla behavior."""
    compiled = state_graph.compile(optimization=None)
    assert isinstance(compiled, CompiledStateGraph)


def test_state_graph_compile_passes_through_compile_params(
    state_graph: StateGraph,
) -> None:
    """StateGraph.compile() passes checkpointer, name, debug, etc."""
    compiled = state_graph.compile(
        name="test_graph",
        debug=False,
    )
    assert isinstance(compiled, CompiledStateGraph)


def test_state_graph_compile_with_interrupt_nodes(state_graph: StateGraph) -> None:
    """StateGraph.compile() accepts interrupt_before and interrupt_after."""
    compiled = state_graph.compile(
        interrupt_before=["a"],
        interrupt_after=["b"],
    )
    assert isinstance(compiled, CompiledStateGraph)


def test_state_graph_compile_with_max_subgraph_depth(state_graph: StateGraph) -> None:
    """StateGraph.compile() accepts max_subgraph_depth."""
    compiled = state_graph.compile(max_subgraph_depth=5)
    assert isinstance(compiled, CompiledStateGraph)


# ---------------------------------------------------------------------------
# CompilableGraph
# ---------------------------------------------------------------------------


def test_compilable_graph_wraps_state_graph(state_graph: StateGraph) -> None:
    """CompilableGraph wraps StateGraph and compiles."""
    compilable = CompilableGraph(state_graph)
    compiled = compilable.compile()
    assert isinstance(compiled, CompiledStateGraph)
    assert "a" in compiled.nodes
    assert "b" in compiled.nodes


def test_compilable_graph_wraps_langgraph_state_graph(
    langgraph_state_graph: LangGraphStateGraph,
) -> None:
    """CompilableGraph wraps LangGraph StateGraph."""
    compilable = CompilableGraph(langgraph_state_graph)
    compiled = compilable.compile()
    assert isinstance(compiled, CompiledStateGraph)
    assert "a" in compiled.nodes
    assert "b" in compiled.nodes


def test_compilable_graph_compile_with_optimization(
    state_graph: StateGraph,
) -> None:
    """CompilableGraph.compile(optimization=...) accepts OptimizationConfig."""
    compilable = CompilableGraph(state_graph)
    compiled = compilable.compile(
        optimization=OptimizationConfig(enable_parallel=True),
    )
    assert compiled is not None
    assert hasattr(compiled, "invoke")


def test_compilable_graph_compile_passes_params(state_graph: StateGraph) -> None:
    """CompilableGraph.compile() passes through compile params."""
    compilable = CompilableGraph(state_graph)
    compiled = compilable.compile(
        name="wrapped_graph",
        interrupt_before=["a"],
        max_subgraph_depth=3,
    )
    assert isinstance(compiled, CompiledStateGraph)


# ---------------------------------------------------------------------------
# with_app_compile
# ---------------------------------------------------------------------------


def test_with_app_compile_returns_compilable_graph(
    state_graph: StateGraph,
) -> None:
    """with_app_compile returns CompilableGraph."""
    compilable = with_app_compile(state_graph)
    assert isinstance(compilable, CompilableGraph)


def test_with_app_compile_accepts_state_graph(state_graph: StateGraph) -> None:
    """with_app_compile accepts langchain_nvidia StateGraph."""
    compilable = with_app_compile(state_graph)
    compiled = compilable.compile()
    assert isinstance(compiled, CompiledStateGraph)


def test_with_app_compile_accepts_langgraph_state_graph(
    langgraph_state_graph: LangGraphStateGraph,
) -> None:
    """with_app_compile accepts LangGraph StateGraph."""
    compilable = with_app_compile(langgraph_state_graph)
    compiled = compilable.compile()
    assert isinstance(compiled, CompiledStateGraph)


def test_with_app_compile_roundtrip(state_graph: StateGraph) -> None:
    """with_app_compile -> compile(optimization=...) produces runnable graph."""
    compilable = with_app_compile(state_graph)
    compiled = compilable.compile(
        optimization=OptimizationConfig(enable_parallel=True),
    )
    result = compiled.invoke({"value": "hello"})
    assert result is not None
    assert "value" in result
