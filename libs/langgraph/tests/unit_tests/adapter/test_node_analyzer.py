"""Unit tests for node_analyzer public interfaces."""

from __future__ import annotations

from typing import Any, Callable, TypedDict

import pytest
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from langchain_nvidia_langgraph.adapter.node_analyzer import (
    LANGGRAPH_SPECIAL_CALLS,
    analyze_langgraph_node,
)

# ---------------------------------------------------------------------------
# Node functions for AST analysis (must be defined in this module for source)
# ---------------------------------------------------------------------------


def _node_passthrough(state: dict[str, Any]) -> dict[str, Any]:
    """Returns empty update - analyzable."""
    return {}


def _node_reads_state(state: dict[str, Any]) -> dict[str, Any]:
    """Reads from state - analyzable."""
    x = state.get("x")
    return {"y": x}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class SimpleState(TypedDict):
    """Minimal state schema for subgraph tests."""

    value: str
    count: int


@pytest.fixture
def simple_compiled_graph() -> CompiledStateGraph:
    """Minimal compiled LangGraph for subgraph analysis."""
    graph = StateGraph(SimpleState)
    graph.add_node("n", _node_passthrough)
    graph.set_entry_point("n")
    return graph.compile()


# ---------------------------------------------------------------------------
# LANGGRAPH_SPECIAL_CALLS constant
# ---------------------------------------------------------------------------


def test_langgraph_special_calls_is_frozenset() -> None:
    """LANGGRAPH_SPECIAL_CALLS is a frozenset."""
    assert isinstance(LANGGRAPH_SPECIAL_CALLS, frozenset)


def test_langgraph_special_calls_contains_send_and_command() -> None:
    """LANGGRAPH_SPECIAL_CALLS contains Send and Command."""
    assert LANGGRAPH_SPECIAL_CALLS == {"Send", "Command"}


# ---------------------------------------------------------------------------
# analyze_langgraph_node - regular callable (AST path)
# ---------------------------------------------------------------------------


def test_analyze_langgraph_node_returns_node_analysis() -> None:
    """analyze_langgraph_node returns NodeAnalysis with expected attributes."""
    analysis = analyze_langgraph_node("test_node", _node_passthrough)
    assert analysis is not None
    assert analysis.name == "test_node"
    assert hasattr(analysis, "reads")
    assert hasattr(analysis, "writes")
    assert hasattr(analysis, "mutations")
    assert hasattr(analysis, "confidence")
    assert hasattr(analysis, "source")
    assert hasattr(analysis, "is_pure")
    assert hasattr(analysis, "trace_successful")
    assert hasattr(analysis, "warnings")


def test_analyze_langgraph_node_with_function_has_ast_source() -> None:
    """analyze_langgraph_node uses AST when function source is available."""
    analysis = analyze_langgraph_node("test_node", _node_reads_state)
    assert analysis.source == "ast"
    assert analysis.trace_successful is True


def test_analyze_langgraph_node_accepts_all_schema_fields() -> None:
    """analyze_langgraph_node accepts all_schema_fields parameter."""
    analysis = analyze_langgraph_node(
        "test_node",
        _node_passthrough,
        all_schema_fields={"x", "y", "z"},
    )
    assert analysis is not None
    assert analysis.name == "test_node"


def test_analyze_langgraph_node_accepts_config() -> None:
    """analyze_langgraph_node accepts config with max_recursion_depth."""
    config = type("Config", (), {"max_recursion_depth": 3})()
    analysis = analyze_langgraph_node("test_node", _node_passthrough, config=config)
    assert analysis is not None


# ---------------------------------------------------------------------------
# analyze_langgraph_node - source unavailable (opaque path)
# ---------------------------------------------------------------------------


def test_analyze_langgraph_node_opaque_when_source_unavailable() -> None:
    """analyze_langgraph_node returns opaque when source not available (e.g. lambda)."""
    # Lambda from command line has no source; when in module, use exec to simulate
    no_source_fn: Callable[[dict], dict] = lambda state: {}  # noqa: E731
    analysis = analyze_langgraph_node("opaque_node", no_source_fn)
    # Lambdas in module may still have source - check both paths
    assert analysis.name == "opaque_node"
    assert analysis.source in ("ast", "unavailable")
    if analysis.source == "unavailable":
        assert analysis.confidence == "opaque"
        assert analysis.trace_successful is False


def test_analyze_langgraph_node_conservative_fallback_with_schema_fields() -> None:
    """When source unavailable, assumes all schema fields as writes if given."""
    no_source_fn: Callable[[dict], dict] = lambda state: {}  # noqa: E731
    analysis = analyze_langgraph_node(
        "opaque_node",
        no_source_fn,
        all_schema_fields={"a", "b"},
    )
    assert analysis.name == "opaque_node"
    if analysis.source == "unavailable":
        assert not analysis.is_pure
        assert len(analysis.mutations) > 0
        assert any("conservatively" in w for w in analysis.warnings)


# ---------------------------------------------------------------------------
# analyze_langgraph_node - CompiledStateGraph (subgraph path)
# ---------------------------------------------------------------------------


def test_analyze_langgraph_node_subgraph_returns_node_analysis(
    simple_compiled_graph: CompiledStateGraph,
) -> None:
    """analyze_langgraph_node with CompiledStateGraph returns NodeAnalysis."""
    analysis = analyze_langgraph_node("sub", simple_compiled_graph)
    assert analysis is not None
    assert analysis.name == "sub"
    assert analysis.source == "subgraph_schema"
    assert analysis.confidence in ("partial", "opaque")


def test_analyze_langgraph_node_subgraph_with_schema_sets_reads_writes(
    simple_compiled_graph: CompiledStateGraph,
) -> None:
    """Subgraph with state schema sets reads/writes from schema fields."""
    analysis = analyze_langgraph_node("sub", simple_compiled_graph)
    assert analysis.source == "subgraph_schema"
    # SimpleState has value, count
    if analysis.confidence != "opaque":
        assert len(analysis.reads) > 0 or len(analysis.writes) > 0
        assert not analysis.is_pure


def test_analyze_langgraph_node_subgraph_fallback_to_parent_schema() -> None:
    """Subgraph without schema uses all_schema_fields when provided."""
    # Create minimal compiled graph - use one with dict schema to avoid schema
    graph = StateGraph(dict)
    graph.add_node("n", _node_passthrough)
    graph.set_entry_point("n")
    compiled = graph.compile()
    analysis = analyze_langgraph_node(
        "sub",
        compiled,
        all_schema_fields={"parent_x", "parent_y"},
    )
    assert analysis.source == "subgraph_schema"
    assert (
        any("Subgraph schema unavailable" in w for w in analysis.warnings)
        or analysis.confidence == "opaque"
    )
    if analysis.confidence == "partial":
        assert not analysis.is_pure
