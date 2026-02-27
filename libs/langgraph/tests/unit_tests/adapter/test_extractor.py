"""Unit tests for LangGraphExtractor public interfaces."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict
from unittest.mock import MagicMock, patch

import pytest
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from langchain_nvidia_langgraph.adapter.extractor import LangGraphExtractor
from langchain_nvidia_langgraph.adapter.llm_detector import LangChainLLMDetector

# ---------------------------------------------------------------------------
# Fixtures: minimal compiled graphs
# ---------------------------------------------------------------------------


class SimpleState(TypedDict):
    """Minimal state schema for tests."""

    value: str
    count: int


def _node_a(state: dict[str, Any]) -> dict[str, Any]:
    """Simple node that returns empty update."""
    return {}


def _node_b(state: dict[str, Any]) -> dict[str, Any]:
    """Simple node that returns empty update."""
    return {}


def _reducer_for_test(left: list, right: list) -> list:
    """Module-level reducer for TypedDict annotation (must be at module scope)."""
    return left + right


@pytest.fixture
def simple_compiled_graph() -> CompiledStateGraph:
    """A minimal compiled LangGraph: a -> b -> END."""
    graph = StateGraph(SimpleState)
    graph.add_node("a", _node_a)
    graph.add_node("b", _node_b)
    graph.add_edge("a", "b")
    graph.set_entry_point("a")
    return graph.compile()


@pytest.fixture
def extractor(simple_compiled_graph: CompiledStateGraph) -> LangGraphExtractor:
    """LangGraphExtractor initialized with a simple compiled graph."""
    return LangGraphExtractor(simple_compiled_graph)


# ---------------------------------------------------------------------------
# Constructor and properties
# ---------------------------------------------------------------------------


def test_init_stores_source(simple_compiled_graph: CompiledStateGraph) -> None:
    """LangGraphExtractor stores the source graph."""
    ext = LangGraphExtractor(simple_compiled_graph)
    assert ext.source is simple_compiled_graph


def test_source_property_returns_constructor_arg(
    simple_compiled_graph: CompiledStateGraph,
) -> None:
    """source property returns the graph passed to __init__."""
    ext = LangGraphExtractor(simple_compiled_graph)
    assert ext.source is simple_compiled_graph


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


def test_extract_returns_nat_app_graph(extractor: LangGraphExtractor) -> None:
    """extract returns a nat_app Graph with nodes and edges."""
    graph = extractor.extract(extractor.source)
    assert graph is not None
    assert hasattr(graph, "node_names")
    assert hasattr(graph, "edge_pairs")
    assert hasattr(graph, "entry_point")


def test_extract_populates_nodes_from_compiled_graph(
    extractor: LangGraphExtractor,
) -> None:
    """extract adds nodes from the compiled graph (excluding __start__/__end__)."""
    graph = extractor.extract(extractor.source)
    assert "a" in graph.node_names
    assert "b" in graph.node_names
    assert "__start__" not in graph.node_names
    assert "__end__" not in graph.node_names


def test_extract_populates_edges(extractor: LangGraphExtractor) -> None:
    """extract adds edges between nodes."""
    graph = extractor.extract(extractor.source)
    edge_pairs = list(graph.edge_pairs)
    assert ("a", "b") in edge_pairs


def test_extract_sets_entry_point(extractor: LangGraphExtractor) -> None:
    """extract sets entry_point from __start__ edge."""
    graph = extractor.extract(extractor.source)
    assert graph.entry_point == "a"


def test_extract_idempotent(extractor: LangGraphExtractor) -> None:
    """extract returns cached graph on second call."""
    g1 = extractor.extract(extractor.source)
    g2 = extractor.extract(extractor.source)
    assert g1 is g2


# ---------------------------------------------------------------------------
# get_node_func
# ---------------------------------------------------------------------------


def test_get_node_func_returns_callable(extractor: LangGraphExtractor) -> None:
    """get_node_func returns the underlying callable for a node."""
    func = extractor.get_node_func("a")
    assert func is not None
    assert callable(func)
    assert func is _node_a


def test_get_node_func_returns_none_for_missing_node(
    extractor: LangGraphExtractor,
) -> None:
    """get_node_func returns None for unknown node."""
    assert extractor.get_node_func("nonexistent") is None


def test_get_node_func_lazy_extracts(simple_compiled_graph: CompiledStateGraph) -> None:
    """get_node_func triggers extract when graph not yet extracted."""
    ext = LangGraphExtractor(simple_compiled_graph)
    # Do not call extract; get_node_func should trigger it
    func = ext.get_node_func("a")
    assert func is _node_a


# ---------------------------------------------------------------------------
# subgraphs property
# ---------------------------------------------------------------------------


def test_subgraphs_empty_for_flat_graph(extractor: LangGraphExtractor) -> None:
    """subgraphs is empty when graph has no nested subgraphs."""
    subgraphs = extractor.subgraphs
    assert subgraphs == {}


def test_subgraphs_lazy_extracts(simple_compiled_graph: CompiledStateGraph) -> None:
    """subgraphs property triggers extract when graph not yet extracted."""
    ext = LangGraphExtractor(simple_compiled_graph)
    subgraphs = ext.subgraphs
    assert isinstance(subgraphs, dict)
    assert subgraphs == {}


# ---------------------------------------------------------------------------
# get_state_schema
# ---------------------------------------------------------------------------


def test_get_state_schema_returns_typed_dict(extractor: LangGraphExtractor) -> None:
    """get_state_schema returns the state TypedDict when available."""
    schema = extractor.get_state_schema()
    assert schema is not None
    assert schema is not dict
    assert schema is SimpleState


def test_get_state_schema_returns_none_when_no_builder() -> None:
    """get_state_schema returns None when source has no builder."""
    mock_source = MagicMock(spec=CompiledStateGraph)
    mock_source.builder = None
    ext = LangGraphExtractor(mock_source)
    assert ext.get_state_schema() is None


def test_get_state_schema_returns_none_when_schema_is_dict() -> None:
    """get_state_schema returns None when state_schema is dict."""
    mock_builder = MagicMock()
    mock_builder.state_schema = dict
    mock_source = MagicMock(spec=CompiledStateGraph)
    mock_source.builder = mock_builder
    ext = LangGraphExtractor(mock_source)
    assert ext.get_state_schema() is None


# ---------------------------------------------------------------------------
# get_reducer_fields
# ---------------------------------------------------------------------------


def test_get_reducer_fields_empty_for_simple_schema(
    extractor: LangGraphExtractor,
) -> None:
    """get_reducer_fields returns empty dict when no Annotated reducer fields."""
    result = extractor.get_reducer_fields()
    assert result == {}


def test_get_reducer_fields_detects_reducer_annotation() -> None:
    """get_reducer_fields returns field names with Annotated[T, reducer_fn]."""

    class StateWithReducer(TypedDict):
        items: Annotated[list[str], _reducer_for_test]
        plain: str

    graph = StateGraph(StateWithReducer)
    graph.add_node("n", _node_a)
    graph.set_entry_point("n")
    compiled = graph.compile()
    ext = LangGraphExtractor(compiled)
    result = ext.get_reducer_fields()
    assert "state" in result
    assert "items" in result["state"]
    assert "plain" not in result["state"]


def test_get_reducer_fields_returns_empty_when_no_schema() -> None:
    """get_reducer_fields returns {} when state schema unavailable."""
    mock_source = MagicMock(spec=CompiledStateGraph)
    mock_source.builder = None
    ext = LangGraphExtractor(mock_source)
    assert ext.get_reducer_fields() == {}


# ---------------------------------------------------------------------------
# get_all_schema_fields
# ---------------------------------------------------------------------------


def test_get_all_schema_fields_returns_field_names(
    extractor: LangGraphExtractor,
) -> None:
    """get_all_schema_fields returns set of state field names."""
    fields = extractor.get_all_schema_fields()
    assert fields is not None
    assert "value" in fields
    assert "count" in fields


def test_get_all_schema_fields_returns_none_when_no_schema() -> None:
    """get_all_schema_fields returns None when state schema unavailable."""
    mock_source = MagicMock(spec=CompiledStateGraph)
    mock_source.builder = None
    ext = LangGraphExtractor(mock_source)
    assert ext.get_all_schema_fields() is None


# ---------------------------------------------------------------------------
# get_special_call_names
# ---------------------------------------------------------------------------


def test_get_special_call_names_returns_send_and_command() -> None:
    """get_special_call_names returns Send and Command."""
    mock_source = MagicMock(spec=CompiledStateGraph)
    ext = LangGraphExtractor(mock_source)
    names = ext.get_special_call_names()
    assert names == {"Send", "Command"}


# ---------------------------------------------------------------------------
# get_llm_detector
# ---------------------------------------------------------------------------


def test_get_llm_detector_returns_langchain_detector() -> None:
    """get_llm_detector returns LangChainLLMDetector instance."""
    mock_source = MagicMock(spec=CompiledStateGraph)
    ext = LangGraphExtractor(mock_source)
    detector = ext.get_llm_detector()
    assert isinstance(detector, LangChainLLMDetector)


# ---------------------------------------------------------------------------
# analyze_node
# ---------------------------------------------------------------------------


def test_analyze_node_returns_node_analysis(extractor: LangGraphExtractor) -> None:
    """analyze_node returns NodeAnalysis from analyze_langgraph_node."""
    analysis = extractor.analyze_node("a", _node_a)
    assert analysis is not None
    assert hasattr(analysis, "name")
    assert analysis.name == "a"
    assert hasattr(analysis, "reads")
    assert hasattr(analysis, "writes")
    assert hasattr(analysis, "confidence")


def test_analyze_node_uses_all_schema_fields_when_provided(
    extractor: LangGraphExtractor,
) -> None:
    """analyze_node passes all_schema_fields to analyze_langgraph_node."""
    analysis = extractor.analyze_node(
        "a",
        _node_a,
        state_schema=SimpleState,
        all_schema_fields={"value", "count"},
    )
    assert analysis is not None
    assert analysis.name == "a"


def test_analyze_node_derives_fields_from_state_schema(
    extractor: LangGraphExtractor,
) -> None:
    """analyze_node derives all_schema_fields from state_schema when not provided."""
    analysis = extractor.analyze_node(
        "a",
        _node_a,
        state_schema=SimpleState,
    )
    assert analysis is not None


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def test_build_calls_build_optimized_langgraph() -> None:
    """build delegates to build_optimized_langgraph and returns result."""
    mock_original = MagicMock(spec=CompiledStateGraph)
    mock_result = MagicMock()
    mock_optimized_graph = MagicMock(spec=CompiledStateGraph)
    mock_opt_result = MagicMock(optimized_graph=mock_optimized_graph)
    ext = LangGraphExtractor(mock_original)
    with patch(
        "langchain_nvidia_langgraph.adapter.extractor.build_optimized_langgraph",
        return_value=mock_opt_result,
    ) as mock_build:
        result = ext.build(mock_original, mock_result)
        mock_build.assert_called_once_with(mock_original, mock_result)
    assert result is mock_opt_result
    assert result.optimized_graph is mock_optimized_graph


def test_build_returns_optimized_graph() -> None:
    """build returns OptimizedGraph from build_optimized_langgraph."""
    from nat_app.graph.scheduling import CompilationResult
    from nat_app.graph.types import Graph

    mock_original = MagicMock(spec=CompiledStateGraph)
    graph = Graph()
    graph.add_node("a")
    graph.entry_point = "a"
    mock_compilation = CompilationResult(
        graph=graph,
        node_analyses={},
        necessary_edges=set(),
        unnecessary_edges=set(),
        optimized_order=[{"a"}],
        topology=None,
    )
    ext = LangGraphExtractor(mock_original)
    with patch(
        "langchain_nvidia_langgraph.adapter.extractor.build_optimized_langgraph",
    ) as mock_build:
        mock_opt = MagicMock()
        mock_opt.optimized_graph = MagicMock(spec=CompiledStateGraph)
        mock_build.return_value = mock_opt
        result = ext.build(mock_original, mock_compilation)
        assert result is mock_opt
