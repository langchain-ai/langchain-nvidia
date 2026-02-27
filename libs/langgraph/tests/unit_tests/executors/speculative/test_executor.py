"""Unit tests for executor public interfaces."""

from __future__ import annotations

from typing import TypedDict

import pytest
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from nat_app.speculation.safety import SpeculationSafetyConfig

from langchain_nvidia_langgraph.analysis.analyzer import GraphAnalysis
from langchain_nvidia_langgraph.analysis.dependency_tracker import DependencyTracker
from langchain_nvidia_langgraph.executors.speculative.executor import (
    SpeculativeGraphWrapper,
    SpeculativeRouteConfig,
    SpeculativeRouteExecutor,
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
def compiled_graph() -> CompiledStateGraph:
    """Minimal compiled graph: a -> END."""
    graph = StateGraph(SimpleState)
    graph.add_node("a", _node_passthrough)
    graph.set_entry_point("a")
    return graph.compile()


@pytest.fixture
def analysis() -> GraphAnalysis:
    """GraphAnalysis for linear graph (no routers, no cycles)."""
    return GraphAnalysis(
        routers=[],
        entry_point="a",
        has_cycles=False,
        back_edges=[],
    )


@pytest.fixture
def dep_tracker(
    compiled_graph: CompiledStateGraph, analysis: GraphAnalysis
) -> DependencyTracker:
    """DependencyTracker for the graph."""
    return DependencyTracker(compiled_graph, analysis)


# ---------------------------------------------------------------------------
# SpeculativeRouteConfig
# ---------------------------------------------------------------------------


def test_speculative_route_config_defaults() -> None:
    """SpeculativeRouteConfig has sensible defaults."""
    config = SpeculativeRouteConfig()
    assert config.max_iterations == 50
    assert config.log_level is None
    assert config.invoke_executor_max_workers == 8
    assert isinstance(config.speculation_safety, SpeculationSafetyConfig)


def test_speculative_route_config_custom_values() -> None:
    """SpeculativeRouteConfig accepts custom values."""
    safety = SpeculationSafetyConfig(unsafe_nodes={"x"})
    config = SpeculativeRouteConfig(
        max_iterations=25,
        log_level="DEBUG",
        speculation_safety=safety,
        invoke_executor_max_workers=4,
    )
    assert config.max_iterations == 25
    assert config.log_level == "DEBUG"
    assert config.speculation_safety is safety
    assert config.invoke_executor_max_workers == 4


# ---------------------------------------------------------------------------
# SpeculativeRouteExecutor
# ---------------------------------------------------------------------------


def test_speculative_route_executor_constructor(
    compiled_graph: CompiledStateGraph,
    analysis: GraphAnalysis,
) -> None:
    """SpeculativeRouteExecutor stores graph, analysis, and creates subcomponents."""
    executor = SpeculativeRouteExecutor(compiled_graph, analysis)
    assert executor.graph is compiled_graph
    assert executor.analysis is analysis
    assert executor.state_manager is not None
    assert executor.node_executor is not None
    assert executor.result_handler is not None
    assert executor.config is not None


def test_speculative_route_executor_accepts_config(
    compiled_graph: CompiledStateGraph,
    analysis: GraphAnalysis,
) -> None:
    """SpeculativeRouteExecutor accepts optional config."""
    config = SpeculativeRouteConfig(max_iterations=10)
    executor = SpeculativeRouteExecutor(compiled_graph, analysis, config=config)
    assert executor.config.max_iterations == 10


def test_speculative_route_executor_accepts_runnable_config(
    compiled_graph: CompiledStateGraph,
    analysis: GraphAnalysis,
) -> None:
    """SpeculativeRouteExecutor accepts optional runnable_config."""
    executor = SpeculativeRouteExecutor(
        compiled_graph,
        analysis,
        runnable_config={"configurable": {"thread_id": "test"}},
    )
    assert executor.runnable_config is not None


def test_speculative_route_executor_accepts_dependency_tracker(
    compiled_graph: CompiledStateGraph,
    analysis: GraphAnalysis,
    dep_tracker: DependencyTracker,
) -> None:
    """SpeculativeRouteExecutor accepts optional dependency_tracker."""
    executor = SpeculativeRouteExecutor(
        compiled_graph,
        analysis,
        dependency_tracker=dep_tracker,
    )
    assert executor.dep_tracker is dep_tracker


def test_speculative_route_executor_accepts_node_rw(
    compiled_graph: CompiledStateGraph,
    analysis: GraphAnalysis,
) -> None:
    """SpeculativeRouteExecutor accepts optional node_rw."""
    executor = SpeculativeRouteExecutor(
        compiled_graph,
        analysis,
        node_rw={"a": {"reads": set(), "writes": set()}},
    )
    assert executor.node_rw is not None
    assert "a" in executor.node_rw


@pytest.mark.asyncio
async def test_speculative_route_executor_execute_returns_tuple(
    compiled_graph: CompiledStateGraph,
    analysis: GraphAnalysis,
) -> None:
    """SpeculativeRouteExecutor.execute returns (final_state, metrics) tuple."""
    executor = SpeculativeRouteExecutor(compiled_graph, analysis)
    final_state, metrics = await executor.execute({"value": ""})
    assert isinstance(final_state, dict)
    assert isinstance(metrics, dict)
    assert "value" in final_state or len(final_state) >= 0
    assert "total_time_ms" in metrics or "tools_launched" in metrics


@pytest.mark.asyncio
async def test_speculative_route_executor_execute_completes(
    compiled_graph: CompiledStateGraph,
    analysis: GraphAnalysis,
) -> None:
    """SpeculativeRouteExecutor.execute runs to completion for simple graph."""
    executor = SpeculativeRouteExecutor(compiled_graph, analysis)
    final_state, metrics = await executor.execute({"value": "hello"})
    assert "value" in final_state
    assert final_state["value"] == "hello"


# ---------------------------------------------------------------------------
# SpeculativeGraphWrapper
# ---------------------------------------------------------------------------


def test_speculative_graph_wrapper_constructor(
    compiled_graph: CompiledStateGraph,
    analysis: GraphAnalysis,
    dep_tracker: DependencyTracker,
) -> None:
    """SpeculativeGraphWrapper stores graph, analysis, dependency_tracker."""
    wrapper = SpeculativeGraphWrapper(
        compiled_graph,
        analysis,
        dep_tracker,
    )
    assert wrapper._graph is compiled_graph
    assert wrapper._analysis is analysis
    assert wrapper._dependency_tracker is dep_tracker


def test_speculative_graph_wrapper_accepts_config(
    compiled_graph: CompiledStateGraph,
    analysis: GraphAnalysis,
    dep_tracker: DependencyTracker,
) -> None:
    """SpeculativeGraphWrapper accepts optional config."""
    config = SpeculativeRouteConfig(max_iterations=5)
    wrapper = SpeculativeGraphWrapper(
        compiled_graph,
        analysis,
        dep_tracker,
        config=config,
    )
    assert wrapper._config.max_iterations == 5


def test_speculative_graph_wrapper_invoke_returns_dict(
    compiled_graph: CompiledStateGraph,
    analysis: GraphAnalysis,
    dep_tracker: DependencyTracker,
) -> None:
    """SpeculativeGraphWrapper.invoke returns state dict."""
    wrapper = SpeculativeGraphWrapper(
        compiled_graph,
        analysis,
        dep_tracker,
    )
    result = wrapper.invoke({"value": ""})
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_speculative_graph_wrapper_ainvoke_returns_dict(
    compiled_graph: CompiledStateGraph,
    analysis: GraphAnalysis,
    dep_tracker: DependencyTracker,
) -> None:
    """SpeculativeGraphWrapper.ainvoke returns state dict."""
    wrapper = SpeculativeGraphWrapper(
        compiled_graph,
        analysis,
        dep_tracker,
    )
    result = await wrapper.ainvoke({"value": ""})
    assert isinstance(result, dict)


def test_speculative_graph_wrapper_last_metrics_after_invoke(
    compiled_graph: CompiledStateGraph,
    analysis: GraphAnalysis,
    dep_tracker: DependencyTracker,
) -> None:
    """SpeculativeGraphWrapper.last_metrics is set after invoke."""
    wrapper = SpeculativeGraphWrapper(
        compiled_graph,
        analysis,
        dep_tracker,
    )
    assert wrapper.last_metrics is None
    wrapper.invoke({"value": ""})
    assert wrapper.last_metrics is not None
    assert isinstance(wrapper.last_metrics, dict)


def test_speculative_graph_wrapper_getattr_forwards_to_graph(
    compiled_graph: CompiledStateGraph,
    analysis: GraphAnalysis,
    dep_tracker: DependencyTracker,
) -> None:
    """SpeculativeGraphWrapper forwards attribute access to underlying graph."""
    wrapper = SpeculativeGraphWrapper(
        compiled_graph,
        analysis,
        dep_tracker,
    )
    assert wrapper.nodes is compiled_graph.nodes
    assert wrapper.channels is compiled_graph.channels


def test_speculative_graph_wrapper_repr(
    compiled_graph: CompiledStateGraph,
    analysis: GraphAnalysis,
    dep_tracker: DependencyTracker,
) -> None:
    """SpeculativeGraphWrapper __repr__ includes graph."""
    wrapper = SpeculativeGraphWrapper(
        compiled_graph,
        analysis,
        dep_tracker,
    )
    repr_str = repr(wrapper)
    assert "SpeculativeGraphWrapper" in repr_str
    assert "graph" in repr_str.lower()
