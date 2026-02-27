"""Unit tests for ExecutionLoop public interfaces."""

from __future__ import annotations

from typing import TypedDict

import pytest
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from nat_app.executors import ExecutionState, ResultHandler
from nat_app.speculation.safety import SpeculationSafetyConfig

from langchain_nvidia_langgraph.analysis.analyzer import GraphAnalysis
from langchain_nvidia_langgraph.analysis.dependency_tracker import DependencyTracker
from langchain_nvidia_langgraph.executors.speculative.channel_state_manager import (
    ChannelStateManager,
)
from langchain_nvidia_langgraph.executors.speculative.execution_loop import (
    ExecutionLoop,
)
from langchain_nvidia_langgraph.executors.speculative.node_executor import (
    LangGraphNodeExecutor,
)
from langchain_nvidia_langgraph.executors.speculative.router_evaluator import (
    RouterEvaluator,
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
def execution_loop(
    compiled_graph: CompiledStateGraph,
    analysis: GraphAnalysis,
) -> ExecutionLoop:
    """ExecutionLoop with real dependencies."""
    dep_tracker = DependencyTracker(compiled_graph, analysis)
    state_manager = ChannelStateManager(compiled_graph)
    node_executor = LangGraphNodeExecutor(compiled_graph)
    router_evaluator = RouterEvaluator(analysis)
    result_handler = ResultHandler(
        command_checker=lambda r: hasattr(r, "update") and not callable(r.update),
    )
    return ExecutionLoop(
        analysis=analysis,
        plans={},
        dep_tracker=dep_tracker,
        state_manager=state_manager,
        node_executor=node_executor,
        router_evaluator=router_evaluator,
        result_handler=result_handler,
        speculation_safety=SpeculationSafetyConfig(),
        node_rw=None,
        graph=compiled_graph,
        max_iterations=50,
    )


@pytest.fixture
def empty_execution_state(compiled_graph: CompiledStateGraph) -> ExecutionState:
    """ExecutionState with no ready nodes, no running tasks."""
    state = ExecutionState()
    state.channels = ChannelStateManager(compiled_graph).create_isolated_channels()
    return state


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_int(
    execution_loop: ExecutionLoop,
    empty_execution_state: ExecutionState,
) -> None:
    """run() returns an integer (iteration count)."""
    current_state: dict[str, str] = {}
    iteration = await execution_loop.run(empty_execution_state, current_state)
    assert isinstance(iteration, int)
    assert iteration >= 1


@pytest.mark.asyncio
async def test_run_terminates_immediately_when_empty(
    execution_loop: ExecutionLoop,
    empty_execution_state: ExecutionState,
) -> None:
    """run() terminates in one iteration when no ready nodes and no running tasks."""
    current_state: dict[str, str] = {}
    iteration = await execution_loop.run(empty_execution_state, current_state)
    assert iteration == 1


@pytest.mark.asyncio
async def test_run_with_entry_point_ready(
    execution_loop: ExecutionLoop,
    compiled_graph: CompiledStateGraph,
) -> None:
    """run() executes when entry point is marked ready."""
    state_manager = ChannelStateManager(compiled_graph)
    channels = state_manager.create_isolated_channels()
    current_state = state_manager.initialize_state(channels, {"value": ""})

    execution_state = ExecutionState()
    execution_state.channels = channels
    execution_state.mark_node_ready("a")

    iteration = await execution_loop.run(execution_state, current_state)
    assert isinstance(iteration, int)
    assert iteration >= 1
    assert "a" in execution_state.completed_nodes


def test_execution_loop_constructor(
    compiled_graph: CompiledStateGraph,
    analysis: GraphAnalysis,
) -> None:
    """ExecutionLoop constructor accepts all required parameters."""
    dep_tracker = DependencyTracker(compiled_graph, analysis)
    state_manager = ChannelStateManager(compiled_graph)
    node_executor = LangGraphNodeExecutor(compiled_graph)
    router_evaluator = RouterEvaluator(analysis)
    result_handler = ResultHandler(
        command_checker=lambda r: hasattr(r, "update") and not callable(r.update),
    )
    loop = ExecutionLoop(
        analysis=analysis,
        plans={},
        dep_tracker=dep_tracker,
        state_manager=state_manager,
        node_executor=node_executor,
        router_evaluator=router_evaluator,
        result_handler=result_handler,
        speculation_safety=SpeculationSafetyConfig(),
        node_rw={"a": {"reads": set(), "writes": set()}},
        graph=compiled_graph,
        max_iterations=50,
    )
    assert loop is not None


def test_execution_loop_accepts_max_iterations(
    compiled_graph: CompiledStateGraph,
    analysis: GraphAnalysis,
) -> None:
    """ExecutionLoop constructor accepts max_iterations parameter."""
    dep_tracker = DependencyTracker(compiled_graph, analysis)
    state_manager = ChannelStateManager(compiled_graph)
    node_executor = LangGraphNodeExecutor(compiled_graph)
    router_evaluator = RouterEvaluator(analysis)
    result_handler = ResultHandler(
        command_checker=lambda r: hasattr(r, "update") and not callable(r.update),
    )
    loop = ExecutionLoop(
        analysis=analysis,
        plans={},
        dep_tracker=dep_tracker,
        state_manager=state_manager,
        node_executor=node_executor,
        router_evaluator=router_evaluator,
        result_handler=result_handler,
        speculation_safety=SpeculationSafetyConfig(),
        node_rw=None,
        graph=compiled_graph,
        max_iterations=25,
    )
    assert loop is not None
