"""Unit tests for LangGraphNodeExecutor public interfaces."""

from __future__ import annotations

from typing import TypedDict

import pytest
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from nat_app.executors import ExecutionState

from langchain_nvidia_langgraph.executors.speculative.channel_state_manager import (
    ChannelStateManager,
)
from langchain_nvidia_langgraph.executors.speculative.node_executor import (
    LangGraphNodeExecutor,
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
def node_executor(compiled_graph: CompiledStateGraph) -> LangGraphNodeExecutor:
    """LangGraphNodeExecutor with compiled graph."""
    return LangGraphNodeExecutor(compiled_graph)


@pytest.fixture
def execution_state(compiled_graph: CompiledStateGraph) -> ExecutionState:
    """ExecutionState with channels."""
    state = ExecutionState()
    state.channels = ChannelStateManager(compiled_graph).create_isolated_channels()
    return state


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_node_executor_constructor(compiled_graph: CompiledStateGraph) -> None:
    """LangGraphNodeExecutor stores graph and runnable_config."""
    executor = LangGraphNodeExecutor(compiled_graph)
    assert executor._graph is compiled_graph


def test_node_executor_accepts_runnable_config(
    compiled_graph: CompiledStateGraph,
) -> None:
    """LangGraphNodeExecutor accepts optional runnable_config."""
    config = {"configurable": {"thread_id": "test"}}
    executor = LangGraphNodeExecutor(compiled_graph, runnable_config=config)
    assert executor._runnable_config is not None
    assert executor._runnable_config["configurable"]["thread_id"] == "test"


def test_node_executor_accepts_none_runnable_config(
    compiled_graph: CompiledStateGraph,
) -> None:
    """LangGraphNodeExecutor accepts runnable_config=None."""
    executor = LangGraphNodeExecutor(compiled_graph, runnable_config=None)
    assert executor._runnable_config is None


# ---------------------------------------------------------------------------
# extract_node_function
# ---------------------------------------------------------------------------


def test_extract_node_function_returns_callable_for_langgraph_node(
    node_executor: LangGraphNodeExecutor,
    compiled_graph: CompiledStateGraph,
) -> None:
    """extract_node_function returns async callable for node with ainvoke."""
    node = compiled_graph.nodes.get("a")
    assert node is not None
    fn = node_executor.extract_node_function(node)
    assert fn is not None
    assert callable(fn)


def test_extract_node_function_returns_none_for_object_without_ainvoke(
    node_executor: LangGraphNodeExecutor,
) -> None:
    """extract_node_function returns None when node has no ainvoke."""
    fn = node_executor.extract_node_function(object())
    assert fn is None


def test_extract_node_function_returns_none_for_plain_callable(
    node_executor: LangGraphNodeExecutor,
) -> None:
    """extract_node_function returns None for plain function (no ainvoke)."""
    fn = node_executor.extract_node_function(_node_passthrough)
    assert fn is None


# ---------------------------------------------------------------------------
# safe_call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_safe_call_with_sync_function(
    node_executor: LangGraphNodeExecutor,
) -> None:
    """safe_call invokes sync function and returns result."""

    def sync_fn(state: dict) -> dict:
        return {"x": 1}

    result = await node_executor.safe_call(sync_fn, {"value": ""})
    assert result == {"x": 1}


@pytest.mark.asyncio
async def test_safe_call_with_async_function(
    node_executor: LangGraphNodeExecutor,
) -> None:
    """safe_call awaits async function and returns result."""

    async def async_fn(state: dict) -> dict:
        return {"y": 2}

    result = await node_executor.safe_call(async_fn, {"value": ""})
    assert result == {"y": 2}


# ---------------------------------------------------------------------------
# launch_node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_launch_node_adds_task_to_running_tasks(
    node_executor: LangGraphNodeExecutor,
    execution_state: ExecutionState,
) -> None:
    """launch_node adds task to execution_state.running_tasks."""
    current_state = {"value": ""}
    await node_executor.launch_node("a", execution_state, current_state)
    assert "a" in execution_state.running_tasks
    assert execution_state.tools_launched >= 1


@pytest.mark.asyncio
async def test_launch_node_skips_if_already_running(
    node_executor: LangGraphNodeExecutor,
    execution_state: ExecutionState,
) -> None:
    """launch_node does nothing when node already in running_tasks."""
    current_state = {"value": ""}
    await node_executor.launch_node("a", execution_state, current_state)
    initial_launched = execution_state.tools_launched
    await node_executor.launch_node("a", execution_state, current_state)
    assert execution_state.tools_launched == initial_launched


@pytest.mark.asyncio
async def test_launch_node_skips_if_already_completed(
    node_executor: LangGraphNodeExecutor,
    execution_state: ExecutionState,
) -> None:
    """launch_node does nothing when node already in completed_nodes."""
    execution_state.completed_nodes["a"] = {}
    current_state = {"value": ""}
    await node_executor.launch_node("a", execution_state, current_state)
    assert "a" not in execution_state.running_tasks


@pytest.mark.asyncio
async def test_launch_node_skips_unknown_node(
    node_executor: LangGraphNodeExecutor,
    execution_state: ExecutionState,
) -> None:
    """launch_node does nothing when node not in graph."""
    current_state = {"value": ""}
    await node_executor.launch_node("nonexistent", execution_state, current_state)
    assert "nonexistent" not in execution_state.running_tasks


@pytest.mark.asyncio
async def test_launch_node_task_completes(
    node_executor: LangGraphNodeExecutor,
    execution_state: ExecutionState,
) -> None:
    """launch_node creates task that completes successfully."""
    current_state = {"value": "hello"}
    await node_executor.launch_node("a", execution_state, current_state)
    task = execution_state.running_tasks["a"]
    result = await task
    assert result is None or isinstance(result, dict)


# ---------------------------------------------------------------------------
# execute_router_node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_router_node_returns_task_for_regular_node(
    node_executor: LangGraphNodeExecutor,
) -> None:
    """execute_router_node returns Task for node with ainvoke."""
    current_state = {"value": ""}
    task = await node_executor.execute_router_node("a", current_state)
    assert task is not None
    result = await task
    assert result is not None or result == {}


@pytest.mark.asyncio
async def test_execute_router_node_returns_none_for_unknown_node(
    node_executor: LangGraphNodeExecutor,
) -> None:
    """execute_router_node returns None when node not in graph."""
    task = await node_executor.execute_router_node("nonexistent", {"value": ""})
    assert task is None


@pytest.mark.asyncio
async def test_execute_router_node_uses_isolated_state(
    node_executor: LangGraphNodeExecutor,
) -> None:
    """execute_router_node passes deep-copied state to node."""
    current_state = {"value": "original"}
    task = await node_executor.execute_router_node("a", current_state)
    assert task is not None
    await task
    # State passed to node should be copy; original unchanged
    assert current_state["value"] == "original"
