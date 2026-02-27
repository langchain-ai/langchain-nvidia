"""Shared fixtures for integration tests."""

from __future__ import annotations

from typing import TypedDict

import pytest
from langgraph.graph import StateGraph as LangGraphStateGraph

from langchain_nvidia_langgraph.graph.state_graph import StateGraph as NvidiaStateGraph

# ---------------------------------------------------------------------------
# State schemas
# ---------------------------------------------------------------------------


class SimpleState(TypedDict):
    """Minimal state schema."""

    value: str


class ExtendedState(TypedDict):
    """State with multiple fields for parallel/speculative tests."""

    value: str
    path: str
    visited: str


# ---------------------------------------------------------------------------
# Node functions (pure, no external calls)
# ---------------------------------------------------------------------------


def _node_passthrough(state: dict) -> dict:
    """Simple passthrough node."""
    return {}


def _node_append_a(state: dict) -> dict:
    """Append 'a' to visited."""
    visited = state.get("visited", "")
    return {"visited": visited + "a"}


def _node_append_b(state: dict) -> dict:
    """Append 'b' to visited."""
    visited = state.get("visited", "")
    return {"visited": visited + "b"}


def _node_append_c(state: dict) -> dict:
    """Append 'c' to visited."""
    visited = state.get("visited", "")
    return {"visited": visited + "c"}


def _router_to_b_or_c(state: dict) -> str:
    """Route to 'b' or 'c' based on state.value."""
    v = state.get("value", "")
    return "b" if "b" in v else "c"


def _node_b_sets_path(state: dict) -> dict:
    """Node b: set path to 'b' (no concurrent write with c)."""
    return {"path": "b"}


def _node_c_sets_path(state: dict) -> dict:
    """Node c: set path to 'c' (no concurrent write with b)."""
    return {"path": "c"}


# ---------------------------------------------------------------------------
# Graph fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def linear_graph() -> NvidiaStateGraph:
    """Linear graph: a -> b -> c -> END."""
    graph = NvidiaStateGraph(ExtendedState)
    graph.add_node("a", _node_append_a)
    graph.add_node("b", _node_append_b)
    graph.add_node("c", _node_append_c)
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph.set_entry_point("a")
    return graph


def _router_passthrough(state: dict) -> dict:
    """Router node passthrough (routing done by conditional_edges)."""
    return {}


@pytest.fixture
def conditional_graph() -> NvidiaStateGraph:
    """Graph with router: a -> router -> b | c -> END.
    b and c write to 'path' (only one runs) to avoid concurrent write errors."""
    graph = NvidiaStateGraph(ExtendedState)
    graph.add_node("a", _node_append_a)
    graph.add_node("router", _router_passthrough)
    graph.add_node("b", _node_b_sets_path)
    graph.add_node("c", _node_c_sets_path)
    graph.add_edge("a", "router")
    graph.add_conditional_edges("router", _router_to_b_or_c, {"b": "b", "c": "c"})
    graph.set_entry_point("a")
    return graph


@pytest.fixture
def simple_linear_graph() -> NvidiaStateGraph:
    """Minimal linear graph: a -> b -> END (SimpleState)."""
    graph = NvidiaStateGraph(SimpleState)
    graph.add_node("a", _node_passthrough)
    graph.add_node("b", _node_passthrough)
    graph.add_edge("a", "b")
    graph.set_entry_point("a")
    return graph


@pytest.fixture
def langgraph_linear_graph() -> LangGraphStateGraph:
    """Same as linear_graph but built with standard LangGraph StateGraph."""
    graph = LangGraphStateGraph(ExtendedState)
    graph.add_node("a", _node_append_a)
    graph.add_node("b", _node_append_b)
    graph.add_node("c", _node_append_c)
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph.set_entry_point("a")
    return graph
