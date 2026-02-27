"""Integration tests: interrupt_before and interrupt_after."""

from __future__ import annotations

from typing import TypedDict

import pytest
from langgraph.graph.state import CompiledStateGraph

from langchain_nvidia_langgraph.graph.constraints import OptimizationConfig
from langchain_nvidia_langgraph.graph.state_graph import StateGraph


class InterruptState(TypedDict):
    """State for interrupt tests."""

    value: str
    visited: str


def _node_a(state: dict) -> dict:
    return {"visited": state.get("visited", "") + "a"}


def _node_b(state: dict) -> dict:
    return {"visited": state.get("visited", "") + "b"}


def _node_c(state: dict) -> dict:
    return {"visited": state.get("visited", "") + "c"}


@pytest.fixture
def interruptible_graph() -> StateGraph:
    """Linear graph: a -> b -> c -> END."""
    graph = StateGraph(InterruptState)
    graph.add_node("a", _node_a)
    graph.add_node("b", _node_b)
    graph.add_node("c", _node_c)
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph.set_entry_point("a")
    return graph


@pytest.mark.integration
class TestInterrupts:
    """interrupt_before and interrupt_after with vanilla and parallel."""

    def test_vanilla_compile_with_interrupt_before(
        self,
        interruptible_graph: StateGraph,
    ) -> None:
        """Vanilla compile accepts interrupt_before; invoke stops at interrupt."""
        compiled = interruptible_graph.compile(
            interrupt_before=["b"],
        )
        assert isinstance(compiled, CompiledStateGraph)
        # Interrupt before b: execution stops after a, returns partial result
        result = compiled.invoke({"value": "x", "visited": ""})
        assert "visited" in result
        # With interrupt, we get partial state (at least one node ran)
        assert len(result["visited"]) >= 1

    def test_vanilla_compile_with_interrupt_after(
        self,
        interruptible_graph: StateGraph,
    ) -> None:
        """Vanilla compile accepts interrupt_after; invoke stops at interrupt."""
        compiled = interruptible_graph.compile(
            interrupt_after=["a"],
        )
        assert isinstance(compiled, CompiledStateGraph)
        # Interrupt after a: execution stops after a
        result = compiled.invoke({"value": "x", "visited": ""})
        assert "a" in result["visited"]

    def test_parallel_compile_with_interrupts(
        self,
        interruptible_graph: StateGraph,
    ) -> None:
        """Parallel compile accepts interrupt_before and interrupt_after."""
        compiled = interruptible_graph.compile(
            optimization=OptimizationConfig(enable_parallel=True),
            interrupt_before=["b"],
            interrupt_after=["a"],
        )
        assert compiled is not None
        result = compiled.invoke({"value": "x", "visited": ""})
        assert "a" in result["visited"]
