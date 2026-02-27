"""LangGraph-style API: ``StateGraph`` and ``with_app_compile``.

Provides a drop-in replacement for ``StateGraph`` with
``.compile(optimization=...)`` and a wrapper for existing graphs. Use
``StateGraph`` for new graphs or ``with_app_compile`` for existing
LangGraph ``StateGraph`` instances.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph as _StateGraph
from langgraph.graph.state import CompiledStateGraph as _CompiledStateGraph

from .constraints import OptimizationConfig

__all__ = [
    "CompilableGraph",
    "StateGraph",
    "with_app_compile",
]


class StateGraph(_StateGraph):
    """StateGraph with parallel/speculative execution via .compile(optimization=...).

    Drop-in replacement for :class:`langgraph.graph.StateGraph`. Use the same
    builder API (add_node, add_edge, add_conditional_edges, etc.); compile()
    accepts an additional ``optimization`` parameter. Defaults give
    vanilla LangGraph behavior.

    Example:
        from langchain_nvidia_langgraph.graph import StateGraph, OptimizationConfig

        graph = StateGraph(ResearchState)
        graph.add_node("a", node_a)
        graph.add_edge("a", "b")

        compiled = graph.compile(
            optimization=OptimizationConfig(enable_parallel=True),
            checkpointer=cp,
        )
    """

    def compile(  # type: ignore[override]
        self,
        checkpointer: Any = None,
        *,
        cache: Any = None,
        store: Any = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        debug: bool = False,
        name: str | None = None,
        optimization: OptimizationConfig | None = None,
        speculative_config: Any | None = None,
        max_subgraph_depth: int = 10,
    ) -> _CompiledStateGraph:
        """Compile the graph with optional parallel/speculative optimization.

        Accepts all standard LangGraph compile parameters plus optimization.
        Default optimization gives vanilla LangGraph behavior.

        Args:
            checkpointer: Optional checkpointer for state persistence.
            cache: Optional cache for compiled graph.
            store: Optional store for graph state.
            interrupt_before: Optional list of node names to interrupt before.
            interrupt_after: Optional list of node names to interrupt after.
            debug: Enable debug mode.
            name: Optional name for the compiled graph.
            optimization: OptimizationConfig for fine-tuning (parallel/speculation).
            speculative_config: Optional SpeculativeRouteConfig; if None and
                optimization.enable_speculation, derived from optimization.
            max_subgraph_depth: Max recursion depth for nested subgraph optimization.

        Returns:
            CompiledStateGraph, optimized when optimization enables
            parallel/speculation.
        """
        # Call parent's compile first to get CompiledStateGraph, then optimize.
        # Avoids recursion: compile_langgraph(StateGraph) would call
        # self.compile() again. Lazy import to avoid circular import when
        # compile is imported before graph.
        from ..compile.compiler import compile_langgraph

        compiled_raw = super().compile(
            checkpointer=checkpointer,
            cache=cache,
            store=store,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            debug=debug,
            name=name,
        )
        return compile_langgraph(
            compiled_raw,
            optimization=optimization,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            checkpointer=checkpointer,
            cache=cache,
            store=store,
            debug=debug,
            name=name,
            speculative_config=speculative_config,
            max_subgraph_depth=max_subgraph_depth,
        )


class CompilableGraph:
    """Wrapper that adds .compile(optimization=...) to an existing StateGraph.

    Use :func:`with_app_compile` to create. After calling compile(), you
    receive a CompiledStateGraph and use it normally (invoke, stream, etc.).
    """

    def __init__(self, graph: StateGraph | _StateGraph) -> None:
        """Initialize the wrapper.

        Args:
            graph: StateGraph or LangGraph StateGraph to wrap.
        """
        self._graph = graph

    def compile(
        self,
        checkpointer: Any = None,
        *,
        cache: Any = None,
        store: Any = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        debug: bool = False,
        name: str | None = None,
        optimization: OptimizationConfig | None = None,
        speculative_config: Any | None = None,
        max_subgraph_depth: int = 10,
    ) -> _CompiledStateGraph:
        """Compile the wrapped graph with optional parallel/speculative optimization.

        Delegates to compile_langgraph with the same parameters.

        Args:
            checkpointer: Optional checkpointer for state persistence.
            cache: Optional cache for compiled graph.
            store: Optional store for graph state.
            interrupt_before: Optional list of node names to interrupt before.
            interrupt_after: Optional list of node names to interrupt after.
            debug: Enable debug mode.
            name: Optional name for the compiled graph.
            optimization: OptimizationConfig for fine-tuning (parallel/speculation).
            speculative_config: Optional SpeculativeRouteConfig; if None and
                optimization.enable_speculation, derived from optimization.
            max_subgraph_depth: Max recursion depth for nested subgraph optimization.

        Returns:
            CompiledStateGraph, optimized when optimization enables
            parallel/speculation.
        """
        from ..compile.compiler import compile_langgraph

        return compile_langgraph(
            self._graph,
            optimization=optimization,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
            checkpointer=checkpointer,
            cache=cache,
            store=store,
            debug=debug,
            name=name,
            speculative_config=speculative_config,
            max_subgraph_depth=max_subgraph_depth,
        )


def with_app_compile(graph: StateGraph | _StateGraph) -> CompilableGraph:
    """Wrap a StateGraph so .compile() accepts optimization params.

    Use when you have an existing graph built with StateGraph and want to
    compile it with parallel or speculative optimization.

    Args:
        graph: StateGraph or LangGraph StateGraph to wrap.

    Returns:
        CompilableGraph wrapping the input graph.

    Example:
        from langchain_nvidia_langgraph.graph import (
            with_app_compile, OptimizationConfig,
        )

        graph = create_baseline_graph()  # returns StateGraph
        compilable = with_app_compile(graph)
        compiled = compilable.compile(
            optimization=OptimizationConfig(enable_parallel=True)
        )
    """
    return CompilableGraph(graph)
