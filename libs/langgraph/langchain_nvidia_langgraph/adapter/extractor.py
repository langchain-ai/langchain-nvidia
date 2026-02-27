"""LangGraph structure and schema extractors.

Implements :class:`~nat_app.graph.protocols.GraphExtractor` and
:class:`~nat_app.graph.protocols.NodeIntrospector` for LangGraph's
``CompiledStateGraph``.
"""

from __future__ import annotations

import logging
import typing
from collections.abc import Callable
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from nat_app.graph.adapter import AbstractFrameworkAdapter
from nat_app.graph.types import Graph

from ..builder.builder import build_optimized_langgraph
from .llm_detector import LangChainLLMDetector
from .node_analyzer import analyze_langgraph_node

logger = logging.getLogger(__name__)

# Special LangGraph node names to skip during extraction
_INFRASTRUCTURE_NODES = frozenset({"__start__", "__end__"})


class LangGraphExtractor(AbstractFrameworkAdapter):
    """LangGraph framework adapter for the DefaultGraphCompiler.

    Extracts a :class:`~nat_app.graph.types.Graph` and node metadata from
    a LangGraph ``CompiledStateGraph``. Extends
    :class:`~nat_app.graph.adapter.AbstractFrameworkAdapter` with LangGraph-specific
    subgraph detection and Send/Command special-call handling.

    Uses the following LangGraph public API:
    - CompiledStateGraph.builder
    - StateGraph.state_schema, branches, edges, nodes
    - PregelNode.bound (via compiled.nodes)
    - Runnable.get_graph()

    Attributes:
        source: The compiled LangGraph passed to the constructor.
        subgraphs: Dict of node name -> CompiledStateGraph for nested graphs.
    """

    def __init__(self, source: CompiledStateGraph) -> None:
        """Initialize the extractor with a compiled LangGraph.

        Args:
            source: The CompiledStateGraph to extract and analyze.
        """
        self._source = source
        self._node_funcs: dict[str, Callable | Any] = {}
        self._subgraphs: dict[str, CompiledStateGraph] = {}
        self._graph: Graph | None = None

    @property
    def source(self) -> CompiledStateGraph:
        """The compiled LangGraph passed to the constructor."""
        return self._source

    @property
    def subgraphs(self) -> dict[str, CompiledStateGraph]:
        """Nodes that are themselves compiled subgraphs.

        Returns:
            Dict mapping node name -> CompiledStateGraph for nested graphs.
        """
        if self._graph is None:
            self.extract(self._source)
        return self._subgraphs

    # -- GraphExtractor protocol -------------------------------------------

    def extract(self, source: Any) -> Graph:
        """Extract a ``Graph`` from a ``CompiledStateGraph``.

        Args:
            source: The CompiledStateGraph to extract (typically self._source).

        Returns:
            A nat_app Graph with nodes, edges, and conditional edges.
        """
        if self._graph is not None:
            return self._graph

        graph = Graph()
        self._node_funcs = {}
        self._subgraphs = {}

        self._extract_nodes(graph)
        self._extract_edges(graph)

        self._graph = graph
        return graph

    # -- NodeIntrospector protocol -----------------------------------------

    def get_node_func(self, node_id: str) -> Callable | None:
        """Return the callable for a LangGraph node, or None.

        Args:
            node_id: Name of the node.

        Returns:
            The underlying callable (func/afunc) or None if not callable.
        """
        if self._graph is None:
            self.extract(self._source)
        func = self._node_funcs.get(node_id)
        return func if callable(func) else None

    def get_state_schema(self) -> type | None:
        """Return the LangGraph state schema (TypedDict class), or None.

        Returns:
            The state schema type from the graph builder, or None.
        """
        builder = self._source.builder
        schema = builder.state_schema if builder else None
        if schema is not None and schema is not dict:
            return schema
        return None

    def _get_type_hints_safe(self, schema: type) -> dict[str, Any] | None:
        """Get type hints from schema, or None if introspection fails.

        Catches TypeError, NameError, AttributeError (known get_type_hints failures
        from forward refs, missing names, malformed schemas). Logs at debug on failure.
        """
        try:
            return typing.get_type_hints(schema, include_extras=True)
        except (TypeError, NameError, AttributeError) as exc:
            logger.debug(
                "Could not introspect state schema type hints: %s",
                exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            return None
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(
                "Unexpected error introspecting state schema: %s",
                exc,
                exc_info=True,
            )
            return None

    def get_reducer_fields(self) -> dict[str, set[str]]:
        """Find fields with reducer annotations (``Annotated[T, reducer_fn]``).

        Returns:
            A dict mapping "state" to the set of field names that use reducers.
            Empty dict if no reducer fields.
        """
        schema = self.get_state_schema()
        if schema is None or schema is dict:
            return {}
        hints = self._get_type_hints_safe(schema)
        if hints is None:
            return {}
        reducer_fields: set[str] = set()
        for field_name, hint in hints.items():
            if typing.get_origin(hint) is typing.Annotated:
                args = typing.get_args(hint)
                if len(args) >= 2 and callable(args[1]):
                    reducer_fields.add(field_name)
        return {"state": reducer_fields} if reducer_fields else {}

    def get_all_schema_fields(self) -> set[str] | None:
        """Get all field names from the state schema, or None.

        Returns:
            Set of state field names, or None if schema is unavailable.
        """
        schema = self.get_state_schema()
        if schema is None or schema is dict:
            return None
        hints = self._get_type_hints_safe(schema)
        return set(hints.keys()) if hints is not None else None

    # -- AbstractFrameworkAdapter overrides ----------------------------------

    def get_special_call_names(self) -> set[str]:
        """LangGraph special calls that act as optimization barriers.

        Returns:
            Set of names (e.g. "Send", "Command") that prevent parallelization.
        """
        return {"Send", "Command"}

    def get_llm_detector(self) -> LangChainLLMDetector:
        """Return a LangChain-aware LLM detector for priority analysis.

        Returns:
            LangChainLLMDetector instance for identifying LLM nodes.
        """
        return LangChainLLMDetector()

    def analyze_node(
        self,
        name: str,
        func: Callable | Any,
        state_schema: type | None = None,
        all_schema_fields: set[str] | None = None,
        *,
        config: Any = None,
    ) -> Any:
        """LangGraph node analysis with subgraph detection.

        Note: state_schema is required by NodeIntrospector protocol;
        all_schema_fields is used for analysis. When all_schema_fields is
        None, it is derived from state_schema when available.

        Args:
            name: Node name.
            func: Node callable.
            state_schema: Optional state TypedDict (protocol-mandated).
            all_schema_fields: Optional set of schema field names.
            config: Optional analysis config.

        Returns:
            Node analysis from analyze_langgraph_node.
        """
        fields = all_schema_fields
        if fields is None and state_schema is not None and state_schema is not dict:
            hints = self._get_type_hints_safe(state_schema)
            fields = set(hints.keys()) if hints else None
        return analyze_langgraph_node(name, func, fields, config=config)

    def build(self, original: Any, result: Any) -> Any:
        """Build optimized LangGraph from TransformationResult.

        Args:
            original: The original CompiledStateGraph.
            result: TransformationResult from the compiler.

        Returns:
            Optimized CompiledStateGraph from build_optimized_langgraph.
        """
        return build_optimized_langgraph(original, result)

    # -- Internal extraction -----------------------------------------------

    def _extract_nodes(self, graph: Graph) -> None:
        """Extract nodes and their callables from the compiled graph.

        Populates graph with nodes, detects subgraphs (CompiledStateGraph
        nodes), and stores callables in _node_funcs.

        Args:
            graph: The nat_app Graph to populate.
        """
        if not hasattr(self._source, "nodes"):
            return

        for name, node in self._source.nodes.items():
            if name in _INFRASTRUCTURE_NODES:
                continue

            bound = getattr(node, "bound", None)
            if isinstance(bound, CompiledStateGraph):
                self._subgraphs[name] = bound
                self._node_funcs[name] = bound
                graph.add_node(name, func=bound, is_subgraph=True)
                continue

            func = self._unwrap_node_func(node)
            if func is not None:
                self._node_funcs[name] = func
                graph.add_node(name, func=func)
            else:
                logger.warning("Could not extract callable for node: %s", name)
                graph.add_node(name)

    def _extract_edges(self, graph: Graph) -> None:
        """Extract edges and conditional edges from the compiled graph.

        Reads graph_obj.edges, handles __start__/__end__, and sets
        entry_point and terminal_nodes.

        Args:
            graph: The nat_app Graph to populate with edges.
        """
        graph_obj = self._source.get_graph()
        entry_point = ""
        conditional_sources: dict[str, dict[str, str]] = {}

        for edge in graph_obj.edges:
            source = edge.source
            target = edge.target

            if source == "__start__":
                if not entry_point:
                    entry_point = target
                continue
            if target == "__end__":
                graph.terminal_nodes.add(source)
                continue

            if edge.conditional:
                conditional_sources.setdefault(source, {})
                conditional_sources[source][target] = target
            else:
                graph.add_edge(source, target)

        for source, targets in conditional_sources.items():
            graph.add_conditional_edges(source, targets)

        if entry_point:
            graph.entry_point = entry_point
        elif graph.node_names:
            targets_set = {tgt for _, tgt in graph.edge_pairs}
            sources_set = {src for src, _ in graph.edge_pairs}
            entry_candidates = sources_set - targets_set
            if entry_candidates:
                graph.entry_point = next(iter(entry_candidates))
            else:
                graph.entry_point = next(iter(graph.node_names))

    @staticmethod
    def _unwrap_node_func(node: Any) -> Callable | None:
        """Unwrap a LangGraph node to get the underlying callable.

        Handles bound nodes with .func or .afunc, and plain callables.

        Args:
            node: A LangGraph node (may be bound, have .func, etc.).

        Returns:
            The underlying callable, or None if not extractable.
        """
        bound = getattr(node, "bound", None)
        if bound is not None:
            if hasattr(bound, "func") and bound.func is not None:
                return bound.func
            if hasattr(bound, "afunc") and bound.afunc is not None:
                afunc = bound.afunc
                if callable(afunc) and not hasattr(afunc, "func"):
                    return afunc
                if hasattr(afunc, "args") and afunc.args:
                    return afunc.args[-1] if afunc.args else None

        if callable(node):
            return node
        if hasattr(node, "func"):
            return node.func
        if hasattr(node, "__call__"):
            return node

        return None
