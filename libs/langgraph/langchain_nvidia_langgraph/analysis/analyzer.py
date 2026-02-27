"""Graph analysis data structures for speculative execution.

``GraphAnalysis`` is the speculative executor's view of the graph.
Construct it from a ``CompilationResult`` via
``GraphAnalysis.from_compilation_result()`` -- this reuses the
pre-computed topology from the compilation pipeline and only adds
LangGraph-specific enrichment (``conditional_edge_fn``, ``path_mapping``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langgraph.graph.state import CompiledStateGraph

if TYPE_CHECKING:
    from nat_app.graph.scheduling import CompilationResult
    from nat_app.graph.topology import GraphTopology

logger = logging.getLogger(__name__)


@dataclass
class RouterInfo:
    """Information about a detected router (conditional branch) node.

    Attributes:
        name: Node name of the router.
        possible_targets: List of nodes this router can route to.
        conditional_edge_fn: The LangGraph conditional edge function, or None if
            unavailable.
        path_mapping: Mapping of branch names to target node names.
    """

    name: str
    possible_targets: list[str]
    conditional_edge_fn: Any
    path_mapping: dict[str, str] = field(default_factory=dict)


@dataclass
class GraphAnalysis:
    """Complete analysis of a LangGraph for speculative execution.

    Built from a ``CompilationResult`` via ``from_compilation_result()``.
    Holds router metadata, entry point, cycle info, and back edges.

    Attributes:
        routers: List of RouterInfo for each conditional branch point.
        entry_point: Name of the graph entry node.
        has_cycles: True if the graph contains cycles.
        back_edges: List of (source, target) back edges that close cycles.
    """

    routers: list[RouterInfo] = field(default_factory=list)
    entry_point: str = ""
    has_cycles: bool = False
    back_edges: list[tuple[str, str]] = field(default_factory=list)

    @classmethod
    def from_compilation_result(
        cls,
        result: CompilationResult,
        graph: CompiledStateGraph,
        *,
        optimized_topology: GraphTopology | None = None,
        optimized_entry_point: str = "",
    ) -> GraphAnalysis:
        """Build from a ``CompilationResult`` produced by ``DefaultGraphCompiler``.

        Reads entry point, cycles, and router names from the pre-computed
        *optimized_topology*.  Only LangGraph-specific enrichment
        (``conditional_edge_fn``, ``path_mapping``) is done here.

        Args:
            result: Compilation result containing the original topology.
            graph: The compiled LangGraph (used for LangGraph-specific enrichment).
            optimized_topology: Pre-computed topology for the optimized graph.
            optimized_entry_point: Entry point of the optimized graph.

        Returns:
            A ``GraphAnalysis`` ready for speculative execution.
        """
        if result.topology is None or optimized_topology is None:
            return cls()

        entry_point = optimized_entry_point or _find_entry_point(graph)
        has_cycles = bool(optimized_topology.cycles)
        back_edges = [c.back_edge for c in optimized_topology.cycles]
        routers = _enrich_routers(graph, optimized_topology)

        analysis = cls(
            routers=routers,
            entry_point=entry_point,
            has_cycles=has_cycles,
            back_edges=back_edges,
        )

        logger.info(
            "GraphAnalysis from CompilationResult: %d routers, cycles=%s",
            len(routers),
            has_cycles,
        )

        return analysis


# ---------------------------------------------------------------------------
# LangGraph-specific helpers
# ---------------------------------------------------------------------------


def _enrich_routers(
    graph: CompiledStateGraph,
    topology: GraphTopology,
) -> list[RouterInfo]:
    """Build ``RouterInfo`` list from pre-computed topology + LangGraph internals.

    Router names and targets come from topology. The LangGraph-specific
    conditional_edge_fn and path_mapping are extracted from graph.builder.branches.

    Args:
        graph: The compiled LangGraph.
        topology: Pre-computed topology with routers and branches.

    Returns:
        List of RouterInfo for each router in the topology.
    """
    builder_branches: dict[str, Any] = graph.builder.branches if graph.builder else {}  # type: ignore[assignment]

    routers: list[RouterInfo] = []
    for topo_router in topology.routers:
        rname = topo_router.node
        all_targets: list[str] = []
        for target_list in topo_router.branches.values():
            all_targets.extend(target_list)
        targets_dedup = list(dict.fromkeys(all_targets))

        conditional_fn = None
        path_mapping: dict[str, str] = {}
        if rname in builder_branches:
            for _bname, branch_spec in builder_branches[rname].items():
                ends: dict[str, str] | None = getattr(branch_spec, "ends", None)
                if ends:
                    path_mapping.update(ends)
                if hasattr(branch_spec, "path") and conditional_fn is None:
                    conditional_fn = branch_spec.path

        routers.append(
            RouterInfo(
                name=rname,
                possible_targets=targets_dedup,
                conditional_edge_fn=conditional_fn,
                path_mapping=path_mapping,
            )
        )

    return routers


def _find_entry_point(graph: CompiledStateGraph) -> str:
    """Find the entry point of a LangGraph.

    Prefers the first edge target from ``__start__``; if the start node has
    conditional edges, returns ``__start__``. Falls back to first non-start
    node if no edges from start exist.

    Args:
        graph: The compiled LangGraph.

    Returns:
        Name of the entry node.
    """
    if "__start__" in graph.nodes:
        graph_obj = graph.get_graph()
        has_conditional = any(
            edge.source == "__start__" and edge.conditional for edge in graph_obj.edges
        )
        if has_conditional:
            return "__start__"
        for edge in graph_obj.edges:
            if edge.source == "__start__":
                return edge.target

    real_nodes = [n for n in graph.nodes if n not in ("__start__", "__end__")]
    return real_nodes[0] if real_nodes else list(graph.nodes.keys())[0]
