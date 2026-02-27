"""LangGraph optimized graph builder.

Takes a ``CompilationResult`` from the compiler and builds a new
``StateGraph`` with proper parallel structure (fan-out / fan-in patterns).

Build paths:
    - ``_build_staged``: Acyclic graphs without routers. Simple linear fan-out/fan-in.
    - ``_build_with_topology``: Graphs with cycles and/or conditional routing.
      Preserves routers and cycles while adding parallelism where safe.

``build()`` dispatches based on ``topology.cycles`` and ``topology.routers``.

Node sections (for topology-aware builds):
    - pre: Nodes before the first router.
    - branch: Nodes inside a conditional branch arm.
    - merge: Nodes after branches join (post-merge).
    - cycle: Nodes inside a loop.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from nat_app.graph.scheduling import BranchInfo, CompilationResult
from nat_app.graph.topology import CycleInfo, GraphTopology, analyze_graph_topology
from nat_app.graph.types import Graph

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CycleContext:
    """Pre-collected cycle data passed through the topology-aware build.

    Attributes:
        all_cycle_nodes: All nodes that participate in cycles.
        cycle_back_edges: Back edges that close cycles.
        absorbed_entries: Maps original cycle entry nodes to synthetic entry
            nodes when the cycle body needs a fan-out before its first
            parallel stage.
    """

    all_cycle_nodes: frozenset[str]
    cycle_back_edges: frozenset[tuple[str, str]]
    absorbed_entries: dict[str, str] = field(default_factory=dict)


def _make_passthrough(label: str) -> Callable[..., dict]:
    """Create a named passthrough function for infrastructure nodes.

    Args:
        label: Name for the passthrough (used in __name__ and debug logs).

    Returns:
        A function that returns {} (no state updates).
    """

    def passthrough(state: Any) -> dict:
        logger.debug("Passthrough: %s", label)
        return {}

    passthrough.__name__ = f"_passthrough_{label}"
    return passthrough


@dataclass
class OptimizedGraph:
    """Result of building an optimized LangGraph.

    Attributes:
        original_graph: The input CompiledStateGraph.
        optimized_graph: The output CompiledStateGraph with parallel structure.
        transformation: CompilationResult used for the build.
        stages: List of node sets per execution stage.
        speedup_estimate: Estimated speedup factor.
        optimized_topology: Topology of the optimized graph.
        optimized_entry_point: Entry node name.
    """

    original_graph: CompiledStateGraph
    optimized_graph: CompiledStateGraph
    transformation: CompilationResult
    stages: list[set[str]]
    speedup_estimate: float = 1.0
    optimized_topology: GraphTopology | None = None
    optimized_entry_point: str = ""

    def __repr__(self) -> str:
        return (
            f"OptimizedGraph(stages={len(self.stages)}, "
            f"speedup={self.speedup_estimate:.2f}x)"
        )


class OptimizedGraphBuilder:
    """Builds an optimized LangGraph from ``CompilationResult``.

    Creates a new ``StateGraph`` where independent nodes run in parallel
    via fan-out / fan-in patterns.
    """

    def __init__(
        self,
        original_graph: CompiledStateGraph,
        transformation: CompilationResult,
        compile_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the builder.

        Args:
            original_graph: The compiled graph to optimize.
            transformation: CompilationResult with optimized_order and topology.
            compile_kwargs: Optional kwargs passed to StateGraph.compile().
        """
        self.original_graph = original_graph
        self.transformation = transformation
        self._compile_kwargs = compile_kwargs or {}
        self._optimized: OptimizedGraph | None = None
        self._collector_counter: int = 0

    def build(self) -> OptimizedGraph:
        """Build the optimized LangGraph from the transformation result.

        Returns:
            OptimizedGraph with the new compiled graph and metadata.
        """
        if self._optimized is not None:
            return self._optimized

        logger.info("Building optimized LangGraph...")

        state_schema = self._extract_state_schema()
        stages = self.transformation.optimized_order

        if not stages:
            logger.warning("No stages found, returning original graph")
            return OptimizedGraph(
                original_graph=self.original_graph,
                optimized_graph=self.original_graph,
                transformation=self.transformation,
                stages=[],
                speedup_estimate=1.0,
            )

        topology = self.transformation.topology
        has_cycles = topology and topology.cycles
        has_routers = topology and topology.routers

        # Use topology-aware path when graph has cycles or conditional routing.
        if has_cycles or has_routers:
            compiled = self._build_with_topology(state_schema, stages, topology)
        else:
            compiled = self._build_staged(state_schema, stages)

        original_steps = len(self.transformation.graph.node_names)
        optimized_steps = len(stages)
        speedup = original_steps / max(optimized_steps, 1)

        opt_topology, opt_entry = self._extract_optimized_topology(compiled)

        self._optimized = OptimizedGraph(
            original_graph=self.original_graph,
            optimized_graph=compiled,
            transformation=self.transformation,
            stages=stages,
            speedup_estimate=speedup,
            optimized_topology=opt_topology,
            optimized_entry_point=opt_entry,
        )

        logger.info(
            "Built optimized graph: %d stages, ~%.2fx speedup", len(stages), speedup
        )
        return self._optimized

    @staticmethod
    def _extract_optimized_topology(
        compiled: CompiledStateGraph,
    ) -> tuple[GraphTopology | None, str]:
        """Build a ``GraphTopology`` for the optimized graph.

        Extracts nodes and edges from the compiled graph's drawable
        representation, wraps them in a lightweight ``Graph``, and
        delegates to ``analyze_graph_topology``.

        Returns:
            ``(topology, entry_point)`` tuple.
        """
        try:
            graph_obj = compiled.get_graph()
        except (RuntimeError, AttributeError, TypeError) as exc:
            logger.debug(
                "Could not extract drawable graph for topology analysis: %s",
                exc,
            )
            return None, ""

        g = Graph()
        conditional_sources: dict[str, dict[str, str]] = {}

        for node_name in graph_obj.nodes:
            if node_name not in ("__start__", "__end__"):
                g.add_node(node_name)

        entry_point = ""
        for edge in graph_obj.edges:
            if edge.source == "__start__":
                if not entry_point:
                    entry_point = edge.target
                if edge.conditional:
                    conditional_sources.setdefault(edge.source, {})[
                        edge.target
                    ] = edge.target
                continue
            if edge.target == "__end__":
                g.terminal_nodes.add(edge.source)
                continue
            if edge.conditional:
                conditional_sources.setdefault(edge.source, {})[
                    edge.target
                ] = edge.target
            else:
                g.add_edge(edge.source, edge.target)

        for source, targets in conditional_sources.items():
            if source in ("__start__",):
                continue
            g.add_conditional_edges(source, targets)

        if entry_point:
            g.entry_point = entry_point
        elif g.node_names:
            g.entry_point = next(iter(g.node_names))

        return analyze_graph_topology(g), entry_point

    def _build_staged(
        self, state_schema: type, stages: list[set[str]]
    ) -> CompiledStateGraph:
        """Build using pure fan-out/fan-in (acyclic graphs without routers).

        Args:
            state_schema: State TypedDict or dict for the graph.
            stages: List of node sets per execution stage.

        Returns:
            Compiled LangGraph with parallel structure.
        """
        new_graph = StateGraph(state_schema)

        for name in self.transformation.graph.node_names:
            node_info = self.transformation.graph.get_node(name)
            func = node_info.func
            if func is not None:
                new_graph.add_node(name, func)

        self._wire_stages_linear(new_graph, stages, set_entry=True)
        self._wire_to_end(new_graph, stages[-1])

        return new_graph.compile(**self._compile_kwargs)

    # -- Topology-aware build -----------------------------------------------

    def _build_with_topology(
        self,
        state_schema: type,
        stages: list[set[str]],
        topology: GraphTopology,
    ) -> CompiledStateGraph:
        """Build preserving routers and cycles.

        Args:
            state_schema: State TypedDict or dict for the graph.
            stages: List of node sets per execution stage.
            topology: Graph topology with cycles and routers.

        Returns:
            Compiled LangGraph with routers and cycles preserved.
        """
        new_graph = StateGraph(state_schema)
        graph = self.transformation.graph
        branch_info = self.transformation.branch_info

        for name in graph.node_names:
            node_info = graph.get_node(name)
            func = node_info.func
            if func is not None:
                new_graph.add_node(name, func)

        cycle_ctx = self._collect_cycle_context(topology)
        orig_builder = self.original_graph.builder
        orig_branches = orig_builder.branches
        orig_edges = set(orig_builder.edges)

        # node_section: maps each node to its section (pre, branch, merge, cycle).
        node_section = self._classify_nodes(
            graph,
            branch_info,
            cycle_ctx.all_cycle_nodes,
        )

        # stages_for_section: returns stages filtered by section id.
        def stages_for_section(section_id: Any) -> list[set[str]]:
            result: list[set[str]] = []
            for stage in stages:
                filtered = {n for n in stage if node_section.get(n) == section_id}
                if filtered:
                    result.append(filtered)
            return result

        pre_stages = stages_for_section("pre")
        # per_branch: (router, target) -> list of stage sets for each branch arm.
        per_branch = self._collect_per_branch(branch_info, stages_for_section)

        # Phase 1: Linear stages before first router; sets entry point.
        self._wire_pre_section(new_graph, pre_stages, graph.entry_point)
        # Phase 2: Conditional edges from routers to branches (or fanout nodes).
        self._wire_router_edges(
            new_graph,
            branch_info,
            orig_branches,
            per_branch,
            cycle_ctx.all_cycle_nodes,
            cycle_ctx.absorbed_entries,
        )
        # Phase 3: Per-branch fan-out/fan-in chains.
        self._wire_branches(new_graph, per_branch)
        # Phase 4: Intra-cycle parallelism or sequential fallback.
        self._wire_cycles(
            new_graph,
            topology,
            orig_branches,
            orig_edges,
            per_branch,
            cycle_ctx,
        )
        # Phase 5: Connect pre-section to cycle entry (when no router leads in).
        self._wire_pre_to_cycle(
            new_graph,
            pre_stages,
            topology,
            node_section,
            branch_info,
            orig_edges,
            cycle_ctx,
            graph=graph,
        )
        # Phase 6: Merge sections and branch terminals to END.
        self._wire_terminals(
            new_graph,
            graph,
            branch_info,
            per_branch,
            orig_edges,
            stages_for_section,
            cycle_ctx,
        )

        return new_graph.compile(**self._compile_kwargs)

    @staticmethod
    def _collect_cycle_context(topology: GraphTopology) -> CycleContext:
        """Extract cycle-related data from topology into a reusable context.

        Args:
            topology: Graph topology with cycle information.

        Returns:
            CycleContext with cycle nodes, back edges, and absorbed entries.
        """
        all_cycle_nodes: set[str] = set()
        cycle_back_edges: set[tuple[str, str]] = set()
        absorbed_entries: dict[str, str] = {}
        for cycle in topology.cycles or []:
            all_cycle_nodes.update(cycle.nodes)
            cycle_back_edges.add(cycle.back_edge)
            ba = cycle.body_analysis
            if ba and ba.needs_synthetic_entry:
                absorbed_entries[cycle.entry_node] = ba.entry_node
        return CycleContext(
            all_cycle_nodes=frozenset(all_cycle_nodes),
            cycle_back_edges=frozenset(cycle_back_edges),
            absorbed_entries=absorbed_entries,
        )

    @staticmethod
    def _classify_nodes(
        graph: Graph,
        branch_info: dict[str, BranchInfo],
        all_cycle_nodes: frozenset[str],
    ) -> dict[str, Any]:
        """Assign each node to a section: pre, branch, merge, or cycle.

        Args:
            graph: The nat_app Graph with nodes.
            branch_info: Branch info from the transformation.
            all_cycle_nodes: Set of nodes that participate in cycles.

        Returns:
            Dict mapping node name -> section (str or tuple).
        """
        node_section: dict[str, Any] = {}
        for name in graph.node_names:
            if name in all_cycle_nodes:
                node_section[name] = "cycle"
                continue
            found = False
            for rnode, binfo in branch_info.items():
                for target, exclusive in binfo.branches.items():
                    if name in exclusive:
                        node_section[name] = ("branch", rnode, target)
                        found = True
                        break
                if found:
                    break
                if name in binfo.merge_nodes:
                    node_section[name] = ("merge", rnode)
                    found = True
                    break
            if not found:
                node_section[name] = "pre"
        return node_section

    @staticmethod
    def _collect_per_branch(
        branch_info: dict[str, BranchInfo],
        stages_for_section: Callable[[Any], list[set[str]]],
    ) -> dict[tuple[str, str], list[set[str]]]:
        """Collect per-branch stage lists.

        Args:
            branch_info: Branch info from the transformation.
            stages_for_section: Callable that returns stages for a section id.

        Returns:
            Dict mapping (router, target) -> list of stage sets.
        """
        per_branch: dict[tuple[str, str], list[set[str]]] = {}
        for rnode, binfo in branch_info.items():
            for target in binfo.branches:
                bstages = stages_for_section(("branch", rnode, target))
                if bstages:
                    per_branch[(rnode, target)] = bstages
        return per_branch

    def _wire_pre_section(
        self,
        new_graph: StateGraph,
        pre_stages: list[set[str]],
        entry_point: str,
    ) -> None:
        """Wire the pre-router linear section.

        Args:
            new_graph: The StateGraph being built.
            pre_stages: Stages for nodes before the first router.
            entry_point: Entry node name.
        """
        if pre_stages:
            self._wire_stages_linear(
                new_graph,
                pre_stages,
                set_entry=True,
                entry_point=entry_point,
                prefix="pre",
            )

    def _wire_router_edges(
        self,
        new_graph: StateGraph,
        branch_info: dict[str, BranchInfo],
        orig_branches: dict[str, Any],
        per_branch: dict[tuple[str, str], list[set[str]]],
        all_cycle_nodes: set[str] | frozenset[str],
        absorbed_entries: dict[str, str],
    ) -> None:
        """Add conditional edges for non-cycle routers.

        Args:
            new_graph: The StateGraph being built.
            branch_info: Branch info from the transformation.
            orig_branches: Original graph's conditional branches.
            per_branch: Per-branch stage lists.
            all_cycle_nodes: Set of nodes that participate in cycles.
            absorbed_entries: Mapping of absorbed entry points.
        """
        for rnode in branch_info:
            if rnode in all_cycle_nodes:
                continue
            if rnode not in orig_branches:
                continue

            for _branch_name, branch_spec in orig_branches[rnode].items():
                routing_func = (
                    branch_spec.path.func
                    if hasattr(branch_spec.path, "func")
                    else branch_spec.path
                )
                new_ends: dict[str, str] = {}
                for outcome, orig_target in branch_spec.ends.items():
                    bstages = per_branch.get((rnode, orig_target))
                    if bstages and len(bstages[0]) > 1:
                        fanout = f"__{rnode}_{orig_target}_fanout__"
                        if fanout not in new_graph.nodes:
                            new_graph.add_node(
                                fanout,
                                _make_passthrough(f"{rnode}_{orig_target}_fanout"),
                            )
                        new_ends[outcome] = fanout
                    elif bstages:
                        new_ends[outcome] = list(bstages[0])[0]
                    elif orig_target in absorbed_entries:
                        new_ends[outcome] = absorbed_entries[orig_target]
                    else:
                        new_ends[outcome] = orig_target
                new_graph.add_conditional_edges(rnode, routing_func, new_ends)  # type: ignore[arg-type]

    def _wire_branches(
        self,
        new_graph: StateGraph,
        per_branch: dict[tuple[str, str], list[set[str]]],
    ) -> None:
        """Wire per-branch fan-out/fan-in sections.

        Args:
            new_graph: The StateGraph being built.
            per_branch: Per-branch stage lists.
        """
        for (rnode, target), bstages in per_branch.items():
            if not bstages:
                continue
            first = bstages[0]
            if len(first) > 1:
                fanout = f"__{rnode}_{target}_fanout__"
                if fanout in new_graph.nodes:
                    for n in first:
                        new_graph.add_edge(fanout, n)
            if len(bstages) > 1:
                self._wire_stages_linear(
                    new_graph, bstages, set_entry=False, prefix=f"{rnode}_{target}"
                )

    def _wire_cycles(
        self,
        new_graph: StateGraph,
        topology: GraphTopology,
        orig_branches: dict[str, Any],
        orig_edges: set[tuple[str, str]],
        per_branch: dict[tuple[str, str], list[set[str]]],
        cycle_ctx: CycleContext,
    ) -> None:
        """Wire cycle body sections (intra-cycle parallelism or sequential fallback).

        Args:
            new_graph: The StateGraph being built.
            topology: Graph topology with cycles.
            orig_branches: Original graph's conditional branches.
            orig_edges: Original graph edges.
            per_branch: Per-branch stage lists.
            cycle_ctx: Cycle context with back edges and absorbed entries.
        """
        absorbed_entries = cycle_ctx.absorbed_entries
        wired_edges: set[tuple[str, str]] = set()
        wired_conditional_nodes: set[str] = set()

        for cycle in topology.cycles or []:
            body = cycle.body_analysis
            cycle_body_fanout: str | None = None
            cycle_body_first_stage: set[str] = set()

            if body is not None and body.has_parallelism:
                (
                    cycle_body_fanout,
                    cycle_body_first_stage,
                ) = self._wire_cycle_parallel_body(
                    new_graph,
                    body,
                    orig_branches,
                )
            else:
                for src, tgt in orig_edges:
                    if (
                        src in cycle.nodes
                        and tgt in cycle.nodes
                        and src != "__start__"
                        and tgt != "__end__"
                        and (src, tgt) not in wired_edges
                    ):
                        new_graph.add_edge(src, tgt)
                        wired_edges.add((src, tgt))

            self._wire_cycle_conditional_edges(
                new_graph,
                cycle,
                orig_branches,
                per_branch,
                cycle_body_fanout,
                cycle_body_first_stage,
                absorbed_entries,
                wired_conditional_nodes,
            )

    def _wire_cycle_parallel_body(
        self,
        new_graph: StateGraph,
        body: Any,
        orig_branches: dict[str, Any],
    ) -> tuple[str | None, set[str]]:
        """Wire intra-cycle fan-out/fan-in and return (fanout_name, first_stage).

        Args:
            new_graph: The StateGraph being built.
            body: Cycle body analysis from topology.
            orig_branches: Original graph's conditional branches.

        Returns:
            Tuple of (fanout node name or None, first stage node set).
        """
        cycle_body_fanout: str | None = None

        if body.needs_synthetic_entry:
            new_graph.add_node(
                body.entry_node,
                _make_passthrough(body.entry_node.strip("_")),
            )

        first_body = body.stages[0]

        if len(first_body) > 1:
            cycle_body_fanout = f"__cycle_{body.entry_node}_fanout__"
            new_graph.add_node(
                cycle_body_fanout,
                _make_passthrough(f"cycle_{body.entry_node}_fanout"),
            )
            for n in first_body:
                new_graph.add_edge(cycle_body_fanout, n)

        entry_is_router = body.entry_node in orig_branches
        if not entry_is_router:
            if cycle_body_fanout:
                new_graph.add_edge(body.entry_node, cycle_body_fanout)
            elif len(first_body) == 1:
                new_graph.add_edge(body.entry_node, list(first_body)[0])

        if len(body.stages) > 1:
            self._wire_stages_linear(
                new_graph,
                body.stages,
                set_entry=False,
                prefix=f"cycle_{body.entry_node}",
            )

        last_body = body.stages[-1]
        for n in last_body:
            if n not in orig_branches:
                new_graph.add_edge(n, body.exit_node)

        exit_is_router = body.exit_node in orig_branches
        if not exit_is_router:
            new_graph.add_edge(body.exit_node, body.entry_node)

        return cycle_body_fanout, first_body

    @staticmethod
    def _wire_cycle_conditional_edges(
        new_graph: StateGraph,
        cycle: CycleInfo,
        orig_branches: dict[str, Any],
        per_branch: dict[tuple[str, str], list[set[str]]],
        cycle_body_fanout: str | None,
        cycle_body_first_stage: set[str],
        absorbed_entries: dict[str, str],
        wired_conditional_nodes: set[str] | None = None,
    ) -> None:
        """Add conditional edges from cycle-member routers.

        Args:
            new_graph: The StateGraph being built.
            cycle: Cycle info from topology.
            orig_branches: Original graph's conditional branches.
            per_branch: Per-branch stage lists.
            cycle_body_fanout: Fanout node for parallel cycle body, or None.
            cycle_body_first_stage: First stage nodes in the cycle body.
            absorbed_entries: Mapping of absorbed entry points.
            wired_conditional_nodes: Set of routers already wired (mutated).
        """
        if wired_conditional_nodes is None:
            wired_conditional_nodes = set()
        for src, bdict in orig_branches.items():
            if src not in cycle.nodes:
                continue
            if src in wired_conditional_nodes:
                continue
            wired_conditional_nodes.add(src)
            for _bname, bspec in bdict.items():
                routing_func = (
                    bspec.path.func if hasattr(bspec.path, "func") else bspec.path
                )
                new_ends: dict[str, str] = {}
                for outcome, orig_target in bspec.ends.items():
                    if cycle_body_first_stage and orig_target in cycle_body_first_stage:
                        new_ends[outcome] = (
                            cycle_body_fanout if cycle_body_fanout else orig_target
                        )
                        continue

                    if orig_target in absorbed_entries:
                        new_ends[outcome] = absorbed_entries[orig_target]
                        continue

                    bstages = per_branch.get((src, orig_target))
                    if bstages and len(bstages[0]) > 1:
                        fanout = f"__{src}_{orig_target}_fanout__"
                        if fanout not in new_graph.nodes:
                            new_graph.add_node(
                                fanout,
                                _make_passthrough(f"{src}_{orig_target}_fanout"),
                            )
                        for n in bstages[0]:
                            new_graph.add_edge(fanout, n)
                        new_ends[outcome] = fanout
                    else:
                        new_ends[outcome] = orig_target
                new_graph.add_conditional_edges(src, routing_func, new_ends)  # type: ignore[arg-type]

    @staticmethod
    def _find_outermost_cycle_entry(
        topology: GraphTopology,
        graph: Graph,
        all_cycle_nodes: frozenset[str],
    ) -> str:
        """Return the entry node of the outermost cycle.

        The outermost cycle is the one whose entry node has at least
        one predecessor that is NOT a cycle member.  Falls back to
        ``topology.cycles[0].entry_node`` when all predecessors are
        cycle nodes (e.g. a fully-cyclic graph).
        """
        for cycle in topology.cycles:
            preds = set(graph.predecessors(cycle.entry_node))
            if preds - all_cycle_nodes:
                return cycle.entry_node
        return topology.cycles[0].entry_node

    @staticmethod
    def _cycle_reachable_via_router(
        node_section: dict[str, Any],
        branch_info: dict[str, BranchInfo],
        orig_edges: set[tuple[str, str]],
        all_cycle_nodes: frozenset[str],
        cycle_back_edges: frozenset[tuple[str, str]],
    ) -> bool:
        """Return True if a pre-router branch node has a forward edge into the cycle."""
        pre_nodes = {n for n, s in node_section.items() if s == "pre"}
        for rnode, binfo in branch_info.items():
            if rnode in all_cycle_nodes or rnode not in pre_nodes:
                continue
            for _target, exclusive in binfo.branches.items():
                for bnode in exclusive:
                    if any(
                        src == bnode and tgt in all_cycle_nodes
                        for src, tgt in orig_edges
                        if (src, tgt) not in cycle_back_edges
                    ):
                        return True
        return False

    @staticmethod
    def _wire_pre_to_cycle(
        new_graph: StateGraph,
        pre_stages: list[set[str]],
        topology: GraphTopology,
        node_section: dict[str, Any],
        branch_info: dict[str, BranchInfo],
        orig_edges: set[tuple[str, str]],
        cycle_ctx: CycleContext,
        graph: Graph | None = None,
    ) -> None:
        """Connect the pre-router section to the outermost cycle entry.

        Args:
            new_graph: The StateGraph being built.
            pre_stages: Stages for nodes before the first router.
            topology: Graph topology with cycles.
            node_section: Node-to-section mapping.
            branch_info: Branch info from the transformation.
            orig_edges: Original graph edges.
            cycle_ctx: Cycle context.
            graph: Optional nat_app Graph for predecessor lookup.
        """
        all_cycle_nodes = cycle_ctx.all_cycle_nodes
        cycle_back_edges = cycle_ctx.cycle_back_edges
        absorbed_entries = cycle_ctx.absorbed_entries

        if pre_stages and topology.cycles:
            orig_cycle_entry = (
                OptimizedGraphBuilder._find_outermost_cycle_entry(
                    topology,
                    graph,
                    all_cycle_nodes,
                )
                if graph is not None
                else topology.cycles[0].entry_node
            )
            cycle_entry = absorbed_entries.get(orig_cycle_entry, orig_cycle_entry)
            last_pre = pre_stages[-1]

            if not OptimizedGraphBuilder._cycle_reachable_via_router(
                node_section,
                branch_info,
                orig_edges,
                all_cycle_nodes,
                cycle_back_edges,
            ):
                if len(last_pre) > 1:
                    collect = "__pre_cycle_collect__"
                    if collect not in new_graph.nodes:
                        new_graph.add_node(
                            collect, _make_passthrough("pre_cycle_collect")
                        )
                    for n in last_pre:
                        new_graph.add_edge(n, collect)
                    new_graph.add_edge(collect, cycle_entry)
                else:
                    new_graph.add_edge(list(last_pre)[0], cycle_entry)
        elif not pre_stages and topology.cycles:
            orig_entry = (
                OptimizedGraphBuilder._find_outermost_cycle_entry(
                    topology,
                    graph,
                    all_cycle_nodes,
                )
                if graph is not None
                else topology.cycles[0].entry_node
            )
            new_graph.set_entry_point(absorbed_entries.get(orig_entry, orig_entry))

    def _wire_terminals(
        self,
        new_graph: StateGraph,
        graph: Graph,
        branch_info: dict[str, BranchInfo],
        per_branch: dict[tuple[str, str], list[set[str]]],
        orig_edges: set[tuple[str, str]],
        stages_for_section: Callable[[Any], list[set[str]]],
        cycle_ctx: CycleContext,
    ) -> None:
        """Wire merge sections and terminal branches to END.

        Args:
            new_graph: The StateGraph being built.
            graph: The nat_app Graph.
            branch_info: Branch info from the transformation.
            per_branch: Per-branch stage lists.
            orig_edges: Original graph edges.
            stages_for_section: Callable that returns stages for a section id.
            cycle_ctx: Cycle context.
        """
        per_merge = self._wire_merge_sections(
            new_graph, branch_info, stages_for_section
        )
        self._wire_branch_terminals(
            new_graph,
            graph,
            per_branch,
            orig_edges,
            branch_info,
            cycle_ctx,
        )
        self._wire_cycle_terminals(
            new_graph,
            per_merge,
            per_branch,
            orig_edges,
            cycle_ctx,
        )

    def _wire_merge_sections(
        self,
        new_graph: StateGraph,
        branch_info: dict[str, BranchInfo],
        stages_for_section: Callable[[Any], list[set[str]]],
    ) -> dict[str, list[set[str]]]:
        """Wire merge stages linearly and connect to END.

        Args:
            new_graph: The StateGraph being built.
            branch_info: Branch info from the transformation.
            stages_for_section: Callable that returns stages for a section id.

        Returns:
            Dict mapping router name -> merge stage lists.
        """
        per_merge: dict[str, list[set[str]]] = {}
        for rnode in branch_info:
            mstages = stages_for_section(("merge", rnode))
            if mstages:
                per_merge[rnode] = mstages
        for rnode, mstages in per_merge.items():
            if mstages:
                self._wire_stages_linear(
                    new_graph, mstages, set_entry=False, prefix=f"merge_{rnode}"
                )
                self._wire_to_end(new_graph, mstages[-1])
        return per_merge

    def _wire_branch_terminals(
        self,
        new_graph: StateGraph,
        graph: Graph,
        per_branch: dict[tuple[str, str], list[set[str]]],
        orig_edges: set[tuple[str, str]],
        branch_info: dict[str, BranchInfo],
        cycle_ctx: CycleContext,
    ) -> None:
        """Wire terminal branch stages to END or to successor nodes.

        Args:
            new_graph: The StateGraph being built.
            graph: The nat_app Graph.
            per_branch: Per-branch stage lists.
            orig_edges: Original graph edges.
            branch_info: Branch info from the transformation.
            cycle_ctx: Cycle context.
        """
        all_cycle_nodes = cycle_ctx.all_cycle_nodes
        cycle_back_edges = cycle_ctx.cycle_back_edges
        absorbed_entries = cycle_ctx.absorbed_entries

        for (rnode, target), bstages in per_branch.items():
            if not bstages:
                continue
            last = bstages[-1]
            has_terminal = bool(last & graph.terminal_nodes)
            branch_has_end = any(
                src in last for src, tgt in orig_edges if tgt == "__end__"
            )
            if has_terminal or branch_has_end:
                self._wire_to_end(new_graph, last)
            else:
                successors = self._find_branch_successors(
                    last,
                    graph,
                    all_cycle_nodes,
                    branch_info,
                    rnode,
                    cycle_back_edges,
                    orig_edges,
                    absorbed_entries,
                )
                for succ in successors:
                    if len(last) > 1:
                        for n in last:
                            new_graph.add_edge(n, succ)
                    else:
                        new_graph.add_edge(list(last)[0], succ)
                if not successors:
                    self._wire_to_end(new_graph, last)

    def _wire_cycle_terminals(
        self,
        new_graph: StateGraph,
        per_merge: dict[str, list[set[str]]],
        per_branch: dict[tuple[str, str], list[set[str]]],
        orig_edges: set[tuple[str, str]],
        cycle_ctx: CycleContext,
    ) -> None:
        """Wire cycle nodes that exit directly to END (no branches/merges).

        Args:
            new_graph: The StateGraph being built.
            per_merge: Per-router merge stage lists.
            per_branch: Per-branch stage lists.
            orig_edges: Original graph edges.
            cycle_ctx: Cycle context.
        """
        topology = self.transformation.topology
        if not per_merge and not per_branch and topology and topology.cycles:
            all_cycle_nodes = cycle_ctx.all_cycle_nodes
            for src, tgt in orig_edges:
                if tgt == "__end__" and src in all_cycle_nodes:
                    new_graph.add_edge(src, END)

    # -- Wiring helpers -----------------------------------------------------

    def _wire_stages_linear(
        self,
        graph: StateGraph,
        stages: list[set[str]],
        *,
        set_entry: bool = False,
        entry_point: str = "",
        prefix: str = "",
    ) -> None:
        """Wire a linear sequence of stages with fan-out / fan-in.

        Args:
            graph: The StateGraph being built.
            stages: List of node sets per stage.
            set_entry: If True, set the graph entry point from the first stage.
            entry_point: Entry node when set_entry is True.
            prefix: Prefix for collector node names.
        """
        if not stages:
            return

        first = stages[0]
        if set_entry:
            if len(first) == 1:
                graph.set_entry_point(list(first)[0])
            else:
                fanout = "__start_fanout__"
                if fanout not in graph.nodes:
                    graph.add_node(fanout, _make_passthrough("start_fanout"))
                graph.set_entry_point(fanout)
                for n in first:
                    graph.add_edge(fanout, n)

        pfx = f"{prefix}_" if prefix else ""
        # Four cases: 1→1 (direct edge), 1→N (fan-out), N→1 (fan-in), N→N (collector).
        for i in range(len(stages) - 1):
            cur, nxt = stages[i], stages[i + 1]

            if len(cur) == 1 and len(nxt) == 1:
                graph.add_edge(list(cur)[0], list(nxt)[0])
            elif len(cur) == 1 and len(nxt) > 1:
                src = list(cur)[0]
                for n in nxt:
                    graph.add_edge(src, n)
            elif len(cur) > 1 and len(nxt) == 1:
                tgt = list(nxt)[0]
                for n in cur:
                    graph.add_edge(n, tgt)
            else:
                self._collector_counter += 1
                collector = f"__{pfx}collect_{self._collector_counter}__"
                graph.add_node(collector, _make_passthrough(collector.strip("_")))
                for n in cur:
                    graph.add_edge(n, collector)
                for n in nxt:
                    graph.add_edge(collector, n)

    @staticmethod
    def _find_branch_successors(
        last_stage: set[str],
        graph: Graph,
        all_cycle_nodes: frozenset[str],
        branch_info: dict[str, BranchInfo],
        current_router: str,
        cycle_back_edges: frozenset[tuple[str, str]],
        orig_edges: set[tuple[str, str]],
        absorbed_entries: dict[str, str],
    ) -> set[str]:
        """Find successors for the last stage of a branch (merge nodes, cycle entries).

        Args:
            last_stage: Last stage node set of the branch.
            graph: The nat_app Graph.
            all_cycle_nodes: Set of nodes in cycles.
            branch_info: Branch info from the transformation.
            current_router: Router node for this branch.
            cycle_back_edges: Back edges that close cycles.
            orig_edges: Original graph edges.
            absorbed_entries: Mapping of absorbed entry points.

        Returns:
            Set of successor node names.
        """
        binfo = branch_info.get(current_router)
        if not binfo:
            return set()
        all_branch_nodes: set[str] = set()
        for nodes in binfo.branches.values():
            all_branch_nodes |= nodes

        successors: set[str] = set()
        for node in last_stage:
            for src, tgt in orig_edges:
                if (src, tgt) in cycle_back_edges:
                    continue
                if src == node and tgt not in all_branch_nodes:
                    actual_target = absorbed_entries.get(tgt, tgt)
                    successors.add(actual_target)
        return successors

    @staticmethod
    def _wire_to_end(graph: StateGraph, last_stage: set[str]) -> None:
        """Connect the last stage to END.

        Args:
            graph: The StateGraph being built.
            last_stage: Node set of the last stage.
        """
        if len(last_stage) > 1:
            end_fanin = "__end_fanin__"
            if end_fanin not in graph.nodes:
                graph.add_node(end_fanin, _make_passthrough("end_fanin"))
            for n in last_stage:
                graph.add_edge(n, end_fanin)
            graph.add_edge(end_fanin, END)
        else:
            graph.add_edge(list(last_stage)[0], END)

    def _extract_state_schema(self) -> type:
        """Extract state schema from the original graph.

        Returns:
            State schema (TypedDict or dict) from the graph builder.
        """
        builder = getattr(self.original_graph, "builder", None)
        if builder is not None:
            schema = builder.state_schema
            if schema is not None and schema is not dict:
                return schema
            schema = getattr(builder, "schema", None)
            if schema is not None and schema is not dict:
                return schema

        logger.warning("Could not extract state schema, using dict")
        return dict


def build_optimized_langgraph(
    original_graph: CompiledStateGraph,
    transformation: CompilationResult,
    compile_kwargs: dict[str, Any] | None = None,
) -> OptimizedGraph:
    """Build an optimized LangGraph from a compilation result.

    Args:
        original_graph: The compiled graph to optimize.
        transformation: CompilationResult with optimized_order and topology.
        compile_kwargs: Optional kwargs for StateGraph.compile().

    Returns:
        OptimizedGraph with the new compiled graph, stages, and speedup estimate.
    """
    builder = OptimizedGraphBuilder(original_graph, transformation, compile_kwargs)
    return builder.build()
