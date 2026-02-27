"""Dependency tracking for graph nodes.

Tracks upstream routers, upstream nodes, and path information for each node
to determine execution readiness in speculative execution.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from nat_app.speculation.plan import SpeculationPlan

from .analyzer import GraphAnalysis

logger = logging.getLogger(__name__)


@dataclass
class NodeDependencies:
    """Dependencies for a single node in the graph.

    Attributes:
        upstream_routers: Router nodes that must decide before this node can run.
        upstream_nodes: Regular nodes that must complete before this node.
        is_on_path: Mapping of router name -> node name for path from router
            to this node (used to check if router chose the path leading here).
    """

    upstream_routers: list[str] = field(default_factory=list)
    upstream_nodes: list[str] = field(default_factory=list)
    is_on_path: dict[str, str] = field(default_factory=dict)


class DependencyTracker:
    """Builds and tracks dependencies for all nodes in the graph.

    Analyzes the graph structure to determine which routers must decide
    before a node can run, which nodes must complete, and how to handle
    cyclic dependencies. Used by the speculative executor to decide
    when a node is ready to run.
    """

    def __init__(
        self,
        graph: CompiledStateGraph,
        analysis: GraphAnalysis,
        plans: dict[str, SpeculationPlan] | None = None,
    ) -> None:
        """Initialize the dependency tracker.

        Args:
            graph: The compiled LangGraph (must have get_graph()).
            analysis: GraphAnalysis with routers, cycles, and back edges.
            plans: Optional mapping of router name -> SpeculationPlan for
                resolution policies.
        """
        self.graph = graph
        self.analysis = analysis
        self.dependencies: dict[str, NodeDependencies] = {}
        self.graph_obj = graph.get_graph()
        self.router_names: set[str] = {r.name for r in analysis.routers}
        self.back_edges_set: set[tuple[str, str]] = set(analysis.back_edges)
        self._plans: dict[str, SpeculationPlan] = plans or {}

        self.predecessors: dict[str, list[str]] = defaultdict(list)
        self.successors: dict[str, list[str]] = defaultdict(list)

        self._build_dependencies()

    def _build_dependencies(self) -> None:
        """Build predecessor/successor maps and compute dependencies per node."""
        logger.debug("Building dependency map...")
        for edge in self.graph_obj.edges:
            self.successors[edge.source].append(edge.target)
            self.predecessors[edge.target].append(edge.source)

        for node_name in self.graph_obj.nodes:
            if node_name in ("__start__", "__end__"):
                continue
            self.dependencies[node_name] = self._compute_node_dependencies(node_name)
            logger.debug(
                "Node '%s': routers=%s, nodes=%s",
                node_name,
                self.dependencies[node_name].upstream_routers,
                self.dependencies[node_name].upstream_nodes,
            )

    def _compute_node_dependencies(self, node_name: str) -> NodeDependencies:
        """Compute upstream routers and nodes for a single node via backward walk.

        Args:
            node_name: Name of the node to compute dependencies for.

        Returns:
            NodeDependencies with upstream_routers, upstream_nodes, is_on_path.
        """
        deps = NodeDependencies()
        visited: set[str] = set()
        in_recursion: set[str] = set()

        def walk_backwards(current: str, from_node: str | None = None) -> None:
            if current in visited:
                return
            if current in in_recursion:
                return

            visited.add(current)
            in_recursion.add(current)

            for pred in self.predecessors.get(current, []):
                if pred in ("__start__", "__end__"):
                    if pred == "__start__" and pred in self.router_names:
                        if pred not in deps.upstream_routers:
                            deps.upstream_routers.append(pred)
                            if from_node:
                                deps.is_on_path[pred] = from_node
                    continue

                if pred in self.router_names:
                    if pred not in deps.upstream_routers:
                        deps.upstream_routers.append(pred)
                        if from_node:
                            deps.is_on_path[pred] = from_node
                else:
                    if pred not in deps.upstream_nodes:
                        deps.upstream_nodes.append(pred)
                    walk_backwards(pred, from_node or current)

            in_recursion.remove(current)

        walk_backwards(node_name, node_name)
        return deps

    def is_ready(
        self,
        node_name: str,
        completed_nodes: dict[str, Any],
        speculation_decisions: dict[str, str],
        cancelled_nodes: set[str],
        allow_reexecution: bool = True,
    ) -> bool:
        """Check whether a node is ready to run.

        Args:
            node_name: Name of the node to check.
            completed_nodes: Dict of node name -> result for completed nodes.
            speculation_decisions: Dict of router name -> chosen next node.
            cancelled_nodes: Set of nodes that were cancelled.
            allow_reexecution: If False, already-completed nodes are not ready.

        Returns:
            True if the node can run, False otherwise.
        """
        if node_name in cancelled_nodes:
            return False
        if node_name in completed_nodes and not allow_reexecution:
            return False
        if node_name not in self.dependencies:
            return True

        deps = self.dependencies[node_name]

        if self.analysis.has_cycles:
            return self._is_ready_cyclic(
                node_name, deps, completed_nodes, speculation_decisions
            )
        return self._is_ready_acyclic(
            node_name, deps, completed_nodes, speculation_decisions
        )

    def _is_ready_cyclic(
        self,
        node_name: str,
        deps: NodeDependencies,
        completed_nodes: dict[str, Any],
        speculation_decisions: dict[str, str],
    ) -> bool:
        """Check readiness for a node in a cyclic graph (forward edges only)."""
        immediate_preds = self.predecessors.get(node_name, [])
        forward_preds = [
            pred
            for pred in immediate_preds
            if (pred, node_name) not in self.back_edges_set
        ]

        if not forward_preds:
            return True

        controlling_decision = None
        for decision_node, chosen in speculation_decisions.items():
            if chosen in forward_preds:
                controlling_decision = decision_node
                break

        if controlling_decision:
            chosen_pred = speculation_decisions[controlling_decision]
            return chosen_pred in completed_nodes

        for pred in forward_preds:
            if pred in ("__start__", "__end__"):
                continue
            if pred not in completed_nodes:
                return False
            if pred in self.router_names and pred in speculation_decisions:
                if speculation_decisions[pred] != node_name:
                    return False

        return True

    def _is_ready_acyclic(
        self,
        node_name: str,
        deps: NodeDependencies,
        completed_nodes: dict[str, Any],
        speculation_decisions: dict[str, str],
    ) -> bool:
        """Check readiness for a node in an acyclic graph."""
        at_least_one_chose_us = False

        for router in deps.upstream_routers:
            if router not in speculation_decisions:
                continue
            chosen_next = speculation_decisions[router]
            expected_next = deps.is_on_path.get(router)

            if expected_next and chosen_next == expected_next:
                at_least_one_chose_us = True
            elif expected_next and chosen_next != expected_next:
                if chosen_next in deps.upstream_nodes:
                    at_least_one_chose_us = True
                else:
                    return False

        if not at_least_one_chose_us and deps.upstream_routers:
            return False

        unchosen_nodes = self._get_unchosen_nodes(speculation_decisions)
        for upstream_node in deps.upstream_nodes:
            if upstream_node in unchosen_nodes:
                continue
            if upstream_node not in completed_nodes:
                return False

        return True

    def _get_unchosen_nodes(self, speculation_decisions: dict[str, str]) -> set[str]:
        """Get all nodes on unchosen branches using plan resolution policies.

        Args:
            speculation_decisions: Dict of router name -> chosen next node.

        Returns:
            Set of node names that are on branches not chosen by routers.
        """
        unchosen: set[str] = set()
        for decision_node, chosen in speculation_decisions.items():
            plan = self._plans.get(decision_node)
            if plan:
                unchosen |= plan.resolution.get_cancel_set(chosen)
        return unchosen
