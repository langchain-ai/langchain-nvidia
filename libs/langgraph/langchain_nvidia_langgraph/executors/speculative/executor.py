"""Speculative route executor: dependency-aware execution for LangGraph.

Coordinates the execution loop, router evaluation, node execution,
and channel state management for speculative execution of LangGraph graphs.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.runnables import Runnable
from nat_app.executors import ExecutionState, ResultHandler
from nat_app.executors.metrics import ExecutionMetrics
from nat_app.speculation.plan import SpeculationPlan
from nat_app.speculation.safety import (
    SpeculationSafetyConfig,
    is_marked_speculation_unsafe,
)

from ...analysis import DependencyTracker, GraphAnalysis
from .channel_state_manager import ChannelStateManager
from .execution_loop import ExecutionLoop
from .node_executor import LangGraphNodeExecutor
from .router_evaluator import (
    RouterEvaluator,
)

logger = logging.getLogger(__name__)


_LANGGRAPH_UNSAFE_NODES: frozenset[str] = frozenset(
    {
        "HumanInTheLoopMiddleware",
        "PIIMiddleware",
        "ToolCallLimitMiddleware",
    }
)


def _default_safety() -> SpeculationSafetyConfig:
    return SpeculationSafetyConfig(unsafe_nodes=set(_LANGGRAPH_UNSAFE_NODES))


@dataclass
class SpeculativeRouteConfig:
    """Configuration for the LangGraph speculative route executor."""

    max_iterations: int = 50
    log_level: str | None = None
    """If None, use system log level; otherwise override (e.g. 'INFO', 'DEBUG')."""
    speculation_safety: SpeculationSafetyConfig = field(default_factory=_default_safety)
    invoke_executor_max_workers: int = 8
    """Max worker threads for sync invoke when called from async context. Default 8."""


def _is_speculation_safe(
    node_name: str,
    config: SpeculationSafetyConfig,
    graph_nodes: dict[str, Any] | None = None,
) -> bool:
    """Check if speculation can safely bypass a node.

    Checks (in order): safe overrides, unsafe set, ``@speculation_unsafe``
    decorator on the node and its middleware.
    """
    base_name = node_name.split(".", maxsplit=1)[0] if "." in node_name else node_name

    if base_name in config.safe_overrides or node_name in config.safe_overrides:
        return True
    if base_name in config.unsafe_nodes or node_name in config.unsafe_nodes:
        return False

    if graph_nodes and node_name in graph_nodes:
        node_obj = graph_nodes[node_name]
        if is_marked_speculation_unsafe(node_obj):
            return False
        if hasattr(node_obj, "middleware") and is_marked_speculation_unsafe(
            node_obj.middleware
        ):
            return False

    return True


class SpeculativeRouteExecutor:
    """Speculative executor for LangGraph router-based graphs.

    Thin coordinator that wires together:
    - :class:`LangGraphNodeExecutor` for node invocation
    - :class:`RouterEvaluator` for router decision evaluation
    - :class:`ChannelStateManager` for LangGraph channel state
    - :class:`ExecutionLoop` for the event-driven execution cycle
    """

    def __init__(
        self,
        graph: Any,
        analysis: GraphAnalysis,
        config: SpeculativeRouteConfig | None = None,
        runnable_config: dict[str, Any] | None = None,
        *,
        dependency_tracker: DependencyTracker | None = None,
        node_rw: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.graph = graph
        self.analysis = analysis
        self.config = config or SpeculativeRouteConfig()
        self.runnable_config = runnable_config
        self.dep_tracker: DependencyTracker | None = dependency_tracker
        self.node_rw = node_rw

        self.state_manager = ChannelStateManager(graph)
        self.node_executor = LangGraphNodeExecutor(graph, runnable_config)
        self.result_handler = ResultHandler(
            command_checker=lambda result: hasattr(result, "update")
            and not callable(result.update),
        )

        if self.config.log_level:
            logger.setLevel(getattr(logging, self.config.log_level, logging.INFO))

    # -- Main execution ----------------------------------------------------

    async def execute(
        self, initial_state: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        start_time = time.time()
        self._ensure_plans()

        logger.info("Starting speculative route execution")
        logger.info("  Entry point: %s", self.analysis.entry_point)
        logger.info("  Routers: %s", len(self.analysis.routers))
        logger.info("  Has cycles: %s", self.analysis.has_cycles)
        logger.info("  Using unified loop with contiguous chain batching")

        execution_state = ExecutionState()
        execution_state.execution_start_time = start_time

        main_channels = self.state_manager.create_isolated_channels()
        current_state = self.state_manager.initialize_state(
            main_channels, initial_state
        )
        execution_state.channels = main_channels

        execution_state.mark_node_ready(self.analysis.entry_point)

        loop = ExecutionLoop(
            analysis=self.analysis,
            plans=self._plans,
            dep_tracker=self.dep_tracker,  # type: ignore[arg-type]
            state_manager=self.state_manager,
            node_executor=self.node_executor,
            router_evaluator=RouterEvaluator(self.analysis),
            result_handler=self.result_handler,
            speculation_safety=self.config.speculation_safety,
            node_rw=self.node_rw,
            graph=self.graph,
            max_iterations=self.config.max_iterations,
        )
        iteration = await loop.run(execution_state, current_state)

        final_state = self.state_manager.get_current_state(execution_state.channels)
        metrics = self._build_metrics(execution_state, start_time, iteration)
        self._log_execution_summary(metrics)

        return final_state, metrics

    def _ensure_plans(self) -> None:
        if not hasattr(self, "_plans"):
            self._plans: dict[str, SpeculationPlan] = self._build_speculation_plans()
        if self.dep_tracker is None:
            self.dep_tracker = DependencyTracker(
                self.graph, self.analysis, plans=self._plans
            )
        elif not self.dep_tracker._plans:
            self.dep_tracker._plans = self._plans

    def _build_speculation_plans(self) -> dict[str, SpeculationPlan]:
        """Build SpeculationPlan objects via plan_speculation().

        Extracts the graph data from the LangGraph compiled graph into
        the format that :func:`plan_speculation` expects, then layers
        on LangGraph-specific safety filtering.
        """
        from nat_app.speculation.plan import plan_speculation

        graph_obj = self.graph.get_graph()
        nodes: dict[str, None] = {}
        edges: list[tuple[str, str]] = []
        conditional_edges: dict[str, dict[str, str]] = {}

        for name in self.graph.nodes:
            if name not in ("__start__", "__end__"):
                nodes[name] = None

        for edge in graph_obj.edges:
            if edge.source in ("__start__",) or edge.target in ("__end__",):
                continue
            if edge.conditional:
                conditional_edges.setdefault(edge.source, {})[edge.target] = edge.target
            else:
                edges.append((edge.source, edge.target))

        for ri in self.analysis.routers:
            if ri.path_mapping:
                conditional_edges[ri.name] = dict(ri.path_mapping)

        unsafe_nodes = set(self.config.speculation_safety.unsafe_nodes)
        for name in nodes:
            if not _is_speculation_safe(
                name, self.config.speculation_safety, self.graph.nodes
            ):
                unsafe_nodes.add(name)

        safety = SpeculationSafetyConfig(
            unsafe_nodes=unsafe_nodes,
            safe_overrides=self.config.speculation_safety.safe_overrides,
        )

        plans = plan_speculation(
            nodes=nodes,
            edges=edges,
            conditional_edges=conditional_edges if conditional_edges else None,
            safety=safety,
        )
        return {p.decision_node: p for p in plans}

    # -- Metrics -----------------------------------------------------------

    @staticmethod
    def _build_metrics(
        execution_state: ExecutionState,
        start_time: float,
        iteration: int,
    ) -> dict[str, Any]:
        elapsed = time.time() - start_time
        sm = ExecutionMetrics.from_execution_state(
            execution_state, elapsed, iterations=iteration
        )

        deepcopy_ms = sum(execution_state.deepcopy_times) * 1000
        merge_ms = sum(execution_state.state_merge_times) * 1000
        task_ms = sum(execution_state.task_creation_times) * 1000
        sm.profiling = {
            "deepcopy_ms": deepcopy_ms,
            "deepcopy_count": len(execution_state.deepcopy_times),
            "state_merge_ms": merge_ms,
            "state_merge_count": len(execution_state.state_merge_times),
            "task_creation_ms": task_ms,
            "total_measured_overhead_ms": deepcopy_ms + merge_ms + task_ms,
        }

        return sm.to_dict()

    @staticmethod
    def _log_execution_summary(metrics: dict[str, Any]) -> None:
        logger.info("Execution complete in %.0fms", metrics["total_time_ms"])
        logger.info("  Tools launched: %s", metrics["tools_launched"])
        logger.info("  Tools cancelled: %s", metrics["tools_cancelled"])
        logger.info("  Tools completed: %s", metrics["tools_completed"])
        if "speedup_ratio" in metrics:
            logger.info("  Speedup: %.2fx", metrics["speedup_ratio"])


class SpeculativeGraphWrapper(Runnable[dict[str, Any], dict[str, Any]]):
    """Wraps a CompiledStateGraph with speculative execution.

    Implements LangChain Runnable so LangGraph accepts it as a node (e.g. in
    nested subgraphs). Delegates ``.ainvoke()`` and ``.invoke()`` to a
    ``SpeculativeRouteExecutor`` pre-initialized with compile-time analysis.
    All other attribute access is forwarded to the underlying compiled graph.

    Limitations: Does not support checkpointer, streaming (stream/astream),
    interrupts (interrupt_before/after), or human-in-the-loop. Use the underlying
    optimized graph directly for those features.
    """

    def __init__(
        self,
        graph: Any,
        analysis: GraphAnalysis,
        dependency_tracker: DependencyTracker,
        config: SpeculativeRouteConfig | None = None,
        runnable_config: dict[str, Any] | None = None,
        node_rw: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self._graph = graph
        self._analysis = analysis
        self._dependency_tracker = dependency_tracker
        self._config = config or SpeculativeRouteConfig()
        self._runnable_config = runnable_config
        self._node_rw = node_rw
        self._last_metrics: dict[str, Any] | None = None
        self._invoke_executor: (concurrent.futures.ThreadPoolExecutor | None) = None

    def _get_invoke_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        """Lazily create thread pool for sync invoke from async context."""
        if self._invoke_executor is None:
            max_workers = self._config.invoke_executor_max_workers
            self._invoke_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="langgraph_invoke_",
            )
        return self._invoke_executor

    def invoke(  # type: ignore[override]
        self,
        input_state: dict[str, Any],
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Sync invoke: runs ainvoke in event loop.

        When already inside an async context (e.g. Jupyter, FastAPI), runs
        ainvoke in a thread pool to avoid conflict with the existing loop.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(input_state, config=config, **kwargs))
        else:
            pool = self._get_invoke_executor()
            future = pool.submit(
                asyncio.run,
                self.ainvoke(input_state, config=config, **kwargs),  # type: ignore[arg-type]
            )
            return future.result()  # type: ignore[return-value]

    async def ainvoke(  # type: ignore[override]
        self,
        input_state: dict[str, Any],
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        runnable_config = config or self._runnable_config
        executor = SpeculativeRouteExecutor(
            graph=self._graph,
            analysis=self._analysis,
            config=self._config,
            runnable_config=runnable_config,
            dependency_tracker=self._dependency_tracker,
            node_rw=self._node_rw,
        )
        final_state, metrics = await executor.execute(input_state)
        self._last_metrics = metrics
        return final_state

    @property
    def last_metrics(self) -> dict[str, Any] | None:
        return self._last_metrics

    def __getattr__(self, name: str) -> Any:
        return getattr(self._graph, name)

    def __repr__(self) -> str:
        return f"SpeculativeGraphWrapper(graph={self._graph!r})"
