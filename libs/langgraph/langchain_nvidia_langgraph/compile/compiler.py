"""LangGraph optimization: one-call API built on ``DefaultGraphCompiler``.

Uses the same ``DefaultGraphCompiler`` + adapter pattern as agno and crewai.
The ``LangGraphExtractor`` adapter provides graph extraction, node
introspection, and builder hooks; ``DefaultGraphCompiler`` runs the 6-stage
analysis pipeline.

One-call optimization::

    from langchain_nvidia_langgraph.compile import compile_langgraph
    optimized = compile_langgraph(my_graph)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import replace
from typing import Any

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from nat_app.compiler import DefaultGraphCompiler, context_to_result
from nat_app.graph.scheduling import CompilationResult, TransformationResult

from ..adapter.extractor import LangGraphExtractor
from ..analysis.analyzer import GraphAnalysis
from ..analysis.dependency_tracker import DependencyTracker
from ..builder.builder import build_optimized_langgraph
from ..graph.constraints import OptimizationConfig

logger = logging.getLogger(__name__)


def compile_langgraph(
    source: StateGraph | CompiledStateGraph,
    *,
    optimization: OptimizationConfig | None = None,
    interrupt_before: list[str] | None = None,
    interrupt_after: list[str] | None = None,
    checkpointer: Any | None = None,
    cache: Any | None = None,
    store: Any | None = None,
    debug: bool = False,
    name: str | None = None,
    speculative_config: Any | None = None,
    max_subgraph_depth: int = 10,
) -> CompiledStateGraph:
    """One-call optimization for a LangGraph.

    Uses :class:`~nat_app.compiler.DefaultGraphCompiler` with
    :class:`~langchain_nvidia_langgraph.adapter.extractor.LangGraphExtractor`
    to analyze the graph, then builds an optimized ``CompiledStateGraph``
    with parallel
    fan-out / fan-in patterns.

    Args:
        source: The LangGraph to optimize (``StateGraph`` or ``CompiledStateGraph``).
        optimization: ``OptimizationConfig`` for fine-tuning. Default gives vanilla
            LangGraph. Note: When enable_speculation=True, checkpointer, streaming,
            and interrupts are not supported. Use enable_parallel=True only for
            full compatibility.
        interrupt_before: Node names to pause before.
        interrupt_after: Node names to pause after.
        checkpointer: LangGraph checkpointer.
        cache: Optional cache for the compiled graph.
        store: Optional store for the compiled graph.
        debug: Enable debug mode.
        name: Name for the compiled graph.
        speculative_config: Config for speculative execution.
        max_subgraph_depth: Max recursion depth for nested subgraph optimization.

    Returns:
        The optimized ``CompiledStateGraph``.
    """
    compile_kwargs: dict[str, Any] = {}
    if interrupt_before is not None:
        compile_kwargs["interrupt_before"] = interrupt_before
    if interrupt_after is not None:
        compile_kwargs["interrupt_after"] = interrupt_after
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
    if cache is not None:
        compile_kwargs["cache"] = cache
    if store is not None:
        compile_kwargs["store"] = store
    if debug:
        compile_kwargs["debug"] = True
    if name is not None:
        compile_kwargs["name"] = name

    compiled = source.compile() if isinstance(source, StateGraph) else source

    our_config = optimization or OptimizationConfig()
    interrupt_nodes = set(interrupt_before or []) | set(interrupt_after or [])
    if interrupt_nodes:
        our_config = replace(
            our_config,
            parallel_unsafe_nodes=(our_config.parallel_unsafe_nodes | interrupt_nodes),
        )
    result = transform_graph(
        compiled, optimization=our_config, max_depth=max_subgraph_depth
    )
    optimized = build_optimized_langgraph(compiled, result, compile_kwargs)

    if our_config.enable_speculation:
        warnings.warn(
            "Speculative execution does not support checkpointer, streaming, or "
            "interrupts. Use enable_parallel=True without enable_speculation for "
            "full LangGraph compatibility.",
            UserWarning,
            stacklevel=2,
        )
        spec_config = (
            speculative_config
            if speculative_config is not None
            else our_config._to_speculative_route_config()
        )
        return _wrap_speculative(optimized, result, spec_config)
    return optimized.optimized_graph


def transform_graph(
    compiled_graph: CompiledStateGraph,
    optimization: OptimizationConfig | None = None,
    max_depth: int = 10,
) -> TransformationResult:
    """Analyze and compute the optimized execution order for a LangGraph.

    Runs the ``DefaultGraphCompiler`` 6-stage pipeline via ``LangGraphExtractor``,
    with recursive subgraph optimization.

    Args:
        compiled_graph: The compiled LangGraph.
        optimization: OptimizationConfig for fine-tuning (parallel/speculation).
        max_depth: Max recursion depth for nested subgraph optimization.

    Returns:
        A :class:`~nat_app.graph.scheduling.TransformationResult` with the
        optimized execution order, node analyses, and edge classifications.
    """
    return _transform_graph(
        compiled_graph,
        optimization=optimization,
        max_depth=max_depth,
        current_depth=0,
    )


def _transform_graph(
    compiled_graph: CompiledStateGraph,
    optimization: OptimizationConfig | None = None,
    max_depth: int = 10,
    *,
    current_depth: int = 0,
) -> TransformationResult:
    """Internal implementation with recursion depth tracking.

    Recursively optimizes nested subgraphs before running the main compiler
    pipeline. Stops when current_depth >= max_depth to avoid unbounded recursion.

    Args:
        compiled_graph: The compiled LangGraph to analyze.
        optimization: OptimizationConfig for fine-tuning. Defaults to vanilla if None.
        max_depth: Maximum recursion depth for nested subgraph optimization.
        current_depth: Current recursion depth (0 at top level). Passed through
            recursive calls; do not set manually.

    Returns:
        TransformationResult with optimized execution order, node analyses,
        and edge classifications.
    """
    opt_config = optimization or OptimizationConfig()

    nat_app_config = opt_config._to_nat_app_optimization_config()

    extractor = LangGraphExtractor(compiled_graph)

    if extractor.subgraphs and current_depth < max_depth:
        _optimize_subgraphs(
            compiled_graph, extractor, opt_config, max_depth, current_depth
        )

    compiler = DefaultGraphCompiler(adapter=extractor, config=nat_app_config)
    context = compiler.compile(compiled_graph)
    return context_to_result(context)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _optimize_subgraphs(
    compiled: CompiledStateGraph,
    extractor: LangGraphExtractor,
    config: OptimizationConfig,
    max_depth: int,
    current_depth: int,
) -> None:
    """Recursively optimize subgraph nodes before the main pipeline.

    For each node that is a CompiledStateGraph (nested subgraph), runs
    _transform_graph on it and replaces the node's bound with the optimized
    graph. Mutates the extractor's graph so the builder sees the optimized
    subgraphs.

    Args:
        compiled: The parent compiled graph containing subgraph nodes.
        extractor: LangGraphExtractor that has already extracted subgraphs.
        config: OptimizationConfig for the transformation.
        max_depth: Maximum recursion depth for nested subgraphs.
        current_depth: Current recursion depth (incremented for each level).
    """
    for name, subgraph in extractor.subgraphs.items():
        logger.info(
            "Recursively optimizing subgraph '%s' (depth %d/%d)",
            name,
            current_depth + 1,
            max_depth,
        )
        sub_result = _transform_graph(
            subgraph,
            optimization=config,
            max_depth=max_depth,
            current_depth=current_depth + 1,
        )
        sub_optimized = build_optimized_langgraph(subgraph, sub_result)

        if hasattr(compiled, "nodes") and name in compiled.nodes:
            node_obj = compiled.nodes[name]
            if hasattr(node_obj, "bound"):
                node_obj.bound = sub_optimized.optimized_graph

        # Update extractor's graph so the builder uses the optimized subgraph
        # (the builder reads func from transformation.graph, not from compiled.nodes)
        if hasattr(extractor, "_graph") and extractor._graph is not None:
            try:
                node_info = extractor._graph.get_node(name)
                node_info.func = sub_optimized.optimized_graph
            except KeyError:
                pass
        if hasattr(extractor, "_node_funcs"):
            extractor._node_funcs[name] = sub_optimized.optimized_graph

        logger.info(
            "Subgraph '%s' optimized: %d stages, ~%.2fx speedup",
            name,
            len(sub_optimized.stages),
            sub_optimized.speedup_estimate,
        )


def _wrap_speculative(
    optimized: Any,
    transformation: CompilationResult,
    speculative_config: Any | None,
) -> Any:
    """Wrap the optimized graph with speculative execution.

    Builds GraphAnalysis and DependencyTracker from the compilation result,
    then wraps the optimized graph with SpeculativeGraphWrapper. The wrapper
    runs router branches speculatively when safe.

    Args:
        optimized: Build result containing optimized_graph and optimized_topology.
        transformation: CompilationResult with node analyses and topology.
        speculative_config: Optional config for speculative execution.

    Returns:
        A graph wrapped for speculative execution (SpeculativeGraphWrapper).
    """
    from ..executors.speculative.executor import SpeculativeGraphWrapper

    optimized_graph = optimized.optimized_graph
    optimized_topology = getattr(optimized, "optimized_topology", None)
    optimized_entry = getattr(optimized, "optimized_entry_point", "")

    analysis = GraphAnalysis.from_compilation_result(
        transformation,
        optimized_graph,
        optimized_topology=optimized_topology,
        optimized_entry_point=optimized_entry,
    )

    dep_tracker = DependencyTracker(optimized_graph, analysis)

    node_analyses = getattr(transformation, "node_analyses", {})
    node_rw = {
        name: {"reads": a.reads, "writes": a.mutations}
        for name, a in node_analyses.items()
    }

    return SpeculativeGraphWrapper(
        graph=optimized_graph,
        analysis=analysis,
        dependency_tracker=dep_tracker,
        config=speculative_config,
        node_rw=node_rw,
    )
