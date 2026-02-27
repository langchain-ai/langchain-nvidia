"""LangGraph-specific node analysis.

Wraps ``nat_app.graph.static_analysis`` and ``nat_app.graph.analysis`` with
LangGraph-specific handling: subgraph schema-based analysis and
Send/Command special-call detection.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, get_type_hints

from langgraph.graph.state import CompiledStateGraph
from nat_app.graph.access import AccessSet
from nat_app.graph.analysis import NodeAnalysis
from nat_app.graph.static_analysis import analyze_function_ast

logger = logging.getLogger(__name__)

# LangGraph calls that act as optimization barriers (prevent parallelization).
LANGGRAPH_SPECIAL_CALLS: frozenset[str] = frozenset({"Send", "Command"})


def analyze_langgraph_node(
    name: str,
    func: Callable | Any,
    all_schema_fields: set[str] | None = None,
    *,
    config: Any = None,
) -> NodeAnalysis:
    """Analyze a single LangGraph node function.

    Handles subgraph nodes (``CompiledStateGraph``) via schema-based analysis,
    regular nodes via AST analysis with ``Send``/``Command`` detection, and
    conservative fallback when confidence is low.

    Args:
        name: Node name for logging and analysis.
        func: Node callable (function or CompiledStateGraph for subgraphs).
        all_schema_fields: Optional set of all schema field names.
        config: Optional analysis config (e.g., max_recursion_depth).

    Returns:
        NodeAnalysis with reads, writes, mutations, and confidence.
    """
    if isinstance(func, CompiledStateGraph):
        return _analyze_subgraph_node(name, func, all_schema_fields)

    analysis = NodeAnalysis(name=name)

    max_depth = config.max_recursion_depth if config else 5
    ast_result = analyze_function_ast(
        func,
        special_call_names=LANGGRAPH_SPECIAL_CALLS,
        max_recursion_depth=max_depth,
    )

    if not ast_result.source_available:
        analysis.source = "unavailable"
        analysis.confidence = "opaque"
        analysis.trace_successful = False
        analysis.warnings.append(
            "Source code not available — node will be kept sequential"
        )
        if all_schema_fields:
            analysis.mutations = AccessSet.from_fields(*all_schema_fields)
            analysis.is_pure = False
            analysis.warnings.append(
                f"Conservatively assuming all {len(all_schema_fields)} schema "
                "fields as writes"
            )
        return analysis

    reads = ast_result.reads
    writes = ast_result.writes
    in_place_mutations = ast_result.mutations

    all_mutations = AccessSet()
    for obj, path in writes:
        all_mutations.add(obj, path)
    for obj, path in in_place_mutations:
        all_mutations.add(obj, path)

    analysis.source = "ast"
    analysis.special_calls = ast_result.detected_special_calls

    if (
        ast_result.has_dynamic_keys
        or ast_result.has_unresolved_calls
        or ast_result.recursion_depth_hit
    ):
        confidence = "partial"
    elif not all_mutations and ast_result.warnings:
        confidence = "partial"
    else:
        confidence = "full"
    analysis.confidence = confidence

    if analysis.confidence != "full" and not all_mutations and all_schema_fields:
        all_mutations = AccessSet.from_fields(*all_schema_fields)
        analysis.warnings.append(
            f"Confidence {confidence!r} with no detected writes — "
            f"conservatively assuming all {len(all_schema_fields)} schema fields"
        )
        logger.warning(
            "Node '%s': %s confidence, assuming all schema fields "
            "(use @depends_on to explicitly declare dependencies)",
            name,
            confidence,
        )

    analysis.reads = reads
    analysis.writes = writes
    analysis.mutations = all_mutations
    analysis.is_pure = not bool(all_mutations)
    analysis.trace_successful = True
    analysis.warnings.extend(ast_result.warnings)

    logger.debug(
        "Analyzed %s: reads=%s, writes=%s, confidence=%s, source=%s",
        name,
        reads,
        all_mutations,
        analysis.confidence,
        analysis.source,
    )
    return analysis


def _analyze_subgraph_node(
    name: str,
    subgraph: CompiledStateGraph,
    all_schema_fields: set[str] | None,
) -> NodeAnalysis:
    """Analyze a node that is itself a CompiledStateGraph.

    Uses subgraph state schema when available; falls back to parent schema
    or opaque analysis when schema is unavailable.

    Args:
        name: Node name for logging.
        subgraph: The nested CompiledStateGraph to analyze.
        all_schema_fields: Optional parent schema field names for fallback.

    Returns:
        NodeAnalysis with subgraph-based reads/writes or conservative fallback.
    """
    analysis = NodeAnalysis(name=name)
    analysis.source = "subgraph_schema"
    analysis.confidence = "partial"

    sub_schema = None
    if hasattr(subgraph, "builder"):
        sub_schema = getattr(subgraph.builder, "state_schema", None)

    if sub_schema is not None and sub_schema is not dict:
        try:
            hints = get_type_hints(sub_schema, include_extras=True)
            sub_fields = set(hints.keys())
        except (TypeError, NameError, AttributeError) as exc:
            logger.debug(
                "Could not introspect subgraph schema type hints: %s",
                exc,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            sub_fields = None
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning(
                "Unexpected error introspecting subgraph schema: %s",
                exc,
                exc_info=True,
            )
            sub_fields = None

        if sub_fields:
            fields_access = AccessSet.from_fields(*sub_fields)
            analysis.reads = fields_access
            analysis.writes = AccessSet.from_fields(*sub_fields)
            analysis.mutations = AccessSet.from_fields(*sub_fields)
            analysis.is_pure = False
            return analysis

    if all_schema_fields:
        analysis.reads = AccessSet.from_fields(*all_schema_fields)
        analysis.writes = AccessSet.from_fields(*all_schema_fields)
        analysis.mutations = AccessSet.from_fields(*all_schema_fields)
        analysis.is_pure = False
        analysis.confidence = "partial"
        analysis.warnings.append(
            "Subgraph schema unavailable — using parent schema (conservative)"
        )
        return analysis

    analysis.confidence = "opaque"
    analysis.warnings.append("Subgraph with no schema — keeping sequential for safety")
    return analysis
