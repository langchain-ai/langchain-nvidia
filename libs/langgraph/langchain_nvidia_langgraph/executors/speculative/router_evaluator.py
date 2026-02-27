"""LangGraph-specific router evaluation.

Handles invocation of LangGraph conditional edge functions
(``ainvoke``/``invoke``/plain callable) and resolution of
edge names to target node names via path mappings.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

from langgraph.graph import END
from langgraph.pregel.types import All as LangGraphAll

from ...analysis import GraphAnalysis, RouterInfo

logger = logging.getLogger(__name__)


class SpeculativeExecutionError(Exception):
    """Base exception for speculative execution failures."""


class RouterEvaluationError(SpeculativeExecutionError):
    """Router conditional function evaluation failed."""

    def __init__(self, router_name: str, cause: Exception):
        self.router_name = router_name
        self.cause = cause
        super().__init__(f"Router '{router_name}' evaluation failed: {cause}")


class InvalidRouterError(SpeculativeExecutionError):
    """Router configuration is invalid."""

    def __init__(self, router_name: str, reason: str):
        self.router_name = router_name
        self.reason = reason
        super().__init__(f"Router '{router_name}' is invalid: {reason}")


class RouterEvaluator:
    """Evaluates LangGraph router decisions.

    Invokes conditional edge functions using the LangGraph dispatch
    chain (``ainvoke`` -> ``invoke`` -> plain callable) and resolves
    edge names to target node names using path mappings.
    """

    def __init__(self, analysis: GraphAnalysis) -> None:
        self._analysis = analysis
        self._router_lookup: dict[str, RouterInfo] = {
            r.name: r for r in analysis.routers
        }

    def get_router_info(self, router_name: str) -> RouterInfo | None:
        return self._router_lookup.get(router_name)

    async def evaluate_decision(
        self,
        router_name: str,
        router_info: RouterInfo,
        state: dict[str, Any],
    ) -> str:
        """Evaluate a router's conditional edge function to determine the chosen target.

        Router failures propagate; they are not silently recovered. Incorrect
        path selection would violate correctness.
        """
        if not router_info.possible_targets:
            raise InvalidRouterError(router_name, "Router has no possible targets")
        if not router_info.conditional_edge_fn:
            return router_info.possible_targets[0]

        edge_name = await self._invoke_router_function(
            router_name,
            router_info.conditional_edge_fn,
            state,
        )
        return self._resolve_edge_to_target(router_name, edge_name, router_info)

    async def _invoke_router_function(
        self,
        router_name: str,
        fn: Any,
        state: dict[str, Any],
    ) -> str:
        try:
            if hasattr(fn, "ainvoke") and callable(fn.ainvoke):
                return await fn.ainvoke(state)
            if hasattr(fn, "invoke") and callable(fn.invoke):
                return fn.invoke(state)
            if callable(fn):
                result = (
                    await fn(state) if inspect.iscoroutinefunction(fn) else fn(state)
                )
                if inspect.iscoroutine(result):
                    result = await result
                return result
            raise RouterEvaluationError(
                router_name, TypeError(f"Not callable: {type(fn)}")
            )
        except RouterEvaluationError:
            raise
        except Exception as e:
            raise RouterEvaluationError(router_name, e) from e

    def _resolve_edge_to_target(
        self,
        router_name: str,
        edge_name: Any,
        router_info: RouterInfo,
    ) -> str:
        if not edge_name:
            raise InvalidRouterError(router_name, f"Invalid edge name: {edge_name!r}")
        if isinstance(edge_name, list):
            if not edge_name:
                raise InvalidRouterError(router_name, "Empty edge list")
            edge_name = edge_name[0]
        if hasattr(edge_name, "node"):
            edge_name = edge_name.node
        if router_info.path_mapping and edge_name in router_info.path_mapping:
            return router_info.path_mapping[edge_name]
        if edge_name == "*" or isinstance(edge_name, type(LangGraphAll)):
            return router_info.possible_targets[0]
        if edge_name in router_info.possible_targets or edge_name in ("__end__", END):
            return edge_name
        raise InvalidRouterError(
            router_name,
            f"Edge name {edge_name!r} not in path_mapping or possible_targets "
            f"{router_info.possible_targets!r}",
        )
