"""Speculative branch executor for LangGraph optimization.

Provides GraphAnalysis, DependencyTracker, ChannelStateManager, ExecutionLoop,
RouterEvaluator, and LangGraphNodeExecutor for speculative execution of LangGraph
graphs with router nodes.
"""

from nat_app.executors import ExecutionState
from nat_app.speculation.safety import (
    SpeculationSafetyConfig,
    is_marked_speculation_unsafe,
    speculation_unsafe,
)

from ...analysis import DependencyTracker, GraphAnalysis, NodeDependencies, RouterInfo
from .channel_state_manager import ChannelStateManager
from .execution_loop import ExecutionLoop
from .executor import (
    SpeculativeGraphWrapper,
    SpeculativeRouteConfig,
    SpeculativeRouteExecutor,
)
from .node_executor import LangGraphNodeExecutor
from .router_evaluator import (
    InvalidRouterError,
    RouterEvaluationError,
    RouterEvaluator,
    SpeculativeExecutionError,
)

__all__ = [
    "ChannelStateManager",
    "DependencyTracker",
    "ExecutionLoop",
    "ExecutionState",
    "GraphAnalysis",
    "InvalidRouterError",
    "LangGraphNodeExecutor",
    "NodeDependencies",
    "RouterEvaluationError",
    "RouterEvaluator",
    "RouterInfo",
    "SpeculationSafetyConfig",
    "SpeculativeExecutionError",
    "SpeculativeGraphWrapper",
    "SpeculativeRouteConfig",
    "SpeculativeRouteExecutor",
    "is_marked_speculation_unsafe",
    "speculation_unsafe",
]
