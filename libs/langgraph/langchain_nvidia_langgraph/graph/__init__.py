"""Graph API for LangGraph-style state graphs and optimization constraints.

Exports ``StateGraph``, ``CompilableGraph``, and ``with_app_compile`` for
building graphs; ``OptimizationConfig``, ``depends_on``, ``sequential``, and
``speculation_unsafe`` for constraints.
"""

from .constraints import OptimizationConfig, depends_on, sequential, speculation_unsafe
from .state_graph import CompilableGraph, StateGraph, with_app_compile

__all__ = [
    "CompilableGraph",
    "StateGraph",
    "with_app_compile",
    "OptimizationConfig",
    "depends_on",
    "sequential",
    "speculation_unsafe",
]
