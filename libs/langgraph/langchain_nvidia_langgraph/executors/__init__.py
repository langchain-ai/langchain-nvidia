"""Speculative execution for LangGraph optimization.

Re-exports ``SpeculativeGraphWrapper``, ``SpeculativeRouteConfig``, and
``SpeculativeRouteExecutor`` from the speculative submodule.
"""

from .speculative.executor import (
    SpeculativeGraphWrapper,
    SpeculativeRouteConfig,
    SpeculativeRouteExecutor,
)

__all__ = [
    "SpeculativeGraphWrapper",
    "SpeculativeRouteConfig",
    "SpeculativeRouteExecutor",
]
