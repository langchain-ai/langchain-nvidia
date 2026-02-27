"""Graph analysis for speculative execution.

Provides ``GraphAnalysis`` (router metadata, cycles, back edges) and
``DependencyTracker`` for determining node execution readiness.
"""

from .analyzer import GraphAnalysis, RouterInfo
from .dependency_tracker import DependencyTracker, NodeDependencies

__all__ = [
    "GraphAnalysis",
    "RouterInfo",
    "DependencyTracker",
    "NodeDependencies",
]
