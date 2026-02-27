"""Parallelism and speculation constraints for LangGraph optimization.

Decorators (re-exported from nat_app): ``sequential`` marks nodes as
non-parallelizable; ``depends_on`` declares explicit dependencies when
auto-analysis cannot infer them; ``speculation_unsafe`` marks nodes as unsafe
for speculative execution.

``OptimizationConfig`` is the unified config for parallel and speculation.
Pass to ``compile(..., optimization=...)``. Attributes include
``parallel_unsafe_nodes``, ``parallel_safe_overrides``,
``speculation_unsafe_nodes``, ``speculation_safe_overrides``, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nat_app.constraints import depends_on, sequential
from nat_app.speculation.safety import speculation_unsafe


@dataclass
class OptimizationConfig:
    """Unified config for parallel and speculative optimization.

    Uses symmetric unsafe/safe-override attributes for both domains.
    Pass to compile(..., optimization=...) or compile_langgraph(..., optimization=...).
    """

    # Parallel
    parallel_unsafe_nodes: set[str] = field(default_factory=set)
    """Node names that must run sequentially (no parallelization)."""
    parallel_safe_overrides: set[str] = field(default_factory=set)
    """Node names allowed to parallelize despite @sequential or analysis."""
    explicit_dependencies: dict[str, set[str]] = field(default_factory=dict)
    """Explicit dependencies when auto-analysis cannot infer them."""
    enable_parallel: bool = False
    """If True, parallelize independent stages (fan-out/fan-in).
    Default False = vanilla LangGraph."""
    enable_speculation: bool = False
    """If True, speculative execution at conditional branches.
    Default False = vanilla LangGraph. Note: Speculative mode does not support
    checkpointer, streaming, or interrupts."""
    trust_analysis: bool = False
    """Whether to trust static analysis (less conservative)."""
    max_recursion_depth: int = 5
    """Max depth for nested subgraph optimization."""

    # Speculation
    speculation_unsafe_nodes: set[str] = field(default_factory=set)
    """Node names where speculation is disabled."""
    speculation_safe_overrides: set[str] = field(default_factory=set)
    """Node names where speculation is allowed despite @speculation_unsafe."""
    speculation_max_iterations: int = 50
    """Max iterations for speculative execution loop."""
    invoke_executor_max_workers: int = 8
    """Max worker threads for sync invoke when called from async context. Default 8."""

    @classmethod
    def conservative(cls) -> OptimizationConfig:
        """Return config with vanilla behavior (no optimization). Alias for cls()."""
        return cls()

    def _to_nat_app_optimization_config(self) -> Any:
        """Convert to nat_app.constraints.OptimizationConfig for internal use."""
        from nat_app.constraints import OptimizationConfig as NatAppOptimizationConfig

        force_sequential = self.parallel_unsafe_nodes - self.parallel_safe_overrides
        return NatAppOptimizationConfig(
            force_sequential=force_sequential,
            explicit_dependencies=dict(self.explicit_dependencies),
            side_effect_nodes=set(),
            disable_parallelization=not self.enable_parallel,
            trust_analysis=self.trust_analysis,
            max_recursion_depth=self.max_recursion_depth,
        )

    def _to_speculative_route_config(self) -> Any:
        """Convert to SpeculativeRouteConfig for internal use."""
        from nat_app.speculation.safety import SpeculationSafetyConfig

        from ..executors.speculative.executor import SpeculativeRouteConfig

        safety = SpeculationSafetyConfig(
            unsafe_nodes=self.speculation_unsafe_nodes,
            safe_overrides=self.speculation_safe_overrides,
        )
        return SpeculativeRouteConfig(
            speculation_safety=safety,
            max_iterations=self.speculation_max_iterations,
            log_level=None,  # Use system log level
            invoke_executor_max_workers=self.invoke_executor_max_workers,
        )


__all__ = [
    "OptimizationConfig",
    "depends_on",
    "sequential",
    "speculation_unsafe",
]
