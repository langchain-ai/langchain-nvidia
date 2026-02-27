"""Unit tests for constraints public interfaces."""

from __future__ import annotations

from langchain_nvidia_langgraph.graph.constraints import (
    OptimizationConfig,
    depends_on,
    sequential,
    speculation_unsafe,
)

# ---------------------------------------------------------------------------
# OptimizationConfig
# ---------------------------------------------------------------------------


def test_optimization_config_defaults() -> None:
    """OptimizationConfig has sensible defaults."""
    config = OptimizationConfig()
    assert config.parallel_unsafe_nodes == set()
    assert config.parallel_safe_overrides == set()
    assert config.explicit_dependencies == {}
    assert config.enable_parallel is False
    assert config.enable_speculation is False
    assert config.trust_analysis is False
    assert config.max_recursion_depth == 5
    assert config.speculation_unsafe_nodes == set()
    assert config.speculation_safe_overrides == set()
    assert config.speculation_max_iterations == 50
    assert config.invoke_executor_max_workers == 8


def test_optimization_config_custom_parallel() -> None:
    """OptimizationConfig accepts custom parallel settings."""
    config = OptimizationConfig(
        parallel_unsafe_nodes={"a", "b"},
        parallel_safe_overrides={"a"},
        explicit_dependencies={"c": {"a", "b"}},
        enable_parallel=True,
        trust_analysis=True,
        max_recursion_depth=10,
    )
    assert config.parallel_unsafe_nodes == {"a", "b"}
    assert config.parallel_safe_overrides == {"a"}
    assert config.explicit_dependencies == {"c": {"a", "b"}}
    assert config.enable_parallel is True
    assert config.trust_analysis is True
    assert config.max_recursion_depth == 10


def test_optimization_config_custom_speculation() -> None:
    """OptimizationConfig accepts custom speculation settings."""
    config = OptimizationConfig(
        speculation_unsafe_nodes={"x"},
        speculation_safe_overrides={"x"},
        speculation_max_iterations=100,
        invoke_executor_max_workers=16,
        enable_speculation=True,
    )
    assert config.speculation_unsafe_nodes == {"x"}
    assert config.speculation_safe_overrides == {"x"}
    assert config.speculation_max_iterations == 100
    assert config.invoke_executor_max_workers == 16
    assert config.enable_speculation is True


def test_optimization_config_conservative() -> None:
    """conservative() returns config with vanilla behavior (no optimization)."""
    config = OptimizationConfig.conservative()
    assert isinstance(config, OptimizationConfig)
    assert config.enable_parallel is False
    assert config.enable_speculation is False
    assert config.parallel_unsafe_nodes == set()
    assert config.speculation_unsafe_nodes == set()


def test_optimization_config_conservative_equivalent_to_default() -> None:
    """conservative() is equivalent to OptimizationConfig()."""
    conservative = OptimizationConfig.conservative()
    default = OptimizationConfig()
    assert conservative.parallel_unsafe_nodes == default.parallel_unsafe_nodes
    assert conservative.enable_parallel == default.enable_parallel
    assert conservative.enable_speculation == default.enable_speculation
    assert conservative.speculation_max_iterations == default.speculation_max_iterations


# ---------------------------------------------------------------------------
# Decorators: sequential
# ---------------------------------------------------------------------------


def test_sequential_decorator_importable() -> None:
    """sequential is importable from constraints."""
    assert callable(sequential)


def test_sequential_decorator_without_args() -> None:
    """@sequential() marks function with force_sequential."""

    @sequential()
    def my_node(state: dict) -> dict:
        return state

    assert hasattr(my_node, "_optimization_constraints")
    constraints = my_node._optimization_constraints
    assert constraints.force_sequential is True
    assert constraints.has_side_effects is True


def test_sequential_decorator_with_reason() -> None:
    """@sequential(reason=...) stores the reason."""

    @sequential(reason="Writes to database")
    def my_node(state: dict) -> dict:
        return state

    constraints = my_node._optimization_constraints
    assert constraints.force_sequential is True
    assert constraints.reason == "Writes to database"


# ---------------------------------------------------------------------------
# Decorators: depends_on
# ---------------------------------------------------------------------------


def test_depends_on_decorator_importable() -> None:
    """depends_on is importable from constraints."""
    assert callable(depends_on)


def test_depends_on_decorator_adds_dependencies() -> None:
    """@depends_on("a", "b") adds explicit dependencies."""

    @depends_on("parse_query", "load_context")
    def merge_node(state: dict) -> dict:
        return state

    assert hasattr(merge_node, "_optimization_constraints")
    constraints = merge_node._optimization_constraints
    assert constraints.depends_on == {"parse_query", "load_context"}


def test_depends_on_decorator_with_reason() -> None:
    """@depends_on(..., reason=...) stores the reason."""

    @depends_on("a", "b", reason="Needs both complete")
    def my_node(state: dict) -> dict:
        return state

    constraints = my_node._optimization_constraints
    assert constraints.depends_on == {"a", "b"}
    assert constraints.reason == "Needs both complete"


# ---------------------------------------------------------------------------
# Decorators: speculation_unsafe
# ---------------------------------------------------------------------------


def test_speculation_unsafe_decorator_importable() -> None:
    """speculation_unsafe is importable from constraints."""
    assert callable(speculation_unsafe)


def test_speculation_unsafe_decorator_marks_function() -> None:
    """@speculation_unsafe marks function as unsafe for speculation."""

    @speculation_unsafe
    def side_effect_node(state: dict) -> dict:
        return state

    assert getattr(side_effect_node, "_speculation_unsafe", False) is True


def test_speculation_unsafe_decorator_marks_class() -> None:
    """@speculation_unsafe marks class as unsafe for speculation."""

    @speculation_unsafe
    class HumanApprovalMiddleware:
        def after_model(self, state: dict, runtime: object) -> None:
            pass

    assert getattr(HumanApprovalMiddleware, "_speculation_unsafe", False) is True


def test_speculation_unsafe_preserves_callable() -> None:
    """@speculation_unsafe preserves the wrapped function."""

    @speculation_unsafe
    def my_node(state: dict) -> dict:
        return {"done": True}

    result = my_node({})
    assert result == {"done": True}
