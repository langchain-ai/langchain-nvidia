"""Unit tests for RouterEvaluator public interfaces."""

from __future__ import annotations

import pytest

from langchain_nvidia_langgraph.analysis.analyzer import GraphAnalysis, RouterInfo
from langchain_nvidia_langgraph.executors.speculative.router_evaluator import (
    InvalidRouterError,
    RouterEvaluationError,
    RouterEvaluator,
    SpeculativeExecutionError,
)

# ---------------------------------------------------------------------------
# Exception classes
# ---------------------------------------------------------------------------


def test_speculative_execution_error_is_exception() -> None:
    """SpeculativeExecutionError is an Exception subclass."""
    assert issubclass(SpeculativeExecutionError, Exception)


def test_router_evaluation_error_inherits_from_speculative() -> None:
    """RouterEvaluationError inherits from SpeculativeExecutionError."""
    assert issubclass(RouterEvaluationError, SpeculativeExecutionError)


def test_router_evaluation_error_stores_router_name_and_cause() -> None:
    """RouterEvaluationError stores router_name and cause."""
    cause = ValueError("test error")
    err = RouterEvaluationError("router1", cause)
    assert err.router_name == "router1"
    assert err.cause is cause
    assert "router1" in str(err)
    assert "test error" in str(err)


def test_invalid_router_error_inherits_from_speculative() -> None:
    """InvalidRouterError inherits from SpeculativeExecutionError."""
    assert issubclass(InvalidRouterError, SpeculativeExecutionError)


def test_invalid_router_error_stores_router_name_and_reason() -> None:
    """InvalidRouterError stores router_name and reason."""
    err = InvalidRouterError("r1", "no targets")
    assert err.router_name == "r1"
    assert err.reason == "no targets"
    assert "r1" in str(err)
    assert "no targets" in str(err)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def analysis_with_router() -> GraphAnalysis:
    """GraphAnalysis with one router."""
    return GraphAnalysis(
        routers=[
            RouterInfo(
                name="router1",
                possible_targets=["a", "b"],
                conditional_edge_fn=None,
                path_mapping={"x": "a", "y": "b"},
            ),
        ],
        entry_point="router1",
        has_cycles=False,
        back_edges=[],
    )


@pytest.fixture
def analysis_empty_routers() -> GraphAnalysis:
    """GraphAnalysis with no routers."""
    return GraphAnalysis(
        routers=[],
        entry_point="a",
        has_cycles=False,
        back_edges=[],
    )


@pytest.fixture
def evaluator(analysis_with_router: GraphAnalysis) -> RouterEvaluator:
    """RouterEvaluator with router in analysis."""
    return RouterEvaluator(analysis_with_router)


# ---------------------------------------------------------------------------
# RouterEvaluator - get_router_info
# ---------------------------------------------------------------------------


def test_get_router_info_returns_router_info(
    evaluator: RouterEvaluator,
) -> None:
    """get_router_info returns RouterInfo when router exists."""
    info = evaluator.get_router_info("router1")
    assert info is not None
    assert info.name == "router1"
    assert info.possible_targets == ["a", "b"]
    assert info.path_mapping == {"x": "a", "y": "b"}


def test_get_router_info_returns_none_for_unknown_router(
    evaluator: RouterEvaluator,
) -> None:
    """get_router_info returns None when router not in analysis."""
    info = evaluator.get_router_info("nonexistent")
    assert info is None


def test_get_router_info_returns_none_when_no_routers(
    analysis_empty_routers: GraphAnalysis,
) -> None:
    """get_router_info returns None when analysis has no routers."""
    evaluator = RouterEvaluator(analysis_empty_routers)
    assert evaluator.get_router_info("any") is None


# ---------------------------------------------------------------------------
# RouterEvaluator - evaluate_decision
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evaluate_decision_returns_first_target_when_no_conditional_fn(
    evaluator: RouterEvaluator,
) -> None:
    """evaluate_decision returns first possible_target when conditional_fn None."""
    router_info = RouterInfo(
        name="r1",
        possible_targets=["a", "b"],
        conditional_edge_fn=None,
        path_mapping={},
    )
    result = await evaluator.evaluate_decision("r1", router_info, {"state": "x"})
    assert result == "a"


@pytest.mark.asyncio
async def test_evaluate_decision_raises_when_no_possible_targets(
    evaluator: RouterEvaluator,
) -> None:
    """evaluate_decision raises InvalidRouterError when possible_targets is empty."""
    router_info = RouterInfo(
        name="r1",
        possible_targets=[],
        conditional_edge_fn=None,
        path_mapping={},
    )
    with pytest.raises(InvalidRouterError) as exc_info:
        await evaluator.evaluate_decision("r1", router_info, {})
    assert exc_info.value.router_name == "r1"
    assert "no possible targets" in exc_info.value.reason.lower()


@pytest.mark.asyncio
async def test_evaluate_decision_with_sync_callable(
    evaluator: RouterEvaluator,
) -> None:
    """evaluate_decision invokes sync callable and resolves via path_mapping."""

    def choose_x(state: dict) -> str:
        return "x"

    router_info = RouterInfo(
        name="r1",
        possible_targets=["a", "b"],
        conditional_edge_fn=choose_x,
        path_mapping={"x": "a", "y": "b"},
    )
    result = await evaluator.evaluate_decision("r1", router_info, {"value": ""})
    assert result == "a"


@pytest.mark.asyncio
async def test_evaluate_decision_with_async_callable(
    evaluator: RouterEvaluator,
) -> None:
    """evaluate_decision invokes async callable and resolves via path_mapping."""

    async def choose_y(state: dict) -> str:
        return "y"

    router_info = RouterInfo(
        name="r1",
        possible_targets=["a", "b"],
        conditional_edge_fn=choose_y,
        path_mapping={"x": "a", "y": "b"},
    )
    result = await evaluator.evaluate_decision("r1", router_info, {"value": ""})
    assert result == "b"


@pytest.mark.asyncio
async def test_evaluate_decision_returns_target_directly_when_in_possible_targets(
    evaluator: RouterEvaluator,
) -> None:
    """evaluate_decision returns edge name when it's in possible_targets."""

    def choose_a(state: dict) -> str:
        return "a"

    router_info = RouterInfo(
        name="r1",
        possible_targets=["a", "b"],
        conditional_edge_fn=choose_a,
        path_mapping={"x": "a", "y": "b"},
    )
    result = await evaluator.evaluate_decision("r1", router_info, {})
    assert result == "a"


@pytest.mark.asyncio
async def test_evaluate_decision_raises_when_edge_not_in_mapping(
    evaluator: RouterEvaluator,
) -> None:
    """evaluate_decision raises InvalidRouterError when edge not in mapping."""

    def choose_unknown(state: dict) -> str:
        return "unknown"

    router_info = RouterInfo(
        name="r1",
        possible_targets=["a", "b"],
        conditional_edge_fn=choose_unknown,
        path_mapping={"x": "a", "y": "b"},
    )
    with pytest.raises(InvalidRouterError) as exc_info:
        await evaluator.evaluate_decision("r1", router_info, {})
    assert exc_info.value.router_name == "r1"


@pytest.mark.asyncio
async def test_evaluate_decision_raises_when_callable_raises(
    evaluator: RouterEvaluator,
) -> None:
    """evaluate_decision raises RouterEvaluationError when conditional fn raises."""

    def failing_fn(state: dict) -> str:
        raise RuntimeError("router failed")

    router_info = RouterInfo(
        name="r1",
        possible_targets=["a", "b"],
        conditional_edge_fn=failing_fn,
        path_mapping={},
    )
    with pytest.raises(RouterEvaluationError) as exc_info:
        await evaluator.evaluate_decision("r1", router_info, {})
    assert exc_info.value.router_name == "r1"
    assert isinstance(exc_info.value.cause, RuntimeError)
    assert "router failed" in str(exc_info.value.cause)
