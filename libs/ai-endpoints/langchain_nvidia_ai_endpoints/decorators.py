"""Inference-priority decorator for LangChain chat models.

Provides a decorator / context manager that sets ``priority`` for every
:class:`~langchain_core.language_models.BaseChatModel` call in scope.

The mechanism is **universal**: any ``BaseChatModel`` subclass whose Pydantic
``model_fields`` include ``priority`` will automatically receive the value as
a keyword argument — no per-model integration required.

Lower number = higher priority (``priority=1`` is most urgent).

Example — decorator (deprioritize background work)::

    from langchain_nvidia_ai_endpoints import ChatNVIDIADynamo, inference_priority

    llm = ChatNVIDIADynamo(model="my-model", base_url="http://localhost:8099/v1")

    @inference_priority(priority=10)
    def background_research(query: str) -> str:
        return llm.invoke(query).content

Example — context manager::

    with inference_priority(priority=10):
        result = llm.invoke("background task")
"""

from __future__ import annotations

import functools
import inspect
from contextvars import ContextVar
from typing import Any, Callable, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# ── context variable ──────────────────────────────────────────────────────────

_inference_priority_var: ContextVar[Optional[int]] = ContextVar(
    "inference_priority", default=None
)


def get_inference_priority() -> Optional[int]:
    """Return the active inference priority, or *None* if unset."""
    return _inference_priority_var.get()


# ── lazy BaseChatModel patch ──────────────────────────────────────────────────

_patched = False


def _ensure_patched() -> None:
    """Apply a one-time patch to ``BaseChatModel`` so that *invoke*,
    *ainvoke*, *stream* and *astream* inject ``priority`` from context.

    The patch is a no-op when:
    * the context variable is unset (``None``), **or**
    * the LLM class does not declare ``priority`` in its Pydantic
      ``model_fields``.
    """
    global _patched
    if _patched:
        return
    _patched = True

    from langchain_core.language_models import BaseChatModel

    def _inject_priority(self: Any, kwargs: dict) -> None:  # type: ignore[type-arg]
        if "priority" in kwargs:
            return  # explicit kwarg always wins
        ctx_priority = _inference_priority_var.get()
        if ctx_priority is None:
            return
        if "priority" in getattr(type(self), "model_fields", {}):
            kwargs["priority"] = ctx_priority

    _orig_invoke = BaseChatModel.invoke
    _orig_ainvoke = BaseChatModel.ainvoke
    _orig_stream = BaseChatModel.stream
    _orig_astream = BaseChatModel.astream

    @functools.wraps(_orig_invoke)
    def _patched_invoke(
        self: Any,
        input: Any,
        config: Any = None,
        *,
        stop: Any = None,
        **kwargs: Any,
    ) -> Any:
        _inject_priority(self, kwargs)
        return _orig_invoke(self, input, config, stop=stop, **kwargs)

    @functools.wraps(_orig_ainvoke)
    async def _patched_ainvoke(
        self: Any,
        input: Any,
        config: Any = None,
        *,
        stop: Any = None,
        **kwargs: Any,
    ) -> Any:
        _inject_priority(self, kwargs)
        return await _orig_ainvoke(self, input, config, stop=stop, **kwargs)

    @functools.wraps(_orig_stream)
    def _patched_stream(
        self: Any,
        input: Any,
        config: Any = None,
        *,
        stop: Any = None,
        **kwargs: Any,
    ) -> Any:
        _inject_priority(self, kwargs)
        yield from _orig_stream(self, input, config, stop=stop, **kwargs)

    @functools.wraps(_orig_astream)
    async def _patched_astream(
        self: Any,
        input: Any,
        config: Any = None,
        *,
        stop: Any = None,
        **kwargs: Any,
    ) -> Any:
        _inject_priority(self, kwargs)
        async for chunk in _orig_astream(self, input, config, stop=stop, **kwargs):
            yield chunk

    BaseChatModel.invoke = _patched_invoke  # type: ignore[assignment]
    BaseChatModel.ainvoke = _patched_ainvoke  # type: ignore[assignment]
    BaseChatModel.stream = _patched_stream  # type: ignore[assignment]
    BaseChatModel.astream = _patched_astream  # type: ignore[assignment]


# ── decorator / context manager ───────────────────────────────────────────────


class inference_priority:  # noqa: N801
    """Set inference priority for all LLM calls within scope.

    Lower number = higher priority (``priority=1`` is most urgent).

    Works as **both** a decorator and a context manager::

        # decorator — deprioritize background work
        @inference_priority(priority=10)
        def background_research(query):
            return llm.invoke(query)

        # context manager
        with inference_priority(priority=1):
            result = llm.invoke(query)

        # async decorator
        @inference_priority(priority=10)
        async def background_async(query):
            return await llm.ainvoke(query)

    **Precedence** (wins first → last):

    1. Active ``inference_priority`` context
    2. Instance default: ``ChatNVIDIADynamo(priority=1)``

    Nesting: inner scopes fully replace outer scopes.
    """

    def __init__(self, *, priority: int) -> None:
        if not isinstance(priority, int) or priority < 0:
            raise ValueError(f"priority must be a non-negative int, got {priority!r}")
        self._priority = priority
        self._token: Any = None
        _ensure_patched()

    # ── context manager ──

    def __enter__(self) -> int:
        self._token = _inference_priority_var.set(self._priority)
        return self._priority

    def __exit__(self, *exc: Any) -> None:
        _inference_priority_var.reset(self._token)

    # ── decorator ──

    def __call__(self, fn: F) -> F:
        priority = self._priority
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def _async_wrapper(*args: Any, **kwargs: Any) -> Any:
                token = _inference_priority_var.set(priority)
                try:
                    return await fn(*args, **kwargs)
                finally:
                    _inference_priority_var.reset(token)

            return _async_wrapper  # type: ignore[return-value]
        else:

            @functools.wraps(fn)
            def _sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                token = _inference_priority_var.set(priority)
                try:
                    return fn(*args, **kwargs)
                finally:
                    _inference_priority_var.reset(token)

            return _sync_wrapper  # type: ignore[return-value]
