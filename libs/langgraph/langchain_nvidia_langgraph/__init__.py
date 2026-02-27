"""LangChain NVIDIA LangGraph integration.

Import from submodules (LangGraph-style)::

    from langchain_nvidia_langgraph.graph import (
        StateGraph,
        CompilableGraph,
        with_app_compile,
        OptimizationConfig,
        depends_on,
        sequential,
        speculation_unsafe,
    )
    from langchain_nvidia_langgraph.compile import compile_langgraph, transform_graph
"""

import warnings as _warnings


class ExperimentalWarning(UserWarning):
    """Issued once when importing an experimental langchain-nvidia-langgraph package."""


_warnings.warn(
    "The langchain-nvidia-langgraph package is experimental and the API may "
    "change in future releases. Future versions may introduce breaking changes "
    "without notice.",
    ExperimentalWarning,
    stacklevel=2,
)
