"""LangGraph-specific node execution.

Handles extraction of callable functions from LangGraph node objects,
``ainvoke`` wrapping with runnable config, and LangGraph ``Runtime``
injection.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import time
from collections.abc import Callable
from typing import Any

from langgraph.runtime import Runtime
from nat_app.executors import ExecutionState

logger = logging.getLogger(__name__)

# Mirrors langgraph._internal._constants.CONF / CONFIG_KEY_RUNTIME.
# Replace with public imports if LangGraph exposes them in a future release.
_CONF = "configurable"
_CONFIG_KEY_RUNTIME = "__pregel_runtime"


class LangGraphNodeExecutor:
    """Executes LangGraph nodes with proper ``ainvoke`` wrapping and state isolation."""

    def __init__(
        self,
        graph: Any,
        runnable_config: dict[str, Any] | None = None,
    ) -> None:
        self._graph = graph
        self._runnable_config = dict(runnable_config) if runnable_config else None
        if self._runnable_config and _CONF in self._runnable_config:
            self._runnable_config[_CONF] = dict(self._runnable_config[_CONF])
        self._inject_runtime()

    def _inject_runtime(self) -> None:
        if self._runnable_config is None:
            return
        store = getattr(self._graph, "store", None)
        runtime = Runtime(context=None, store=store, stream_writer=None, previous=None)  # type: ignore[arg-type]
        if _CONF not in self._runnable_config:
            self._runnable_config[_CONF] = {}
        self._runnable_config[_CONF][_CONFIG_KEY_RUNTIME] = runtime

    def extract_node_function(self, node: Any) -> Callable | None:
        """Extract an async callable from a LangGraph node object.

        Args:
            node: A LangGraph node (typically has ainvoke).

        Returns:
            An async wrapper that invokes node.ainvoke with runnable config,
            or None if the node has no ainvoke.
        """
        if hasattr(node, "ainvoke"):
            runnable_config = self._runnable_config

            async def wrapper(state: dict[str, Any]) -> Any:
                return await node.ainvoke(state, config=runnable_config)

            return wrapper
        return None

    async def safe_call(self, fn: Callable, state: dict[str, Any]) -> Any:
        """Call a node function, awaiting coroutines.

        Args:
            fn: The node function to call (may be sync or async).
            state: State dict to pass to the function.

        Returns:
            The result of the call (awaited if fn returns a coroutine).
        """
        result = fn(state)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    async def launch_node(
        self,
        node_name: str,
        execution_state: ExecutionState,
        current_state: dict[str, Any],
    ) -> None:
        """Launch a single node as an async task with state isolation.

        Args:
            node_name: Name of the node to launch.
            execution_state: ExecutionState to update with the new task.
            current_state: Current state dict (deep-copied before passing to node).
        """
        if (
            node_name in execution_state.running_tasks
            or node_name in execution_state.completed_nodes
        ):
            return
        node = self._graph.nodes.get(node_name)
        if not node:
            return

        node_fn = self.extract_node_function(node)
        if not node_fn:
            return

        deepcopy_start = time.perf_counter()
        isolated_state = copy.deepcopy(current_state)
        execution_state.deepcopy_times.append(time.perf_counter() - deepcopy_start)

        task_start = time.perf_counter()
        task = asyncio.create_task(self.safe_call(node_fn, isolated_state))
        execution_state.task_creation_times.append(time.perf_counter() - task_start)

        execution_state.running_tasks[node_name] = task
        execution_state.tools_launched += 1
        execution_state.node_start_times[node_name] = time.time()

    async def execute_router_node(
        self,
        router_name: str,
        current_state: dict[str, Any],
    ) -> asyncio.Task | None:
        """Execute a router node and return its task.

        Does not register the task in execution_state; caller handles that.

        Args:
            router_name: Name of the router node to execute.
            current_state: State dict (deep-copied before passing to node).

        Returns:
            The asyncio.Task for the router execution, or None if node not found.
        """
        router_node = self._graph.nodes.get(router_name)
        if router_node:
            node_fn = self.extract_node_function(router_node)
            if node_fn:
                isolated_state = copy.deepcopy(current_state)
                return asyncio.create_task(self.safe_call(node_fn, isolated_state))
        return None
