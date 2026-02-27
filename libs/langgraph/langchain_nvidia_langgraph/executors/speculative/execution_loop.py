"""Event-driven execution loop for speculative LangGraph execution.

Implements the ready-launch-await-process-update cycle with support for
multi-router concurrent evaluation, cyclic re-execution, contiguous
router chain expansion, and stale state invalidation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any

from langgraph.graph import END
from nat_app.executors import ExecutionState, ResultHandler
from nat_app.speculation.plan import SpeculationPlan
from nat_app.speculation.safety import SpeculationSafetyConfig

from ...analysis import DependencyTracker, GraphAnalysis, RouterInfo
from .channel_state_manager import ChannelStateManager
from .node_executor import LangGraphNodeExecutor
from .router_evaluator import RouterEvaluator

logger = logging.getLogger(__name__)


class ExecutionLoop:
    """Event-driven speculative execution loop.

    Coordinates node launching, router evaluation, task completion,
    dependency-driven readiness, and unchosen-path cancellation.
    """

    def __init__(
        self,
        analysis: GraphAnalysis,
        plans: dict[str, SpeculationPlan],
        dep_tracker: DependencyTracker,
        state_manager: ChannelStateManager,
        node_executor: LangGraphNodeExecutor,
        router_evaluator: RouterEvaluator,
        result_handler: ResultHandler,
        speculation_safety: SpeculationSafetyConfig,
        node_rw: dict[str, dict[str, Any]] | None = None,
        graph: Any = None,
        max_iterations: int = 50,
    ) -> None:
        self._analysis = analysis
        self._plans = plans
        self._dep_tracker = dep_tracker
        self._state_manager = state_manager
        self._node_executor = node_executor
        self._router_evaluator = router_evaluator
        self._result_handler = result_handler
        self._speculation_safety = speculation_safety
        self._node_rw = node_rw
        self._graph = graph
        self._max_iterations = max_iterations
        self._router_names: frozenset[str] = frozenset(r.name for r in analysis.routers)

    # -- Main loop ----------------------------------------------------------

    async def run(
        self,
        execution_state: ExecutionState,
        current_state: dict[str, Any],
    ) -> int:
        """Run the execution loop until completion or max iterations."""
        iteration = 0

        while iteration < self._max_iterations:
            iteration += 1
            await self._launch_ready_nodes(execution_state, current_state, iteration)

            if self._should_terminate(execution_state):
                break

            if execution_state.running_tasks:
                done, _ = await asyncio.wait(
                    execution_state.running_tasks.values(),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                completion_time = time.time()
                task_end_times = {task: completion_time for task in done}
                recently_completed: set[str] = set()
                await self._process_completed_tasks(
                    done,
                    execution_state,
                    current_state,
                    recently_completed,
                    task_end_times,
                )
                self._update_ready_nodes(execution_state, recently_completed)

        if iteration >= self._max_iterations:
            logger.warning("Hit max iterations (%s)", self._max_iterations)
        return iteration

    def _should_terminate(self, execution_state: ExecutionState) -> bool:
        if not execution_state.ready_nodes and not execution_state.running_tasks:
            logger.info("No more nodes to execute - done!")
            return True
        if not execution_state.running_tasks and execution_state.ready_nodes:
            return False
        if not execution_state.running_tasks:
            logger.info("No running tasks and no ready nodes - execution complete")
            return True
        return False

    # -- Node launching -----------------------------------------------------

    async def _launch_ready_nodes(
        self,
        execution_state: ExecutionState,
        current_state: dict[str, Any],
        iteration: int,
    ) -> None:
        if not execution_state.ready_nodes:
            return
        nodes_to_launch = list(execution_state.ready_nodes)
        execution_state.ready_nodes.clear()

        routers = [n for n in nodes_to_launch if n in self._router_names]
        tools = [n for n in nodes_to_launch if n not in self._router_names]

        if routers:
            expanded_routers = self._expand_to_contiguous_chains(
                routers, execution_state
            )
            await self._launch_routers_parallel(
                expanded_routers,
                execution_state,
                current_state,
                iteration,
            )

        for node_name in tools:
            await self._node_executor.launch_node(
                node_name, execution_state, current_state
            )

    def _expand_to_contiguous_chains(
        self,
        initial_routers: list[str],
        execution_state: ExecutionState,
    ) -> list[str]:
        expanded = set(initial_routers)
        for router in initial_routers:
            current = router
            while current is not None:
                plan = self._plans.get(current)
                nxt = plan.chain_next if plan else None
                if (
                    nxt
                    and nxt not in expanded
                    and nxt not in execution_state.running_tasks
                    and nxt not in execution_state.completed_nodes
                ):
                    expanded.add(nxt)
                current = nxt
        return list(expanded)

    async def _launch_routers_parallel(
        self,
        router_names: list[str],
        execution_state: ExecutionState,
        current_state: dict[str, Any],
        iteration: int,
    ) -> None:
        router_tasks: list[tuple[str, asyncio.Task]] = []
        router_infos: dict[str, RouterInfo] = {}
        router_start_times: dict[str, float] = {}

        for router_name in router_names:
            router_info = self._router_evaluator.get_router_info(router_name)
            if not router_info:
                continue
            router_infos[router_name] = router_info
            router_start_times[router_name] = time.time()
            task = await self._node_executor.execute_router_node(
                router_name, current_state
            )
            if task:
                router_tasks.append((router_name, task))
                execution_state.tools_launched += 1

        await self._launch_speculative_tools(
            router_names,
            execution_state,
            current_state,
        )

        if router_tasks:
            await self._await_router_tasks(
                router_tasks,
                router_infos,
                router_start_times,
                execution_state,
                current_state,
                iteration,
            )
            self._cancel_unchosen_tasks(execution_state)

    async def _launch_speculative_tools(
        self,
        router_names: list[str],
        execution_state: ExecutionState,
        current_state: dict[str, Any],
    ) -> None:
        from .executor import _is_speculation_safe

        for router_name in router_names:
            if not _is_speculation_safe(router_name, self._speculation_safety):
                logger.info(
                    "Speculation blocked: '%s' is marked as unsafe.", router_name
                )
                return

        tools_to_launch: set[str] = set()
        for router_name in router_names:
            plan = self._plans.get(router_name)
            if plan:
                tools_to_launch |= plan.targets_to_launch

        for tool_name in tools_to_launch:
            if self._analysis.has_cycles:
                if tool_name in execution_state.completed_nodes:
                    execution_state.clear_for_reexecution(tool_name)
                if tool_name in execution_state.cancelled_nodes:
                    execution_state.cancelled_nodes.discard(tool_name)

            await self._node_executor.launch_node(
                tool_name, execution_state, current_state
            )

    async def _await_router_tasks(
        self,
        router_tasks: list[tuple[str, asyncio.Task]],
        router_infos: dict[str, RouterInfo],
        router_start_times: dict[str, float],
        execution_state: ExecutionState,
        current_state: dict[str, Any],
        iteration: int,
    ) -> None:
        task_to_router = {task: name for name, task in router_tasks}
        pending_router_tasks = {task for _, task in router_tasks}
        router_results: dict[str, Any] = {}
        router_end_times: dict[str, float] = {}

        task_to_tool = {
            task: name
            for name, task in execution_state.running_tasks.items()
            if name not in task_to_router.values()
        }
        all_pending = pending_router_tasks | set(task_to_tool.keys())

        while pending_router_tasks:
            done, all_pending = await asyncio.wait(
                all_pending, return_when=asyncio.FIRST_COMPLETED
            )
            completion_time = time.time()

            for task in done:
                if task in task_to_router:
                    router_name = task_to_router[task]
                    router_end_times[router_name] = completion_time
                    pending_router_tasks.discard(task)
                    try:
                        router_results[router_name] = await task
                    except Exception as e:
                        router_results[router_name] = e
                elif task in task_to_tool:
                    await self._process_interleaved_tool(
                        task,
                        task_to_tool,
                        execution_state,
                        current_state,
                        completion_time,
                        all_pending,
                    )

        routers_that_wrote_state: set[str] = set()
        for router_name, _ in router_tasks:
            start_time = router_start_times[router_name]
            end_time = router_end_times[router_name]
            result = router_results[router_name]

            execution_state.record_node_duration(router_name, end_time - start_time)
            execution_state.node_execution_count[router_name] += 1
            execution_state.record_timeline_event(router_name, start_time, end_time)

            if isinstance(result, Exception):
                logger.error("Router %s failed: %s", router_name, result)
            else:
                execution_state.mark_node_completed(router_name, result or {})
                if result:
                    updated = self._state_manager.merge_update(
                        execution_state.channels,
                        result,
                    )
                    current_state.update(updated)
                    routers_that_wrote_state.add(router_name)

        for router_name in [name for name, _ in router_tasks]:
            router_info = router_infos.get(router_name)
            if not router_info:
                continue
            chosen = await self._router_evaluator.evaluate_decision(
                router_name,
                router_info,
                current_state,
            )
            execution_state.record_decision(router_name, chosen, iteration)
            logger.info("  Router chose: '%s'", chosen)

            if router_name in routers_that_wrote_state:
                if self._has_stale_overlap(router_name, chosen):
                    self._invalidate_stale_speculation(chosen, execution_state)

            self._mark_chosen_target_ready(chosen, execution_state, router_name)

    async def _process_interleaved_tool(
        self,
        task: asyncio.Task,
        task_to_tool: dict[asyncio.Task, str],
        execution_state: ExecutionState,
        current_state: dict[str, Any],
        completion_time: float,
        all_pending: set[asyncio.Task],
    ) -> None:
        """Handle a speculative tool that finished while routers are pending."""
        tool_name = task_to_tool[task]
        try:
            tool_result = await task
            execution_state.running_tasks.pop(tool_name, None)
            execution_state.mark_node_completed(tool_name, tool_result or {})

            if tool_name in execution_state.node_start_times:
                start_t = execution_state.node_start_times.pop(tool_name)
                execution_state.record_node_duration(
                    tool_name, completion_time - start_t
                )
                execution_state.record_timeline_event(
                    tool_name, start_t, completion_time
                )
            execution_state.prerecorded_end_times.pop(tool_name, None)

            if tool_result:
                updated = self._state_manager.merge_update(
                    execution_state.channels,
                    tool_result,
                )
                current_state.update(updated)

            for succ in self._dep_tracker.successors.get(tool_name, []):
                if (
                    succ not in execution_state.running_tasks
                    and succ not in execution_state.completed_nodes
                    and succ != "__end__"
                    and self._immediate_preds_completed(succ, execution_state)
                ):
                    await self._node_executor.launch_node(
                        succ,
                        execution_state,
                        current_state,
                    )
                    if succ in execution_state.running_tasks:
                        new_task = execution_state.running_tasks[succ]
                        task_to_tool[new_task] = succ
                        all_pending.add(new_task)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(
                "Speculative tool '%s' failed during cascade: %s",
                tool_name,
                e,
                exc_info=True,
            )

    def _immediate_preds_completed(
        self,
        node_name: str,
        execution_state: ExecutionState,
    ) -> bool:
        preds = self._dep_tracker.predecessors.get(node_name, [])
        for pred in preds:
            if pred in ("__start__", "__end__"):
                continue
            if (pred, node_name) in self._dep_tracker.back_edges_set:
                continue
            if pred not in execution_state.completed_nodes:
                return False
        return True

    # -- Stale state management ---------------------------------------------

    def _invalidate_stale_speculation(
        self,
        chosen: str,
        execution_state: ExecutionState,
    ) -> None:
        if chosen in ("__end__", END):
            return
        if chosen in execution_state.running_tasks:
            task = execution_state.running_tasks[chosen]
            if not task.done():
                task.cancel()
            del execution_state.running_tasks[chosen]
            execution_state.tools_cancelled += 1
        if chosen in execution_state.completed_nodes:
            del execution_state.completed_nodes[chosen]
            execution_state.tools_completed -= 1
            execution_state.tools_cancelled += 1

    def _has_stale_overlap(self, router_name: str, target_name: str) -> bool:
        if self._node_rw is None:
            return True
        from nat_app.graph.access import AccessSet

        router_writes = self._node_rw.get(router_name, {}).get("writes")
        if not router_writes:
            return False
        if isinstance(router_writes, set):
            router_writes = AccessSet.from_set(router_writes)

        visited: set[str] = set()
        queue: deque[str] = deque([target_name])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            node_reads = self._node_rw.get(node, {}).get("reads")
            if node_reads:
                if isinstance(node_reads, set):
                    node_reads = AccessSet.from_set(node_reads)
                if router_writes.overlaps(node_reads):
                    return True
            for succ in self._dep_tracker.successors.get(node, []):
                if succ not in visited and succ != "__end__":
                    queue.append(succ)
        return False

    def _mark_chosen_target_ready(
        self,
        chosen: str,
        execution_state: ExecutionState,
        router_name: str | None = None,
    ) -> None:
        if chosen in ("__end__", END):
            return
        if chosen in execution_state.completed_nodes and not self._analysis.has_cycles:
            return

        if self._analysis.has_cycles:
            already_active = (
                chosen in execution_state.running_tasks
                or chosen in execution_state.completed_nodes
            )
            if (
                already_active
                and router_name
                and not self._has_stale_overlap(router_name, chosen)
            ):
                return
            execution_state.clear_for_reexecution(chosen)

        execution_state.mark_node_ready(chosen)

    # -- Task processing ----------------------------------------------------

    async def _process_completed_tasks(
        self,
        done_tasks: set[asyncio.Task],
        execution_state: ExecutionState,
        current_state: dict[str, Any],
        recently_completed: set[str],
        task_end_times: dict[asyncio.Task, float] | None = None,
    ) -> None:
        prerecorded = getattr(execution_state, "prerecorded_end_times", {})
        for task in done_tasks:
            node_name = self._find_task_node(task, execution_state)
            if not node_name:
                continue

            if node_name in prerecorded:
                end_time = prerecorded.pop(node_name)
            elif task_end_times and task in task_end_times:
                end_time = task_end_times[task]
            else:
                end_time = time.time()

            del execution_state.running_tasks[node_name]

            if task.cancelled():
                execution_state.mark_node_cancelled(node_name)
                continue

            try:
                result = await task
                await self._handle_task_result(
                    node_name,
                    result,
                    execution_state,
                    current_state,
                    recently_completed,
                    end_time,
                )
            except Exception as e:
                logger.error("'%s' failed: %s", node_name, e)
                raise

    @staticmethod
    def _find_task_node(
        task: asyncio.Task, execution_state: ExecutionState
    ) -> str | None:
        for name, t in execution_state.running_tasks.items():
            if t == task:
                return name
        return None

    async def _handle_task_result(
        self,
        node_name: str,
        result: Any,
        execution_state: ExecutionState,
        current_state: dict[str, Any],
        recently_completed: set[str],
        end_time: float | None = None,
    ) -> None:
        execution_state.mark_node_completed(node_name, result or {})
        recently_completed.add(node_name)

        should_merge, type_desc = self._result_handler.should_merge(result)
        self._result_handler.log_result(node_name, result, should_merge, type_desc)

        is_on_chosen = self._is_on_chosen_path(node_name, execution_state)
        self._record_node_timing(
            node_name,
            execution_state,
            status="completed" if is_on_chosen else "cancelled",
            end_time=end_time,
        )

        if should_merge and result:
            if is_on_chosen:
                merge_start = time.perf_counter()
                updated = self._state_manager.merge_update(
                    execution_state.channels, result
                )
                execution_state.state_merge_times.append(
                    time.perf_counter() - merge_start
                )
                current_state.update(updated)
            else:
                if node_name in execution_state.completed_nodes:
                    del execution_state.completed_nodes[node_name]
                recently_completed.discard(node_name)
                execution_state.tools_completed -= 1
                execution_state.tools_cancelled += 1

    def _is_on_chosen_path(
        self, node_name: str, execution_state: ExecutionState
    ) -> bool:
        is_cyclic = self._analysis.has_cycles

        back_edge_sources: frozenset[str] = (
            frozenset(src for src, tgt in self._analysis.back_edges if tgt == node_name)
            if is_cyclic
            else frozenset()
        )

        if back_edge_sources:
            for source in back_edge_sources:
                if execution_state.speculation_decisions.get(source) == node_name:
                    return True
                if source in execution_state.completed_nodes:
                    return True

        for router_info in self._analysis.routers:
            plan = self._plans.get(router_info.name)
            if not plan or node_name not in plan.targets_to_launch:
                continue
            if router_info.name in back_edge_sources:
                continue

            chosen_label = execution_state.speculation_decisions.get(router_info.name)
            if chosen_label is None:
                if is_cyclic:
                    continue
                return False
            if not plan.resolution.is_on_chosen_path(node_name, chosen_label):
                return False

        return True

    @staticmethod
    def _record_node_timing(
        node_name: str,
        execution_state: ExecutionState,
        status: str = "completed",
        end_time: float | None = None,
    ) -> None:
        if node_name not in execution_state.node_start_times:
            return
        if end_time is None:
            end_time = time.time()
        start_time = execution_state.node_start_times[node_name]
        duration = end_time - start_time
        execution_state.record_node_duration(node_name, duration)
        execution_state.record_timeline_event(
            node_name, start_time, end_time, status=status
        )
        del execution_state.node_start_times[node_name]

    # -- Dependency updates -------------------------------------------------

    def _update_ready_nodes(
        self,
        execution_state: ExecutionState,
        recently_completed: set[str],
    ) -> None:
        nodes_to_check = self._get_nodes_to_check(recently_completed)
        for node_name in nodes_to_check:
            if node_name in ("__start__", "__end__"):
                continue
            self._check_node_readiness(node_name, execution_state, recently_completed)
        self._cancel_unchosen_tasks(execution_state)

    def _get_nodes_to_check(self, recently_completed: set[str]) -> set[str]:
        if recently_completed:
            nodes: set[str] = set()
            for completed in recently_completed:
                if completed in self._dep_tracker.successors:
                    nodes.update(self._dep_tracker.successors[completed])
            return nodes
        return set(self._graph.get_graph().nodes) if self._graph else set()

    def _check_node_readiness(
        self,
        node_name: str,
        execution_state: ExecutionState,
        recently_completed: set[str],
    ) -> None:
        is_ready = self._dep_tracker.is_ready(
            node_name,
            execution_state.completed_nodes,
            execution_state.speculation_decisions,
            execution_state.cancelled_nodes,
            allow_reexecution=False,
        )
        if is_ready and node_name not in execution_state.running_tasks:
            execution_state.mark_node_ready(node_name)

        if self._analysis.has_cycles and node_name in execution_state.completed_nodes:
            self._handle_cyclic_reexecution(
                node_name, execution_state, recently_completed
            )

    def _handle_cyclic_reexecution(
        self,
        node_name: str,
        execution_state: ExecutionState,
        recently_completed: set[str],
    ) -> None:
        is_router = node_name in self._router_names
        if is_router:
            if self._should_reexecute_router(
                node_name, execution_state, recently_completed
            ):
                execution_state.clear_for_reexecution(node_name)
                execution_state.mark_node_ready(node_name)
        elif self._should_reexecute_node(node_name, execution_state):
            execution_state.clear_for_reexecution(node_name)
            execution_state.mark_node_ready(node_name)

    def _should_reexecute_router(
        self,
        router_name: str,
        execution_state: ExecutionState,
        recently_completed: set[str],
    ) -> bool:
        if (
            router_name in execution_state.running_tasks
            or router_name not in self._dep_tracker.dependencies
        ):
            return False
        deps = self._dep_tracker.dependencies[router_name]
        return any(upstream in recently_completed for upstream in deps.upstream_nodes)

    def _should_reexecute_node(
        self, node_name: str, execution_state: ExecutionState
    ) -> bool:
        if node_name in execution_state.running_tasks:
            return False
        exec_count = execution_state.node_execution_count.get(node_name, 0)
        cycle_routers = {source for source, _ in self._analysis.back_edges}

        for decision_node, chosen in execution_state.speculation_decisions.items():
            if decision_node not in cycle_routers:
                continue
            if chosen == node_name:
                decision_iter = execution_state.last_decision_iteration.get(
                    decision_node, 0
                )
                if decision_iter >= exec_count:
                    return True

        if node_name in self._dep_tracker.dependencies:
            deps = self._dep_tracker.dependencies[node_name]
            for upstream in deps.upstream_nodes:
                upstream_exec_count = execution_state.node_execution_count.get(
                    upstream, 0
                )
                if upstream_exec_count > exec_count:
                    return True
        return False

    def _cancel_unchosen_tasks(self, execution_state: ExecutionState) -> None:
        nodes_to_cancel: set[str] = set()
        for (
            decision_name,
            chosen_label,
        ) in execution_state.speculation_decisions.items():
            plan = self._plans.get(decision_name)
            if plan:
                nodes_to_cancel |= plan.resolution.get_cancel_set(chosen_label)

        for node_name, task in list(execution_state.running_tasks.items()):
            should_cancel = (
                node_name in nodes_to_cancel
                or not self._dep_tracker.is_ready(
                    node_name,
                    execution_state.completed_nodes,
                    execution_state.speculation_decisions,
                    execution_state.cancelled_nodes,
                    allow_reexecution=self._analysis.has_cycles,
                )
            )
            if should_cancel:
                if not task.done():
                    task.cancel()
                    if node_name in execution_state.node_start_times:
                        start_time = execution_state.node_start_times[node_name]
                        end_time = time.time()
                        execution_state.record_timeline_event(
                            node_name,
                            start_time,
                            end_time,
                            status="cancelled",
                        )
                        del execution_state.node_start_times[node_name]

                del execution_state.running_tasks[node_name]
                if not self._analysis.has_cycles:
                    execution_state.mark_node_cancelled(node_name)
                else:
                    execution_state.tools_cancelled += 1
                logger.info("  Cancelling '%s' (unchosen path)", node_name)
