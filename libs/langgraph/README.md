# langchain-nvidia-langgraph

LangGraph integration for NVIDIA AI optimization using a set of foundational agent performance primitives.

## Usage

### Quick Start

Use ``StateGraph`` as a drop-in replacement for LangGraph. Build your graph with the same API (``add_node``, ``add_edge``, ``add_conditional_edges``, ``set_entry_point``), then compile with optional optimization:

```python
from langchain_nvidia_langgraph.graph import StateGraph, OptimizationConfig

graph = StateGraph(ResearchState)
graph.add_node("parse_query", parse_query)
graph.add_edge("parse_query", "classify_intent")
# ... add nodes and edges as usual

# Baseline (vanilla LangGraph)
compiled = graph.compile()

# With parallel optimization
compiled = graph.compile(optimization=OptimizationConfig(enable_parallel=True))
```

### Optimization Modes

Compile the same graph in three modes: baseline (sequential), parallel, or parallel + speculative:

```python
# Baseline — no optimization
baseline_compiled = graph.compile()

# Parallel — independent stages parallelized (compile-time graph rewrite)
optimized_parallel = graph.compile(optimization=OptimizationConfig(enable_parallel=True))

# Speculative — parallel + speculative branch execution (runtime speculation)
optimized_speculative = graph.compile(
    optimization=OptimizationConfig(enable_parallel=True, enable_speculation=True),
)
```

See [Limitations](#limitations) for speculative execution constraints.

### OptimizationConfig

| Setting | Purpose |
|---------|---------|
| ``enable_parallel=True`` | Parallelize independent stages (fan-out/fan-in) |
| ``enable_speculation=True`` | Speculative execution at conditional branches |
| ``parallel_unsafe_nodes`` | Node names that must run sequentially |
| ``parallel_safe_overrides`` | Override ``@sequential`` for specific nodes |
| ``speculation_unsafe_nodes`` | Nodes where speculation is disabled |
| ``speculation_safe_overrides`` | Override ``@speculation_unsafe`` for specific nodes |
| ``explicit_dependencies`` | Declare dependencies when auto-analysis fails |

### Escape Hatch Decorators

When auto-analysis cannot infer dependencies or safety, use decorators to constrain optimization:

| Decorator | Purpose |
|-----------|---------|
| ``@sequential`` | Mark node as non-parallelizable (must run in order) |
| ``@depends_on("node_a", "node_b")`` | Declare explicit dependencies when auto-analysis cannot infer them |
| ``@speculation_unsafe`` | Mark node as unsafe for speculative execution |

```python
from langchain_nvidia_langgraph.graph import (
    StateGraph,
    OptimizationConfig,
    depends_on,
    sequential,
    speculation_unsafe,
)

@sequential
async def critical_node(state): ...

@depends_on("parse_query", "load_context")
async def merge_node(state): ...

@speculation_unsafe
async def side_effect_node(state): ...
```

### Existing Graphs

To add optimization to an existing ``StateGraph`` built with standard LangGraph, use ``with_app_compile``:

```python
from langchain_nvidia_langgraph.graph import with_app_compile, OptimizationConfig

graph = create_baseline_graph()  # returns standard StateGraph
compilable = with_app_compile(graph)
compiled = compilable.compile(optimization=OptimizationConfig(enable_parallel=True))
```


## Limitations

**Speculative execution** (``enable_speculation=True``) uses a custom execution strategy
and does not support:

- Checkpointer / state persistence
- Streaming (``stream``, ``astream``)
- Interrupts (``interrupt_before``, ``interrupt_after``)
- Human-in-the-loop

For full LangGraph compatibility, use ``enable_parallel=True`` only.

## LangGraph API Compatibility

This package uses the following LangGraph public API:

- **CompiledStateGraph.builder** — access to the StateGraph builder
- **StateGraph.state_schema** — state schema (TypedDict or similar)
- **StateGraph.branches** — conditional edge branches
- **StateGraph.edges** — graph edges
- **StateGraph.nodes** — graph nodes
- **PregelNode.bound** — node runnable (via compiled.nodes)
- **Runnable.get_graph()** — graph structure for topology

**Supported LangGraph versions:** `>=1.0.0,<2.0.0` (see pyproject.toml).
