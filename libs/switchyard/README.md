# Switchyard routing middleware for LangChain and Deep Agents

This experimental package lets a [LangChain Deep Agent](https://docs.langchain.com/oss/python/deepagents/overview)
route each model call through Switchyard's published `libsy` Python bindings. You configure ordinary
LangChain chat models, wrap them as Switchyard targets, choose a routing algorithm, and add one
middleware object to `create_deep_agent`.

The package is intentionally small:

- `LangChainLlmClient` adapts any LangChain `BaseChatModel` to the client interface expected by
  `switchyard.libsy.LlmTarget`.
- `SwitchyardRoutingMiddleware` replaces Deep Agents' model with a buffered model backed by any
  preconstructed `switchyard.libsy.Algorithm`.
- Algorithm construction remains in `nemo-switchyard`; this package does not copy or reinterpret
  Switchyard configuration.

The included paid example uses OpenRouter and Stage routing. A simple provider-native structured
output turn goes to an efficient model, while a turn containing a critical failed-tool result goes
to a capable model.

> [!WARNING]
> This integration is experimental. Its APIs and behavior are subject to breaking changes without
> notice. The `langchain-nvidia-switchyard` distribution is not yet a stable release and currently
> supports buffered text, reasoning, tool calling, and structured output. See
> [Limitations](#limitations) before using it in an application.

## The mental model

Deep Agents still owns the agent loop, tools, state, and middleware composition. Switchyard owns
the routing decision and calls whichever target the selected `libsy` algorithm requests.

```text
Deep Agent model turn
    -> Deep Agents and LangChain middleware
    -> SwitchyardRoutingMiddleware
    -> internal buffered Switchyard chat model
    -> configured libsy Algorithm
    -> selected LangChainLlmClient target
    -> target LangChain BaseChatModel
    -> AIMessage with Switchyard decision metadata
```

The middleware follows LangChain's dynamic-model pattern: it calls the existing handler with
`request.override(model=...)`. It does not short-circuit the handler, so inner middleware, tool
binding, callbacks, and LangChain tracing continue to run normally.

## Prerequisites

- Python 3.12 or newer.
- A source checkout of [NVIDIA-NeMo/Switchyard](https://github.com/NVIDIA-NeMo/Switchyard).
- A checkout of this `langchain-nvidia` repository.
- [Poetry](https://python-poetry.org/) 2.3 or newer for development commands.
- For the paid example only:
  - an [OpenRouter account and API key](https://openrouter.ai/keys);
  - account access and sufficient credits for both configured models;
  - explicit acknowledgment of spend through `SWITCHYARD_LANGCHAIN_E2E=1`.

The default OpenRouter targets are:

| Switchyard target | OpenRouter model |
|---|---|
| `efficient` | `nvidia/nemotron-3-ultra-550b-a55b` |
| `capable` | `anthropic/claude-sonnet-4.6` |

Both IDs can be overridden in `.env`. Model availability and pricing can change, so check the
[OpenRouter model catalog](https://openrouter.ai/models) before a paid run.

## Install from source

Switchyard's `libsy` Python bindings currently need to be built from the Switchyard source tree.
Install that checkout first, then install this integration:

```bash
git clone https://github.com/NVIDIA-NeMo/Switchyard.git
python -m pip install -e ./Switchyard

git clone https://github.com/langchain-ai/langchain-nvidia.git
python -m pip install -e "./langchain-nvidia/libs/switchyard[openrouter]"
```

The published metadata declares a normal `nemo-switchyard>=0.2.0` dependency. Installing the
Switchyard checkout first satisfies that dependency while making the source-built `libsy` bindings
available to `langchain-nvidia-switchyard`.

For an application that only needs the generic LangChain adapter and supplies its own provider
integration:

```bash
python -m pip install -e ./langchain-nvidia/libs/switchyard
```

To include Deep Agents:

```bash
python -m pip install -e "./langchain-nvidia/libs/switchyard[deepagents]"
```

To include Deep Agents, OpenRouter, and `.env` loading:

```bash
python -m pip install -e "./langchain-nvidia/libs/switchyard[openrouter]"
```

The generic runtime depends on `nemo-switchyard>=0.2.0` and LangChain 1.x. Provider packages are
extras so applications do not acquire OpenRouter or Deep Agents unless they ask for them.

For development, install the locked tool groups from the package directory. Its test group resolves
`nemo-switchyard` from the canonical GitHub source so contributors can build the bindings without
changing the distribution metadata:

```bash
cd langchain-nvidia/libs/switchyard
poetry install --with test,lint,typing
```

## Configure OpenRouter

The runnable example deliberately reads the `langchain-nvidia` repository-root `.env`, not an
environment file inside the package directory. From the repository root, create it only if you do
not already have one:

```bash
cp libs/switchyard/.env.example .env
```

Edit `.env`:

```dotenv
OPENROUTER_API_KEY=sk-or-v1-your-key
OPENROUTER_EFFICIENT_MODEL=nvidia/nemotron-3-ultra-550b-a55b
OPENROUTER_CAPABLE_MODEL=anthropic/claude-sonnet-4.6
```

Existing process environment variables take precedence over `.env`. The key is gitignored by the
repository and must never be committed. The example never prints it.

## Run the paid end-to-end example

The spend opt-in is required even when the key is present:

```bash
cd libs/switchyard
SWITCHYARD_LANGCHAIN_E2E=1 poetry run python example.py
```

The script makes two buffered Deep Agent model calls:

1. A simple request asks for provider-native structured output. It has no tool-failure signal, so
   Stage routing selects `efficient`, forwards the JSON schema, and returns a validated Pydantic
   result.
2. A synthetic prior `read_file` call returns a critical out-of-memory error, so Stage routing
   selects `capable`.

Expected output has this shape:

```text
simple: selected=efficient
{"message": "<short generated confirmation>", "status": "ok"}
failed-tool: selected=capable
<non-empty response from the capable model>
```

The responses are generated by paid APIs and therefore are not byte-for-byte deterministic. The
routing decisions are deterministic for the supplied histories.

## Minimal Deep Agent setup

The example below is the essential integration. The order of arguments to `stage_router` matters:
the capable target is first and the efficient target is second.

```python
from deepagents import create_deep_agent
from langchain_openrouter import ChatOpenRouter

from switchyard.libsy import LlmTarget, algorithms
from langchain_nvidia_switchyard import LangChainLlmClient, SwitchyardRoutingMiddleware

efficient_model = ChatOpenRouter(model="nvidia/nemotron-3-ultra-550b-a55b")
capable_model = ChatOpenRouter(model="anthropic/claude-sonnet-4.6")

router = algorithms.stage_router(
    LlmTarget("capable", LangChainLlmClient(capable_model)),
    LlmTarget("efficient", LangChainLlmClient(efficient_model)),
    picker="efficient_first",
    confidence_threshold=0.5,
    recent_window=3,
)

agent = create_deep_agent(
    # Deep Agents requires a base model. The middleware replaces it for each routed call,
    # so reusing a configured target avoids constructing an unused third model.
    model=efficient_model,
    middleware=[SwitchyardRoutingMiddleware(router)],
)

result = await agent.ainvoke({
    "messages": [
        {
            "role": "user",
            "content": "Summarize the important files in this project.",
        }
    ]
})
```

`LangChainLlmClient` calls the target's asynchronous `ainvoke` method. It does not own the target
model or its HTTP transport, so normal LangChain construction, callbacks, rate limiters, and model
configuration continue to apply.

## Structured output

Structured output is returned through the normal Deep Agent state key:

```python
structured = result["structured_response"]
```

The integration supports both LangChain strategies. Choose based on the capabilities shared by
all targets that your router may select.

### Portable tool strategy

Pass a Pydantic model, dataclass, TypedDict, or JSON Schema directly to `response_format`:

```python
from pydantic import BaseModel


class ContactInfo(BaseModel):
    name: str
    email: str


agent = create_deep_agent(
    model=efficient_model,
    middleware=[SwitchyardRoutingMiddleware(router)],
    response_format=ContactInfo,
)

result = await agent.ainvoke({
    "messages": [{"role": "user", "content": "Return Ada Lovelace's contact details."}]
})
contact: ContactInfo = result["structured_response"]
```

For a raw schema, LangChain selects its portable `ToolStrategy`. The response schema becomes a
tool definition, Switchyard preserves it while routing, and LangChain validates the selected
target's tool call. This is the safest default for a router whose targets have different
provider-native capabilities.

### Provider-native strategy

Opt in explicitly when every possible routed target accepts OpenAI-compatible provider-native
structured output:

```python
from langchain.agents.structured_output import ProviderStrategy

agent = create_deep_agent(
    model=efficient_model,
    middleware=[SwitchyardRoutingMiddleware(router)],
    response_format=ProviderStrategy(ContactInfo),
)

result = await agent.ainvoke({
    "messages": [{"role": "user", "content": "Return Ada Lovelace's contact details."}]
})
contact: ContactInfo = result["structured_response"]
```

The middleware copies LangChain's `response_format` JSON schema into libsy's neutral request. The
selected `LangChainLlmClient` forwards it to its target model, and LangChain parses and validates
the returned JSON. The routed model deliberately does not advertise provider-native capability,
because an opaque algorithm may select heterogeneous targets. Consequently, raw schemas use the
portable tool strategy and provider-native output always requires explicit `ProviderStrategy`.

## Use a different routing algorithm

`SwitchyardRoutingMiddleware` accepts an opaque `switchyard.libsy.Algorithm`. It does not branch on
the algorithm's concrete type. Any algorithm exposed by the installed `nemo-switchyard` Python
bindings can be supplied.

### Random routing

```python
router = algorithms.random(
    [
        LlmTarget("efficient", LangChainLlmClient(efficient_model)),
        LlmTarget("capable", LangChainLlmClient(capable_model)),
    ],
    weights=[3, 1],
    seed=42,
)

middleware = SwitchyardRoutingMiddleware(router)
```

Weights are relative. A seed makes the process-local selection sequence reproducible; it does not
make model output deterministic.

### LLM task classifier

The task classifier uses a judge model before selecting the efficient or capable target. Its
structured classifier schema is forwarded through `LangChainLlmClient` to the judge model.

```python
from switchyard.libsy import TaskClassifierConfig

judge_model = ChatOpenRouter(model="openai/gpt-5-mini", max_tokens=256)

router = algorithms.llm_task_classifier(
    LlmTarget("judge", LangChainLlmClient(judge_model)),
    LlmTarget("efficient", LangChainLlmClient(efficient_model)),
    LlmTarget("capable", LangChainLlmClient(capable_model)),
    config=TaskClassifierConfig(
        0.5,
        threshold_step=0.1,
        recent_turn_window=3,
    ),
)

middleware = SwitchyardRoutingMiddleware(router)
```

This route normally spends one judge call plus one selected target call per model turn. Session
affinity and message-hash fallback remain Switchyard configuration; the middleware does not add
its own sticky routing.

### Stage routing

Stage routing is signal-driven and does not require a judge by default. It examines actual message
history, including assistant tool calls and tool results. This package preserves call IDs,
arguments, result text, and LangChain's `ToolMessage.status` error flag across the neutral boundary.

```python
router = algorithms.stage_router(
    LlmTarget("capable", LangChainLlmClient(capable_model)),
    LlmTarget("efficient", LangChainLlmClient(efficient_model)),
    picker="efficient_first",  # or "capable_first"
    confidence_threshold=0.5,
    recent_window=3,
)
```

### No-op reference algorithm

`algorithms.noop()` requires no target and returns libsy's reference `OK` response. It is useful for
checking the middleware boundary without a provider call:

```python
middleware = SwitchyardRoutingMiddleware(algorithms.noop())
```

## Inspect routing decisions

Every routed `AIMessage` contains the complete ordered decision trace:

```python
from langchain_core.messages import AIMessage

message = next(
    message
    for message in reversed(result["messages"])
    if isinstance(message, AIMessage)
)

routing = message.response_metadata["switchyard"]
print(routing["selected_model"])
print(routing["decisions"])
```

The metadata shape is:

```python
{
    "decisions": [
        {
            "selected_model": "efficient",
            "reasoning": "...",
        }
    ],
    "selected_model": "efficient",  # convenience copy of the final decision
}
```

Some algorithms make more than one decision. Always inspect `decisions` when you need the full
trace; `selected_model` is only the final selection.

## Async, sync, and streaming

Asynchronous invocation is the canonical path:

```python
result = await agent.ainvoke({"messages": [{"role": "user", "content": "Hello"}]})
```

Ordinary synchronous applications may use:

```python
result = agent.invoke({"messages": [{"role": "user", "content": "Hello"}]})
```

The synchronous adapter runs the same async libsy algorithm on a persistent `asyncio.Runner`, so
async model clients remain bound to one event loop across multiple turns. Calling `agent.invoke`
from an already running event loop—such as a Jupyter notebook, async web handler, or async test—
raises a clear error. Use `await agent.ainvoke(...)` there.

Token-by-token model streaming is not supported because the published Python binding currently
exposes buffered `Algorithm.run`. LangGraph's `agent.astream(...)` can still emit graph or node
events, but each routed model response arrives as one buffered result.

## Deep Agent tools and subagents

Deep Agents binds its filesystem and subagent tools after the middleware replaces the model. The
internal routed model converts those tools into libsy's neutral tool definitions, and the selected
`LangChainLlmClient` binds them to its target model. Tool calls and tool results then round-trip in
subsequent turns, allowing Stage routing to react to actual agent progress and failures.

The top-level `middleware=[...]` list applies to the main Deep Agent. Independently configured
subagents do not automatically acquire this routing middleware. Add it to each declarative
subagent's middleware list when that subagent must route too:

```python
subagents = [
    {
        "name": "researcher",
        "description": "Research a focused question.",
        "system_prompt": "Research carefully and return a concise result.",
        "model": efficient_model,
        "tools": [],
        "middleware": [SwitchyardRoutingMiddleware(router)],
    }
]

agent = create_deep_agent(
    model=efficient_model,
    middleware=[SwitchyardRoutingMiddleware(router)],
    subagents=subagents,
)
```

For concurrent agents, decide deliberately whether to share one algorithm instance or construct
one per agent. Stateful algorithms retain their own process-local routing state.

## Supported request and response data

The current adapter supports:

- system and developer instructions;
- human text messages;
- assistant text, reasoning, and tool calls;
- text tool results with success/error status;
- tool definitions and `auto`, `any`/`required`, `none`, or named tool choice;
- temperature, top-p, maximum output tokens, stop sequences, and reasoning effort;
- portable tool-based and explicit provider-native structured output;
- libsy classifier response schemas sent to judge targets;
- response text, reasoning, tool calls, IDs, normalized finish reasons, and common token usage;
- complete Switchyard decision traces.

Unsupported content is rejected with a path-specific `ValueError` before a target model is called.

## Limitations

- Buffered responses only; no token streaming through libsy's current Python API.
- Text, reasoning, and tool workflows only. Images, audio, video, files, refusals, and
  provider-unknown blocks are rejected rather than silently dropped.
- Provider-specific response metadata without a neutral Switchyard equivalent is not retained.
- The middleware does not add retries or fallback. Target failures propagate through libsy as
  `switchyard.libsy.LibsyError`; use an algorithm or model configuration that implements the
  fallback policy you want.
- Routing applies only where the middleware is installed. Separately constructed and remote
  subagents must be configured independently.

## Run tests without spending money

The package defaults exclude the `e2e` marker, so this command never calls OpenRouter:

```bash
cd libs/switchyard
poetry run pytest tests/unit_tests -v
```

Run the strict package checks:

```bash
poetry check --lock
poetry run mypy langchain_nvidia_switchyard
make lint_package
make lint_tests
poetry build
```

## Run the paid E2E test

The paid pytest invokes both default models again. It requires the same explicit spend opt-in:

```bash
cd libs/switchyard
SWITCHYARD_LANGCHAIN_E2E=1 poetry run pytest \
  tests/unit_tests/test_e2e.py -v -m e2e -o addopts=
```

The test skips before model construction if either the opt-in or key is absent. When enabled, it
asserts both responses are non-empty and that routing metadata reports `efficient` then `capable`.

## Troubleshooting

### `SWITCHYARD_LANGCHAIN_E2E=1` is required

The paid example is fail-closed. Add the variable to the command line exactly as shown; placing a
key in `.env` alone does not authorize spend.

### `OPENROUTER_API_KEY` is required

Confirm the key is non-empty in the repository-root `.env`, or export it in the current shell.
Do not put `.env` inside `libs/switchyard`; the example intentionally uses the repository root.

### A model returns an authorization or availability error

Confirm your OpenRouter account can access both model IDs. Override either model in `.env` with a
tool-capable OpenRouter model available to your account. A direct provider failure is different
from a Switchyard routing failure.

### The capable target is selected unexpectedly

Inspect `response_metadata["switchyard"]["decisions"]`. For Stage routing, also inspect the recent
assistant tool calls and tool result text. Critical phrases such as `out of memory` intentionally
escalate. For `llm_task_classifier`, an invalid or unavailable judge fails safely to the configured
capable target.

### Stage routing does not react to a failed tool

Use a real `AIMessage.tool_calls` entry followed by a matching `ToolMessage`. Preserve the same
tool-call ID and set `status="error"`. Putting failure prose in an ordinary user message is not the
same signal.

### `synchronous Switchyard routing cannot run inside an active event loop`

Replace `agent.invoke(...)` with `await agent.ainvoke(...)`. Do not start a second event loop from a
notebook or async server handler.

### A target error appears as `LibsyError`

The Python binding wraps exceptions raised by `LangChainLlmClient.call` at the native boundary.
Read the chained error message for the original provider or validation failure. The adapter does
not hide it or retry independently.

## Package layout

```text
libs/switchyard/
├── .env.example                   # names and non-secret defaults
├── Makefile                       # package validation commands
├── README.md                      # this guide
├── example.py                     # paid two-route OpenRouter demo
├── poetry.lock                    # reproducible development environment
├── pyproject.toml                 # install metadata and test config
├── langchain_nvidia_switchyard/
│   ├── __init__.py                # two public exports
│   ├── client.py                  # LangChainLlmClient
│   ├── conversion.py              # neutral request/response translation
│   ├── middleware.py              # SwitchyardRoutingMiddleware
│   └── routed_chat_model.py       # internal buffered BaseChatModel
└── tests/
    ├── integration_tests/
    └── unit_tests/
        ├── test_algorithms.py     # every currently bound algorithm + Deep Agent
        ├── test_client.py
        ├── test_conversion.py
        ├── test_e2e.py            # safety checks and paid E2E
        ├── test_middleware.py
        └── test_packaging.py
```

## API reference

### `LangChainLlmClient(model)`

Wrap a LangChain `BaseChatModel` for use as an `LlmTarget` client.

```python
target = LlmTarget("efficient", LangChainLlmClient(efficient_model))
```

The public async method is the structural libsy client contract:

```python
await client.call(neutral_request)
```

Most applications do not call it directly; `Algorithm.run` does.

### `SwitchyardRoutingMiddleware(algorithm)`

Wrap an already configured Python-bound `Algorithm` and pass the object to Deep Agents:

```python
middleware = SwitchyardRoutingMiddleware(router)
agent = create_deep_agent(model=base_model, middleware=[middleware])
```

The middleware implements both `wrap_model_call` and `awrap_model_call`. It delegates exactly once
to LangChain's handler with the routed model override.

## Further reading

- [NVIDIA-NeMo/Switchyard](https://github.com/NVIDIA-NeMo/Switchyard)
- [LangChain custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom)
- [Deep Agents models and dynamic selection](https://docs.langchain.com/oss/python/deepagents/models)
- [Deep Agents customization](https://docs.langchain.com/oss/python/deepagents/customization)
- [LangChain ChatOpenRouter integration](https://docs.langchain.com/oss/python/integrations/chat/openrouter)
