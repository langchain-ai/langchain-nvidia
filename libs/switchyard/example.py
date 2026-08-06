#!/usr/bin/env python3
"""Run two paid Deep Agent turns through Switchyard Stage routing and OpenRouter."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openrouter import ChatOpenRouter
from pydantic import BaseModel, Field
from switchyard.libsy import LlmTarget, algorithms

from langchain_nvidia_switchyard import LangChainLlmClient, SwitchyardRoutingMiddleware

EFFICIENT_MODEL_DEFAULT = "nvidia/nemotron-3-ultra-550b-a55b"
CAPABLE_MODEL_DEFAULT = "anthropic/claude-sonnet-4.6"


class RoutingCheck(BaseModel):
    """Validated provider-native result returned by the structured routing check."""

    status: Literal["ok"] = Field(description="Whether the routing check succeeded.")
    message: str = Field(description="A short confirmation of the routing check.")


@dataclass(frozen=True)
class DemoResult:
    """One routed Deep Agent result displayed by the example."""

    case: str
    selected_model: str
    text: str
    structured_response: RoutingCheck | None = None


def repository_env_path() -> Path:
    """Return the repository-root environment file used by this example."""
    return Path(__file__).resolve().parents[2] / ".env"


def load_repository_environment() -> None:
    """Load repository credentials without overriding caller-provided values."""
    load_dotenv(repository_env_path(), override=False)


def _require_paid_environment() -> None:
    if os.environ.get("SWITCHYARD_LANGCHAIN_E2E") != "1":
        raise RuntimeError(
            "set SWITCHYARD_LANGCHAIN_E2E=1 to acknowledge that this example makes paid calls"
        )
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required in the repository .env or environment")


def _create_models() -> tuple[ChatOpenRouter, ChatOpenRouter]:
    efficient_name = os.environ.get("OPENROUTER_EFFICIENT_MODEL", EFFICIENT_MODEL_DEFAULT)
    capable_name = os.environ.get("OPENROUTER_CAPABLE_MODEL", CAPABLE_MODEL_DEFAULT)
    return (
        ChatOpenRouter(model=efficient_name, max_tokens=256),
        ChatOpenRouter(model=capable_name, max_tokens=256),
    )


def _agent(
    efficient_model: ChatOpenRouter,
    capable_model: ChatOpenRouter,
    *,
    response_format: ProviderStrategy[RoutingCheck] | None = None,
) -> Any:
    router = algorithms.stage_router(
        LlmTarget("capable", LangChainLlmClient(capable_model)),
        LlmTarget("efficient", LangChainLlmClient(efficient_model)),
        picker="efficient_first",
        confidence_threshold=0.5,
        recent_window=3,
    )
    return create_deep_agent(
        model=efficient_model,
        middleware=[SwitchyardRoutingMiddleware(router)],
        response_format=response_format,
    )


def _last_ai_message(result: dict[str, object]) -> AIMessage:
    messages = result.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Deep Agent result has no message list")
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    raise ValueError("Deep Agent result has no AIMessage")


def _demo_result(case: str, result: dict[str, object]) -> DemoResult:
    message = _last_ai_message(result)
    switchyard = message.response_metadata.get("switchyard")
    if not isinstance(switchyard, dict):
        raise ValueError("Deep Agent response has no Switchyard routing metadata")
    selected = switchyard.get("selected_model")
    if not isinstance(selected, str):
        raise ValueError("Deep Agent response has no selected Switchyard model")
    if not message.text.strip():
        raise ValueError("Deep Agent response has no text")
    structured = result.get("structured_response")
    if structured is not None and not isinstance(structured, RoutingCheck):
        raise ValueError("Deep Agent returned an unexpected structured response")
    return DemoResult(
        case=case,
        selected_model=selected,
        text=message.text,
        structured_response=structured,
    )


async def run_demo() -> list[DemoResult]:
    """Run the deterministic efficient and capable routing demonstrations."""
    load_repository_environment()
    _require_paid_environment()
    efficient_model, capable_model = _create_models()
    structured_agent = _agent(
        efficient_model,
        capable_model,
        response_format=ProviderStrategy(RoutingCheck),
    )
    agent = _agent(efficient_model, capable_model)

    simple = await structured_agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    "Return status 'ok' and one short message confirming the simple "
                    "structured routing check. Do not call any tools."
                )
            ]
        }
    )
    failed_tool = await agent.ainvoke(
        {
            "messages": [
                HumanMessage("Help recover from the failed test inspection."),
                AIMessage(
                    "",
                    tool_calls=[
                        {
                            "id": "call-stage-e2e",
                            "name": "read_file",
                            "args": {"file_path": "tests/failing_test.py"},
                            "type": "tool_call",
                        }
                    ],
                ),
                ToolMessage(
                    "fatal runtime error: out of memory",
                    tool_call_id="call-stage-e2e",
                    status="error",
                ),
                HumanMessage(
                    "This is a synthetic routing check. Reply with one short sentence "
                    "confirming the capable routing check. Do not call any tools."
                ),
            ]
        }
    )
    return [
        _demo_result("simple", cast(dict[str, object], simple)),
        _demo_result("failed-tool", cast(dict[str, object], failed_tool)),
    ]


async def main() -> None:
    """Run the paid demo and print only routing outcomes and response text."""
    results = await run_demo()
    for result in results:
        print(f"{result.case}: selected={result.selected_model}")
        if result.structured_response is not None:
            print(json.dumps(result.structured_response.model_dump(), sort_keys=True))
        else:
            print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
