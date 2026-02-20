"""ChatNVIDIA subclass with Dynamo KV cache optimization support."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from pydantic import Field

from langchain_nvidia_ai_endpoints.chat_models import ChatNVIDIA, _deep_merge

_DYNAMO_KEYS = ("osl", "iat", "latency_sensitivity", "priority")


class ChatNVIDIADynamo(ChatNVIDIA):
    """ChatNVIDIA subclass that injects ``nvext.agent_hints`` into requests
    for Dynamo KV cache routing optimization.

    A unique ``prefix_id`` is auto-generated for every request.

    Example:
        ```python
        from langchain_nvidia_ai_endpoints import ChatNVIDIADynamo

        llm = ChatNVIDIADynamo(model="meta/llama3-8b-instruct")
        # override per-invocation:
        llm.invoke("Hello", osl=2048, iat=50)
        ```
    """

    osl: int = Field(
        default=512,
        description="Expected output sequence length (tokens).",
    )
    iat: int = Field(
        default=250,
        description="Expected inter-arrival time (ms).",
    )
    latency_sensitivity: float = Field(
        default=1.0,
        description="Latency sensitivity hint for Dynamo routing.",
    )
    priority: int = Field(
        default=1,
        description="Request priority hint for Dynamo routing.",
    )

    def __init__(
        self,
        *,
        model: str | None = None,
        nvidia_api_key: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        default_headers: dict[str, str] | None = None,
        osl: int = 512,
        iat: int = 250,
        latency_sensitivity: float = 1.0,
        priority: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            nvidia_api_key=nvidia_api_key,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            top_p=top_p,
            seed=seed,
            stop=stop,
            default_headers=default_headers,
            osl=osl,
            iat=iat,
            latency_sensitivity=latency_sensitivity,
            priority=priority,
            **kwargs,
        )

    @property
    def _llm_type(self) -> str:
        return "chat-nvidia-ai-playground-dynamo"

    def _get_payload(self, inputs: Sequence[dict], **kwargs: Any) -> dict:
        # Pop dynamo-specific overrides from kwargs so they don't leak upstream
        osl_value = kwargs.pop("osl", self.osl)
        iat_value = kwargs.pop("iat", self.iat)
        latency_sensitivity = kwargs.pop(
            "latency_sensitivity", self.latency_sensitivity
        )
        priority = kwargs.pop("priority", self.priority)

        payload = super()._get_payload(inputs, **kwargs)

        agent_hints: dict[str, Any] = {
            "prefix_id": f"langchain-dynamo-{uuid.uuid4().hex[:12]}",
            "osl": osl_value,
            "iat": iat_value,
            "latency_sensitivity": float(latency_sensitivity),
            "priority": priority,
        }

        payload = _deep_merge(payload, {"nvext": {"agent_hints": agent_hints}})

        return payload
