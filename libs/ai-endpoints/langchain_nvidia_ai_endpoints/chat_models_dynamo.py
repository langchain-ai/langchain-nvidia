"""ChatNVIDIA subclass with Dynamo KV cache optimization support."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any

from pydantic import Field

from langchain_nvidia_ai_endpoints.chat_models import ChatNVIDIA, _deep_merge


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

    @property
    def _llm_type(self) -> str:
        return "chat-nvidia-ai-playground-dynamo"

    @staticmethod
    def _validate_dynamo_params(
        osl: int, iat: int, latency_sensitivity: float, priority: int
    ) -> None:
        if not isinstance(osl, int) or osl < 0:
            raise ValueError(f"osl must be a non-negative int, got {osl!r}")
        if not isinstance(iat, int) or iat < 0:
            raise ValueError(f"iat must be a non-negative int, got {iat!r}")
        if not isinstance(latency_sensitivity, (int, float)):
            raise ValueError(
                "latency_sensitivity must be a number, "
                f"got {latency_sensitivity!r}"
            )
        if not isinstance(priority, int) or priority < 0:
            raise ValueError(
                f"priority must be a non-negative int, got {priority!r}"
            )

    def _get_payload(self, inputs: Sequence[dict], **kwargs: Any) -> dict:
        # Pop dynamo-specific overrides from kwargs so they don't leak upstream
        osl_value = kwargs.pop("osl", self.osl)
        iat_value = kwargs.pop("iat", self.iat)
        latency_sensitivity = kwargs.pop(
            "latency_sensitivity", self.latency_sensitivity
        )
        priority = kwargs.pop("priority", self.priority)

        self._validate_dynamo_params(
            osl_value, iat_value, latency_sensitivity, priority
        )

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
