"""Map responses between LangChain and Switchyard response dictionaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, cast

from langchain_core.messages import AIMessage

from .content_mapper import _ContentMapper
from .utils.type_validation import require_mapping, require_sequence


class SwitchyardResponseMapper:
    """Map one buffered response between LangChain and Switchyard.

    A LangChain ``AIMessage`` represents one selected completion, so the reverse
    mapping accepts exactly one Switchyard assistant output. Text/tool-call order,
    finish reasons, common usage fields, and the routing decision trace are preserved.
    """

    _FINISH_TO_SWITCHYARD: ClassVar[dict[str, str]] = {
        "stop": "end_turn",
        "end_turn": "end_turn",
        "length": "max_tokens",
        "max_tokens": "max_tokens",
        "tool_calls": "tool_use",
        "tool_use": "tool_use",
        "content_filter": "content_filter",
        "error": "error",
    }
    _SWITCHYARD_TO_FINISH: ClassVar[dict[str, str]] = {
        "end_turn": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
        "content_filter": "content_filter",
        "error": "error",
        "unknown": "unknown",
    }

    @classmethod
    def to_switchyard(cls, message: AIMessage, *, model_name: str) -> dict[str, object]:
        """Build a Switchyard response from a target ``AIMessage``.

        Raises:
            ValueError: If the message contains unsupported or invalid content.
        """
        content = _ContentMapper.to_switchyard(
            message,
            path="response",
            allow_tool_calls=True,
        )
        return {
            "id": message.id,
            "model": model_name,
            "outputs": [
                {
                    "role": "assistant",
                    "content": content,
                    "stop_reason": cls._stop_reason_to_switchyard(
                        message,
                        content=content,
                    ),
                }
            ],
            "usage": cls._usage_to_switchyard(message),
        }

    @classmethod
    def from_switchyard(
        cls,
        response: Mapping[str, object],
        *,
        decisions: Sequence[Mapping[str, object]],
    ) -> AIMessage:
        """Build one routed ``AIMessage`` and attach its Switchyard decision trace.

        Raises:
            ValueError: If the response does not contain exactly one supported
                assistant output or contains malformed usage data.
        """
        output = cls._assistant_output(response)
        content = _ContentMapper.from_switchyard(
            output.get("content"),
            path="outputs[0].content",
            allow_tool_calls=True,
        )
        message_id = response.get("id")
        return AIMessage(
            content_blocks=cast(Any, content),
            id=message_id if isinstance(message_id, str) else None,
            response_metadata=cls._response_metadata(response, output, decisions),
            usage_metadata=cls._usage_from_switchyard(response.get("usage", {})),
        )

    # LangChain -> Switchyard

    @classmethod
    def _stop_reason_to_switchyard(
        cls,
        message: AIMessage,
        *,
        content: Sequence[Mapping[str, object]],
    ) -> str:
        """Normalize common provider finish reasons, with a content-based fallback."""
        raw_finish = message.response_metadata.get("finish_reason")
        if isinstance(raw_finish, str):
            return cls._FINISH_TO_SWITCHYARD.get(raw_finish, "unknown")
        has_tool_call = any(block.get("type") == "tool_call" for block in content)
        return "tool_use" if has_tool_call else "end_turn"

    @staticmethod
    def _usage_to_switchyard(message: AIMessage) -> dict[str, int]:
        """Map LangChain's common and detailed token counters."""
        usage = message.usage_metadata
        if usage is None:
            return {}

        input_details = usage.get("input_token_details") or {}
        output_details = usage.get("output_token_details") or {}
        details: dict[str, int] = {}
        for source, target in (
            ("cache_read", "cached_input_tokens"),
            ("cache_creation", "cache_creation_input_tokens"),
        ):
            value = input_details.get(source)
            if value is not None:
                if not isinstance(value, int):
                    raise ValueError(f"response input token detail {source!r} must be an integer")
                details[target] = value
        reasoning = output_details.get("reasoning")
        if reasoning is not None:
            if not isinstance(reasoning, int):
                raise ValueError("response reasoning token count must be an integer")
            details["reasoning_tokens"] = reasoning

        result: dict[str, int] = {
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
        }
        result.update(details)
        return result

    # Switchyard -> LangChain

    @staticmethod
    def _assistant_output(response: Mapping[str, object]) -> Mapping[str, object]:
        """Return the single assistant output represented by an ``AIMessage``."""
        outputs = require_sequence(response.get("outputs"), "outputs")
        if not outputs:
            raise ValueError("Switchyard response has no outputs")
        if len(outputs) != 1:
            raise ValueError("Switchyard response must contain exactly one output")
        output = require_mapping(outputs[0], "outputs[0]")
        if output.get("role") != "assistant":
            raise ValueError("outputs[0].role must be assistant")
        return output

    @classmethod
    def _response_metadata(
        cls,
        response: Mapping[str, object],
        output: Mapping[str, object],
        decisions: Sequence[Mapping[str, object]],
    ) -> dict[str, object]:
        """Attach normalized response fields and the complete routing trace."""
        decision_trace = [dict(decision) for decision in decisions]
        switchyard: dict[str, object] = {"decisions": decision_trace}
        if decision_trace:
            selected_model = decision_trace[-1].get("selected_model")
            if selected_model is not None:
                switchyard["selected_model"] = selected_model

        metadata: dict[str, object] = {"switchyard": switchyard}
        model = response.get("model")
        if isinstance(model, str):
            metadata["model_name"] = model
        stop_reason = output.get("stop_reason")
        if isinstance(stop_reason, str):
            metadata["finish_reason"] = cls._SWITCHYARD_TO_FINISH.get(
                stop_reason,
                "unknown",
            )
        return metadata

    @staticmethod
    def _usage_from_switchyard(raw_usage: object) -> dict[str, Any] | None:
        """Rebuild LangChain token counts and cache details from Switchyard usage."""
        usage = require_mapping(raw_usage, "usage")
        counts: dict[str, int] = {}
        for name in (
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_input_tokens",
            "cache_creation_input_tokens",
            "reasoning_tokens",
        ):
            value = usage.get(name)
            if value is not None:
                if not isinstance(value, int):
                    raise ValueError(f"usage.{name} must be an integer")
                counts[name] = value
        if not counts:
            return None

        result: dict[str, Any] = {
            "input_tokens": counts.get("input_tokens", 0),
            "output_tokens": counts.get("output_tokens", 0),
            "total_tokens": counts.get(
                "total_tokens",
                counts.get("input_tokens", 0) + counts.get("output_tokens", 0),
            ),
        }
        input_details = {
            target: counts[source]
            for source, target in (
                ("cached_input_tokens", "cache_read"),
                ("cache_creation_input_tokens", "cache_creation"),
            )
            if source in counts
        }
        if input_details:
            result["input_token_details"] = input_details
        if "reasoning_tokens" in counts:
            result["output_token_details"] = {"reasoning": counts["reasoning_tokens"]}
        return result
