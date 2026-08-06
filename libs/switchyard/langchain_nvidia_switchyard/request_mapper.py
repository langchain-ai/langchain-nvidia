"""Map model calls between LangChain and Switchyard request dictionaries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    convert_to_messages,
    convert_to_openai_messages,
)
from langchain_core.utils.function_calling import convert_to_openai_tool

from .content_mapper import _ContentMapper
from .invocation import LangChainInvocation
from .utils.type_validation import (
    require_mapping,
    require_sequence,
    require_string_list,
)


class SwitchyardRequestMapper:
    """Map one complete buffered call between LangChain and Switchyard.

    The libsy algorithm API consumes a Switchyard request dictionary. That schema is
    provider-neutral, but the mapper names it after Switchyard to keep the direction clear.
    This mapper supports text, parsed tool calls, text tool results, common model
    options, and reasoning effort. Values without a supported Switchyard field are
    rejected rather than silently discarded.

    The mapper is stateless. ``to_switchyard`` and ``from_switchyard`` are the public
    operations; private methods below them own the policy for each request field.
    """

    @classmethod
    def to_switchyard(
        cls,
        messages: Sequence[BaseMessage],
        *,
        tools: Sequence[object],
        tool_choice: object | None,
        model_settings: Mapping[str, object],
        stop: list[str] | None,
    ) -> dict[str, object]:
        """Build a Switchyard request from one LangChain model invocation.

        ``model_settings`` contains per-call overrides supplied by LangChain
        middleware. Settings configured on a target model's constructor remain on
        that target and do not pass through this mapping.

        Raises:
            ValueError: If a message, tool, setting, or stop sequence cannot be
                represented by the supported Switchyard subset.
        """
        instructions, turns = cls._messages_to_switchyard(messages)
        settings = cls._model_settings_to_switchyard(model_settings)

        switchyard_tools = [
            cls._tool_to_switchyard(tool, index) for index, tool in enumerate(tools)
        ]
        switchyard_choice = cls._tool_choice_to_switchyard(tool_choice)

        request: dict[str, object] = {
            "model": "auto",
            "instructions": instructions,
            "messages": turns,
            "tools": switchyard_tools,
            **settings,
            "stream": False,
        }
        if switchyard_choice is not None:
            request["tool_choice"] = switchyard_choice

        if stop is not None:
            request["extensions"] = {"fields": {"stop": stop}}
        return request

    @classmethod
    def from_switchyard(cls, request: Mapping[str, object]) -> LangChainInvocation:
        """Build target-model arguments from one Switchyard request.

        Returns:
            A single object containing messages, tools, tool choice, per-call model
            options, and stop sequences for ``BaseChatModel.ainvoke``.

        Raises:
            ValueError: If the request cannot be represented by the supported
                LangChain subset.
        """
        tools = cls._tools_from_switchyard(request)
        tool_choice = cls._tool_choice_from_switchyard(request)
        cls._require_tools_for_choice(tool_choice, tools=tools)
        return LangChainInvocation(
            messages=cls._messages_from_switchyard(request),
            tools=tools,
            tool_choice=tool_choice,
            options=cls._options_from_switchyard(request),
            stop=cls._stop_from_switchyard(request),
        )

    # LangChain -> Switchyard

    @classmethod
    def _messages_to_switchyard(
        cls,
        messages: Sequence[BaseMessage],
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        """Separate instruction messages from ordered conversation turns."""
        if not messages:
            raise ValueError("messages must not be empty")

        instructions: list[dict[str, object]] = []
        turns: list[dict[str, object]] = []
        conversation_started = False
        for index, message in enumerate(messages):
            path = f"messages[{index}]"
            if isinstance(message, SystemMessage):
                if conversation_started:
                    raise ValueError(
                        f"{path} is an instruction after the conversation has started; "
                        "Switchyard requests store instructions before all turns"
                    )
                instructions.append(cls._instruction_to_switchyard(message, index=index))
            else:
                conversation_started = True
                turns.append(cls._turn_to_switchyard(message, index=index))
        return instructions, turns

    @staticmethod
    def _instruction_to_switchyard(
        message: SystemMessage,
        *,
        index: int,
    ) -> dict[str, object]:
        """Map a LangChain system/developer message to a Switchyard instruction."""
        path = f"messages[{index}]"
        # LangChain owns the internal marker that distinguishes developer messages.
        converted = convert_to_openai_messages(
            message,
            text_format="block",
            pass_through_unknown_blocks=False,
        )
        value = require_mapping(converted, path)
        role = value.get("role")
        if role not in {"system", "developer"}:
            raise ValueError(f"{path} has unsupported instruction role {role!r}")
        return {
            "role": role,
            "content": _ContentMapper.to_switchyard(message, path=path),
        }

    @classmethod
    def _turn_to_switchyard(
        cls,
        message: BaseMessage,
        *,
        index: int,
    ) -> dict[str, object]:
        """Map one non-instruction LangChain message to a Switchyard turn."""
        path = f"messages[{index}]"
        if isinstance(message, HumanMessage):
            return {
                "role": "user",
                "content": _ContentMapper.to_switchyard(message, path=path),
            }
        if isinstance(message, AIMessage):
            return {
                "role": "assistant",
                "content": _ContentMapper.to_switchyard(
                    message,
                    path=path,
                    allow_tool_calls=True,
                ),
            }
        if isinstance(message, ToolMessage):
            if not isinstance(message.tool_call_id, str):
                raise ValueError(f"{path}.tool_call_id must be a string")
            return {
                "role": "tool",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_call_id": message.tool_call_id,
                        "content": _ContentMapper.to_switchyard(message, path=path),
                        "is_error": message.status == "error",
                    }
                ],
            }
        raise ValueError(f"{path} has unsupported LangChain message type {type(message).__name__}")

    @classmethod
    def _tool_to_switchyard(cls, raw_tool: object, index: int) -> dict[str, object]:
        """Normalize one LangChain tool through its public OpenAI-tool converter."""
        converted = convert_to_openai_tool(cast(Any, raw_tool))
        return cls._switchyard_tool_fields(
            converted.get("function"),
            path=f"tools[{index}].function",
        )

    @staticmethod
    def _tool_choice_to_switchyard(choice: object) -> dict[str, object] | None:
        """Map LangChain's named and policy tool-choice forms."""
        if choice is None:
            return None
        if isinstance(choice, str):
            if choice == "any":
                return {"type": "required"}
            if choice in {"auto", "required", "none"}:
                return {"type": choice}
            return {"type": "tool", "data": {"name": choice}}

        value = require_mapping(choice, "tool_choice")
        choice_type = value.get("type")
        if choice_type == "function":
            function = require_mapping(value.get("function"), "tool_choice.function")
            name = function.get("name")
            if not isinstance(name, str):
                raise ValueError("tool_choice.function.name must be a string")
            return {"type": "tool", "data": {"name": name}}
        if choice_type in {"auto", "required", "none"}:
            return {"type": choice_type}
        raise ValueError("tool_choice is not supported")

    @staticmethod
    def _require_tools_for_choice(choice: object | None, *, tools: Sequence[object]) -> None:
        """Prevent a tool choice from disappearing when no tools can be bound."""
        if choice is not None and not tools:
            raise ValueError("tool_choice requires at least one tool")

    @staticmethod
    def _model_settings_to_switchyard(
        model_settings: Mapping[str, object],
    ) -> dict[str, object]:
        """Consume supported per-call settings and reject anything left over."""
        settings = dict(model_settings)

        sampling: dict[str, object] = {}
        for name in ("temperature", "top_p"):
            value = settings.pop(name, None)
            if value is not None:
                sampling[name] = value

        output: dict[str, object] = {}
        max_tokens = settings.pop("max_tokens", None)
        max_completion_tokens = settings.pop("max_completion_tokens", None)
        if max_tokens is not None and max_completion_tokens is not None:
            raise ValueError(
                "model settings 'max_tokens' and 'max_completion_tokens' cannot both be set"
            )
        token_limit = max_completion_tokens if max_completion_tokens is not None else max_tokens
        if token_limit is not None:
            output["max_output_tokens"] = token_limit

        response_format = settings.pop("response_format", None)
        if response_format is not None:
            output["response_format"] = dict(
                require_mapping(
                    response_format,
                    "model_settings.response_format",
                )
            )

        # Reasoning controls
        reasoning: dict[str, object] = {}
        raw_reasoning = settings.pop("reasoning", None)
        reasoning_effort = settings.pop("reasoning_effort", None)
        nested_effort: object | None = None
        if raw_reasoning is not None:
            reasoning_fields = require_mapping(
                raw_reasoning,
                "model_settings.reasoning",
            )
            unknown_field = next(
                (field for field in reasoning_fields if field != "effort"),
                None,
            )
            if unknown_field is not None:
                raise ValueError(
                    f"model setting 'reasoning.{unknown_field}' has no Switchyard representation"
                )
            nested_effort = reasoning_fields.get("effort")

        if nested_effort is not None and reasoning_effort is not None:
            raise ValueError("model settings 'reasoning' and 'reasoning_effort' cannot both be set")
        effort = nested_effort if nested_effort is not None else reasoning_effort
        if effort is not None:
            reasoning["effort"] = effort

        if settings:
            name = sorted(settings)[0]
            raise ValueError(f"model setting {name!r} has no Switchyard representation")

        return {
            "sampling": sampling,
            "output": output,
            "reasoning": reasoning,
        }

    # Switchyard -> LangChain

    @classmethod
    def _messages_from_switchyard(cls, request: Mapping[str, object]) -> list[BaseMessage]:
        """Rebuild instructions followed by ordered conversation messages."""
        instructions = require_sequence(request.get("instructions", []), "instructions")
        messages: list[BaseMessage] = [
            cls._instruction_from_switchyard(raw, path=f"instructions[{index}]")
            for index, raw in enumerate(instructions)
        ]

        turns = require_sequence(request.get("messages"), "messages")
        for index, raw_turn in enumerate(turns):
            messages.extend(cls._turn_from_switchyard(raw_turn, index=index))
        if not messages:
            raise ValueError("messages must not be empty")
        return messages

    @staticmethod
    def _instruction_from_switchyard(raw_instruction: object, *, path: str) -> BaseMessage:
        """Let LangChain encode a Switchyard system/developer instruction."""
        instruction = require_mapping(raw_instruction, path)
        role = instruction.get("role")
        if role not in {"system", "developer"}:
            raise ValueError(f"{path}.role must be system or developer")
        content = _ContentMapper.from_switchyard(
            instruction.get("content"),
            path=f"{path}.content",
        )
        text = "\n".join(block["text"] for block in content)

        # LangChain represents both roles as SystemMessage and owns the internal
        # developer-role marker. Its public converter keeps that detail out of this mapper.
        converted = convert_to_messages(
            [
                cast(Any, {"role": role, "content": text}),
            ]
        )
        return converted[0]

    @classmethod
    def _turn_from_switchyard(
        cls,
        raw_turn: object,
        *,
        index: int,
    ) -> list[BaseMessage]:
        """Map one Switchyard turn; tool results may expand to multiple messages."""
        path = f"messages[{index}]"
        turn = require_mapping(raw_turn, path)
        role = turn.get("role")

        if role in {"system", "developer"}:
            return [cls._instruction_from_switchyard(turn, path=path)]
        if role == "user":
            content = _ContentMapper.from_switchyard(
                turn.get("content"),
                path=f"{path}.content",
            )
            return [HumanMessage(content_blocks=cast(Any, content))]
        if role == "assistant":
            content = _ContentMapper.from_switchyard(
                turn.get("content"),
                path=f"{path}.content",
                allow_tool_calls=True,
            )
            return [AIMessage(content_blocks=cast(Any, content))]
        if role == "tool":
            return cls._tool_messages_from_switchyard(
                turn.get("content"),
                path=f"{path}.content",
            )
        raise ValueError(f"{path}.role {role!r} is not supported")

    @staticmethod
    def _tool_messages_from_switchyard(
        raw_content: object,
        *,
        path: str,
    ) -> list[BaseMessage]:
        """Split Switchyard tool-result blocks into LangChain ToolMessages."""
        content = require_sequence(raw_content, path)
        if not content:
            raise ValueError(f"{path} must contain at least one tool_result")

        messages: list[BaseMessage] = []
        for index, raw_block in enumerate(content):
            block_path = f"{path}[{index}]"
            block = require_mapping(raw_block, block_path)
            if block.get("type") != "tool_result":
                raise ValueError(f"{block_path} must be a tool_result")
            call_id = block.get("tool_call_id")
            is_error = block.get("is_error")
            if not isinstance(call_id, str):
                raise ValueError(f"{block_path}.tool_call_id must be a string")
            result_content = _ContentMapper.from_switchyard(
                block.get("content"),
                path=f"{block_path}.content",
            )
            messages.append(
                ToolMessage(
                    content_blocks=cast(Any, result_content),
                    tool_call_id=call_id,
                    status="error" if is_error is True else "success",
                )
            )
        return messages

    @classmethod
    def _tools_from_switchyard(cls, request: Mapping[str, object]) -> list[dict[str, object]]:
        """Map Switchyard tools to LangChain's OpenAI-compatible form."""
        tools: list[dict[str, object]] = []
        raw_tools = require_sequence(request.get("tools", []), "tools")
        for index, raw_tool in enumerate(raw_tools):
            fields = cls._switchyard_tool_fields(raw_tool, path=f"tools[{index}]")
            tools.append({"type": "function", "function": fields})
        return tools

    @staticmethod
    def _switchyard_tool_fields(value: object, *, path: str) -> dict[str, object]:
        """Validate the tool fields shared by both mapping directions."""
        tool = require_mapping(value, path)
        name = tool.get("name")
        parameters = tool.get("parameters")
        if not isinstance(name, str) or not isinstance(parameters, Mapping):
            raise ValueError(f"{path} must define a function name and parameters")

        result: dict[str, object] = {
            "name": name,
            "description": tool.get("description"),
            "parameters": dict(parameters),
        }
        if "strict" in tool:
            result["strict"] = tool["strict"]
        return result

    @staticmethod
    def _tool_choice_from_switchyard(request: Mapping[str, object]) -> object | None:
        """Map a Switchyard tool-choice policy to LangChain's invocation form."""
        raw_choice = request.get("tool_choice")
        if raw_choice is None:
            return None

        choice = require_mapping(raw_choice, "tool_choice")
        choice_type = choice.get("type")
        if choice_type in {"auto", "required", "none"}:
            return choice_type
        if choice_type == "tool":
            data = require_mapping(choice.get("data"), "tool_choice.data")
            name = data.get("name")
            if not isinstance(name, str):
                raise ValueError("tool_choice.data.name must be a string")
            return {"type": "function", "function": {"name": name}}
        raise ValueError("tool_choice is not supported")

    @staticmethod
    def _options_from_switchyard(request: Mapping[str, object]) -> dict[str, object]:
        """Map normalized request options to per-call LangChain model options."""
        options: dict[str, object] = {}

        sampling = require_mapping(request.get("sampling", {}), "sampling")
        if sampling.get("top_k") is not None:
            raise ValueError("sampling.top_k has no supported LangChain model option")
        for name in ("temperature", "top_p"):
            value = sampling.get(name)
            if value is not None:
                options[name] = value

        output = require_mapping(request.get("output", {}), "output")
        max_tokens = output.get("max_output_tokens")
        if max_tokens is not None:
            options["max_tokens"] = max_tokens
        response_format = output.get("response_format")
        if response_format is not None:
            options["response_format"] = dict(
                require_mapping(response_format, "output.response_format")
            )

        reasoning = require_mapping(request.get("reasoning", {}), "reasoning")
        if reasoning.get("raw") is not None:
            raise ValueError("reasoning.raw has no supported LangChain model option")
        effort = reasoning.get("effort")
        if effort is not None:
            options["reasoning"] = {"effort": effort}
        return options

    @staticmethod
    def _stop_from_switchyard(request: Mapping[str, object]) -> list[str] | None:
        """Read the OpenAI-compatible stop extension from a Switchyard request."""
        extensions = require_mapping(request.get("extensions", {}), "extensions")
        fields = require_mapping(extensions.get("fields", {}), "extensions.fields")
        stop = fields.get("stop")
        return None if stop is None else require_string_list(stop, "extensions.fields.stop")
