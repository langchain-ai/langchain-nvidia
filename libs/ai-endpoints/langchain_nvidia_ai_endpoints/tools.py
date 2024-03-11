"""OpenAI chat wrapper."""

from __future__ import annotations

import logging
from operator import itemgetter
from typing import (
    Any,
    Callable,
    Dict,
    Sequence,
    Type,
    TypeVar,
    Union,
)

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

logger = logging.getLogger(__name__)

_BM = TypeVar("_BM", bound=BaseModel)


## Directly Inspired by OpenAI/MistralAI's server-side support.
## Moved here for versioning/additional integration options.


class ServerToolsMixin(Runnable):
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        tool_arg: str = "tools",
        conversion_fn: Callable = convert_to_openai_tool,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with OpenAI tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.

        EXPERIMENTAL: This method is intended for future support. Invoked in a class:
        ```
        class TooledChatNVIDIA(ChatNVIDIA, ToolsMixin):
            pass

        llm = TooledChatNVIDIA(model="mixtral_8x7b")
        tooled_llm = llm.bind_tools(tools)
        tooled_llm.invoke("Hello world!!")
        ```

        See langchain-mistralal/openai's implementation for more documentation.
        """
        formatted_tools = [conversion_fn(tool) for tool in tools]
        tool_kw = {tool_arg: formatted_tools}
        return super().bind(**tool_kw, **kwargs)

    def with_structured_output(
        self,
        schema: Union[Dict, Type[BaseModel]],
        *,
        include_raw: bool = False,
        tool_arg: str = "tools",
        conversion_fn: Callable = convert_to_openai_tool,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema as a dict or a Pydantic class. If a Pydantic class
                then the model output will be an object of that class. If a dict then
                the model output will be a dict. With a Pydantic class the returned
                attributes will be validated, whereas with a dict they will not be. If
                `method` is "function_calling" and `schema` is a dict, then the dict
                must match the OpenAI function-calling spec.
            include_raw: If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".

        Returns:
            A Runnable that takes any ChatModel input and returns as output:

                If include_raw is True then a dict with keys:
                    raw: BaseMessage
                    parsed: Pydantic BaseModel or Dictionary
                    parsing_error: Optional[BaseException]

                If include_raw is False then just BaseModel/Dictionary is returned
                (depending on schema type).

        EXPERIMENTAL: This method is intended for future support. Invoked in a class:
        ```
        class TooledChatNVIDIA(ChatNVIDIA, ToolsMixin):
            pass
        ```

        See langchain-mistralal/openai's implementation for more documentation.
        """
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = isinstance(schema, type) and issubclass(schema, BaseModel)
        llm = self.bind_tools([schema], tool_arg=tool_arg, conversion_fn=conversion_fn)
        if is_pydantic_schema and isinstance(schema, BaseModel):
            schema_cls: Type[BaseModel] = schema
            output_parser: OutputParserLike = PydanticToolsParser(
                tools=[schema_cls], first_tool_only=True
            )
        else:
            key_name = conversion_fn(schema)["function"]["name"]
            output_parser = JsonOutputKeyToolsParser(
                key_name=key_name, first_tool_only=True
            )

        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            return llm | output_parser
