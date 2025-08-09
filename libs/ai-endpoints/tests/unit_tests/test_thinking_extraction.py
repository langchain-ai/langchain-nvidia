import pytest
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.outputs import Generation

from langchain_nvidia_ai_endpoints.chat_models import (
    _create_thinking_aware_parser,
    _extract_content_after_thinking,
)


@pytest.mark.parametrize(
    "input_content,expected_output",
    [
        # Basic extraction
        (
            '<think>This is my reasoning.</think>{"result": "success"}',
            '{"result": "success"}',
        ),
        # No thinking tags
        ('{"result": "success"}', '{"result": "success"}'),
        # Multiple thinking blocks
        (
            "<think>First thought</think>Some text<think>Second thought</think>"
            "Final answer",
            "Final answer",
        ),
        # Whitespace handling
        ('<think>Reasoning...</think>\n\n  {"key": "value"}  \n', '{"key": "value"}'),
        # Empty content after thinking
        ("<think>Only thinking, no output</think>", ""),
        # Empty string
        ("", ""),
    ],
    ids=[
        "with-thinking-tags",
        "without-thinking-tags",
        "multiple-thinking-blocks",
        "whitespace-handling",
        "empty-after-thinking",
        "empty-string",
    ],
)
def test_extract_content_after_thinking(
    input_content: str, expected_output: str
) -> None:
    """Test extraction of content after thinking tags."""
    result = _extract_content_after_thinking(input_content)
    assert result == expected_output


# Tests for _create_thinking_aware_parser
class MockParser(BaseOutputParser):
    """Mock parser for testing the thinking aware wrapper."""

    def parse(self, text: str) -> str:
        return f"parsed: {text}"

    def parse_result(self, result: list[Generation], *, partial: bool = False) -> str:
        return f"parsed_result: {result[0].text}"


def test_create_thinking_aware_parser_parse_method() -> None:
    """Test that thinking aware parser extracts content in parse method."""
    ThinkingAwareMockParser = _create_thinking_aware_parser(MockParser)
    parser = ThinkingAwareMockParser()

    # Test with thinking tags
    result = parser.parse("<think>reasoning</think>actual content")
    assert result == "parsed: actual content"

    # Test without thinking tags
    result = parser.parse("just content")
    assert result == "parsed: just content"


def test_create_thinking_aware_parser_parse_result_method() -> None:
    """Test that thinking aware parser extracts content in parse_result method."""
    ThinkingAwareMockParser = _create_thinking_aware_parser(MockParser)
    parser = ThinkingAwareMockParser()

    # Test with thinking tags
    generation = Generation(text="<think>reasoning</think>actual content")
    result = parser.parse_result([generation])
    assert result == "parsed_result: actual content"

    # Test without thinking tags
    generation = Generation(text="just content")
    result = parser.parse_result([generation])
    assert result == "parsed_result: just content"


def test_create_thinking_aware_parser_preserves_functionality() -> None:
    """Test that the wrapper preserves the original parser's functionality."""
    ThinkingAwareMockParser = _create_thinking_aware_parser(MockParser)
    parser = ThinkingAwareMockParser()

    # The wrapped parser should still be an instance of the base parser
    assert isinstance(parser, MockParser)
    assert isinstance(parser, BaseOutputParser)
