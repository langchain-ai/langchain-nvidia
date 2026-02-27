"""Unit tests for ChannelStateManager public interfaces."""

from __future__ import annotations

from typing import TypedDict

import pytest
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from langchain_nvidia_langgraph.executors.speculative.channel_state_manager import (
    ChannelStateManager,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class SimpleState(TypedDict):
    """Minimal state schema with two fields."""

    value: str
    count: int


def _node_passthrough(state: dict) -> dict:
    """Simple node."""
    return {}


@pytest.fixture
def compiled_graph() -> CompiledStateGraph:
    """Compiled graph with channels for value and count."""
    graph = StateGraph(SimpleState)
    graph.add_node("a", _node_passthrough)
    graph.set_entry_point("a")
    return graph.compile()


@pytest.fixture
def manager(compiled_graph: CompiledStateGraph) -> ChannelStateManager:
    """ChannelStateManager initialized with compiled graph."""
    return ChannelStateManager(compiled_graph)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_init_stores_channel_templates(compiled_graph: CompiledStateGraph) -> None:
    """ChannelStateManager stores non-__ channels from graph.channels."""
    mgr = ChannelStateManager(compiled_graph)
    assert "value" in mgr.channel_templates
    assert "count" in mgr.channel_templates
    assert not any(k.startswith("__") for k in mgr.channel_templates)


def test_init_excludes_internal_channels(compiled_graph: CompiledStateGraph) -> None:
    """ChannelStateManager excludes channels starting with __."""
    mgr = ChannelStateManager(compiled_graph)
    assert "__start__" not in mgr.channel_templates
    assert "__pregel_tasks" not in mgr.channel_templates


# ---------------------------------------------------------------------------
# create_isolated_channels
# ---------------------------------------------------------------------------


def test_create_isolated_channels_returns_dict(manager: ChannelStateManager) -> None:
    """create_isolated_channels returns a dict."""
    channels = manager.create_isolated_channels()
    assert isinstance(channels, dict)


def test_create_isolated_channels_returns_copies(manager: ChannelStateManager) -> None:
    """create_isolated_channels returns copies, not original templates."""
    channels = manager.create_isolated_channels()
    for k, ch in channels.items():
        assert ch is not manager.channel_templates[k]


def test_create_isolated_channels_same_keys(manager: ChannelStateManager) -> None:
    """create_isolated_channels has same keys as channel_templates."""
    channels = manager.create_isolated_channels()
    assert set(channels.keys()) == set(manager.channel_templates.keys())


# ---------------------------------------------------------------------------
# initialize_state
# ---------------------------------------------------------------------------


def test_initialize_state_updates_channels(manager: ChannelStateManager) -> None:
    """initialize_state updates channels and returns state."""
    channels = manager.create_isolated_channels()
    initial = {"value": "hello", "count": 42}
    state = manager.initialize_state(channels, initial)
    assert state["value"] == "hello"
    assert state["count"] == 42


def test_initialize_state_partial_values(manager: ChannelStateManager) -> None:
    """initialize_state handles partial initial values."""
    channels = manager.create_isolated_channels()
    initial = {"value": "only_value"}
    state = manager.initialize_state(channels, initial)
    assert state["value"] == "only_value"


def test_initialize_state_key_not_in_channels(manager: ChannelStateManager) -> None:
    """initialize_state stores keys not in channels directly in state."""
    channels = manager.create_isolated_channels()
    initial = {"value": "x", "extra_key": "not_a_channel"}
    state = manager.initialize_state(channels, initial)
    assert state["value"] == "x"
    assert state["extra_key"] == "not_a_channel"


# ---------------------------------------------------------------------------
# merge_update
# ---------------------------------------------------------------------------


def test_merge_update_dict(manager: ChannelStateManager) -> None:
    """merge_update merges dict update into channels."""
    channels = manager.create_isolated_channels()
    update = {"value": "merged", "count": 10}
    state = manager.merge_update(channels, update)
    assert state["value"] == "merged"
    assert state["count"] == 10


def test_merge_update_list(manager: ChannelStateManager) -> None:
    """merge_update processes list of updates sequentially."""
    channels = manager.create_isolated_channels()
    updates = [{"value": "first"}, {"count": 5}]
    state = manager.merge_update(channels, updates)
    assert state["value"] == "first"
    assert state["count"] == 5


def test_merge_update_command_like(manager: ChannelStateManager) -> None:
    """merge_update extracts update from Command-like object (has .update attr)."""
    channels = manager.create_isolated_channels()
    cmd = type("Command", (), {"update": {"value": "from_cmd", "count": 99}})()
    state = manager.merge_update(channels, cmd)
    assert state["value"] == "from_cmd"
    assert state["count"] == 99


def test_merge_update_skips_non_dict(manager: ChannelStateManager) -> None:
    """merge_update returns empty dict for non-dict, non-list update."""
    channels = manager.create_isolated_channels()
    state = manager.merge_update(channels, "not a dict")
    assert state == {}


def test_merge_update_key_not_in_channels(manager: ChannelStateManager) -> None:
    """merge_update includes keys not in channels in returned state."""
    channels = manager.create_isolated_channels()
    update = {"value": "x", "unknown_field": 123}
    state = manager.merge_update(channels, update)
    assert state["value"] == "x"
    assert state["unknown_field"] == 123


# ---------------------------------------------------------------------------
# get_current_state
# ---------------------------------------------------------------------------


def test_get_current_state_after_initialize(manager: ChannelStateManager) -> None:
    """get_current_state returns values after initialize_state."""
    channels = manager.create_isolated_channels()
    manager.initialize_state(channels, {"value": "x", "count": 1})
    state = manager.get_current_state(channels)
    assert state["value"] == "x"
    assert state["count"] == 1


def test_get_current_state_skips_empty_channels(manager: ChannelStateManager) -> None:
    """get_current_state skips channels that raise EmptyChannelError."""
    channels = manager.create_isolated_channels()
    # Don't initialize - some channels may be empty
    state = manager.get_current_state(channels)
    # Should not raise; may have fewer keys if some channels are empty
    assert isinstance(state, dict)
