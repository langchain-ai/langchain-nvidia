"""Channel-based state management for LangGraph execution.

Provides ``ChannelStateManager`` for merging updates into LangGraph channels
and extracting current state during speculative execution.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.checkpoint.base import EmptyChannelError

logger = logging.getLogger(__name__)


class ChannelStateManager:
    """Manages state updates using LangGraph's channel system.

    Works with any state schema, any reducer, and overwrite patterns.
    LangGraph's channels encapsulate all reducer logic.
    """

    def __init__(self, graph: Any) -> None:
        """Initialize with channel templates from the graph.

        Args:
            graph: CompiledStateGraph with a channels attribute.
        """
        self.channel_templates: dict[str, Any] = {
            k: v for k, v in graph.channels.items() if not k.startswith("__")
        }
        logger.debug(
            "ChannelStateManager initialized with channels: %s",
            list(self.channel_templates.keys()),
        )

    def create_isolated_channels(self) -> dict[str, Any]:
        """Create a copy of channel templates for isolated execution.

        Returns:
            Dict of channel name -> channel copy.
        """
        return {k: v.copy() for k, v in self.channel_templates.items()}

    def initialize_state(
        self, channels: dict[str, Any], initial_values: dict[str, Any]
    ) -> dict[str, Any]:
        """Initialize channels with initial values and return current state.

        Args:
            channels: Isolated channel dict from create_isolated_channels.
            initial_values: Initial state values to inject.

        Returns:
            Dict of channel name -> current value after update.
        """
        state: dict[str, Any] = {}
        for key, value in initial_values.items():
            if key in channels:
                channels[key].update([value])
                state[key] = channels[key].get()
            else:
                state[key] = value
        return state

    def merge_update(
        self, channels: dict[str, Any], update: dict[str, Any] | list | Any
    ) -> dict[str, Any]:
        """Merge an update into channels and return the resulting state.

        Handles Command objects, lists of updates, and dict updates.

        Args:
            channels: Isolated channel dict.
            update: Update to merge (dict, list of dicts, or Command-like).

        Returns:
            Dict of channel name -> value after merge.
        """
        if hasattr(update, "update") and not callable(update.update):
            logger.debug("Extracting update from Command object")
            update = update.update

        if isinstance(update, list):
            logger.debug("Processing list of %s updates", len(update))
            merged_state: dict[str, Any] = {}
            for single_update in update:
                merged_state.update(self.merge_update(channels, single_update))
            return merged_state

        if not isinstance(update, dict):
            logger.warning("Unexpected update type: %s, skipping", type(update))
            return {}

        state: dict[str, Any] = {}
        for key, value in update.items():
            if key not in channels:
                state[key] = value
                continue
            channel = channels[key]
            channel.update([value])
            state[key] = channel.get()
            logger.debug("Channel '%s' updated via %s", key, type(channel).__name__)
        return state

    def get_current_state(self, channels: dict[str, Any]) -> dict[str, Any]:
        """Get all channel values as a state dict.

        Args:
            channels: Isolated channel dict.

        Returns:
            Dict of channel name -> current value (skips empty channels).
        """
        state: dict[str, Any] = {}
        for k, ch in channels.items():
            try:
                state[k] = ch.get()
            except EmptyChannelError:
                logger.debug("Channel '%s' is empty, skipping", k)
        return state
