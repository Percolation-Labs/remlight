"""Streaming module for agent responses.

Provides SSE event types and simulator for testing.
"""

from remlight.agentic.streaming.simulator import (
    is_simulator_agent,
    stream_simulator_plain,
    stream_simulator_sse,
)
from remlight.agentic.streaming.events import (
    ActionEvent,
    DoneEvent,
    ErrorEvent,
    MetadataEvent,
    ProgressEvent,
    ToolCallEvent,
)
from remlight.agentic.streaming.formatters import (
    format_content_chunk,
    format_done,
    format_sse_event,
)
from remlight.agentic.streaming.state import StreamingState

__all__ = [
    # Simulator
    "stream_simulator_sse",
    "stream_simulator_plain",
    "is_simulator_agent",
    # Event types
    "ActionEvent",
    "ToolCallEvent",
    "MetadataEvent",
    "ProgressEvent",
    "DoneEvent",
    "ErrorEvent",
    # State
    "StreamingState",
    # Formatters
    "format_sse_event",
    "format_content_chunk",
    "format_done",
]
