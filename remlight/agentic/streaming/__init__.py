"""Streaming module for agent responses.

Provides two streaming modes:
- SSE (Server-Sent Events): OpenAI-compatible format for HTTP APIs
- Plain: Raw text chunks for CLI

Components:
- events.py: SSE event types (Pydantic models)
- state.py: StreamingState for tracking execution
- formatters.py: SSE formatting functions
- handlers.py: Event handlers (child events, tool events)
- core.py: Main streaming generators
"""

from remlight.agentic.streaming.core import (
    save_user_message,
    stream_plain,
    stream_sse,
    stream_sse_with_save,
)
from remlight.agentic.streaming.events import (
    ActionEvent,
    ContentChunk,
    DoneEvent,
    ErrorEvent,
    MetadataEvent,
    ProgressEvent,
    ToolCallEvent,
)
from remlight.agentic.streaming.formatters import (
    format_content_chunk,
    format_done,
    format_plain_text,
    format_sse_event,
)
from remlight.agentic.streaming.state import StreamingState

__all__ = [
    # Core streaming functions
    "stream_sse",
    "stream_sse_with_save",
    "stream_plain",
    "save_user_message",
    # Event types
    "ActionEvent",
    "ToolCallEvent",
    "MetadataEvent",
    "ProgressEvent",
    "DoneEvent",
    "ErrorEvent",
    "ContentChunk",
    # State
    "StreamingState",
    # Formatters
    "format_sse_event",
    "format_content_chunk",
    "format_done",
    "format_plain_text",
]
