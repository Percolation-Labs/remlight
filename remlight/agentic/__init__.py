"""REMLight agentic module - declarative agents with YAML schemas.

Core Components:
- AgentContext: Session and configuration context
- AgentSchema: YAML-based agent definitions
- create_agent: Agent factory from schema
- run_streaming/run_sync: Unified entry points for API and CLI

Streaming:
- stream_sse: OpenAI-compatible SSE streaming
- stream_plain: Plain text streaming for CLI
"""

from remlight.agentic.schema import AgentSchema, schema_from_yaml, schema_from_yaml_file, schema_to_yaml
from remlight.agentic.provider import (
    create_agent,
    AgentRuntime,
    clear_agent_cache,
    get_agent_cache_stats,
)
from remlight.agentic.context import (
    AgentContext,
    get_current_context,
    set_current_context,
    agent_context_scope,
    get_event_sink,
    set_event_sink,
    event_sink_scope,
    push_event,
)
from remlight.agentic.runner import run_streaming, run_sync
from remlight.agentic.streaming import (
    stream_sse,
    stream_plain,
    StreamingState,
    # Event types
    ToolCallEvent,
    MetadataEvent,
    ProgressEvent,
    DoneEvent,
    ErrorEvent,
    # Formatters
    format_sse_event,
    format_content_chunk,
)

__all__ = [
    # Schema
    "AgentSchema",
    "schema_from_yaml",
    "schema_from_yaml_file",
    "schema_to_yaml",
    # Agent factory
    "create_agent",
    "AgentRuntime",
    # Agent cache
    "clear_agent_cache",
    "get_agent_cache_stats",
    # Context
    "AgentContext",
    "get_current_context",
    "set_current_context",
    "agent_context_scope",
    "get_event_sink",
    "set_event_sink",
    "event_sink_scope",
    "push_event",
    # Runner (unified entry points)
    "run_streaming",
    "run_sync",
    # Streaming
    "stream_sse",
    "stream_plain",
    "StreamingState",
    # Events
    "ToolCallEvent",
    "MetadataEvent",
    "ProgressEvent",
    "DoneEvent",
    "ErrorEvent",
    # Formatters
    "format_sse_event",
    "format_content_chunk",
]
