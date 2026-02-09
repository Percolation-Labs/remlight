"""SSE event types for streaming responses.

All events are Pydantic models for consistent serialization.
"""

from typing import Any

from pydantic import BaseModel


class ToolCallEvent(BaseModel):
    """SSE event for tool call start/end."""

    type: str = "tool_call"
    tool_name: str
    tool_id: str
    status: str = "started"  # "started" | "completed"
    arguments: dict[str, Any] | None = None
    result: Any = None


class ActionEvent(BaseModel):
    """SSE event for generic action (from action() tool).

    Used for action types other than 'observation' (which uses MetadataEvent).
    """

    type: str = "action"
    action_type: str  # "elicit", "delegate", etc.
    payload: dict[str, Any] | None = None


class MetadataEvent(BaseModel):
    """SSE event for observation metadata (from action(type='observation')).

    Emitted when agents record observations about their response.
    """

    type: str = "metadata"
    message_id: str | None = None
    in_reply_to: str | None = None
    session_id: str | None = None
    agent_schema: str | None = None
    responding_agent: str | None = None
    session_name: str | None = None
    confidence: float | None = None
    sources: list[str] | None = None
    model_version: str | None = None
    latency_ms: int | None = None
    token_count: int | None = None
    # Risk assessment fields
    risk_level: str | None = None
    risk_score: int | None = None
    risk_reasoning: str | None = None
    recommended_action: str | None = None
    # Generic extension
    extra: dict[str, Any] | None = None


class ProgressEvent(BaseModel):
    """SSE event for progress updates."""

    type: str = "progress"
    step: int = 1
    total_steps: int = 3
    label: str = "Processing"
    status: str = "in_progress"


class DoneEvent(BaseModel):
    """SSE event for stream completion."""

    type: str = "done"
    reason: str = "stop"  # "stop" | "error"


class ErrorEvent(BaseModel):
    """SSE event for errors."""

    type: str = "error"
    code: str = "stream_error"
    message: str
    details: dict[str, Any] | None = None
    recoverable: bool = True


class ContentChunk(BaseModel):
    """Internal representation of a content chunk (not SSE event)."""

    content: str
    is_first: bool = False
    finish_reason: str | None = None
