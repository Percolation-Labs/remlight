"""SSE formatting functions.

Converts events to SSE wire format:
- Custom events: event: {type}\ndata: {json}\n\n
- OpenAI content chunks: data: {json}\n\n
"""

import json
from typing import Any

from pydantic import BaseModel

from remlight.agentic.streaming.state import StreamingState


def format_sse_event(event: BaseModel) -> str:
    """Format a Pydantic event model as SSE.

    Args:
        event: Pydantic model with a 'type' field

    Returns:
        SSE-formatted string: "event: {type}\\ndata: {json}\\n\\n"
    """
    event_type = getattr(event, "type", "message")
    data = event.model_dump(exclude_none=True)
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def format_content_chunk(
    content: str,
    state: StreamingState,
    finish_reason: str | None = None,
) -> str:
    """Format content chunk in OpenAI SSE format.

    Args:
        content: Text content to send
        state: Streaming state (updated in place)
        finish_reason: Optional finish reason ("stop", "error")

    Returns:
        SSE-formatted string: "data: {json}\\n\\n"
    """
    # Update state
    if content:
        state.append_content(content)

    # Build OpenAI-compatible chunk
    chunk: dict[str, Any] = {
        "id": state.request_id,
        "object": "chat.completion.chunk",
        "created": state.created_at,
        "model": state.model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }
        ],
    }

    # Set delta content
    delta = chunk["choices"][0]["delta"]
    if state.is_first_chunk:
        delta["role"] = "assistant"
    if content:
        delta["content"] = content

    # Clean up empty delta
    if not delta:
        chunk["choices"][0]["delta"] = {}

    state.mark_first_chunk_sent()
    return f"data: {json.dumps(chunk)}\n\n"


def format_done() -> str:
    """Format the final [DONE] marker."""
    return "data: [DONE]\n\n"


def format_plain_text(content: str) -> str:
    """Format plain text for CLI output (no SSE wrapper)."""
    return content
