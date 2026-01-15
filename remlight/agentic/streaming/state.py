"""Streaming state management."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class StreamingState:
    """Tracks state during streaming.

    Maintains all context needed during a streaming session:
    - Request/message identifiers
    - Accumulated content
    - Tool call tracking
    - Metadata from register_metadata
    - Child agent state
    """

    # Request identification
    request_id: str = field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: int = field(default_factory=lambda: int(datetime.now(timezone.utc).timestamp()))
    start_time: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    model: str = "unknown"

    # Content accumulation
    current_text: str = ""
    accumulated_chunks: list[str] = field(default_factory=list)
    token_count: int = 0
    is_first_chunk: bool = True

    # Tool call tracking
    tool_calls: list[dict] = field(default_factory=list)
    active_tool_calls: dict[int, tuple[str, str]] = field(default_factory=dict)
    pending_tool_completions: list[tuple[str, str]] = field(default_factory=list)
    pending_tool_data: dict[str, dict] = field(default_factory=dict)
    current_tool_id: str | None = None

    # Metadata tracking (from register_metadata tool)
    metadata: dict[str, Any] = field(default_factory=dict)
    metadata_registered: bool = False

    # Progress tracking
    current_step: int = 1
    total_steps: int = 3

    # Child agent tracking (for multi-agent)
    child_content_streamed: bool = False
    responding_agent: str | None = None

    def latency_ms(self) -> int:
        """Calculate latency in milliseconds since stream started."""
        return int((datetime.now(timezone.utc).timestamp() - self.start_time) * 1000)

    def append_content(self, content: str) -> None:
        """Append content to accumulated text."""
        self.current_text += content
        self.accumulated_chunks.append(content)
        self.token_count += len(content.split())

    def get_full_content(self) -> str:
        """Get all accumulated content."""
        return "".join(self.accumulated_chunks)

    def mark_first_chunk_sent(self) -> None:
        """Mark that the first chunk has been sent."""
        self.is_first_chunk = False

    def register_tool_call(self, tool_name: str, tool_id: str, index: int, args: dict) -> None:
        """Register a new tool call."""
        self.current_tool_id = tool_id
        self.active_tool_calls[index] = (tool_name, tool_id)
        self.pending_tool_completions.append((tool_name, tool_id))
        self.pending_tool_data[tool_id] = {
            "tool_name": tool_name,
            "tool_id": tool_id,
            "arguments": args,
        }

    def complete_tool_call(self, result: Any) -> dict | None:
        """Complete a pending tool call and return its data."""
        if not self.pending_tool_completions:
            return None

        tool_name, tool_id = self.pending_tool_completions.pop(0)

        if tool_id in self.pending_tool_data:
            tool_data = self.pending_tool_data.pop(tool_id)
            tool_data["result"] = result
            return tool_data

        return {"tool_name": tool_name, "tool_id": tool_id, "result": result}

    def register_metadata(self, metadata: dict) -> None:
        """Register metadata from register_metadata tool."""
        self.metadata.update(metadata)
        self.metadata_registered = True

        if not self.responding_agent and metadata.get("agent_schema"):
            self.responding_agent = metadata["agent_schema"]

    def mark_child_content(self, agent_name: str) -> None:
        """Mark that child agent is streaming content."""
        self.child_content_streamed = True
        self.responding_agent = agent_name

    @classmethod
    def create(
        cls,
        model: str = "unknown",
        request_id: str | None = None,
        message_id: str | None = None,
    ) -> "StreamingState":
        """Create a new streaming state with optional overrides."""
        state = cls(model=model)
        if request_id:
            state.request_id = request_id
        if message_id:
            state.message_id = message_id
        return state
