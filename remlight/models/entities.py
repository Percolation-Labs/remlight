"""REMLight entity models."""

from typing import Any
from uuid import UUID

from pydantic import Field

from remlight.models.core import CoreModel


class Ontology(CoreModel):
    """Domain entity: people, projects, concepts with semantic links."""

    name: str
    description: str | None = None
    category: str | None = None
    entity_type: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)


class Resource(CoreModel):
    """Document or content chunk with embeddings."""

    name: str | None = None
    uri: str | None = None
    ordinal: int = 0
    content: str | None = None
    category: str | None = None
    related_entities: list[dict[str, Any]] = Field(default_factory=list)


class User(CoreModel):
    """User profile with AI-generated summary."""

    name: str | None = None
    email: str | None = None
    summary: str | None = None  # AI-generated profile summary
    interests: list[str] = Field(default_factory=list)
    preferred_topics: list[str] = Field(default_factory=list)
    activity_level: str | None = None


class Session(CoreModel):
    """Conversation session."""

    name: str | None = None
    description: str | None = None
    agent_name: str | None = None
    status: str = "active"


class Message(CoreModel):
    """Chat message in a session.

    Message roles:
    - user: User input messages
    - assistant: Agent responses
    - tool: Tool call results (NEVER compressed - contains structured metadata)
    - system: System prompts (usually not stored)
    """

    session_id: UUID | str | None = None
    role: str = "assistant"  # 'user', 'assistant', 'tool', 'system' - matches DB column
    content: str = ""
    tool_calls: dict[str, Any] | None = None  # JSONB in DB
    # OTEL tracing
    trace_id: str | None = None
    span_id: str | None = None
