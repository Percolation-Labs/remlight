"""REMLight entity models."""

import json
from typing import Any
from uuid import UUID

from pydantic import Field, field_validator

from remlight.models.core import CoreModel


class Ontology(CoreModel):
    """Domain entity: people, projects, concepts with semantic links.

    Used for wiki-style knowledge bases with:
    - entity_key (name): Unique identifier for LOOKUP queries
    - content: Full markdown content for SEARCH/FUZZY queries
    - properties: YAML frontmatter metadata (parent, children, related, tags)
    """

    name: str  # entity_key from frontmatter
    content: str | None = None  # Full markdown content
    description: str | None = None
    category: str | None = None
    entity_type: str | None = None
    uri: str | None = None  # Source file URI
    properties: dict[str, Any] = Field(default_factory=dict)  # Frontmatter metadata

    @field_validator("properties", mode="before")
    @classmethod
    def parse_properties(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        return v


class Resource(CoreModel):
    """Document or content chunk with embeddings."""

    name: str | None = None
    uri: str | None = None
    ordinal: int = 0
    content: str | None = None
    category: str | None = None
    related_entities: list[dict[str, Any]] = Field(default_factory=list)

    @field_validator("related_entities", mode="before")
    @classmethod
    def parse_related_entities(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return []
        return v


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

    @field_validator("tool_calls", mode="before")
    @classmethod
    def parse_tool_calls(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                # Empty list or empty dict means no tool calls
                if parsed == [] or parsed == {}:
                    return None
                # Convert list to dict wrapper
                if isinstance(parsed, list):
                    return {"items": parsed} if parsed else None
                return parsed
            except json.JSONDecodeError:
                return None
        # Handle list passed directly (not as string)
        if isinstance(v, list):
            return {"items": v} if v else None
        # Handle empty dict
        if v == {}:
            return None
        return v


class Scenario(CoreModel):
    """Labeled scenario linked to a session for replay and search.

    Scenarios allow users to:
    - Label sessions with descriptive metadata for later retrieval
    - Search by description (semantic), tags, dates, and title
    - Replay old sessions by loading the associated session
    - Build context by finding relevant past interactions
    """

    name: str | None = None  # Scenario title
    description: str | None = None  # Searchable description (embeddings generated)
    session_id: UUID | str | None = None  # Link to the session
    agent_name: str | None = None  # Agent used in this scenario
    status: str = "active"  # active, archived, completed
