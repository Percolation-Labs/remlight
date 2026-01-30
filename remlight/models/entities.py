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

    # Embeddings: uses default precedence (description → content)
    model_config = {"embedding_field": True}

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

    # Embeddings: specify content since no description field
    model_config = {"embedding_field": "content"}

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


class Agent(CoreModel):
    """Stored agent schema from database.

    Agents can be defined in YAML files (schemas directory) or stored in the database.
    Database agents allow runtime creation and modification without code changes.

    The 'content' field stores the full YAML schema that can be loaded
    via schema_from_yaml() to create an AgentSchema instance.

    The 'description' field is used for semantic search embeddings. If not provided,
    embeddings are generated from 'content' as fallback.

    Registry URI: Agents can specify their registry source for federation:
    - "local" or None: Built-in local agent
    - URL: Remote registry that owns this agent definition

    The agent's ID is deterministic: hash(registry_uri + name)

    Time Machine: When agents are upserted, a trigger automatically records
    version history in agent_timemachine table if content changes.
    """

    name: str  # Unique agent identifier (matches json_schema_extra.name)
    description: str | None = None  # Optional short description for search
    content: str  # Full YAML content - source of truth
    version: str = "1.0.0"  # Schema version
    enabled: bool = True  # Whether agent is active
    registry_uri: str | None = None  # Registry source (None = "local")
    icon: str | None = None  # Icon URL or emoji

    # Embeddings: uses default precedence (description → content)
    model_config = {"embedding_field": True}


class AgentTimeMachine(CoreModel):
    """Version history entry for an agent.

    Automatically populated by database trigger when agents change.
    Records the full content at each version for audit and rollback.
    """

    agent_id: UUID | str  # Reference to agents.id
    agent_name: str  # Agent name at time of change
    content: str  # Full YAML content at this version
    version: str | None = None  # Version at time of change
    content_hash: str  # SHA256 hash for change detection
    change_type: str  # 'created', 'updated', 'deleted'


class Server(CoreModel):
    """MCP tool server configuration.

    Servers provide tools that agents can use. They can be:
    - local: In-process Python module (built-in)
    - rest: Remote REST/HTTP server
    - stdio: MCP stdio transport (subprocess)

    The `registry_uri` field enables future federation where servers
    can be discovered from remote registries.

    Attributes:
        name: Unique server identifier/alias
        description: Server description (used for embeddings/search)
        server_type: Type of server (local, rest, stdio)
        endpoint: URL or command for remote/stdio servers
        config: Server-specific configuration (auth, headers, etc.)
        enabled: Whether server is active
        registry_uri: Parent registry URI for federation (nullable)
        icon: Display icon (URL or emoji)
    """

    name: str
    description: str | None = None
    server_type: str = "mcp"  # mcp (local), rest, stdio
    endpoint: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    registry_uri: str | None = None  # Federation support
    icon: str | None = None

    # Embeddings: uses description for search
    model_config = {"embedding_field": "description"}

    @field_validator("config", mode="before")
    @classmethod
    def parse_config(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        return v


class Tool(CoreModel):
    """Registered tool definition.

    Tools are functions that agents can call. Each tool belongs to a server
    and has a schema describing its input parameters.

    Attributes:
        name: Tool function name
        description: Tool description (used for embeddings/search)
        server_id: Reference to parent server
        input_schema: JSON Schema for tool parameters
        enabled: Whether tool is active
        icon: Display icon (URL or emoji)
    """

    name: str
    description: str | None = None
    server_id: UUID | str | None = None  # FK to servers.id
    input_schema: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    icon: str | None = None

    # Embeddings: uses description for search
    model_config = {"embedding_field": "description"}

    @field_validator("input_schema", mode="before")
    @classmethod
    def parse_input_schema(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        return v


class File(CoreModel):
    """File metadata and parsed content.

    Files represent uploaded or processed documents (PDFs, images, audio, etc.)
    with their extracted content stored in parsed_output.

    Attributes:
        name: Original filename
        uri: Source URI (s3://, file://, https://)
        uri_hash: SHA256 hash of URI for deduplication
        content: Extracted text content
        mime_type: MIME type (application/pdf, text/markdown, etc.)
        size_bytes: File size in bytes
        processing_status: pending, processing, completed, failed
        parsed_output: Rich parsing result with text, tables, images, metadata
    """

    name: str
    uri: str
    uri_hash: str | None = None  # Computed from URI if not provided
    content: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    processing_status: str = "pending"
    parsed_output: dict[str, Any] = Field(default_factory=dict)

    @field_validator("parsed_output", mode="before")
    @classmethod
    def parse_parsed_output(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {}
        return v


class Feedback(CoreModel):
    """User feedback on agent responses.

    Feedback is decoupled from Scenarios:
    - Feedback: End-user ratings/comments on agent quality (thumbs up/down, scores)
    - Scenarios: Admin-created test cases for evaluation

    Feedback is stored locally AND optionally sent to Phoenix for observability.
    Each feedback record links to a session/message via trace_id or span_id.

    Attributes:
        session_id: Reference to the session being rated
        message_id: Optional reference to specific message
        trace_id: Phoenix/OTEL trace ID for observability
        span_id: Phoenix/OTEL span ID for observability
        name: Annotation type (e.g., "user_feedback", "quality_rating")
        score: Numeric rating (0.0 to 1.0)
        label: Categorical label (e.g., "thumbs_up", "thumbs_down", "helpful")
        comment: Free text feedback
        source: Feedback source ("user", "evaluator", "automated")
    """

    session_id: UUID | str | None = None
    message_id: UUID | str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    name: str = "user_feedback"
    score: float | None = None
    label: str | None = None
    comment: str | None = None
    source: str = "user"  # user, evaluator, automated


class Collection(CoreModel):
    """Collection of sessions for batch evaluation and testing.

    Collections group sessions together for:
    - Batch evaluation: Run evaluator agents across all sessions
    - Test suites: Create reproducible test sets from real sessions
    - Analysis: Compare agent performance across session groups

    Sessions can be added to collections via:
    - Manual selection
    - Query-based rules (sessions matching certain criteria)
    - Clone operations (copy sessions from other collections)

    Collections are searchable by:
    - Name (fuzzy text search)
    - Description (semantic vector search via embeddings)
    - Tags (array containment)

    Attributes:
        name: Collection name/identifier
        description: Purpose and contents description (embedded for semantic search)
        session_count: Cached count of sessions (updated on add/remove)
        status: Collection status (active, archived, running, completed)
        query_filter: Optional JSON query to auto-populate (saved search)
    """

    name: str
    description: str | None = None
    session_count: int = 0
    status: str = "active"  # active, archived, running, completed
    query_filter: dict[str, Any] | None = None  # Saved query for auto-population

    # Embeddings: uses description for semantic search
    model_config = {"embedding_field": "description"}

    @field_validator("query_filter", mode="before")
    @classmethod
    def parse_query_filter(cls, v: Any) -> Any:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return None
        return v


class CollectionSession(CoreModel):
    """Junction table linking sessions to collections.

    Allows many-to-many relationship between collections and sessions.
    A session can belong to multiple collections.

    Attributes:
        collection_id: Reference to the parent collection
        session_id: Reference to the session
        ordinal: Optional ordering within the collection
        notes: Optional notes about why session was included
    """

    collection_id: UUID | str
    session_id: UUID | str
    ordinal: int = 0
    notes: str | None = None
