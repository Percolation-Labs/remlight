# REMLight Agentic Framework

This module implements the core agentic framework for REMLight, providing declarative agent definitions with YAML schemas, multi-agent orchestration, and session context management.

**IMPORTANT**: This implementation MUST match the reference implementation in `remstack/rem/src/rem/agentic/`. Any deviations are bugs.

## Architecture Overview

```
agentic/
├── README.md           # This file - design documentation
├── __init__.py         # Public exports
├── context.py          # AgentContext - session and configuration (Pydantic BaseModel)
├── schema.py           # AgentSchema - YAML agent definitions
└── provider.py         # Agent factory - creates pydantic-ai agents
```

## Critical Design Patterns

### 1. AgentContext - Pydantic BaseModel (NOT dataclass)

**Reference**: `rem/src/rem/agentic/context.py`

AgentContext MUST be a `Pydantic BaseModel`, not a Python dataclass. This enables:
- JSON serialization/deserialization
- Field validation
- `from_request()` method for FastAPI integration

```python
from pydantic import BaseModel, Field

class AgentContext(BaseModel):
    """Session and configuration context for agent execution."""

    user_id: str | None = Field(default=None)
    tenant_id: str = Field(default="default")
    session_id: str | None = Field(default=None)
    default_model: str = Field(default_factory=lambda: settings.llm.default_model)
    agent_schema_uri: str | None = Field(default=None)
    is_eval: bool = Field(default=False)      # REQUIRED - marks evaluation sessions
    client_id: str | None = Field(default=None)  # REQUIRED - e.g., "web", "mobile", "cli"

    model_config = {"populate_by_name": True}
```

#### Required Methods

```python
@classmethod
def from_request(cls, request: "Request") -> "AgentContext":
    """
    Construct from FastAPI Request - PREFERRED for API endpoints.

    Extracts user_id from JWT token in request.state (set by auth middleware),
    NOT from X-User-Id header. This is the secure path.

    Priority for user_id:
    1. request.state.user.id - From validated JWT token (SECURE)
    2. X-User-Id header - Fallback only
    """

@classmethod
def from_headers(cls, headers: dict[str, str]) -> "AgentContext":
    """Construct from HTTP headers dict - for testing/CLI."""

@staticmethod
def get_user_id_or_default(user_id: str | None, source: str, default: str | None = None) -> str | None:
    """
    Get user_id or return None for anonymous access.

    User ID convention:
    - user_id is a deterministic UUID5 hash of user's email
    - Returns None for anonymous - queries WHERE user_id IS NULL (shared data)
    - NO fake user IDs generated
    """

def child_context(self, agent_schema_uri: str | None = None, model_override: str | None = None) -> "AgentContext":
    """Create child context for nested agent calls - inherits user_id, tenant_id, session_id, is_eval, client_id."""
```

#### Context Headers Mapping

| Header | Field | Default |
|--------|-------|---------|
| `X-User-Id` | `user_id` | `None` |
| `X-Tenant-Id` | `tenant_id` | `"default"` |
| `X-Session-Id` | `session_id` | `None` |
| `X-Model-Name` | `default_model` | from settings |
| `X-Agent-Schema` | `agent_schema_uri` | `None` |
| `X-Is-Eval` | `is_eval` | `False` |
| `X-Client-Id` | `client_id` | `None` |

### 2. Multi-Agent Context Propagation

Context flows through nested agent calls via `ContextVar`:

```python
from contextvars import ContextVar

# Thread-local context for current agent execution
_current_agent_context: ContextVar["AgentContext | None"] = ContextVar(
    "current_agent_context", default=None
)

# Event sink for streaming child agent events to parent
_parent_event_sink: ContextVar["asyncio.Queue | None"] = ContextVar(
    "parent_event_sink", default=None
)
```

#### Context Functions

```python
def get_current_context() -> AgentContext | None:
    """Get current agent context from ContextVar."""

def set_current_context(ctx: AgentContext | None) -> None:
    """Set current agent context."""

@contextmanager
def agent_context_scope(ctx: AgentContext) -> Generator[AgentContext, None, None]:
    """Context manager - automatically restores previous context on exit."""
```

#### Event Sink Functions

```python
def get_event_sink() -> asyncio.Queue | None:
    """Get parent's event sink for streaming child events."""

def set_event_sink(sink: asyncio.Queue | None) -> None:
    """Set event sink for child agents."""

@contextmanager
def event_sink_scope(sink: asyncio.Queue) -> Generator[asyncio.Queue, None, None]:
    """Context manager for scoped event sink setting."""

async def push_event(event: Any) -> bool:
    """Push event to parent's sink. Returns True if pushed, False if no sink."""
```

### 3. Agent Schemas (YAML)

Agents are defined declaratively in YAML files:

```yaml
type: object
description: |
  You are a helpful assistant with access to a knowledge base.
  Use the search tool to find relevant information.
  Use the action tool to emit typed events (observation, elicit, etc.).

properties:
  answer:
    type: string
    description: Your response to the user

required:
  - answer

json_schema_extra:
  kind: agent
  name: default-agent
  version: "1.0.0"
  tools:
    - name: search
    - name: action
```

#### Schema Structure

| Field | Purpose |
|-------|---------|
| `description` | System prompt - agent's instructions |
| `properties` | Structured output schema |
| `json_schema_extra.kind` | Always `"agent"` |
| `json_schema_extra.name` | Agent identifier |
| `json_schema_extra.version` | Semantic version |
| `json_schema_extra.tools` | MCP tools available to agent |

### 4. Structured Output Pattern

**By default, structured output is DISABLED** for agents. This is intentional because:

1. **Most agents are conversational** - They generate free-form text responses
2. **Structured output constrains creativity** - It forces the LLM into a rigid schema
3. **The `action` tool captures structure** - Agents call this tool to emit typed events

#### The Pattern

Instead of forcing structured output from the agent, we use a hybrid approach:

```yaml
# Agent schema - NO structured output enforced
type: object
description: |
  You are a helpful assistant.
  Use action tool to emit observations (confidence, sources, etc.).

properties:
  answer:
    type: string
    description: Your response

json_schema_extra:
  structured_output: false  # Default - free-form text
  tools:
    - name: action  # Emits typed action events
```

The agent can:
1. Generate natural conversational responses
2. Call `action(type="observation", payload={"confidence": 0.85, "sources": ["doc-1"]})`
3. Both are streamed - text as content chunks, actions as typed SSE events

#### When to Enable Structured Output

Enable `structured_output: true` only for:
- **Data extraction agents** - Parse documents into schemas
- **Classification agents** - Return specific categories
- **API integration agents** - Generate structured API payloads

```yaml
json_schema_extra:
  structured_output: true  # Force agent output to match properties schema
```

#### Benefits

| Approach | Pros | Cons |
|----------|------|------|
| Structured output | Type-safe, predictable | Constrains responses |
| action tool | Flexible, natural | Requires agent to call tool |

The `action` tool pattern is generally preferred because it:
- Lets agents reason freely
- Captures metadata when relevant
- Streams both text AND structured data
- Works with any LLM provider

### 5. Repository Pattern for DB Access

**CRITICAL**: All database operations MUST use the Repository pattern, not raw SQL.

```python
from remlight.services.repository import Repository
from remlight.models.entities import Message, Session

class SessionMessageStore:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.repo = Repository(Message)
        self._session_repo = Repository(Session, table_name="sessions")

    async def store_message(self, ...):
        msg = Message(...)
        await self.repo.upsert(msg)  # NOT raw SQL

    async def load_messages(self, session_id: str):
        return await self.repo.find(
            {"session_id": session_id, "tenant_id": self.user_id},
            order_by="created_at ASC"
        )
```

### 5. Settings Checks

All database operations MUST check `settings.postgres.enabled`:

```python
from remlight.settings import settings

async def store_message(...):
    if not settings.postgres.enabled:
        logger.debug("Postgres disabled, skipping message storage")
        return
    # ... proceed with storage
```

### 6. Logging with Loguru

All modules MUST use `loguru` for logging:

```python
from loguru import logger

logger.debug(f"Loaded {len(messages)} messages for session {session_id}")
logger.info(f"Created session {session_id} for user {user_id}")
logger.warning(f"Failed to ensure session exists: {e}")
logger.error(f"Failed to load session messages: {e}")
```

## Session Message Flow

### Storage Pattern

1. **All messages stored UNCOMPRESSED** in database for full audit trail
2. **Compression happens only on RELOAD** when reconstructing context for LLM
3. **Tool messages NEVER compressed** - contain structured metadata

### Message Types

| Type | Storage | Compression on Load |
|------|---------|---------------------|
| `user` | As-is | Never |
| `tool` | As-is with metadata | **NEVER** |
| `assistant` | As-is | May compress if >400 chars |

### Tool Message Metadata

Tool messages include call details in `metadata`:

```python
msg_metadata = {
    "message_index": idx,
    "timestamp": message.get("timestamp"),
    "tool_call_id": message.get("tool_call_id"),
    "tool_name": message.get("tool_name"),
    "tool_arguments": message.get("tool_arguments"),  # For parent calls
}
```

### REM LOOKUP Pattern

Long assistant messages get truncated with lookup hint:

```
{start_content}

... [Message truncated - REM LOOKUP session-{id}-msg-{idx} to recover full content] ...

{end_content}
```

## MCP Tools Protocol

### action Tool

Generic action emitter for typed events:

```python
async def action(
    type: str,                              # "observation", "elicit", "delegate"
    payload: dict[str, Any] | None = None,  # Action-specific data
) -> dict[str, Any]:
    """Returns dict with _action_event: True marker for SSE layer."""
```

For `type="observation"`, payload supports:
- `confidence`: float (0.0-1.0)
- `sources`: list[str] - Entity keys used
- `session_name`: str - For UI display
- `references`: list[str] - Doc links
- `flags`: list[str] - Special handling
- `risk_level`: str - low/moderate/high/critical
- `risk_score`: int - 0-100
- `extra`: dict - Extension fields

### ask_agent Tool

Multi-agent orchestration:

```python
async def ask_agent(
    agent_name: str,           # Agent to invoke
    input_text: str,           # User message
    input_data: dict | None,   # Optional structured input
    user_id: str | None,       # Override user
    timeout_seconds: int = 300,
) -> dict:
    # 1. Get parent context via get_current_context()
    # 2. Create child context via parent.child_context()
    # 3. Load session history via SessionMessageStore
    # 4. Convert to pydantic-ai format via session_to_pydantic_messages()
    # 5. Run agent with message_history
    # 6. Stream events to parent via event sink
```

## Streaming Architecture

```
User Request
    │
    ▼
stream_agent_response_with_save()
    │
    ├── Save user message FIRST
    │
    ├── Load session history
    │
    ├── Convert to pydantic-ai messages
    │
    ├── stream_agent_response()
    │   │
    │   ├── Set context via set_current_context()
    │   ├── Set event sink via set_event_sink()
    │   │
    │   ├── agent.iter() loop
    │   │   ├── TextPart → content chunks (skip if child_content_streamed)
    │   │   ├── ToolCallPart → tool_call SSE events
    │   │   └── ToolReturnPart → tool results + action events
    │   │
    │   └── Child events via _stream_with_child_events()
    │       ├── child_content → content chunks (sets child_content_streamed=True)
    │       ├── child_tool_start → tool_call events
    │       └── child_tool_result → action extraction
    │
    └── Save assistant + tool messages AFTER streaming
```

### Child Content Deduplication

**CRITICAL**: When child streams content, set `state.child_content_streamed = True`.
Parent MUST skip its own TextPart events when this flag is True.

## Pydantic Message Reconstruction

`session_to_pydantic_messages()` converts stored format → pydantic-ai native:

```python
# Storage format (simplified):
{"role": "user", "content": "..."}
{"role": "assistant", "content": "..."}
{"role": "tool", "content": "{...}", "tool_name": "...", "tool_call_id": "..."}

# Pydantic-ai format (what LLM expects):
ModelRequest(parts=[UserPromptPart(content="...")])
ModelResponse(parts=[TextPart(content="..."), ToolCallPart(...)])  # SYNTHESIZED
ModelRequest(parts=[ToolReturnPart(...)])
```

**Key insight**: LLM APIs require matching `ToolCallPart` for each `ToolReturnPart`.
We synthesize `ToolCallPart` from stored metadata.

## Required Dependencies

```python
# In all modules
from loguru import logger
from pydantic import BaseModel, Field
from remlight.settings import settings
from remlight.services.repository import Repository
```

## Testing Checklist

- [ ] AgentContext is Pydantic BaseModel
- [ ] AgentContext.from_request() extracts user from JWT
- [ ] AgentContext has is_eval and client_id fields
- [ ] SessionMessageStore uses Repository pattern
- [ ] All DB ops check settings.postgres.enabled
- [ ] All modules use loguru for logging
- [ ] action tool emits typed events
- [ ] ask_agent loads session history
- [ ] Child content deduplication works
- [ ] Tool messages never compressed
