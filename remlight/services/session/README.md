# Session Message Storage

This module implements session message persistence and reconstruction for multi-turn conversations and multi-agent context sharing.

**Reference**: `remstack/rem/src/rem/services/session/`

## Design Principles

### 1. Store Uncompressed, Compress on Load

```
STORE: All messages → Database (UNCOMPRESSED)
       ↓
       Full audit trail preserved
       ↓
LOAD:  Database → Messages (COMPRESS long assistant messages)
       ↓
       Efficient LLM context window
```

### 2. Never Compress Tool Messages

Tool messages contain structured metadata that MUST stay intact:
- `tool_call_id`
- `tool_name`
- `tool_arguments`
- Result content (may contain `_action_event`)

### 3. REM LOOKUP for Full Recovery

Long assistant messages get truncated with lookup hint:

```
{first 200 chars}

... [Message truncated - REM LOOKUP session-{id}-msg-{idx} to recover full content] ...

{last 200 chars}
```

Agent can retrieve full content on-demand using the LOOKUP key.

## Components

### MessageCompressor

```python
class MessageCompressor:
    def __init__(self, truncate_length: int = 200):
        self.truncate_length = truncate_length
        self.min_length_for_compression = truncate_length * 2  # 400 chars

    def compress_message(self, message: dict, entity_key: str | None) -> dict:
        """Truncate long content, add REM LOOKUP hint."""

    def decompress_message(self, message: dict, full_content: str) -> dict:
        """Restore full content from compressed message."""

    def is_compressed(self, message: dict) -> bool:
        """Check if message has _compressed flag."""

    def get_entity_key(self, message: dict) -> str | None:
        """Get REM lookup key from compressed message."""
```

### SessionMessageStore

**MUST use Repository pattern**:

```python
from remlight.services.repository import Repository
from remlight.models.entities import Message, Session

class SessionMessageStore:
    def __init__(self, user_id: str, compressor: MessageCompressor | None = None):
        self.user_id = user_id
        self.compressor = compressor or MessageCompressor()
        self._message_repo = Repository(Message)
        self._session_repo = Repository(Session)
```

#### Methods

```python
async def _ensure_session_exists(self, session_id: str, user_id: str | None) -> None:
    """Create session if not exists - best effort, don't fail."""

async def store_message(self, session_id: str, message: dict, message_index: int, user_id: str | None) -> str:
    """Store individual message, return entity_key for LOOKUP."""

async def retrieve_message(self, entity_key: str) -> str | None:
    """Retrieve full message content by REM lookup key."""

async def store_session_messages(self, session_id: str, messages: list[dict], user_id: str | None, compress: bool = False) -> list[dict]:
    """Store all session messages, return optionally compressed versions."""

async def load_session_messages(self, session_id: str, user_id: str | None, compress_on_load: bool = True) -> list[dict]:
    """Load messages from database, optionally compress long assistant messages."""

async def retrieve_full_message(self, session_id: str, message_index: int) -> str | None:
    """Retrieve full content by session and index (convenience for REM LOOKUP)."""
```

### session_to_pydantic_messages

Converts stored format to pydantic-ai native format:

```python
def session_to_pydantic_messages(
    session_history: list[dict],
    system_prompt: str | None = None,
) -> list[ModelMessage]:
    """
    IMPORTANT: pydantic-ai only auto-adds system prompts when message_history is empty.
    You MUST pass system_prompt here for multi-turn conversations.

    Tool arguments sources:
    - Parent tool calls (ask_agent): tool_arguments in metadata
    - Child tool calls (action): parsed from content JSON
    """
```

#### Conversion Logic

```python
# Input (storage format):
{"role": "user", "content": "Hello"}
{"role": "assistant", "content": "Hi there"}
{"role": "tool", "content": "{\"result\": ...}", "tool_name": "search", "tool_call_id": "call_123"}

# Output (pydantic-ai format):
ModelRequest(parts=[SystemPromptPart(content="...")])  # If system_prompt provided
ModelRequest(parts=[UserPromptPart(content="Hello")])
ModelResponse(parts=[
    ToolCallPart(tool_name="search", args={...}, tool_call_id="call_123"),  # SYNTHESIZED
    TextPart(content="Hi there")
], model_name="recovered")
ModelRequest(parts=[ToolReturnPart(tool_name="search", content={...}, tool_call_id="call_123")])
```

### audit_session_history

Debug function for dumping session history:

```python
def audit_session_history(
    session_id: str,
    agent_name: str,
    prompt: str,
    raw_session_history: list[dict],
    pydantic_messages_count: int,
) -> None:
    """
    Only runs when DEBUG_AUDIT_SESSION environment variable is set.
    Writes to DEBUG_AUDIT_DIR env var path (default /tmp).
    """
```

## Message Metadata Schema

### User Messages

```python
{
    "role": "user",
    "content": "User's message",
    "timestamp": "2024-01-15T10:30:00Z",
}
```

### Assistant Messages

```python
{
    "role": "assistant",
    "content": "Assistant's response",
    "timestamp": "2024-01-15T10:30:05Z",
    "id": "pre-generated-uuid",  # Optional, for frontend feedback
}
```

### Tool Messages

```python
{
    "role": "tool",
    "content": "{\"status\": \"success\", ...}",  # JSON string
    "timestamp": "2024-01-15T10:30:03Z",
    "tool_call_id": "call_abc123",
    "tool_name": "action",
    "tool_arguments": {"type": "observation", "payload": {"confidence": 0.85}},  # Optional
}
```

## Database Schema

Messages table:

```sql
CREATE TABLE messages (
    id UUID PRIMARY KEY,
    session_id UUID REFERENCES sessions(id),
    role VARCHAR(50),  -- 'user', 'assistant', 'tool'
    content TEXT,
    user_id VARCHAR(255),
    tenant_id VARCHAR(255),
    metadata JSONB,
    trace_id VARCHAR(255),
    span_id VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    deleted_at TIMESTAMPTZ
);
```

Sessions table:

```sql
CREATE TABLE sessions (
    id UUID PRIMARY KEY,
    name VARCHAR(255),
    user_id VARCHAR(255),
    tenant_id VARCHAR(255),
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    deleted_at TIMESTAMPTZ
);
```

## Usage Pattern

```python
from remlight.services.session import (
    SessionMessageStore,
    session_to_pydantic_messages,
)

# In API endpoint:
store = SessionMessageStore(user_id=context.user_id or "anonymous")

# Save user message BEFORE agent execution
await store.store_session_messages(session_id, [{"role": "user", "content": prompt}])

# Load history for agent
raw_history = await store.load_session_messages(session_id, compress_on_load=True)
pydantic_history = session_to_pydantic_messages(raw_history, system_prompt=agent.system_prompt)

# Run agent with history
result = await agent.run(prompt, message_history=pydantic_history)

# Save assistant response AFTER execution
await store.store_session_messages(session_id, [{"role": "assistant", "content": result}])
```

## Settings Integration

```python
import os
from remlight.settings import settings

# All DB operations must check this
if not settings.postgres.enabled:
    logger.debug("Postgres disabled, skipping storage")
    return []

# Audit logging (enabled via environment variable)
if os.environ.get("DEBUG_AUDIT_SESSION"):
    audit_session_history(...)
```

## Error Handling

- Session creation is **best-effort** - don't fail on errors
- Message loading returns **empty list** on errors
- Always log errors with `logger.error()`
- Never let persistence failures break streaming
