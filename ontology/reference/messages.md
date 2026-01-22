---
entity_key: messages
title: Message Persistence
tags: [reference, messages, sessions, postgres]
---

# Message Persistence

How conversation messages are stored in PostgreSQL and reconstructed for LLM replay.

## PostgreSQL Schema

### Sessions Table

```sql
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(512),
    description TEXT,
    agent_name VARCHAR(256),
    status VARCHAR(64) DEFAULT 'active',
    graph_edges JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    user_id VARCHAR(256),
    tenant_id VARCHAR(256),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);
```

### Messages Table

```sql
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    role VARCHAR(64) NOT NULL,  -- 'user', 'assistant', 'tool'
    content TEXT NOT NULL,
    tool_calls JSONB DEFAULT '[]',
    graph_edges JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    embedding VECTOR(1536),
    trace_id VARCHAR(256),
    span_id VARCHAR(256),
    user_id VARCHAR(256),
    tenant_id VARCHAR(256),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);
```

## Message Roles

| Role | Description | Stored | Compressed |
|------|-------------|--------|------------|
| `user` | User input | Yes | No |
| `assistant` | Agent response | Yes | Yes (if long) |
| `tool` | Tool call result | Yes | Never |
| `system` | System prompt | No | N/A |

## Storage Format

Messages stored as JSON in the `content` and `tool_calls` columns:

### User Message
```json
{
    "role": "user",
    "content": "Find documents about machine learning",
    "timestamp": "2024-01-15T10:30:00Z"
}
```

### Assistant Message
```json
{
    "role": "assistant",
    "content": "I found 3 relevant documents about machine learning..."
}
```

### Tool Message
```json
{
    "role": "tool",
    "content": "{\"results\": [...], \"total\": 3}",
    "tool_name": "search",
    "tool_call_id": "call_abc123",
    "tool_arguments": {"query": "SEARCH ML IN ontology"}
}
```

## Tool Calls Column

The `tool_calls` JSONB column stores tool invocations for replay:

```json
[
    {
        "tool_name": "search",
        "tool_call_id": "call_abc123",
        "arguments": {"query": "SEARCH ML IN ontology", "limit": 10},
        "result": {"results": [...], "total": 3}
    }
]
```

## Compression on Load

Long assistant messages are compressed with REM LOOKUP hints for context efficiency:

```text
[First 200 chars of the message]...

... [Message truncated - REM LOOKUP session-abc-msg-5 to recover full content] ...

...[Last 200 chars]
```

**Important:** Tool messages are NEVER compressed - they contain structured metadata needed for accurate replay.

## Pydantic-AI Format Conversion

Stored messages are converted to pydantic-ai types for LLM replay:

### User → ModelRequest
```python
ModelRequest(parts=[UserPromptPart(content="Find documents about ML")])
```

### Assistant with Tool Call → ModelResponse
```python
ModelResponse(parts=[
    ToolCallPart(tool_name="search", args={"query": "..."}, tool_call_id="call_abc"),
    TextPart(content="I found 3 documents...")
])
```

### Tool Result → ModelRequest
```python
ModelRequest(parts=[
    ToolReturnPart(tool_name="search", content={"results": [...]}, tool_call_id="call_abc")
])
```

## Session Store API

```python
from remlight.services.session import SessionMessageStore

store = SessionMessageStore(user_id="user-123")

# Store messages (uncompressed in DB)
await store.store_session_messages(
    session_id="sess-456",
    messages=[
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
)

# Load with optional compression
history = await store.load_session_messages(
    session_id="sess-456",
    compress_on_load=True  # Compress long assistant messages
)

# Convert to pydantic-ai format
from remlight.services.session import session_to_pydantic_messages
pydantic_messages = session_to_pydantic_messages(
    history,
    system_prompt="You are a helpful assistant..."
)
```

## Embedding Support

Messages have optional embeddings for semantic search:

```sql
-- Find similar past conversations
SELECT m.*, 1 - (m.embedding <=> query_embedding) as similarity
FROM messages m
WHERE m.embedding IS NOT NULL
  AND m.user_id = 'user-123'
ORDER BY m.embedding <=> query_embedding
LIMIT 10;
```

Embeddings are automatically queued for generation via trigger:

```sql
CREATE TRIGGER messages_embedding_queue
AFTER INSERT OR UPDATE OF content ON messages
FOR EACH ROW EXECUTE FUNCTION queue_message_embedding();
```

## See also

- `REM LOOKUP sessions` - Session management
- `REM LOOKUP sse-events` - Streaming protocol
- `REM LOOKUP headers` - Request context headers
