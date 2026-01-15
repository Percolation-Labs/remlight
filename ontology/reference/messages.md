---
entity_key: messages
title: Message Persistence
tags: [reference, messages, sessions]
---

# Message Persistence

How conversation messages are stored and reconstructed.

## Message Roles

| Role | Description | Compressed |
| ---- | ----------- | ---------- |
| `user` | User input | No |
| `assistant` | Agent response | Yes (if long) |
| `tool` | Tool call result | Never |
| `system` | System prompt | Not stored |

## Storage Format

Messages stored uncompressed in `messages` table:

```python
{
    "role": "user",
    "content": "Find documents about ML",
    "timestamp": "2024-01-15T10:30:00Z"
}

{
    "role": "assistant",
    "content": "I found 3 relevant documents..."
}

{
    "role": "tool",
    "content": "{\"results\": [...]}",
    "tool_name": "search",
    "tool_call_id": "call_abc123",
    "tool_arguments": {"query": "SEARCH ML IN ontology"}
}
```

## Compression on Load

Long assistant messages compressed with REM LOOKUP hints:

```text
[First 200 chars]...

... [Message truncated - REM LOOKUP session-abc-msg-5 to recover full content] ...

...[Last 200 chars]
```

Tool messages are NEVER compressed - contain structured metadata.

## Pydantic-AI Format

Stored messages converted to pydantic-ai types for LLM replay:

```python
# User message
ModelRequest(parts=[UserPromptPart(content="...")])

# Assistant with tool call
ModelResponse(parts=[
    ToolCallPart(tool_name="search", args={...}, tool_call_id="call_abc"),
    TextPart(content="I found...")
])

# Tool result
ModelRequest(parts=[
    ToolReturnPart(tool_name="search", content={...}, tool_call_id="call_abc")
])
```

## Session Store API

```python
from remlight.services.session import SessionMessageStore

store = SessionMessageStore(user_id="user-123")

# Store messages (uncompressed in DB)
await store.store_session_messages(session_id, messages)

# Load with optional compression
history = await store.load_session_messages(
    session_id,
    compress_on_load=True
)
```

## See also

- `REM LOOKUP entities` - Entity types including Message
- `REM LOOKUP architecture` - System data flow
- `REM LOOKUP multi-agent` - Multi-turn conversation examples
