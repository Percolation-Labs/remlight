# REMLight Future Features

This document outlines possible features and their implementation approach.

---

## 1. Tool Response Persistence (`X-TOOL-RESPONSES` Header)

### Overview

By default, REMLight stores tool **calls** (role: "tool") but discards tool **responses**. This feature introduces a mechanism to optionally persist tool responses, controlled via an HTTP header that can be set per-session.

### Current Behavior

- Tool calls are stored with `role: "tool"` in the messages table
- Tool call metadata includes: `tool_call_id`, `tool_name`, `tool_arguments`
- Tool responses (the actual results returned by tools) are used during agent execution but **not persisted**
- When sessions are reloaded, only the tool invocation is visible, not what the tool returned - this is by design because agents can often act and possibly even take note of what they need from a tool response e.g. a large database lookup while not actually saving it into context. This of course is not always safe to do - it is just one pattern.

### Proposed Behavior

When `X-TOOL-RESPONSES: true` is set:
1. Tool responses are stored alongside tool calls
2. When the session is reloaded, tool responses are included in the message history
3. This enables agents/clients to have full context of previous tool executions

### Implementation Details

#### 1. New HTTP Header

| Header | Type | Default | Description |
|--------|------|---------|-------------|
| `X-TOOL-RESPONSES` | boolean | `false` | When `true`, persist tool responses for this session |

#### 2. Settings Default

Add to `remlight/settings.py`:

```python
class SessionSettings(BaseModel):
    keep_tool_responses: bool = os.getenv("SESSION__KEEP_TOOL_RESPONSES", "false").lower() == "true"
```

This provides a system-wide default that can be overridden per-request via the header.

#### 3. AgentContext Extension

Extend `AgentContext` in `remlight/agentic/context.py`:

```python
class AgentContext(BaseModel):
    # ... existing fields ...
    keep_tool_responses: bool = False  # Session-level override
```

Extract from headers in `from_headers_with_profile()`:

```python
# Check header first (session override), fall back to settings default
header_value = headers.get("x-tool-responses", "").lower()
if header_value == "true":
    keep_tool_responses = True
elif header_value == "false":
    keep_tool_responses = False
else:
    keep_tool_responses = settings.session.keep_tool_responses
```

#### 4. Message Storage Changes

In `remlight/services/session/store.py`, modify `store_session_messages()`:

```python
async def store_session_messages(
    self,
    session_id: UUID,
    messages: list[dict],
    keep_tool_responses: bool = False,  # New parameter
    ...
):
    for message in messages:
        if message.get("role") == "tool":
            msg_metadata = {
                "tool_call_id": message["tool_call_id"],
                "tool_name": message["tool_name"],
                "tool_arguments": message["tool_arguments"],
                "agent_schema": message.get("agent_schema"),
                "model": message.get("model"),
            }

            # NEW: Conditionally include tool response
            if keep_tool_responses and "tool_response" in message:
                msg_metadata["tool_response"] = message["tool_response"]
```

#### 5. Streaming Layer Changes

In `remlight/agentic/streaming/core.py`, capture tool responses during execution:

```python
# In the tool execution callback
tool_result = {
    "tool_name": tool_name,
    "tool_id": tool_id,
    "tool_arguments": arguments,
    "tool_response": result,  # Include the actual response
    "status": "completed",
}
tool_calls_out.append(tool_result)
```

Pass `keep_tool_responses` from context through to storage:

```python
await store.store_session_messages(
    session_id=session_id,
    messages=messages_to_store,
    keep_tool_responses=context.keep_tool_responses,
)
```

#### 6. Session Reload Behavior

When loading session messages in `load_session_messages()`:

```python
# Tool responses are loaded from metadata if they were persisted
for msg in messages:
    if msg.role == "tool" and msg.metadata.get("tool_response"):
        # Include tool response in reconstructed message
        loaded_msg["tool_response"] = msg.metadata["tool_response"]
```

### Use Cases

1. **Debugging**: Inspect exactly what tools returned in previous turns
2. **Agent Memory**: Allow agents to reference previous tool outputs without re-executing
3. **Audit Trail**: Complete record of tool interactions for compliance
4. **Client Replay**: Clients can reconstruct full conversation with tool results

### Storage Considerations

- Tool responses can be large (search results, file contents, API responses)
- Consider adding a size limit or truncation strategy
- May want to add compression for large responses
- Database `metadata` JSONB column can handle variable-size responses

### API Example

```bash
# Enable tool response persistence for this request/session
curl -X POST /api/v1/chat/completions \
  -H "X-Session-Id: 550e8400-e29b-41d4-a716-446655440000" \
  -H "X-TOOL-RESPONSES: true" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Search for Python tutorials"}]}'
```

### Migration Notes

- No database schema changes required (uses existing JSONB `metadata` column)
- Backwards compatible: existing sessions continue to work without tool responses
- New sessions can opt-in via header

---

## 2. OAuth Implementation

*Details to be added.*

---

## 3. Document Parsing Extraction from REM

*Details to be added.*

---
