---
entity_key: headers
title: HTTP Headers
tags: [reference, headers, api, protocols]
---

# HTTP Headers

REM uses HTTP headers for context propagation across API requests.

## Session ID

Session ID can be provided in two ways (URL parameter takes precedence):

1. **URL path (preferred)**: `POST /api/v1/chat/completions/{session_id}`
2. **Header (fallback)**: `X-Session-Id`

## Request Headers

| Header | Type | Required | Description |
|--------|------|----------|-------------|
| `X-User-Id` | string | No | User identifier (UUID5 hash of email). Fallback if JWT not present. |
| `X-Tenant-Id` | string | No | Tenant/org identifier for data isolation. Default: `"default"` |
| `X-Session-Id` | string | No | Session ID (fallback - prefer URL path parameter) |
| `X-Agent-Schema` | string | No | Agent schema name (e.g., `"orchestrator-agent"`) |
| `X-Model-Name` | string | No | LLM model override (e.g., `"openai:gpt-4.1"`) |
| `X-Is-Eval` | boolean | No | Mark as evaluation session (`"true"`, `"1"`, `"yes"`) |
| `X-Client-Id` | string | No | Client application identifier (`"web"`, `"mobile"`, `"cli"`) |

## Header Flow

```text
HTTP Request
     │
     ▼
┌─────────────────────────────────┐
│  AgentContext.from_request()    │
│                                 │
│  Priority for user_id:          │
│  1. JWT token (secure)          │
│  2. X-User-Id header (fallback) │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  AgentContext                   │
│  - user_id                      │
│  - tenant_id                    │
│  - session_id                   │
│  - agent_schema_uri             │
│  - default_model                │
└──────────────┬──────────────────┘
               │
               ▼
        Agent Execution
```

## Example: Chat Request

**Using URL path for session ID (preferred):**

```bash
curl -N -X POST http://localhost:8000/api/v1/chat/completions/sess-456 \
  -H "Content-Type: application/json" \
  -H "X-User-Id: user-123" \
  -H "X-Agent-Schema: orchestrator-agent" \
  -H "X-Model-Name: anthropic:claude-sonnet-4-5-20250929" \
  -d '{
    "messages": [{"role": "user", "content": "Research neural networks"}],
    "stream": true
  }'
```

**Using header for session ID (fallback):**

```bash
curl -N -X POST http://localhost:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-User-Id: user-123" \
  -H "X-Session-Id: sess-456" \
  -H "X-Agent-Schema: orchestrator-agent" \
  -H "X-Model-Name: anthropic:claude-sonnet-4-5-20250929" \
  -d '{
    "messages": [{"role": "user", "content": "Research neural networks"}],
    "stream": true
  }'
```

## Context Inheritance (Multi-Agent)

When parent agents delegate via `ask_agent`, child agents inherit:

| Field | Inherited | Can Override |
|-------|-----------|--------------|
| `user_id` | Yes | No |
| `tenant_id` | Yes | No |
| `session_id` | Yes | No |
| `agent_schema_uri` | No | Yes (different agent) |
| `default_model` | No | Yes (cheaper model for subtasks) |

```python
# In ask_agent tool
parent_context = get_current_context()
child_context = parent_context.child_context(
    agent_schema_uri="sentiment-analyzer",
    model_override="openai:gpt-4.1-mini",
)
```

## Anonymous Access

When `user_id` is None (unauthenticated):

- Database queries use `WHERE user_id IS NULL`
- Returns **shared/public data only**
- User cannot see other users' private data
- No fake IDs generated (explicit None handling)

## See also

- `REM LOOKUP sse-events` - SSE streaming protocol
- `REM LOOKUP messages` - Message persistence
- `REM LOOKUP multi-agent` - Multi-agent orchestration
