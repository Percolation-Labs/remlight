# REMLight API

This module provides both a **FastAPI REST server** and an **MCP server** that share the same tool implementations.

## Architecture

```
api/
├── README.md           # This file
├── __init__.py         # Module exports
├── main.py             # FastAPI application (REST API)
├── mcp_main.py         # MCP server (stdio + HTTP)
└── routers/
    ├── __init__.py     # Router exports
    ├── chat.py         # OpenAI-compatible chat completions
    ├── query.py        # REM query endpoint
    ├── tools.py        # Tool functions (shared with MCP)
    ├── agents.py       # List available agent schemas
    ├── sessions.py     # Session CRUD and message retrieval
    ├── models.py       # List available LLM models
    └── scenarios.py    # Scenario management and feedback
```

## Design Principles

### 1. Single Source of Truth for Tools

Tools are defined **once** in `routers/tools.py` as async functions:

```python
# routers/tools.py
async def search(query: str, limit: int = 20, user_id: str | None = None) -> dict:
    """Execute REM queries to search the knowledge base."""
    ...
```

These functions are then:
- **Exposed as REST endpoints** via FastAPI router
- **Registered with MCP** via `mcp_main.py`
- **Callable directly** from Python code

### 2. Router-Style Tool Definitions

Tools follow FastAPI router conventions, making them testable and reusable:

```python
# Can be called directly
result = await search("LOOKUP sarah-chen")

# Or via REST API
# POST /api/v1/tools/search?query=LOOKUP+sarah-chen

# Or via MCP
# Tools available in Claude Desktop, etc.
```

### 3. Clean Separation

| File | Responsibility |
|------|----------------|
| `main.py` | FastAPI app, CORS, routing |
| `mcp_main.py` | MCP server, tool registration |
| `routers/tools.py` | Tool implementations (search, action, ask_agent) |
| `routers/chat.py` | OpenAI-compatible chat completions |
| `routers/query.py` | Direct REM query execution |
| `routers/agents.py` | List and get agent schemas |
| `routers/sessions.py` | Session CRUD and message retrieval |
| `routers/models.py` | List available LLM models |
| `routers/scenarios.py` | Scenario management and Phoenix feedback |

## Endpoints

### REST API (FastAPI)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/chat/completions` | POST | OpenAI-compatible chat |
| `/api/v1/query` | POST | REM query execution |
| `/api/v1/tools/search` | POST | Search tool |
| `/api/v1/tools/parse-file` | POST | Parse file and extract content |
| `/api/v1/agents` | GET | List available agent schemas |
| `/api/v1/agents/{name}` | GET | Get specific agent |
| `/api/v1/sessions` | GET | List user sessions |
| `/api/v1/sessions/{id}/messages` | GET | Get session messages |
| `/api/v1/sessions/{id}/export` | GET | Export session as YAML |
| `/api/v1/models` | GET | List available LLM models |
| `/api/v1/scenarios` | GET/POST | Scenario CRUD |
| `/api/v1/scenarios/feedback` | POST | Phoenix feedback |
| `/api/v1/mcp/*` | * | MCP HTTP endpoint |
| `/docs` | GET | OpenAPI documentation |

### MCP Tools

| Tool | Description |
|------|-------------|
| `search` | Execute REM queries (LOOKUP, FUZZY, SEARCH, TRAVERSE) |
| `action` | Emit typed action events (observation, elicit, delegate) for SSE streaming |
| `ask_agent` | Multi-agent orchestration |
| `parse_file` | Parse files (PDF, DOCX, images) and extract content |

### MCP Resources

| Resource URI | Description |
|--------------|-------------|
| `user://profile/{user_id}` | Load user profile |
| `project://{project_key}` | Load project details (JSON) |
| `rem://status` | System status |

## Running

### FastAPI Server (REST + MCP HTTP)

```bash
# Development
uvicorn remlight.api.main:app --reload

# Production
uvicorn remlight.api.main:app --host 0.0.0.0 --port 8000
```

### MCP Server (stdio mode)

For use with MCP clients like Claude Desktop:

```bash
python -m remlight.api.mcp_main
```

Or add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "remlight": {
      "command": "python",
      "args": ["-m", "remlight.api.mcp_main"]
    }
  }
}
```

## Usage Examples

### Direct Tool Usage

```python
from remlight.api.routers import search, action, ask_agent

# Search the knowledge base
result = await search("LOOKUP sarah-chen")

# Emit an observation action
await action(type="observation", payload={"confidence": 0.85, "sources": ["doc-1"]})

# Invoke another agent
response = await ask_agent(
    agent_name="query-agent",
    input_text="Find documents about machine learning"
)
```

### Chat Completions (OpenAI-compatible)

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/v1/chat/completions",
        headers={
            "X-User-Id": "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",  # Must be a valid UUID
            "X-Session-Id": "550e8400-e29b-41d4-a716-446655440000",  # Must be a valid UUID
        },
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
    )
```

### REM Query

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/api/v1/query",
        json={
            "query": "SEARCH machine learning IN ontologies",
            "limit": 10
        }
    )
```

### Parse File

Parse and extract content from documents (PDF, DOCX, images, etc.):

```python
import httpx

async with httpx.AsyncClient() as client:
    # Parse a local file
    response = await client.post(
        "http://localhost:8000/api/v1/tools/parse-file",
        json={"uri": "./document.pdf"}
    )

    # Parse from S3
    response = await client.post(
        "http://localhost:8000/api/v1/tools/parse-file",
        json={"uri": "s3://bucket/report.docx"}
    )

    # Parse from URL (don't save to database)
    response = await client.post(
        "http://localhost:8000/api/v1/tools/parse-file",
        json={
            "uri": "https://example.com/paper.pdf",
            "save_to_db": False
        }
    )
```

**Note**: Files are stored globally by default. Avoid setting `user_id` unless you specifically need per-user file isolation.

## Standard Headers

REMLight uses standard HTTP headers for context propagation. These headers are automatically extracted by `AgentContext.from_headers()` and propagated through agent invocations.

### Header Reference

| Header | Type | Default | Description |
|--------|------|---------|-------------|
| `X-User-Id` | UUID | `null` | **MUST be a valid UUID.** User identifier for scoping, personalization, and access control. Generate with `uuid.uuid4()` (Python) or `crypto.randomUUID()` (JS). Prefer JWT-based auth when available. |
| `X-Session-Id` | UUID | `null` | **MUST be a valid UUID.** Session/conversation identifier for multi-turn context. Messages are stored and retrieved by session. Generate with `uuid.uuid4()` (Python) or `crypto.randomUUID()` (JS). |
| `X-Tenant-Id` | string | `"default"` | Tenant identifier for multi-tenancy isolation. Used for REM data partitioning. |
| `X-Agent-Schema` | string | `null` | Agent schema name or file path. Determines which agent handles the request. |
| `X-Model-Name` | string | config default | LLM model override (e.g., `openai:gpt-4.1`, `anthropic:claude-sonnet-4-5-20250929`). |
| `X-Client-Id` | string | `null` | Client identifier for analytics (e.g., `web`, `mobile`, `cli`, `api`). |
| `X-Is-Eval` | boolean | `false` | Marks session as evaluation. Accepts `true`, `1`, or `yes` as truthy. |

### Header Propagation

Headers are automatically propagated through the request lifecycle:

1. **API Request** → `AgentContext.from_headers()` extracts values
2. **Parent Agent** → Context is passed to agent creation
3. **Child Agents** → Context is inherited via `ask_agent` tool
4. **Session Storage** → Messages are stored with context (user_id, session_id)

### Security Considerations

- **Production**: Use JWT-based authentication. User ID is extracted from `request.state.user.id` (set by auth middleware).
- **Development**: `X-User-Id` header fallback for backwards compatibility.
- **Multi-tenancy**: `X-Tenant-Id` isolates data between tenants.

### Example Request

```bash
curl -X POST http://localhost:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-User-Id: a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11" \
  -H "X-Session-Id: 550e8400-e29b-41d4-a716-446655440000" \
  -H "X-Tenant-Id: acme-corp" \
  -H "X-Agent-Schema: query-agent" \
  -H "X-Client-Id: web" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

### Multi-Agent SSE Streaming

Test multi-agent orchestration with SSE streaming:

```bash
# Start the server first: rem serve

# Test orchestrator agent (delegates to worker-agent)
curl -N -X POST http://localhost:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Agent-Schema: orchestrator-agent" \
  -d '{"messages": [{"role": "user", "content": "Count to 3"}], "stream": true}'
```

Expected SSE events:

```
event: progress
data: {"type":"progress","step":1,"total_steps":3,"label":"Processing request"}

event: tool_call
data: {"type":"tool_call","tool_name":"ask_agent","status":"started",...}

data: {"choices":[{"delta":{"content":"The worker..."}}]}

event: tool_call
data: {"type":"tool_call","tool_name":"ask_agent","status":"completed",...}

event: done
data: {"type":"done","reason":"stop"}

data: [DONE]
```

CLI equivalent:

```bash
rem ask "Count to 3" --schema orchestrator-agent
```

## Adding New Tools

1. Add the tool function to `routers/tools.py`:

```python
async def my_tool(param: str) -> dict[str, Any]:
    """My tool description."""
    return {"result": param}
```

2. Add REST endpoint wrapper (optional):

```python
@router.post("/my-tool")
async def my_tool_endpoint(param: str) -> dict[str, Any]:
    return await my_tool(param)
```

3. Register with MCP in `mcp_main.py`:

```python
mcp.tool(name="my_tool")(my_tool)
```

4. Export from `routers/__init__.py`:

```python
from remlight.api.routers.tools import my_tool
__all__ = [..., "my_tool"]
```

## Testing

Tools can be tested directly without HTTP:

```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_search():
    with patch("remlight.api.routers.tools.get_tools_db") as mock_db:
        mock_db.return_value.rem_lookup = AsyncMock(return_value={"key": "value"})

        result = await search("LOOKUP test-key")

        assert result["status"] == "success"
        assert result["query_type"] == "LOOKUP"
```
