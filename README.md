# REMLight

Minimal declarative agent framework with PostgreSQL memory and multi-agent support.

## Features

- **Declarative Agents**: Define agents in YAML with JSON Schema
- **PostgreSQL Memory**: Vector search, graph traversal, fuzzy matching
- **MCP Server**: Tools for `search`, `action`, and `ask_agent`
- **Multi-Agent Orchestration**: Child agents stream through parent
- **OpenAI-compatible API**: Streaming SSE responses
- **CLI**: `ask` command for agent queries

## Quick Start

### Docker

```bash
# Set your API key
export OPENAI_API_KEY=your-key

# Start services
docker-compose up -d

# API available at http://localhost:8000
```

### Local Development

```bash
# Install
pip install -e .

# Start PostgreSQL with pgvector
docker run -d --name remlight-pg \
  -e POSTGRES_DB=remlight \
  -e POSTGRES_USER=remlight \
  -e POSTGRES_PASSWORD=remlight \
  -p 5432:5432 \
  pgvector/pgvector:pg16

# Install schema
rem install

# Start server
rem serve
```

## CLI Commands

```bash
# Ask an agent
rem ask "What is machine learning?"
rem ask "Find documents about AI" --schema schemas/query-agent.yaml

# Execute REM query
rem query "LOOKUP sarah-chen"
rem query "SEARCH projects IN ontologies"
rem query "FUZZY alpha project"

# Start API server
rem serve

# Install database schema
rem install

# Start MCP server (stdio mode for Claude Desktop)
rem mcp-serve
```

## Agent Schema (YAML)

```yaml
type: object
description: |
  You are a helpful assistant...

properties:
  answer:
    type: string
    description: Your response

required:
  - answer

json_schema_extra:
  kind: agent
  name: my-agent
  version: 1.0.0
  tools:
    - name: search
    - name: action
    - name: ask_agent
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/chat/completions` | POST | OpenAI-compatible chat |
| `/api/v1/query` | POST | Execute REM query |
| `/api/v1/mcp` | POST | MCP protocol |

## MCP Tools

### `search`

Execute REM queries:

```
LOOKUP <key>           - O(1) exact lookup
SEARCH <text> IN <table>  - Semantic search
FUZZY <text>           - Fuzzy text matching
TRAVERSE <key>         - Graph traversal
```

### `action`

Register response metadata:

```json
{
  "sources": ["entity-1", "entity-2"],
  "confidence": 0.85,
  "session_name": "My Session"
}
```

### `ask_agent`

Invoke another agent (multi-agent orchestration):

```python
ask_agent(
    agent_name="query-agent",
    input_text="Find documents about machine learning"
)
```

Features:
- Child agents inherit parent's context (user_id, session_id)
- Child agent content streams through parent's SSE connection
- Supports orchestrator, workflow, and ensemble patterns

## Multi-Agent Streaming

When a parent agent calls `ask_agent`, child events are streamed in real-time:

```
Parent Agent
    │
    ▼
ask_agent("query-agent", "Find docs")
    │
    ├── child_tool_start ──────► SSE: tool_call event
    ├── child_content ─────────► SSE: content chunks
    └── child_tool_result ─────► SSE: tool_call complete
```

The streaming architecture prevents content duplication - when child content is streamed, parent text output is skipped.

## Database Tables

- `ontologies` - Domain entities (people, projects, concepts)
- `resources` - Documents and content chunks
- `sessions` - Conversation sessions
- `messages` - Chat messages
- `kv_store` - Key-value lookup cache (regular table, not unlogged)
- `embedding_queue` - Queue for async embedding generation

## REM Functions (PostgreSQL)

- `rem_lookup(key, user_id)` - O(1) KV store lookup
- `rem_search(embedding, table, limit, min_sim, user_id)` - Vector search
- `rem_fuzzy(text, user_id, threshold, limit)` - Trigram matching
- `rem_traverse(key, edge_types, depth, user_id)` - Graph traversal

## Environment Variables

```bash
POSTGRES__CONNECTION_STRING=postgresql://user:pass@host:5432/db
LLM__DEFAULT_MODEL=openai:gpt-4o-mini
LLM__TEMPERATURE=0.5
LLM__MAX_ITERATIONS=20
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
ENVIRONMENT=development
```

## Project Structure

```
remlight/
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── sql/
│   └── install.sql          # Tables, triggers, functions
├── schemas/
│   └── query-agent.yaml     # Sample agent
└── remlight/
    ├── settings.py          # Environment config
    ├── models/
    │   ├── core.py          # CoreModel base
    │   └── entities.py      # Ontology, Resource, Session, Message
    ├── agentic/
    │   ├── schema.py        # AgentSchema + YAML
    │   ├── provider.py      # Pydantic AI agent creation
    │   └── context.py       # AgentContext + event sink
    ├── api/
    │   ├── main.py          # FastAPI app
    │   ├── mcp_main.py      # MCP server with search, action, ask_agent
    │   ├── streaming.py     # SSE with child streaming
    │   └── routers/         # API endpoints (chat, query, tools)
    ├── cli/
    │   └── main.py          # ask, query, serve, install
    └── services/
        └── database.py      # PostgreSQL + rem_* functions
```

## License

MIT
