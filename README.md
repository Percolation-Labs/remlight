# REMLight

Minimal declarative agent framework with PostgreSQL memory and multi-agent support.

Read the article: [A Really Simple Declarative Agent Framework (Part I)](https://medium.com/@mrsirsh/a-really-simply-declarative-agent-framework-part-i-6ae2b05fb2a1)

## Features

- **Declarative Agents**: Define agents in YAML with JSON Schema
- **PostgreSQL Memory**: Vector search, graph traversal, fuzzy matching
- **MCP Server**: Tools for `search`, `action`, and `ask_agent`
- **Multi-Agent Orchestration**: Child agents stream through parent
- **OpenAI-compatible API**: Streaming SSE responses
- **CLI**: `ask`, `query`, and `ingest` commands
- **Sample Ontology**: Self-documenting knowledge base

## Quick Start

### Docker

```bash
# Set your API key
export OPENAI_API_KEY=your-key

# Start services (PostgreSQL, Phoenix, API)
docker-compose up -d

# API available at http://localhost:8080
# Phoenix UI at http://localhost:6016
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
  pgvector/pgvector:pg18

# Install schema
rem install

# Ingest sample ontology
rem ingest ontology/

# Start server
rem serve
```

## CLI Commands

```bash
# Ask an agent
rem ask "What is machine learning?"
rem ask "Find documents" --schema query-agent

# Multi-turn conversations (session must be a UUID)
rem ask "What is REMLight?" --session 550e8400-e29b-41d4-a716-446655440000
rem ask "Tell me more" --session 550e8400-e29b-41d4-a716-446655440000

# Multi-agent orchestration
rem ask "Count to 5" --schema orchestrator-agent

# Execute REM query
rem query "LOOKUP architecture"
rem query "SEARCH agent schema IN ontology"
rem query "FUZZY query"

# Ingest ontology files
rem ingest ontology/
rem ingest ontology/ --dry-run

# Start API server
rem serve

# Install database schema
rem install
```

## REM Query Examples

After ingesting the ontology, test REM queries:

### LOOKUP - O(1) Exact Key Match

```bash
# Lookup by entity_key (from YAML frontmatter)
rem query "LOOKUP architecture"
rem query "LOOKUP rem-query"
rem query "LOOKUP cli"
rem query "LOOKUP multi-agent"
```

Returns exact match from kv_store in O(1) time.

### SEARCH - Semantic Vector Search

```bash
# Vector similarity search in ontology table
rem query "SEARCH agent schema IN ontology"
rem query "SEARCH message persistence IN ontology"
rem query "SEARCH multi-agent orchestration IN ontology"
```

Returns top results ranked by embedding cosine similarity.

### FUZZY - Trigram Text Matching

```bash
# Fuzzy text search (handles typos)
rem query "FUZZY archtecture"  # typo: archtecture
rem query "FUZZY mesages"      # typo: mesages
```

Uses PostgreSQL trigram similarity for approximate matching.

## Sample Ontology

REMLight includes self-documenting ontology with interlinked entities:

```
ontology/
├── README.md                # Main index
├── design/
│   ├── architecture.md      # System architecture
│   └── entities.md          # Database entity types
├── reference/
│   ├── rem-query.md         # Query language spec
│   ├── agent-schema.md      # YAML schema format
│   ├── mcp-tools.md         # Available tools
│   └── messages.md          # Message persistence
└── guides/
    ├── quick-start.md       # Installation
    ├── cli.md               # CLI commands
    └── multi-agent.md       # Multi-agent orchestration
```

### Entity Linking

Each page has a "See also" section with REM LOOKUP hints for graph traversal:

```markdown
## See also

- `REM LOOKUP architecture` - System architecture
- `REM LOOKUP rem-query` - Query language reference
```

## How the Ontology Loader Works

The `rem ingest` command processes markdown files:

1. **Parse Frontmatter**: Extract YAML metadata (entity_key, title, tags, etc.)
2. **Generate Entity Key**: Uses `entity_key` from frontmatter, or filename
3. **Store Content**: Full markdown stored in `ontology` table
4. **Generate Embeddings**: Content embedded for SEARCH queries
5. **Index for LOOKUP**: Entity key indexed in `kv_store` for O(1) access

```
rem ingest ontology/
    │
    ├── Parse: architecture.md
    │   └── entity_key: architecture
    │   └── content: Full markdown
    │   └── properties: {tags, related}
    │
    ├── Store in ontology table
    │   └── Generate embedding for content
    │
    └── Index in kv_store
        └── key: architecture → {name, content, metadata}
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

## Chat Client App

REMLight includes a React-based chat client for testing agents interactively.

### Running the Chat Client

```bash
# Start the API server
rem serve --port 8000

# In another terminal, start the React app
cd app
npm install
npm run dev
```

The chat client will be available at http://localhost:5173 (or next available port).

### Features

- **Agent Selection**: Choose from available agent schemas
- **Model Selection**: Switch between LLM models (GPT-4.1, Claude Sonnet 4.5, etc.)
- **Streaming**: Real-time SSE streaming with tool call visualization
- **Session History**: Browse and resume past conversations
- **Export Sessions**: Download session data as YAML
- **Add to Scenario**: Save conversations for evaluation testing

### Running Tests

```bash
cd app
npm run test               # Run Playwright tests
npm run test:ui            # Run tests with UI
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/v1/chat/completions` | POST | OpenAI-compatible chat |
| `/api/v1/query` | POST | Execute REM query |
| `/api/v1/mcp` | POST | MCP protocol |
| `/api/v1/agents` | GET | List available agent schemas |
| `/api/v1/agents/{name}` | GET | Get specific agent info |
| `/api/v1/models` | GET | List available LLM models |
| `/api/v1/sessions` | GET | List chat sessions |
| `/api/v1/sessions/{id}/messages` | GET | Get session messages |
| `/api/v1/sessions/{id}/export` | GET | Export session as YAML |

## MCP Tools

### `search`

Execute REM queries:

```
LOOKUP <key>              - O(1) exact lookup
SEARCH <text> IN <table>  - Semantic search
FUZZY <text>              - Fuzzy text matching
TRAVERSE <key>            - Graph traversal
```

### `action`

Emit typed action events:

```python
action(
    type="observation",
    payload={
        "confidence": 0.85,
        "sources": ["transformer", "attention"],
        "session_name": "ML Research"
    }
)
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

- `ontology` - Knowledge base entities with embeddings
- `resources` - Documents and content chunks
- `sessions` - Conversation sessions
- `messages` - Chat messages
- `kv_store` - Key-value lookup cache for O(1) LOOKUP

## REM Functions (PostgreSQL)

- `rem_lookup(key, user_id)` - O(1) KV store lookup
- `rem_search(embedding, table, limit, min_sim, user_id)` - Vector search
- `rem_fuzzy(text, user_id, threshold, limit)` - Trigram matching
- `rem_traverse(key, edge_types, depth, user_id)` - Graph traversal

## Environment Variables

```bash
# Database
POSTGRES__CONNECTION_STRING=postgresql://user:pass@host:5432/db

# LLM
LLM__DEFAULT_MODEL=openai:gpt-4.1
LLM__TEMPERATURE=0.5
LLM__MAX_ITERATIONS=20
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# OpenTelemetry (Phoenix tracing)
OTEL__ENABLED=true
OTEL__COLLECTOR_ENDPOINT=http://phoenix:6006
OTEL__SERVICE_NAME=remlight-api

# Debugging
LOGURU_LEVEL=DEBUG  # Show tool calls and results

ENVIRONMENT=development
```

## Debugging

Enable debug logging to see tool calls, results, and session operations:

```bash
LOGURU_LEVEL=DEBUG rem ask "Find declarative agents"
```

Output shows tool invocations and results:
```
Tool call: search({'query': 'declarative agents'})
Tool result: {'status': 'success', 'results': [...]}
```

Session debugging shows message loading:
```bash
LOGURU_LEVEL=DEBUG rem ask "Tell me more" --session 550e8400-e29b-41d4-a716-446655440000
```
```
Loaded 3 messages for session ... (compress_on_load=True)
Loaded 3 messages -> 4 pydantic messages
```

Control tool result truncation in logs:
```bash
MAX_TOOL_RESULT_CHARS=500 LOGURU_LEVEL=DEBUG rem ask "Search for..."
```

## Project Structure

```
remlight/
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── ontology/                # Self-documenting knowledge base
│   ├── design/              # Architecture docs
│   ├── reference/           # API reference
│   └── guides/              # How-to guides
├── sql/
│   └── install.sql          # Tables, triggers, functions
├── schemas/
│   ├── query-agent.yaml     # Search agent
│   ├── action-agent.yaml    # Observation agent
│   └── orchestrator-agent.yaml
├── app/                     # React chat client
│   ├── src/
│   │   ├── api/             # API clients (agents, models, sessions, chat)
│   │   ├── components/
│   │   │   ├── chat/        # Chat components (message-list, input, toolbar)
│   │   │   ├── sidebar/     # Session history sidebar
│   │   │   └── ui/          # Radix UI components
│   │   ├── hooks/           # React hooks (use-chat, use-sse)
│   │   └── types/           # TypeScript types
│   ├── e2e/                 # Playwright tests
│   └── playwright.config.ts
└── remlight/
    ├── settings.py          # Environment config
    ├── models/
    │   ├── core.py          # CoreModel base
    │   └── entities.py      # Ontology, Resource, Session, Message
    ├── agentic/
    │   ├── schema.py        # AgentSchema + YAML
    │   ├── provider.py      # Pydantic AI agent creation
    │   └── streaming/       # SSE streaming with child agent support
    ├── api/
    │   ├── main.py          # FastAPI app
    │   ├── mcp_main.py      # MCP server with search, action, ask_agent
    │   └── routers/         # API endpoints (chat, query, tools, agents, sessions, models)
    ├── cli/
    │   └── main.py          # ask, query, ingest, serve, install
    └── services/
        └── database.py      # PostgreSQL + rem_* functions
```

## License

MIT
