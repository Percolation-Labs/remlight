# REMLight

A lightweight agentic framework built on [REM](https://github.com/mr-saoirse/rem) - the declarative memory and query system for PostgreSQL.

Read the article: [A Really Simple Declarative Agent Framework (Part I)](https://medium.com/@mrsirsh/a-really-simply-declarative-agent-framework-part-i-6ae2b05fb2a1)

## What is REMLight?

REMLight adds agent orchestration on top of REM's memory primitives:

- **Declarative Agents**: Define agents in YAML with JSON Schema
- **Multi-Agent Orchestration**: Child agents stream through parent SSE connections
- **OpenAI-compatible API**: Drop-in replacement for chat completions
- **MCP Server**: Tools for `search`, `action`, and `ask_agent`
- **React Chat Client**: Test agents with streaming tool call visualization

REM provides the underlying memory layer:
- PostgreSQL with pgvector for semantic search
- O(1) key-value lookups via `kv_store`
- Graph traversal with `graph_edges`
- Fuzzy text matching with trigrams

## Quick Start

### Prerequisites

- Docker (for PostgreSQL with pgvector)
- Python 3.12+ with [uv](https://docs.astral.sh/uv/)
- Node.js 22+ (for chat client)

### 1. Start PostgreSQL

```bash
docker run -d --name remlight-pg \
  -e POSTGRES_DB=remlight \
  -e POSTGRES_USER=remlight \
  -e POSTGRES_PASSWORD=remlight \
  -p 5432:5432 \
  pgvector/pgvector:pg18
```

### 2. Install REMLight

```bash
git clone https://github.com/mr-saoirse/remlight
cd remlight

# Create virtual environment and install
uv venv && source .venv/bin/activate
uv sync

# Set your API key
export OPENAI_API_KEY=your-key
```

### 3. Initialize Database

```bash
rem install       # Create tables and functions
rem ingest ontology/  # Load sample knowledge base
```

### 4. Start the API

```bash
rem serve --port 8080
```

### 5. Try It Out

```bash
# Ask an agent
rem ask "What is machine learning?"

# Execute REM queries
rem query "LOOKUP architecture"
rem query "SEARCH agents IN ontology"

# Multi-agent orchestration
rem ask "Count to 5" --schema orchestrator-agent
```

## Chat Client

REMLight includes a React chat client for testing agents interactively:

```bash
cd app
npm install
VITE_API_BASE_URL=http://localhost:8080/api npm run dev
```

Open http://localhost:3000

Features:
- Agent/model selection
- Streaming with tool call visualization
- Session history and search
- Export sessions as YAML

See [app/README.md](app/README.md) for full documentation.

## Project Structure

```
remlight/
├── remlight/           # Python package
│   ├── agentic/        # Agent schema, streaming, context
│   ├── api/            # FastAPI endpoints
│   ├── cli/            # CLI commands (ask, query, serve)
│   └── services/       # Database, sessions
├── schemas/            # Agent YAML definitions
├── ontology/           # Self-documenting knowledge base
├── sql/                # Database schema and ERD
└── app/                # React chat client
```

## Agent Schema

Agents are defined in YAML with JSON Schema:

```yaml
type: object
description: |
  You are a helpful assistant...

properties:
  answer:
    type: string

json_schema_extra:
  kind: agent
  name: my-agent
  tools:
    - name: search
    - name: action
```

See [schemas/](schemas/) for examples.

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/chat/completions` | OpenAI-compatible chat with SSE |
| `POST /api/v1/query` | Execute REM queries |
| `GET /api/v1/agents` | List available agents |
| `GET /api/v1/sessions` | List chat sessions |
| `GET /api/v1/sessions/{id}/messages` | Get session messages |

## Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...

# Database (defaults work with docker-compose)
POSTGRES__CONNECTION_STRING=postgresql://remlight:remlight@localhost:5432/remlight

# Optional
ANTHROPIC_API_KEY=sk-ant-...
LLM__DEFAULT_MODEL=openai:gpt-4.1
LOGURU_LEVEL=DEBUG  # Show tool calls
```

## More Information

- **REM Query Language**: See [ontology/reference/rem-query.md](ontology/reference/rem-query.md)
- **Database Schema**: See [sql/erd.md](sql/erd.md)
- **API Documentation**: See [remlight/api/README.md](remlight/api/README.md)
- **Streaming Architecture**: See [remlight/agentic/README.md](remlight/agentic/README.md)

## License

MIT
