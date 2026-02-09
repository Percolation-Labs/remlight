# REMLight

A lightweight agentic framework built ideas from [REM](https://pypi.org/project/remdb/) - the declarative memory and query system for PostgreSQL.

## Blog Series

| Part | Description |
|------|-------------|
| [Part I](https://medium.com/@mrsirsh/a-really-simply-declarative-agent-framework-part-i-6ae2b05fb2a1) | Introduction & architecture |
| [Part II](https://medium.com/@mrsirsh/part-ii-of-a-really-simply-declarative-agent-framework-320da34e5b4d) | Agent construction & tool signatures |
| [Part III](https://medium.com/@mrsirsh/part-iii-of-a-really-simply-declarative-agent-framework-fc96cc950c11) | Streaming, SSE events & database persistence |

## Quick Start

### 1. Start Services (Docker)

```bash
docker compose up -d
```

This starts:
- **postgres** (port 5432) - pgvector database, auto-runs `sql/install.sql` on first startup
- **phoenix** (port 6016) - Arize Phoenix for tracing/observability
- **api** (port 8080) - REMLight API server with hot reload

### 2. Set Environment

```bash
cp .env.example .env
# Edit .env and add your API key:
# OPENAI_API_KEY=sk-...
```

### 3. Install Dependencies (Local Development)

This project uses [uv](https://docs.astral.sh/uv/) for fast Python package management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Tesseract OCR (required for PDF/image processing)
# macOS
brew install tesseract
# Ubuntu/Debian
# sudo apt-get install tesseract-ocr

# Create virtual environment and install dependencies
uv sync

# Activate the environment
source .venv/bin/activate

# Or run commands directly with uv (auto-activates)
uv run rem --help
uv run rem serve --port 8001 --reload
```

### 4. Ingest Ontology

```bash
rem ingest ontology/
```

### 5. Query the Knowledge Base

```bash
# Exact key lookup
rem query "LOOKUP architecture"

# Semantic search
rem query "SEARCH declarative agents IN ontologies"

# Fuzzy text match
uv run rem query "FUZZY multi-agent"
```

### 6. Ask an Agent

```bash
# Simple question
uv run rem ask "What can you help me with?"

# Use specific agent schema
rem ask "Find documents about AI" --schema query-agent

# Multi-turn conversation (session UUID)
rem ask "What is REM?" --session 550e8400-e29b-41d4-a716-446655440000
rem ask "Tell me more" --session 550e8400-e29b-41d4-a716-446655440000
```

## API Usage

### Start Server (Local Development)

```bash
rem serve --port 8001 --reload
```

### Chat Completions (OpenAI-compatible)

```bash
# Use UUIDs for user and session IDs to ensure uniqueness
# Generate with: uuidgen or python -c "import uuid; print(uuid.uuid4())"
curl -X POST http://localhost:8001/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-User-Id: a1b2c3d4-e5f6-7890-abcd-ef1234567890" \
  -H "X-Session-Id: 550e8400-e29b-41d4-a716-446655440000" \
  -H "X-Agent-Schema: orchestrator-agent" \
  -d '{
    "messages": [{"role": "user", "content": "What is machine learning?"}],
    "stream": true
  }'
```

### Direct Query

```bash
curl -X POST http://localhost:8001/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "LOOKUP architecture"}'
```

### Search Tool

```bash
curl -X POST http://localhost:8001/api/v1/tools/search \
  -H "Content-Type: application/json" \
  -d '{"query": "SEARCH declarative agents IN ontologies", "limit": 5}'
```

## React App

```bash
cd app
npm install
npm run dev
```

Open http://localhost:3000 - connects to API at localhost:8001.

## Phoenix Tracing

When running with Docker, traces are automatically sent to Phoenix:

1. Open Phoenix UI: http://localhost:6016
2. View traces for all agent executions
3. Inspect tool calls, latencies, token usage

To enable tracing in local development:

```bash
export OTEL__ENABLED=true
export OTEL__COLLECTOR_ENDPOINT=http://localhost:6016
```

## Docker Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  docker compose up -d                                       │
└─────────────────────────────────────────────────────────────┘
         │
         ├── postgres (pgvector/pgvector:pg18)
         │     ├── Port: 5432
         │     ├── Auto-runs: sql/install.sql (creates tables, triggers)
         │     └── Volume: postgres_data
         │
         ├── phoenix (arizephoenix/phoenix)
         │     ├── Port: 6016 (UI + OTLP collector)
         │     └── Volume: phoenix_data
         │
         └── api (Dockerfile)
               ├── Port: 8080 → 8000  #separate port to what you might server from codebase e.g. 8001
               ├── Hot reload: ./remlight, ./schemas mounted
               └── Connects to: postgres:5432, phoenix:6006
```

## Documentation

| Document | Description |
|----------|-------------|
| [code-walkthrough.md](code-walkthrough.md) | Agent construction, streaming, tool signatures |
| [remlight/cli/README.md](remlight/cli/README.md) | CLI commands |
| [remlight/api/README.md](remlight/api/README.md) | API endpoints |
| [remlight/agentic/README.md](remlight/agentic/README.md) | Agent runtime |
| [schemas/](schemas/) | Agent YAML examples |
| [app/README.md](app/README.md) | React chat client |



```bash

uv run rem serve --port 8012 

cd app & npm run dev
```

## License

MIT


