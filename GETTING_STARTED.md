# Getting Started with REMLight

REMLight is a minimal declarative agent framework with PostgreSQL memory.

## Quick Start (Docker)

The fastest way to get started is with Docker Compose:

```bash
cd remlight

# Start PostgreSQL and API (database schema installs automatically)
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f api
```

That's it! The API is now running at http://localhost:8080

## Verify Installation

```bash
# Health check
curl http://localhost:8080/health
# {"status":"ok"}

# API info
curl http://localhost:8080/
# {"name":"REMLight API","version":"0.1.0",...}
```

## Using the CLI

Install the package locally to use the CLI:

```bash
# Install in development mode
pip install -e .

# Ask an agent (uses docker postgres)
rem ask "What is machine learning?"

# Execute a REM query
rem query "FUZZY machine learning"

# Start the server locally (without docker)
rem serve
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /` | API info |
| `POST /api/v1/chat/completions` | OpenAI-compatible chat |
| `POST /api/v1/query` | Execute REM query |
| `POST /api/v1/mcp` | MCP protocol endpoint |
| `GET /docs` | Swagger UI |

## Chat Example

```bash
curl -X POST http://localhost:8080/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-User-Id: demo-user" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

## Query Example

```bash
curl -X POST http://localhost:8080/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "FUZZY machine learning",
    "limit": 10
  }'
```

## MCP Integration

The MCP server is mounted at `/api/v1/mcp`. Tools available:

- `search` - Execute REM queries (LOOKUP, SEARCH, FUZZY, TRAVERSE)
- `action` - Register response metadata
- `ask_agent` - Invoke other agents (multi-agent orchestration)

## Environment Variables

Create a `.env` file (copy from the example):

```bash
# LLM API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# LLM Configuration
LLM__DEFAULT_MODEL=openai:gpt-4o-mini
LLM__TEMPERATURE=0.5

# Database
POSTGRES__CONNECTION_STRING=postgresql://remlight:remlight@localhost:5432/remlight
```

## Custom Agents

Create agents in YAML format in the `schemas/` directory:

```yaml
# schemas/my-agent.yaml
type: object
description: |
  You are a helpful assistant...

properties:
  answer:
    type: string

required:
  - answer

json_schema_extra:
  kind: agent
  name: my-agent
  version: 1.0.0
  tools:
    - name: search
    - name: action
```

Use it:

```bash
rem ask "Hello" --schema my-agent
```

## Stopping

```bash
docker compose down

# To also remove the database volume:
docker compose down -v
```

## Next Steps

- Read the full [README.md](README.md) for architecture details
- Explore the `schemas/query-agent.yaml` example
- Check the API docs at http://localhost:8080/docs
