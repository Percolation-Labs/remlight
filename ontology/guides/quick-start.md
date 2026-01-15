---
entity_key: quick-start
title: Quick Start
tags: [guides, setup]
---

# Quick Start

Get REMLight running in minutes.

## Docker

```bash
export OPENAI_API_KEY=your-key
docker-compose up -d
# API at http://localhost:8000
```

## Local Development

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

## Environment Variables

```bash
POSTGRES__CONNECTION_STRING=postgresql://user:pass@host:5432/db
LLM__DEFAULT_MODEL=openai:gpt-4o-mini
LLM__TEMPERATURE=0.5
LLM__MAX_ITERATIONS=20
OPENAI_API_KEY=sk-...
```

## API Endpoints

| Endpoint | Method | Description |
| -------- | ------ | ----------- |
| `/health` | GET | Health check |
| `/api/v1/chat/completions` | POST | OpenAI-compatible chat |
| `/api/v1/query` | POST | Execute REM query |
| `/api/v1/mcp` | POST | MCP protocol |

API docs at `http://localhost:8000/docs`

## See also

- `REM LOOKUP cli` - Full CLI reference
- `REM LOOKUP architecture` - System architecture
- `REM LOOKUP entities` - Database entity types
