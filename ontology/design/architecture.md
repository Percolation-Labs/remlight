---
entity_key: architecture
title: Architecture
tags: [design]
---

# Architecture

REMLight is a declarative agent framework with PostgreSQL memory.

## Components

| Component | Purpose |
| --------- | ------- |
| `agentic/` | Agent runtime (schema parsing, pydantic-ai integration) |
| `models/` | Data models (Ontology, Resource, Session, Message) |
| `services/` | Database, repository, REM query execution |
| `api/` | FastAPI app, MCP server, routers |
| `cli/` | Commands: ask, query, serve, ingest, install |

## Data Flow

```text
User Input → CLI/API → Agent Runner → Pydantic AI Agent → PostgreSQL
                           ↓                    ↓
                   Load session history    Tool calls (REM queries)
                           ↓                    ↓
                   Save messages         Embeddings + KV store
```

## Key Decisions

**Declarative agents**: YAML schemas define system prompts, tools, and output structure. See [Agent Schema](../reference/agent-schema.md).

**Schema-agnostic queries**: LOOKUP and FUZZY don't require table names. The unified KV store resolves entities automatically. See [REM Query](../reference/rem-query.md).

**Streaming-first**: CLI and API share the same streaming infrastructure. SSE for API, plain text for CLI.

**Session persistence**: Messages saved automatically. History loaded on each request for multi-turn conversations.

## Dependencies

- **pydantic-ai**: Agent runtime
- **FastAPI**: HTTP API
- **asyncpg**: PostgreSQL driver
- **pgvector**: Vector similarity search
