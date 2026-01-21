# Tools and Servers Module

This document describes the architecture for registering, discovering, and using MCP tool servers in REMLight.

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        REMLight Registry                             │
├─────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   Server A   │    │   Server B   │    │   Remote Registry    │  │
│  │   (mcp)      │    │   (rest)     │    │   registry_uri:...   │  │
│  ├──────────────┤    ├──────────────┤    └──────────────────────┘  │
│  │  search      │    │  fetch_data  │              │               │
│  │  action      │    │  transform   │              ▼               │
│  │  ask_agent   │    └──────────────┘    ┌──────────────────────┐  │
│  │  parse_file  │                        │   Federated Servers   │  │
│  └──────────────┘                        └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Deterministic IDs

All entities use deterministic UUIDs generated from their identity:

| Entity | ID Formula |
|--------|------------|
| Server | `hash(endpoint or "local://{name}")` |
| Tool | `hash(server_id + ":" + tool_name)` |
| Agent | `hash(registry_uri + ":" + agent_name)` |

This enables idempotent upserts without unique constraints.

## API Endpoints

### Servers

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/servers` | List all servers |
| GET | `/api/v1/servers/{name}` | Get server by name |
| PUT | `/api/v1/servers` | Create/update server |
| DELETE | `/api/v1/servers/{name}` | Soft delete server |
| POST | `/api/v1/servers/search` | Search servers |
| GET | `/api/v1/servers/{name}/tools` | Get server's tools |

### Tools (Registry)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/tools` | List all registered tools |
| GET | `/api/v1/tools/{server}/{name}` | Get tool by server and name |
| PUT | `/api/v1/tools` | Create/update tool registration |
| DELETE | `/api/v1/tools/{id}` | Soft delete tool |
| POST | `/api/v1/tools/search` | Search tools |
| POST | `/api/v1/tools/register` | Register project tools |

### MCP Tools (Execution)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/mcp-tools/search` | Execute search tool |
| POST | `/api/v1/mcp-tools/parse-file` | Execute parse-file tool |

---

## Examples

### List All Servers

```bash
curl http://localhost:8080/api/v1/servers
```

Response:
```json
{
  "servers": [
    {
      "name": "local",
      "description": "Built-in REMLight MCP server with search, action, and agent tools...",
      "server_type": "mcp",
      "endpoint": null,
      "enabled": true,
      "icon": "🏠",
      "tags": ["builtin", "mcp", "local"],
      "registry_uri": null
    }
  ],
  "total": 1
}
```

### Get Server by Name

```bash
curl http://localhost:8080/api/v1/servers/local
```

Response:
```json
{
  "name": "local",
  "description": "Built-in REMLight MCP server...",
  "server_type": "mcp",
  "endpoint": null,
  "enabled": true,
  "icon": "🏠",
  "tags": ["builtin", "mcp", "local"],
  "registry_uri": null
}
```

### Create/Update a Server

```bash
curl -X PUT http://localhost:8080/api/v1/servers \
  -H "Content-Type: application/json" \
  -d '{
    "name": "data-service",
    "description": "External data service for fetching and transforming data",
    "server_type": "rest",
    "endpoint": "http://data-service:8000/api/v1",
    "icon": "📊",
    "tags": ["data", "external"]
  }'
```

Response:
```json
{
  "name": "data-service",
  "created": true,
  "message": "Server 'data-service' created successfully"
}
```

### Search Servers

```bash
curl -X POST http://localhost:8080/api/v1/servers/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "data",
    "search_type": "name",
    "limit": 10
  }'
```

Response:
```json
{
  "results": [
    {
      "name": "data-service",
      "description": "External data service...",
      "server_type": "rest",
      "endpoint": "http://data-service:8000/api/v1",
      "icon": "📊",
      "tags": ["data", "external"],
      "similarity": 0.85
    }
  ],
  "total": 1
}
```

### Get Server's Tools

```bash
curl http://localhost:8080/api/v1/servers/local/tools
```

Response:
```json
[
  {
    "name": "search",
    "description": "Execute REM queries to search the knowledge base...",
    "input_schema": {"type": "object", "properties": {...}},
    "icon": "🔍",
    "tags": ["search", "knowledge", "semantic"]
  },
  {
    "name": "action",
    "description": "Emit a typed action event for SSE streaming...",
    "input_schema": {"type": "object", "properties": {...}},
    "icon": "⚡",
    "tags": ["action", "metadata", "events"]
  }
]
```

---

### List All Tools

```bash
curl http://localhost:8080/api/v1/tools
```

Response:
```json
{
  "tools": [
    {
      "name": "search",
      "description": "Execute REM queries to search the knowledge base...",
      "server_name": "local",
      "input_schema": {
        "type": "object",
        "required": ["query"],
        "properties": {
          "query": {"type": "string"},
          "limit": {"type": "integer", "default": 20},
          "user_id": {"type": "string", "default": null}
        }
      },
      "enabled": true,
      "icon": "🔍",
      "tags": ["search", "knowledge", "semantic"]
    },
    {
      "name": "action",
      "description": "Emit a typed action event...",
      "server_name": "local",
      "input_schema": {...},
      "enabled": true,
      "icon": "⚡",
      "tags": ["action", "metadata", "events"]
    },
    {
      "name": "ask_agent",
      "description": "Invoke another agent by name...",
      "server_name": "local",
      "input_schema": {...},
      "enabled": true,
      "icon": "🤖",
      "tags": ["agent", "delegation", "multi-agent"]
    },
    {
      "name": "parse_file",
      "description": "Parse a file and extract content...",
      "server_name": "local",
      "input_schema": {...},
      "enabled": true,
      "icon": "🔧",
      "tags": ["tool"]
    }
  ],
  "total": 4
}
```

### List Tools by Server

```bash
curl "http://localhost:8080/api/v1/tools?server_name=local"
```

### Get Tool by Server and Name

```bash
curl http://localhost:8080/api/v1/tools/local/search
```

Response:
```json
{
  "name": "search",
  "description": "Execute REM queries to search the knowledge base...",
  "server_name": "local",
  "input_schema": {
    "type": "object",
    "required": ["query"],
    "properties": {
      "query": {"type": "string"},
      "limit": {"type": "integer", "default": 20}
    }
  },
  "enabled": true,
  "icon": "🔍",
  "tags": ["search", "knowledge", "semantic"]
}
```

### Create/Update a Tool

```bash
curl -X PUT http://localhost:8080/api/v1/tools \
  -H "Content-Type: application/json" \
  -d '{
    "name": "fetch_data",
    "server_name": "data-service",
    "description": "Fetch data from external APIs with caching and retry",
    "input_schema": {
      "type": "object",
      "required": ["url"],
      "properties": {
        "url": {"type": "string", "description": "API endpoint URL"},
        "method": {"type": "string", "default": "GET"},
        "cache_ttl": {"type": "integer", "default": 300}
      }
    },
    "icon": "🌐",
    "tags": ["data", "api", "fetch"]
  }'
```

Response:
```json
{
  "name": "fetch_data",
  "server_name": "data-service",
  "created": true,
  "message": "Tool 'fetch_data' on 'data-service' created successfully"
}
```

### Search Tools

**By Name (fuzzy matching):**
```bash
curl -X POST http://localhost:8080/api/v1/tools/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "search",
    "search_type": "name",
    "limit": 10
  }'
```

**By Tags:**
```bash
curl -X POST http://localhost:8080/api/v1/tools/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "semantic",
    "search_type": "tags",
    "limit": 10
  }'
```

**Semantic Search (requires embeddings):**
```bash
curl -X POST http://localhost:8080/api/v1/tools/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "find information in knowledge base",
    "search_type": "semantic",
    "limit": 5
  }'
```

**Combined Search (all methods):**
```bash
curl -X POST http://localhost:8080/api/v1/tools/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "agent",
    "search_type": "all",
    "limit": 10
  }'
```

Response:
```json
{
  "results": [
    {
      "name": "ask_agent",
      "description": "Invoke another agent by name and return its response...",
      "server_name": "local",
      "input_schema": {...},
      "icon": "🤖",
      "tags": ["agent", "delegation", "multi-agent"],
      "similarity": 1.0
    }
  ],
  "total": 1
}
```

### Register Project Tools

Trigger registration of all tools from the MCP server module:

```bash
curl -X POST http://localhost:8080/api/v1/tools/register
```

Force re-registration (ignores MD5 hash check):
```bash
curl -X POST "http://localhost:8080/api/v1/tools/register?force=true"
```

Response:
```json
{
  "servers_registered": 1,
  "servers_created": 0,
  "tools_registered": 4,
  "tools_created": 0,
  "skipped": 0
}
```

---

## Agent Schema with Remote Tools

Agents can reference tools from different servers:

```yaml
type: object
description: |
  You are a data analyst assistant with access to search and external data.

json_schema_extra:
  kind: agent
  name: data-analyst
  version: "1.0.0"
  tools:
    - name: search
      server: local           # default - built-in MCP server
    - name: fetch_data
      server: data-service    # registered remote server
    - name: transform
      server: data-service
```

When `server` is omitted or set to "local", the tool is loaded from the built-in MCP server.

---

## Server Types

| Type | Description | Configuration |
|------|-------------|---------------|
| `mcp` | Built-in or local MCP server | No endpoint needed for built-in |
| `rest` | Remote REST/HTTP server | `endpoint`: Base URL |
| `stdio` | MCP stdio transport (future) | `endpoint`: Command to execute |

---

## Database Schema

### Servers Table

```sql
CREATE TABLE servers (
    id UUID PRIMARY KEY,
    name VARCHAR(512) NOT NULL,
    description TEXT,
    server_type VARCHAR(64) DEFAULT 'mcp',  -- mcp, rest, stdio
    endpoint VARCHAR(2048),
    config JSONB DEFAULT '{}',
    enabled BOOLEAN DEFAULT TRUE,
    registry_uri VARCHAR(2048),  -- for federation
    icon VARCHAR(512),
    -- system fields...
    embedding VECTOR(1536)
);
```

### Tools Table

```sql
CREATE TABLE tools (
    id UUID PRIMARY KEY,
    name VARCHAR(512) NOT NULL,
    description TEXT,
    server_id UUID REFERENCES servers(id) ON DELETE CASCADE,
    input_schema JSONB DEFAULT '{}',
    enabled BOOLEAN DEFAULT TRUE,
    icon VARCHAR(512),
    -- system fields...
    embedding VECTOR(1536)
);
```

---

## Registration with Change Detection

The registration utility uses MD5 hashing to detect description changes:

```python
from remlight.services.registration import register_project_tools

# Only updates if descriptions changed (saves embedding costs)
stats = await register_project_tools(force=False)

# Force re-registration
stats = await register_project_tools(force=True, generate_embeddings=True)
```

Each server/tool stores `metadata.description_hash` which is compared during registration.
