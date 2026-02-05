---
entity_key: agent-schema
title: Agent Schema
tags: [reference, agents]
---

# Agent Schema

YAML format for declarative agent definition.

## Structure

```yaml
type: object
description: |
  System prompt goes here.

properties:
  answer:
    type: string

required:
  - answer

json_schema_extra:
  kind: agent
  name: my-agent
  version: "1.0.0"
  tools:
    - name: search
    - name: action
  structured_output: false
```

## Fields

### Standard JSON Schema

- `type`: Always "object"
- `description`: System prompt
- `properties`: Output schema
- `required`: Required fields

### REMLight Extensions (json_schema_extra)

| Field | Description |
| ----- | ----------- |
| `kind` | Always "agent" |
| `name` | Agent identifier |
| `version` | Semantic version |
| `tools` | Available tools |
| `mcp_servers` | MCP server configs |
| `structured_output` | Use Pydantic model output |
| `override_temperature` | LLM temperature |
| `override_max_iterations` | Max tool iterations |

## Tool Access

Only explicitly listed tools are available to agents. This prevents global tool leakage and ensures agents only have access to the tools they declare.

```yaml
# No tools (default - omit or empty list)
tools: []

# Specific tools only
tools:
  - name: search
  - name: action

# Tools from remote MCP server
tools:
  - name: fetch_data
    server: data-service
```

**Important:** If no `tools` list is specified or if it's empty, the agent has NO tools available.

## Loading

```python
from remlight.agentic import schema_from_yaml_file, create_agent

schema = schema_from_yaml_file("agent.yaml")
runtime = await create_agent(schema=schema, tools=tools)
```

Or via CLI: `rem ask "Hello" --schema agent.yaml`

## See also

- `REM LOOKUP mcp-tools` - Available tools for agents
- `REM LOOKUP multi-agent` - Multi-agent orchestration
- `REM LOOKUP cli` - CLI usage with schemas
