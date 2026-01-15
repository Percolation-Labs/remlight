---
entity_key: multi-agent
title: Multi-Agent Orchestration
tags: [guides, multi-agent]
---

# Multi-Agent Orchestration

Running multiple agents with `ask_agent` tool.

## Quick Test

```bash
# Action agent - records observations
rem ask "Record that the system is healthy" --schema action-agent

# Query agent - searches knowledge base
rem ask "What is machine learning?" --schema query-agent

# Orchestrator - delegates to worker agents
rem ask "Research neural networks" --schema orchestrator-agent
```

## Agent Schemas

### Action Agent

Records typed observations:

```yaml
# schemas/action-agent.yaml
type: object
description: |
  You are an Action Agent that records observations.
  Call action(type='observation', payload={...}).

json_schema_extra:
  name: action-agent
  tools:
    - name: action
```

### Orchestrator Agent

Delegates to sub-agents:

```yaml
# schemas/orchestrator-agent.yaml
type: object
description: |
  Delegate tasks using ask_agent(agent_name="worker-agent", input_text="...").

json_schema_extra:
  name: orchestrator-agent
  tools:
    - name: ask_agent
```

## How ask_agent Works

```text
Parent Agent
    │
    ▼
ask_agent("action-agent", "Record observation X")
    │
    ├── Child inherits: user_id, session_id
    ├── Child streams through parent SSE
    └── Result returned to parent
```

## Streaming

Child events stream in real-time through parent:

```text
child_tool_start ──► SSE: tool_call event
child_content ─────► SSE: content chunks
child_tool_result ─► SSE: tool_call complete
```

## Example Session

```bash
# Start server
rem serve

# In another terminal - test multi-agent
curl -X POST http://localhost:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Agent-Schema: orchestrator-agent" \
  -d '{"messages":[{"role":"user","content":"Research attention mechanisms"}]}'
```

## See also

- `REM LOOKUP mcp-tools` - Tools: search, action, ask_agent
- `REM LOOKUP agent-schema` - Agent schema format
- `REM LOOKUP messages` - Message persistence in sessions
