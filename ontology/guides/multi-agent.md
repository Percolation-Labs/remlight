---
entity_key: multi-agent
title: Multi-Agent Orchestration
tags: [guides, multi-agent, structured, unstructured]
---

# Multi-Agent Orchestration

Running multiple agents with `ask_agent` tool and choosing between structured and unstructured output modes.

## Output Modes

REM agents support two output modes:

| Mode | Use Case | Output | Streaming |
|------|----------|--------|-----------|
| **Unstructured** | Chat, Q&A, creative tasks | Free-form text | Full streaming |
| **Structured** | Data extraction, actions | Typed Pydantic model | Partial streaming |

### Unstructured Mode (Default)

Agent returns free-form text. Use for conversational agents:

```yaml
# schemas/chat-agent.yaml
type: object
description: You are a helpful assistant.

json_schema_extra:
  kind: agent
  name: chat-agent
  output_type: unstructured  # default
  tools:
    - name: search
```

**Output:**
```python
result = await agent.run("What is machine learning?")
print(result.output)  # "Machine learning is a subset of AI..."
```

### Structured Mode

Agent returns a typed Pydantic model. Use for data extraction and deterministic actions:

```yaml
# schemas/extractor-agent.yaml
type: object
description: Extract structured information from text.

properties:
  entities:
    type: array
    items:
      type: object
      properties:
        name: { type: string }
        type: { type: string }
  sentiment:
    type: string
    enum: [positive, negative, neutral]
  confidence:
    type: number

required: [entities, sentiment, confidence]

json_schema_extra:
  kind: agent
  name: extractor-agent
  output_type: structured
  tools:
    - name: search
```

**Output:**
```python
result = await agent.run("Apple released the iPhone 16 today")
print(result.output)
# ExtractorAgentOutput(
#   entities=[{"name": "Apple", "type": "company"}, {"name": "iPhone 16", "type": "product"}],
#   sentiment="positive",
#   confidence=0.92
# )
```

---

## Quick Test

```bash
# Action agent - records observations (structured)
rem ask "Record that the system is healthy" --schema action-agent

# Query agent - searches knowledge base (unstructured)
rem ask "What is machine learning?" --schema query-agent

# Orchestrator - delegates to worker agents
rem ask "Research neural networks" --schema orchestrator-agent
```

## Agent Schemas

### Action Agent (Structured Actions)

Records typed observations with confidence:

```yaml
# schemas/action-agent.yaml
type: object
description: |
  You are an Action Agent that records observations.
  Call action(type='observation', payload={confidence, sources}).

json_schema_extra:
  kind: agent
  name: action-agent
  tools:
    - name: action
```

### Query Agent (Unstructured Search)

Free-form responses based on knowledge base:

```yaml
# schemas/query-agent.yaml
type: object
description: |
  You are a Query Agent. Search the knowledge base and provide helpful answers.

json_schema_extra:
  kind: agent
  name: query-agent
  tools:
    - name: search
```

### Orchestrator Agent (Multi-Agent Delegation)

Delegates to sub-agents for complex tasks:

```yaml
# schemas/orchestrator-agent.yaml
type: object
description: |
  Delegate tasks using ask_agent(agent_name="worker-agent", input_text="...").

json_schema_extra:
  kind: agent
  name: orchestrator-agent
  tools:
    - name: ask_agent
    - name: search
```

---

## How ask_agent Works

```text
Parent Agent (orchestrator-agent)
    │
    ▼
ask_agent("action-agent", "Record observation X")
    │
    ├── Child inherits: user_id, session_id, tenant_id
    ├── Child streams through parent SSE
    ├── Child events: child_tool_start, child_content, child_tool_result
    └── Result returned to parent
```

### Context Inheritance

| Field | Inherited | Overridable |
|-------|-----------|-------------|
| `user_id` | Yes | No |
| `tenant_id` | Yes | No |
| `session_id` | Yes | No |
| `agent_schema_uri` | No | Yes |
| `default_model` | No | Yes |

```python
# Parent passes context to child
child_context = parent_context.child_context(
    agent_schema_uri="sentiment-analyzer",
    model_override="openai:gpt-4.1-mini",  # Cheaper for subtasks
)
```

---

## Streaming in Multi-Agent

Child events stream in real-time through parent:

```text
event: tool_call
data: {"tool_name":"ask_agent","status":"started"}

event: child_content
data: {"content":"Analyzing sentiment..."}

event: child_tool_start
data: {"tool_name":"search","tool_id":"call_123"}

event: child_tool_result
data: {"tool_name":"search","result":{...}}

event: tool_call
data: {"tool_name":"ask_agent","status":"completed","result":"Sentiment: positive"}
```

---

## Structured vs Unstructured Decision Guide

| Scenario | Mode | Why |
|----------|------|-----|
| Chatbot | Unstructured | Natural conversation flow |
| Entity extraction | Structured | Typed data for downstream processing |
| Risk assessment | Structured | Numeric scores, enum categories |
| Research summary | Unstructured | Free-form markdown output |
| Action recording | Structured | Typed observations with confidence |
| Creative writing | Unstructured | No schema constraints |

---

## Example Session

```bash
# Start server
rem serve

# In another terminal - test multi-agent
curl -X POST http://localhost:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Agent-Schema: orchestrator-agent" \
  -H "X-Session-Id: sess-001" \
  -d '{"messages":[{"role":"user","content":"Research attention mechanisms"}], "stream": true}'
```

**Expected event flow:**

1. `progress` - "Starting research"
2. `tool_call` started - ask_agent("query-agent", ...)
3. `child_tool_start` - search tool
4. `child_tool_result` - search results
5. `child_content` - streaming child response
6. `tool_call` completed - ask_agent result
7. `content` - parent's final response
8. `metadata` - confidence, sources
9. `done` - stream complete

---

## See also

- `REM LOOKUP agent-schema` - Agent schema format
- `REM LOOKUP sse-events` - SSE streaming protocol
- `REM LOOKUP headers` - HTTP headers for context
- `REM LOOKUP messages` - Message persistence in sessions
- `REM LOOKUP mcp-tools` - Tools: search, action, ask_agent
