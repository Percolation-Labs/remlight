---
entity_key: mcp-tools
title: MCP Tools
tags: [reference, tools]
---

# MCP Tools

Tools available to agents via MCP protocol.

## search

Execute [REM queries](./rem-query.md).

```
LOOKUP <key>              - O(1) exact lookup
SEARCH <text> IN <table>  - Semantic search
FUZZY <text>              - Fuzzy text matching
TRAVERSE <key>            - Graph traversal
```

Example:

```python
search(query="LOOKUP transformer")
search(query="SEARCH neural networks IN ontology")
```

## action

Emit typed action events.

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

Action types:

- `observation`: Record findings
- `decision`: Log decisions
- `error`: Report errors
- Custom types supported

## ask_agent

Invoke another agent (multi-agent orchestration).

```python
ask_agent(
    agent_name="query-agent",
    input_text="Find documents about machine learning"
)
```

Features:

- Child agents inherit parent context (user_id, session_id)
- Child content streams through parent SSE connection
- Supports orchestrator, workflow, ensemble patterns

## Streaming

Child agent events stream in real-time:

```text
Parent Agent
    ↓
ask_agent("query-agent", "Find docs")
    ├── child_tool_start ──► SSE: tool_call event
    ├── child_content ─────► SSE: content chunks
    └── child_tool_result ─► SSE: tool_call complete
```

## See also

- `REM LOOKUP rem-query` - Query language used by search tool
- `REM LOOKUP agent-schema` - Agent schema with tools config
- `REM LOOKUP multi-agent` - Multi-agent orchestration guide
