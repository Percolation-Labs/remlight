---
entity_key: declarative-agents
title: Declarative Agents
tags: [design, agents, core-concept]
---

# Declarative Agents

Agents are trivially declarative. Here's why.

## The Insight

Every LLM API call is already declarative:

- **System prompt** - text describing behavior
- **Output schema** - JSON Schema for structured responses
- **Tools** - function signatures the model can call

When you build an agent in Python, you're just wrapping these declarative inputs in code. But the code doesn't run on the LLM - only the text gets sent.

The only thing that actually executes is **tool code**. But if tools are external services (MCP servers, APIs, microservices), then the agent definition itself is purely declarative.

```text
┌─────────────────────────────────────────┐
│  What gets sent to LLM (declarative)    │
├─────────────────────────────────────────┤
│  • System prompt (text)                 │
│  • Output schema (JSON Schema)          │
│  • Tool signatures (names + params)     │
└─────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│  What actually runs (imperative)        │
├─────────────────────────────────────────┤
│  • Tool implementations                 │
│  • But these can be external services!  │
└─────────────────────────────────────────┘
```

## Pydantic 2 Example

Define an agent as a Pydantic model with `model_config`:

```python
from pydantic import BaseModel, Field

class ResearchAgent(BaseModel):
    """You are a research assistant that finds information.

    When asked a question:
    1. Search the knowledge base using the search tool
    2. Synthesize findings into a clear answer
    3. Cite your sources

    Be concise and factual.
    """

    answer: str = Field(description="Your research findings")
    confidence: float = Field(description="Confidence score 0-1")
    sources: list[str] = Field(description="Entity keys used")

    model_config = {
        "json_schema_extra": {
            "kind": "agent",
            "name": "research-agent",
            "version": "1.0.0",
            "tools": [
                {"name": "search"},
                {"name": "action"},
            ],
        }
    }
```

The docstring becomes the system prompt. The fields define structured output. The `model_config` adds agent metadata.

## Equivalent YAML

The same agent in REMLight YAML:

```yaml
type: object
description: |
  You are a research assistant that finds information.

  When asked a question:
  1. Search the knowledge base using the search tool
  2. Synthesize findings into a clear answer
  3. Cite your sources

  Be concise and factual.

properties:
  answer:
    type: string
    description: Your research findings
  confidence:
    type: number
    description: Confidence score 0-1
  sources:
    type: array
    items:
      type: string
    description: Entity keys used

required:
  - answer
  - confidence
  - sources

json_schema_extra:
  kind: agent
  name: research-agent
  version: "1.0.0"
  tools:
    - name: search
    - name: action
```

It's the same information - just data instead of code.

## Why This Matters

1. **Version control** - YAML diffs are cleaner than Python diffs
2. **Non-programmers** - Domain experts can edit agent behavior
3. **Hot reload** - Change agents without redeploying code
4. **Portability** - Same schema works across runtimes
5. **Auditing** - Declarative configs are easier to review

## The Pattern

```text
Pydantic Model                    →  YAML Schema
───────────────────────────────────────────────────────
class docstring                   →  description
Field(description=...)            →  properties.*.description
field: type                       →  properties.*.type
model_config["json_schema_extra"] →  json_schema_extra
```

Tools remain external services. The agent is just configuration.

## See also

- `REM LOOKUP agent-schema` - Full YAML schema reference
- `REM LOOKUP architecture` - How schemas fit in the system
- `REM LOOKUP mcp-tools` - External tool services
