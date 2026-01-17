# Agent Schemas

Agents are defined using YAML files with JSON Schema structure. This enables both Pydantic validation and LLM structured output.

## Schema Structure

```yaml
type: object
description: |
  System prompt for the agent...

properties:
  answer:
    type: string
    description: The agent's response

required:
  - answer

json_schema_extra:
  kind: agent
  name: my-agent
  version: "1.0.0"
  structured_output: false
  tools:
    - name: search
    - name: action
```

### Core Fields

| Field | Purpose |
|-------|---------|
| `type` | Always `object` |
| `description` | System prompt - defines agent behavior |
| `properties` | Output schema fields |
| `required` | Required output fields |
| `json_schema_extra` | Agent metadata and configuration |

### `json_schema_extra` Options

| Field | Default | Description |
|-------|---------|-------------|
| `kind` | `agent` | Schema type identifier |
| `name` | Required | Agent name (used in `--schema` flag) |
| `version` | `1.0.0` | Semantic version |
| `structured_output` | `false` | Force JSON output matching `properties` |
| `tools` | `[]` | List of available tools |
| `tags` | `[]` | Categorization tags |

## Output Modes

### Free-form Output (`structured_output: false`)

Agent responds with natural text. The `properties` schema documents expected fields but doesn't enforce them.

```yaml
json_schema_extra:
  structured_output: false
```

### Structured Output (`structured_output: true`)

Agent must return valid JSON matching the `properties` schema exactly. Useful for:
- Evaluation agents returning scores
- Automation pipelines needing parseable output
- Strict API contracts

```yaml
properties:
  score:
    type: number
    minimum: 0
    maximum: 1
  reasoning:
    type: string

json_schema_extra:
  structured_output: true
```

## Tool References

Available tools are declared in `json_schema_extra.tools`:

```yaml
tools:
  - name: search
    description: Execute REM queries
  - name: action
    description: Emit metadata (confidence, sources)
  - name: ask_agent
    description: Delegate to another agent
```

The agent can only call tools listed here. Tool implementations live in `remlight/api/mcp_main.py`.

## Included Agents

| Agent | Purpose | Tools |
|-------|---------|-------|
| `orchestrator-agent` | Delegates to sub-agents | `ask_agent`, `action` |
| `query-agent` | Searches knowledge base | `search`, `action` |
| `action-agent` | Records observations | `action` |
| `worker-agent` | General task execution | `action` |

## Evaluators (`evaluators/`)

Evaluator agents are specialized for testing and validation. They use `structured_output: true` to return parseable results.

```yaml
# schemas/evaluators/self-awareness-evaluator.yaml
properties:
  overall_score:
    type: number
  issues_found:
    type: array
    items:
      type: string

json_schema_extra:
  structured_output: true
  tools: []  # No tools - pure evaluation
```

Evaluators can be:
- **Automated**: Called programmatically in test pipelines
- **Manual**: Document expected behavior for human testing
- **Hybrid**: Run via CLI to generate structured test reports

```bash
# Run an evaluator
rem ask "Evaluate query-agent's self-awareness" --schema self-awareness-evaluator
```

## Pydantic Integration

Schemas are loaded and validated using Pydantic. The YAML maps to:

```python
from pydantic import BaseModel

class AgentOutput(BaseModel):
    answer: str
    confidence: float | None = None
    sources: list[str] | None = None
```

See `remlight/agentic/schema.py` for the loader implementation.

## Related Documentation

- [remlight/agentic/README.md](../remlight/agentic/README.md) - Streaming architecture, context propagation, message persistence
- [examples/README.md](../examples/README.md) - Session storage format and SSE streaming examples
- [app/README.md](../app/README.md) - Chat client for testing agents interactively
