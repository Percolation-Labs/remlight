# Schema-as-Architecture

A concept for deriving multi-agent execution topology from a Pydantic model.

The schema is a **pure domain model**. It describes:

- **Structure**: fields, types, nesting
- **Behavior guidance**: field descriptions guide extraction and validation
- **Routing hints**: `triggers` in field metadata indicate which queries route where
- **Tool access**: `tools` in field metadata specify available capabilities

A **compiler** walks the schema and derives:

- Orchestrator (sees top-level fields)
- Fragment handlers (each handles one nested type)
- Routing table (trigger patterns → fragments)

## Core Concepts

**The schema is independent.** It's a domain model first. Execution topology is derived, not embedded.

**Model docstrings become prompts.** The `TeamMember.__doc__` becomes the system prompt for the handler that processes team member data.

**Field metadata drives routing.** Fields with `triggers` in `json_schema_extra` become routable fragments.

**No handler sees the full schema.** The orchestrator sees field names and descriptions. Fragment handlers see only their type's schema.

---

## Example

See `example_model.py` for the schema and `example_compiler.py` for compilation.

### Schema (Domain Model)

```python
class Task(BaseModel):
    """
    A unit of work within a project.

    ## Extraction Rules
    - Each task needs task_id, title, status
    - Use status: pending | in_progress | done | blocked
    """

    task_id: str = Field(description="Unique identifier (e.g., 'T-001')")
    title: str = Field(description="Brief task title")
    status: str = Field(default="pending", description="pending | in_progress | done | blocked")


class Project(BaseModel):
    """A project with team, tasks, and milestones."""

    name: str = Field(description="Project name")

    tasks: list[Task] = Field(
        description="Project tasks and work items",
        json_schema_extra={
            "triggers": ["task", "todo", "backlog"],
            "tools": ["create_task", "update_status"],
            "structured_output": True,
        },
    )
```

### Compiled Output

```python
compiled = compile_schema(Project)
```

```text
orchestrator:
  name: project
  prompt: "A project with team, tasks, and milestones."
  fields: [name, tasks, milestones, team]

fragments:
  tasks:
    type: Task
    prompt: "A unit of work within a project..."
    tools: [create_task, update_status]
    structured_output: true

routing:
  | Trigger | Fragment |
  |---------|----------|
  | task    | tasks    |
  | todo    | tasks    |
  | backlog | tasks    |
```

### Generated YAML

The compiler can emit YAML definitions:

```yaml
type: object

description: |
  A unit of work within a project.

  ## Extraction Rules
  - Each task needs task_id, title, status
  - Use status: pending | in_progress | done | blocked

properties:
  # Schema derived from Task

json_schema_extra:
  kind: agent
  name: tasks-handler
  version: "1.0.0"
  structured_output: true
  tools:
    - name: create_task
    - name: update_status
```

---

## Routing

Routing is a function of three inputs:

1. **User/project state** — from MCP resource (`user://profile`)
2. **Conversation state** — completed actions in session history
3. **Schema triggers** — pattern match question to field triggers

The routing table (generated from schema) is absorbed into the orchestrator prompt.

---

## Fragment Handoff

```text
User: "Add a task for API integration"

Orchestrator sees:
  fields: {tasks: {triggers: ["task"], ...}}
  → matches "task" → delegate to tasks fragment

Tasks handler receives:
  - Task schema only (not Project, not team, not milestones)
  - Prompt from Task.__doc__
  - Tools: create_task, update_status

  → Returns structured JSON

Orchestrator aggregates into Project.
```

---

## Integration Points

**Session history**: Tool call actions indicate what's been done → informs next routing.

**MCP resources**: `user://profile`, `project://config` provide context for routing overrides.

**Pydantic**: Rich field metadata (`description`, `json_schema_extra`) drives both schema validation and execution behavior.
