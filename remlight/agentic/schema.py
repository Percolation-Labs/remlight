"""Agent schema definition and YAML serialization."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class MCPToolReference(BaseModel):
    """
    Reference to an MCP tool available to the agent.

    Two usage patterns:
    1. Local tools: Just declare name, tools loaded from configured MCP servers
    2. With mcp_server: Specify which MCP server provides this tool

    Example:
        tools:
          - name: search
          - name: lookup_entity
            mcp_server: rem
            description: Lookup entities by exact key

    FUTURE: Remote URI support (mcp://host/tool) - not yet implemented
    """

    name: str
    mcp_server: str | None = None  # Optional: which MCP server provides this tool
    description: str | None = None


class MCPResourceReference(BaseModel):
    """
    Reference to MCP resources accessible to the agent.

    Resources are data sources that can be read by agents.
    Resources declared in agent schemas become callable tools.

    Two formats supported:
    1. uri: Exact URI or URI with query params
    2. uri_pattern: Regex pattern for flexible matching

    Example:
        resources:
          - uri: rem://agents
            name: Agent Schemas
            description: List all available agent schemas

    FUTURE: Remote resource URI support - not yet implemented
    """

    uri: str | None = None
    uri_pattern: str | None = None
    name: str | None = None
    description: str | None = None
    mcp_server: str | None = None


class MCPServerConfig(BaseModel):
    """
    MCP server configuration for tool loading.

    Currently only supports 'local' (in-process) servers.

    FUTURE: Remote MCP server support via HTTP/SSE - not yet implemented
    """

    type: Literal["local"] = "local"  # FUTURE: Add "remote" when implemented
    module: str | None = None  # For local: Python module path
    # FUTURE: url: str | None = None  # For remote: HTTP URL
    id: str | None = None


class AgentSchemaMetadata(BaseModel):
    """REM-specific metadata in json_schema_extra."""

    kind: str | None = "agent"
    name: str
    version: str = "1.0.0"
    system_prompt: str | None = None
    structured_output: bool | None = None
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    tools: list[MCPToolReference] = Field(default_factory=list)
    resources: list[MCPResourceReference] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    author: str | None = None
    override_temperature: float | None = None
    override_max_iterations: int | None = None

    model_config = {"extra": "allow"}  # Allow additional custom metadata


class AgentSchema(BaseModel):
    """JSON Schema with REM extensions for declarative agent definition."""

    type: Literal["object"] = "object"
    description: str  # System prompt
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)
    json_schema_extra: AgentSchemaMetadata
    definitions: dict[str, Any] | None = None
    additionalProperties: bool = False

    def get_system_prompt(self) -> str:
        """Get combined system prompt."""
        parts = [self.description]
        if self.json_schema_extra.system_prompt:
            parts.append(self.json_schema_extra.system_prompt)
        return "\n\n".join(parts)


def get_system_prompt(schema: AgentSchema | dict[str, Any]) -> str:
    """
    Extract system prompt from schema (works with AgentSchema or dict).

    Combines description and optional extended system_prompt.
    """
    if isinstance(schema, AgentSchema):
        return schema.get_system_prompt()

    # Handle dict
    base = schema.get("description", "")
    extra = schema.get("json_schema_extra", {})
    custom = extra.get("system_prompt") if isinstance(extra, dict) else None

    if custom:
        return f"{base}\n\n{custom}"
    return base


def schema_from_yaml(yaml_content: str) -> AgentSchema:
    """Parse agent schema from YAML string."""
    data = yaml.safe_load(yaml_content)
    return AgentSchema(**data)


def schema_from_yaml_file(file_path: str | Path) -> AgentSchema:
    """Load agent schema from YAML file."""
    content = Path(file_path).read_text()
    return schema_from_yaml(content)


def schema_to_yaml(schema: AgentSchema) -> str:
    """Serialize agent schema to YAML."""
    return yaml.dump(
        schema.model_dump(exclude_none=True),
        default_flow_style=False,
        sort_keys=False,
    )


def build_agent_spec(
    name: str,
    description: str,
    properties: dict[str, Any] | None = None,
    tools: list[str] | None = None,
    version: str = "1.0.0",
) -> dict[str, Any]:
    """Build a minimal agent spec dictionary."""
    tools = tools or ["search", "action"]
    return {
        "type": "object",
        "description": description,
        "properties": properties or {
            "answer": {"type": "string", "description": "Response to the user"},
        },
        "required": ["answer"],
        "json_schema_extra": {
            "kind": "agent",
            "name": name,
            "version": version,
            "tools": [{"name": t} for t in tools],
        },
    }
