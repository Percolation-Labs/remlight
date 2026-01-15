"""Agent schema definition and YAML serialization."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class MCPToolReference(BaseModel):
    """Reference to an MCP tool available to the agent."""

    name: str
    description: str | None = None


class MCPServerConfig(BaseModel):
    """MCP server configuration."""

    type: Literal["local", "remote"] = "local"
    module: str | None = None  # For local: Python module path
    url: str | None = None  # For remote: HTTP URL
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
    tags: list[str] = Field(default_factory=list)
    author: str | None = None
    override_temperature: float | None = None
    override_max_iterations: int | None = None


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
