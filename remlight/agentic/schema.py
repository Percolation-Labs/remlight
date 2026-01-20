"""
Agent Schema Definition - The Declarative Agent Format
=======================================================

This module defines the YAML schema format for declarative agent definitions.
Agents are defined as JSON Schema objects with REMLight-specific extensions.

THE DECLARATIVE PATTERN
-----------------------

Instead of writing agent code:

    class QueryAgent:
        def __init__(self):
            self.system_prompt = "You are a helpful assistant..."
            self.tools = [search_tool, action_tool]
            self.model = "openai:gpt-4.1"

You define agents declaratively in YAML:

    type: object
    description: |
      You are a helpful assistant with access to a knowledge base.
      Use the search tool to find relevant information.

    properties:
      answer:
        type: string
        description: Your response to the user

    required:
      - answer

    json_schema_extra:
      kind: agent
      name: query-agent
      version: "1.0.0"
      tools:
        - name: search
        - name: action

BENEFITS:
1. **Configuration over code**: Change agent behavior without code changes
2. **Version control friendly**: YAML diffs are readable
3. **Non-programmers can modify**: Product/ops can tune prompts
4. **Runtime loading**: Hot-reload agents without restarting
5. **Standardization**: All agents follow the same structure

SCHEMA STRUCTURE
----------------

The format is based on JSON Schema with REMLight extensions:

    TYPE: Always "object" (JSON Schema type)

    DESCRIPTION: The agent's system prompt. This is the most important field.
                 It defines the agent's personality, capabilities, and instructions.

    PROPERTIES: Defines the output structure (for structured_output mode) or
                provides internal tracking schema (for unstructured mode).

    REQUIRED: Which properties are mandatory in structured output.

    JSON_SCHEMA_EXTRA: REMLight-specific metadata:
      - kind: Always "agent"
      - name: Unique identifier (e.g., "query-agent")
      - version: Semantic version
      - tools: Which MCP tools the agent can use
      - resources: Which MCP resources the agent can access
      - structured_output: Whether to enforce JSON output
      - override_temperature: Agent-specific LLM temperature
      - override_max_iterations: Max tool call loops

"""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field


class MCPToolReference(BaseModel):
    """
    Reference to an MCP tool available to the agent.

    MCP (Model Context Protocol) tools are functions the agent can call.
    This schema element declares which tools an agent can access.

    TOOL FILTERING
    -------------
    The agent factory (provider.py) uses these references to filter
    which tools are actually attached to the agent:

    - tools: []                     → NO tools available
    - tools: [{name: "search"}]     → Only search tool available
    - (tools not specified)         → ALL tools available

    TOOL SPECIFICATION
    -----------------
    Currently, only local tools are supported. Tools are identified by name
    and matched against the tools provided to create_agent().

    FUTURE: Remote MCP tool URIs (mcp://host/tool) for distributed tools.

    Example YAML:
        tools:
          - name: search
            description: Search the knowledge base
          - name: action
          - name: ask_agent
            description: Delegate to another agent

    Attributes:
        name: Tool function name (matches tool.__name__)
        mcp_server: Optional server identifier (for future remote support)
        description: Optional description override
    """

    name: str
    mcp_server: str | None = None  # Optional: which MCP server provides this tool
    description: str | None = None


class MCPResourceReference(BaseModel):
    """
    Reference to MCP resources accessible to the agent.

    MCP resources are data sources that agents can read. Unlike tools
    (which are functions), resources are data endpoints.

    RESOURCE PATTERNS
    ----------------
    Resources can be specified as:
    - Exact URI: rem://agents (specific resource)
    - URI pattern: rem://agents/* (wildcard matching)

    RESOURCE AS TOOL
    ---------------
    When a resource is declared, it can be exposed as a tool that reads
    the resource. The name is derived from the URI:
        resource://users/profile → tool: resource_users_profile

    FUTURE: Full MCP resource support with remote endpoints.

    Example YAML:
        resources:
          - uri: user://profile
            name: User Profile
            description: Get current user's profile information
          - uri: rem://agents
            description: List available agent schemas

    Attributes:
        uri: Exact resource URI
        uri_pattern: Regex pattern for URI matching
        name: Human-readable name
        description: Resource description
        mcp_server: Which server provides this resource
    """

    uri: str | None = None
    uri_pattern: str | None = None
    name: str | None = None
    description: str | None = None
    mcp_server: str | None = None


class MCPServerConfig(BaseModel):
    """
    MCP server configuration for tool/resource loading.

    Defines how to connect to MCP servers that provide tools and resources.

    CURRENT SUPPORT: Local (in-process) servers only.
    The server is a Python module that registers tools/resources.

    FUTURE: Remote MCP servers via HTTP/SSE transport.

    Example YAML (future):
        mcp_servers:
          - type: local
            module: remlight.api.mcp_main
          - type: remote
            url: https://mcp.example.com/v1

    Attributes:
        type: Server type ("local" only for now)
        module: Python module path for local servers
        id: Optional server identifier
    """

    type: Literal["local"] = "local"  # FUTURE: Add "remote" when implemented
    module: str | None = None  # For local: Python module path
    # FUTURE: url: str | None = None  # For remote: HTTP URL
    id: str | None = None


class AgentSchemaMetadata(BaseModel):
    """
    REMLight-specific metadata in json_schema_extra.

    This is the "extension point" where REMLight-specific configuration
    goes. It's stored in json_schema_extra to keep the main schema
    valid JSON Schema while adding our custom fields.

    KEY FIELDS
    ----------

    kind: Always "agent" - identifies this as an agent schema

    name: Unique identifier used for:
          - Loading schemas by name
          - Logging/tracing
          - Multi-agent references (ask_agent target)

    version: Semantic version for schema evolution

    structured_output: Controls output mode:
          - true: Agent returns JSON matching properties schema
          - false/None: Agent returns free-form text
          - Default is false (unstructured) for conversational agents

    tools: List of MCPToolReference - which tools agent can use

    resources: List of MCPResourceReference - which resources agent can access

    override_model: Per-agent model (forces this model, ignoring request/default)
                    Use when an agent REQUIRES a specific model capability.
                    Model priority: override_model > request model > default_model

    override_temperature: Per-agent temperature (overrides global setting)
                         Lower = more deterministic, higher = more creative

    override_max_iterations: Per-agent max tool loops
                            Prevents infinite tool call loops

    Attributes:
        kind: Schema type (always "agent")
        name: Unique agent identifier
        version: Semantic version string
        system_prompt: Optional additional system prompt text
        structured_output: Whether to enforce JSON output schema
        mcp_servers: MCP server configurations
        tools: Available MCP tools
        resources: Available MCP resources
        tags: Classification tags
        author: Schema author
        override_temperature: Per-agent LLM temperature
        override_max_iterations: Per-agent max tool iterations
    """

    kind: str | None = "agent"
    name: str
    version: str = "1.0.0"
    system_prompt: str | None = None  # Additional system prompt (appended to description)
    structured_output: bool | None = None  # None = False = unstructured text output
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)
    tools: list[MCPToolReference] = Field(default_factory=list)
    resources: list[MCPResourceReference] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    author: str | None = None
    override_model: str | None = None  # Per-agent model override (forces this model)
    override_temperature: float | None = None  # Per-agent temperature override
    override_max_iterations: int | None = None  # Per-agent max iterations override

    model_config = {"extra": "allow"}  # Allow additional custom metadata


class AgentSchema(BaseModel):
    """
    JSON Schema with REMLight extensions for declarative agent definition.

    This is the ROOT schema class representing a complete agent definition.
    It follows JSON Schema structure with custom extensions.

    THE DUAL PURPOSE OF description
    -------------------------------
    In standard JSON Schema, 'description' describes the schema itself.
    In REMLight, 'description' IS the agent's system prompt.

    This dual-use is intentional:
    - The YAML is self-documenting (description describes the agent)
    - The same text configures the agent (becomes system prompt)

    PROPERTIES AND OUTPUT
    --------------------
    The 'properties' field serves two purposes:

    1. When structured_output=true:
       Properties define the required output JSON structure.
       The agent MUST return JSON matching this schema.

    2. When structured_output=false (default):
       Properties provide "thinking structure" guidance in the prompt.
       The agent returns free-form text but can track internal fields.

    Example YAML:
        type: object
        description: |
          You are a helpful assistant with access to a knowledge base.

        properties:
          answer:
            type: string
            description: Your response to the user
          confidence:
            type: number
            description: Your confidence (0.0-1.0)

        required:
          - answer

        json_schema_extra:
          kind: agent
          name: query-agent
          version: "1.0.0"
          tools:
            - name: search

    Attributes:
        type: Always "object" (JSON Schema type)
        description: The agent's system prompt
        properties: Output structure definition
        required: Required output fields
        json_schema_extra: REMLight-specific metadata
        definitions: Shared schema definitions (JSON Schema $defs)
        additionalProperties: Whether extra fields allowed (usually false)
    """

    type: Literal["object"] = "object"
    description: str  # THE SYSTEM PROMPT - Most important field!
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)
    json_schema_extra: AgentSchemaMetadata
    definitions: dict[str, Any] | None = None
    additionalProperties: bool = False

    def get_system_prompt(self) -> str:
        """
        Get the complete system prompt from schema.

        Combines:
        1. description (main system prompt)
        2. json_schema_extra.system_prompt (optional extension)

        The extension is useful for long prompts that would clutter
        the main description, or for prompts that are dynamically generated.

        Returns:
            Combined system prompt string
        """
        parts = [self.description]
        if self.json_schema_extra.system_prompt:
            parts.append(self.json_schema_extra.system_prompt)
        return "\n\n".join(parts)


def get_system_prompt(schema: AgentSchema | dict[str, Any]) -> str:
    """
    Extract system prompt from schema (polymorphic version).

    Works with both parsed AgentSchema objects and raw dict (from YAML load).
    This is useful in places where the schema might not be fully parsed yet.

    Combines:
    - description: The main system prompt
    - json_schema_extra.system_prompt: Optional extension

    Args:
        schema: Either an AgentSchema instance or a raw dict from YAML

    Returns:
        Combined system prompt string
    """
    if isinstance(schema, AgentSchema):
        return schema.get_system_prompt()

    # Handle raw dict (e.g., fresh YAML load)
    base = schema.get("description", "")
    extra = schema.get("json_schema_extra", {})
    custom = extra.get("system_prompt") if isinstance(extra, dict) else None

    if custom:
        return f"{base}\n\n{custom}"
    return base


def schema_from_yaml(yaml_content: str) -> AgentSchema:
    """
    Parse agent schema from YAML string.

    This is the primary way to load agent schemas. The YAML is parsed
    to a dict, then validated and converted to an AgentSchema.

    Args:
        yaml_content: YAML string containing agent definition

    Returns:
        Validated AgentSchema instance

    Raises:
        ValidationError: If schema doesn't match expected format

    Example:
        yaml_str = '''
        type: object
        description: You are a helpful assistant.
        properties:
          answer:
            type: string
        json_schema_extra:
          kind: agent
          name: my-agent
        '''
        schema = schema_from_yaml(yaml_str)
    """
    data = yaml.safe_load(yaml_content)
    return AgentSchema(**data)


def schema_from_yaml_file(file_path: str | Path) -> AgentSchema:
    """
    Load agent schema from a YAML file.

    Convenience wrapper that reads file content and parses.

    Args:
        file_path: Path to YAML file (str or Path object)

    Returns:
        Validated AgentSchema instance

    Example:
        schema = schema_from_yaml_file("schemas/query-agent.yaml")
    """
    content = Path(file_path).read_text()
    return schema_from_yaml(content)


def schema_to_yaml(schema: AgentSchema) -> str:
    """
    Serialize agent schema back to YAML string.

    Useful for:
    - Debugging (print schema)
    - Saving modified schemas
    - Generating schema templates

    Args:
        schema: AgentSchema to serialize

    Returns:
        YAML string representation
    """
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
    """
    Build a minimal agent spec dictionary programmatically.

    This is a HELPER for creating agent schemas in code rather than YAML.
    Useful for:
    - Unit tests (quick agent setup)
    - Dynamic agent creation
    - Default agent configuration

    The result can be passed directly to create_agent() or converted
    to AgentSchema via AgentSchema(**result).

    Args:
        name: Unique agent identifier
        description: System prompt for the agent
        properties: Output schema properties (defaults to simple answer field)
        tools: List of tool names (defaults to search + action)
        version: Schema version

    Returns:
        Dict representing a valid agent schema

    Example:
        # Create a simple test agent
        spec = build_agent_spec(
            name="test-agent",
            description="You are a test assistant.",
            tools=["search"],
        )
        runtime = await create_agent(spec, tools=mcp_tools)
    """
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


async def schema_from_database(
    agent_name: str,
    database_first: bool = False,
) -> AgentSchema | None:
    """
    Load agent schema from database, optionally checking filesystem first.

    This provides a unified way to load agent schemas from either source,
    respecting the database_first preference.

    Loading Order (default - database_first=False):
    1. Check filesystem (schemas directory)
    2. If not found, check database

    Loading Order (database_first=True):
    1. Check database
    2. If not found, check filesystem

    Args:
        agent_name: The agent name (without .yaml extension)
        database_first: If True, prefer database over filesystem

    Returns:
        AgentSchema if found, None otherwise

    Example:
        # Default: filesystem first
        schema = await schema_from_database("my-agent")

        # Database first
        schema = await schema_from_database("my-agent", database_first=True)
    """
    from remlight.models.entities import Agent
    from remlight.services.repository import Repository

    # Lazy import to avoid circular dependencies
    schemas_dir = Path(__file__).parent.parent.parent / "schemas"

    file_schema = None
    db_schema = None

    # Try filesystem
    yaml_file = schemas_dir / f"{agent_name}.yaml"
    if yaml_file.exists():
        try:
            file_schema = schema_from_yaml_file(yaml_file)
        except Exception:
            pass

    # Try database
    try:
        repo = Repository(Agent, table_name="agents")
        db_agent = await repo.get_by_name(agent_name)
        if db_agent and db_agent.enabled:
            db_schema = schema_from_yaml(db_agent.content)
    except Exception:
        pass

    # Apply precedence
    if database_first:
        return db_schema or file_schema
    else:
        return file_schema or db_schema


def get_available_agents(schemas_dir: Path | None = None) -> list[str]:
    """
    Get list of available agent names from filesystem.

    This is a synchronous helper for listing file-based agents.
    For full listing including database agents, use the API endpoint.

    Args:
        schemas_dir: Optional path to schemas directory

    Returns:
        List of agent names (without .yaml extension)
    """
    if schemas_dir is None:
        schemas_dir = Path(__file__).parent.parent.parent / "schemas"

    agents = []
    if schemas_dir.exists():
        for yaml_file in schemas_dir.glob("*.yaml"):
            try:
                schema = schema_from_yaml_file(yaml_file)
                agents.append(schema.json_schema_extra.name)
            except Exception:
                pass

    return sorted(agents)
