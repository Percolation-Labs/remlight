"""Agent provider using Pydantic AI."""

from typing import Any

from pydantic import BaseModel, create_model
from pydantic_ai import Agent

from remlight.agentic.schema import AgentSchema, AgentSchemaMetadata
from remlight.agentic.context import AgentContext
from remlight.settings import settings


class AgentRuntime(BaseModel):
    """Container for agent with resolved runtime configuration."""

    agent: Any  # pydantic_ai.Agent (Any to avoid type issues)
    temperature: float
    max_iterations: int
    schema_name: str

    model_config = {"arbitrary_types_allowed": True}


def _build_output_model(properties: dict[str, Any], required: list[str]) -> type:
    """Build a Pydantic model from JSON schema properties."""
    fields = {}
    for name, prop in properties.items():
        field_type = str  # Default
        if prop.get("type") == "number":
            field_type = float
        elif prop.get("type") == "integer":
            field_type = int
        elif prop.get("type") == "boolean":
            field_type = bool
        elif prop.get("type") == "array":
            field_type = list
        elif prop.get("type") == "object":
            field_type = dict

        default = ... if name in required else None
        fields[name] = (field_type, default)

    return create_model("AgentOutput", **fields)


def _build_system_prompt(schema: AgentSchema, context: AgentContext | None) -> str:
    """Build system prompt with optional user profile hint."""
    parts = []

    # Add user profile hint if available
    if context and context.user_profile_hint:
        parts.append(f"## User Context\n{context.user_profile_hint}\n")

    # Add main system prompt
    parts.append(schema.get_system_prompt())

    return "\n".join(parts)


async def create_agent(
    schema: dict[str, Any] | AgentSchema,
    model_name: str | None = None,
    tools: list | None = None,
    context: AgentContext | None = None,
) -> AgentRuntime:
    """
    Create a Pydantic AI Agent from a JSON/YAML schema.

    Args:
        schema: Agent schema (dict or AgentSchema)
        model_name: Model identifier (e.g., "openai:gpt-4o-mini")
        tools: List of tool functions to make available
        context: Agent execution context (includes user_profile_hint)

    Returns:
        AgentRuntime with configured agent
    """
    # Parse schema if dict
    if isinstance(schema, dict):
        schema = AgentSchema(**schema)

    meta: AgentSchemaMetadata = schema.json_schema_extra
    model_name = model_name or settings.llm.default_model

    # Resolve temperature and max_iterations
    temperature = meta.override_temperature or settings.llm.temperature
    max_iterations = meta.override_max_iterations or settings.llm.max_iterations

    # Build output model if structured output
    output_type = None
    if meta.structured_output is not False and schema.properties:
        output_type = _build_output_model(schema.properties, schema.required)

    # Create system prompt with user profile hint
    system_prompt = _build_system_prompt(schema, context)

    # Convert tools: FastMCP returns dict of FunctionTool, pydantic-ai needs list of callables
    agent_tools = []
    if tools:
        if isinstance(tools, dict):
            # FastMCP format: {name: FunctionTool}
            for tool in tools.values():
                if hasattr(tool, "fn"):
                    agent_tools.append(tool.fn)
                elif callable(tool):
                    agent_tools.append(tool)
        elif isinstance(tools, list):
            # Already a list - extract callables
            for tool in tools:
                if hasattr(tool, "fn"):
                    agent_tools.append(tool.fn)
                elif callable(tool):
                    agent_tools.append(tool)

    # Create agent
    agent = Agent(
        model=model_name,
        system_prompt=system_prompt,
        output_type=output_type,
        tools=agent_tools,
    )

    return AgentRuntime(
        agent=agent,
        temperature=temperature,
        max_iterations=max_iterations,
        schema_name=meta.name,
    )
