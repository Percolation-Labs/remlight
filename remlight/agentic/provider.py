"""Agent provider using Pydantic AI.

Key Design Patterns:
1. JsonSchema → Pydantic Model conversion
2. Agent schema contains both system prompt (description) AND output schema (properties)
3. When structured_output=true, description is STRIPPED from JSON schema sent to LLM
4. When structured_output=false, properties are converted to prompt guidance
5. MCP tools loaded dynamically from schema metadata

FUTURE: Remote MCP tool support via FastMCP client - not yet implemented
"""

from typing import Any

from loguru import logger
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


def _create_schema_wrapper(
    result_type: type[BaseModel], strip_description: bool = True
) -> type[BaseModel]:
    """
    Create wrapper model that strips description from JSON schema.

    When structured_output is enabled, the schema.description contains the system prompt.
    Including it in the output schema would duplicate information sent to the LLM.
    This wrapper removes the model-level description from the JSON schema.

    Args:
        result_type: Original Pydantic model
        strip_description: If True, removes description from schema

    Returns:
        Wrapper model with customized schema generation
    """
    if not strip_description:
        return result_type

    class SchemaWrapper(result_type):  # type: ignore
        @classmethod
        def model_json_schema(cls, **kwargs):
            schema = super().model_json_schema(**kwargs)
            # Remove model-level description to avoid duplication with system prompt
            schema.pop("description", None)
            return schema

    SchemaWrapper.__name__ = result_type.__name__
    return SchemaWrapper


def _render_schema_recursive(schema: dict[str, Any], indent: int = 0) -> list[str]:
    """
    Recursively render a JSON schema as YAML-like text with exact field names.

    Ensures the LLM sees actual field names for nested objects.
    """
    lines = []
    prefix = "  " * indent

    schema_type = schema.get("type", "any")

    if schema_type == "object":
        props = schema.get("properties", {})
        required = schema.get("required", [])

        for field_name, field_def in props.items():
            field_type = field_def.get("type", "any")
            field_desc = field_def.get("description", "")
            is_required = field_name in required

            req_marker = " (required)" if is_required else ""
            if field_type == "object":
                lines.append(f"{prefix}{field_name}:{req_marker}")
                if field_desc:
                    lines.append(f"{prefix}  # {field_desc}")
                nested_lines = _render_schema_recursive(field_def, indent + 1)
                lines.extend(nested_lines)
            elif field_type == "array":
                items = field_def.get("items", {})
                items_type = items.get("type", "any")
                lines.append(f"{prefix}{field_name}: [{items_type}]{req_marker}")
                if field_desc:
                    lines.append(f"{prefix}  # {field_desc}")
                if items_type == "object":
                    lines.append(f"{prefix}  # Each item has:")
                    nested_lines = _render_schema_recursive(items, indent + 2)
                    lines.extend(nested_lines)
            else:
                enum_vals = field_def.get("enum")
                if enum_vals:
                    type_str = f"{field_type} (one of: {', '.join(str(v) for v in enum_vals)})"
                else:
                    type_str = field_type
                lines.append(f"{prefix}{field_name}: {type_str}{req_marker}")
                if field_desc:
                    lines.append(f"{prefix}  # {field_desc}")

    return lines


def _convert_properties_to_prompt(properties: dict[str, Any]) -> str:
    """
    Convert schema properties to prompt guidance text.

    When structured_output is disabled, this converts the properties
    definition into natural language guidance that informs the agent
    about the expected response structure without forcing JSON output.

    Recursively renders nested schemas so the LLM sees exact field names.
    """
    if not properties:
        return ""

    # Separate answer (output) from other fields (internal tracking)
    answer_field = properties.get("answer")
    internal_fields = {k: v for k, v in properties.items() if k != "answer"}

    lines = ["## Internal Thinking Structure (DO NOT output these labels)"]
    lines.append("")
    lines.append("Use this structure to organize your thinking, but ONLY output the answer content:")
    lines.append("")

    # If there's an answer field, emphasize it's the ONLY output
    if answer_field:
        answer_desc = answer_field.get("description", "Your response")
        lines.append(f"**OUTPUT (what the user sees):** {answer_desc}")
        lines.append("")

    # Document internal fields with FULL recursive schema
    if internal_fields:
        lines.append("**INTERNAL (for your tracking only - do NOT include in output):**")
        lines.append("")
        lines.append("Schema (use these EXACT field names):")
        lines.append("```yaml")

        for field_name, field_def in internal_fields.items():
            field_type = field_def.get("type", "any")
            field_desc = field_def.get("description", "")

            if field_type == "object":
                lines.append(f"{field_name}:")
                if field_desc:
                    lines.append(f"  # {field_desc}")
                nested_lines = _render_schema_recursive(field_def, indent=1)
                lines.extend(nested_lines)
            elif field_type == "array":
                items = field_def.get("items", {})
                items_type = items.get("type", "any")
                lines.append(f"{field_name}: [{items_type}]")
                if field_desc:
                    lines.append(f"  # {field_desc}")
                if items_type == "object":
                    lines.append(f"  # Each item has:")
                    nested_lines = _render_schema_recursive(items, indent=2)
                    lines.extend(nested_lines)
            else:
                lines.append(f"{field_name}: {field_type}")
                if field_desc:
                    lines.append(f"  # {field_desc}")

        lines.append("```")

    lines.append("")
    lines.append("CRITICAL: Your response must be ONLY the conversational answer text.")
    lines.append("Do NOT output field names like 'answer:' or JSON - just the response itself.")

    return "\n".join(lines)


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
        model_name: Model identifier (e.g., "openai:gpt-4.1")
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

    # Create system prompt with user profile hint
    system_prompt = _build_system_prompt(schema, context)

    # Build output model if structured output is enabled
    # When structured_output: true with properties, use generated Pydantic model
    # The model is wrapped to STRIP the description field from JSON schema (avoids duplication)
    # When structured_output: false or no properties, use str (pydantic-ai text mode)
    output_type: type | None = str  # Default: text output mode
    if meta.structured_output is True and schema.properties:
        base_model = _build_output_model(schema.properties, schema.required)
        # Wrap to strip description from schema sent to LLM
        output_type = _create_schema_wrapper(base_model, strip_description=True)
        logger.debug(f"Created structured output model: {output_type.__name__} (description stripped)")
    elif schema.properties:
        # Convert properties to inline YAML prompt guidance (like remstack)
        # This informs the LLM about expected structure without forcing JSON
        properties_prompt = _convert_properties_to_prompt(schema.properties)
        if properties_prompt:
            system_prompt = system_prompt + "\n\n" + properties_prompt

    # Filter and convert tools based on schema configuration
    # Schema tools list specifies which tools this agent can access:
    # - tools: [] = NO tools allowed
    # - tools: [{name: "x"}] = only tool "x" allowed
    # - tools not specified (empty) = ALL tools allowed
    agent_tools = []
    schema_tools = meta.tools  # List of MCPToolReference or dicts
    has_tool_filter = len(schema_tools) > 0 if schema_tools else False

    # Extract tool names from schema - handle both MCPToolReference objects and dicts
    allowed_tool_names: set[str] = set()
    for t in (schema_tools or []):
        if hasattr(t, "name"):  # MCPToolReference
            allowed_tool_names.add(t.name)
        elif isinstance(t, dict) and "name" in t:  # dict
            allowed_tool_names.add(t["name"])

    if tools:
        if isinstance(tools, dict):
            # FastMCP format: {name: FunctionTool}
            for name, tool in tools.items():
                # Filter: if schema specifies tools, only include those listed
                if has_tool_filter and name not in allowed_tool_names:
                    continue
                if hasattr(tool, "fn"):
                    agent_tools.append(tool.fn)
                elif callable(tool):
                    agent_tools.append(tool)
        elif isinstance(tools, list):
            # Already a list - extract callables
            for tool in tools:
                tool_name = getattr(tool, "__name__", None) or getattr(tool, "name", None)
                # Filter: if schema specifies tools, only include those listed
                if has_tool_filter and tool_name and tool_name not in allowed_tool_names:
                    continue
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
