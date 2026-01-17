"""
Agent Provider - The Pydantic AI Factory
=========================================

This module is the CORE of the declarative agent system. It transforms YAML agent
schemas into executable pydantic-ai Agent instances with proper tool bindings.

ARCHITECTURE OVERVIEW
---------------------

The flow is: YAML Schema → AgentSchema → Pydantic Model → pydantic_ai.Agent

    1. User defines agent in YAML with JSON Schema format
    2. AgentSchema parses the YAML (schema.py)
    3. This module (provider.py) creates the pydantic-ai Agent:
       - Extracts system prompt from schema.description
       - Builds output model from schema.properties (if structured_output=true)
       - Filters and attaches MCP tools based on schema.tools list
       - Configures LLM model, temperature, max_iterations

KEY DESIGN PATTERNS
-------------------

1. **JsonSchema → Pydantic Model Conversion**
   The _build_output_model() function dynamically creates a Pydantic model from
   JSON Schema properties. This enables type-safe structured outputs from agents.

2. **Dual-Use Schema: System Prompt + Output Structure**
   The YAML schema serves two purposes:
   - schema.description = The agent's system prompt (instructions)
   - schema.properties = The expected output structure

   This is different from typical JSON Schema usage where description describes
   the schema itself. Here, description IS the agent's behavior specification.

3. **Description Stripping for Structured Output**
   CRITICAL: When structured_output=true, we MUST strip the description field
   from the JSON schema sent to the LLM. Why?

   - The description (system prompt) is sent separately via pydantic-ai's system_prompt
   - If we also include it in the output schema, the LLM receives it TWICE
   - This wastes tokens and can confuse the model about what to output

   The _create_schema_wrapper() function handles this by overriding model_json_schema()
   to remove the description field from the generated JSON schema.

4. **Properties-to-Prompt Conversion (structured_output=false)**
   When structured_output is disabled (default), the agent generates free-form text.
   But we still want the LLM to know about the expected structure for internal tracking.

   _convert_properties_to_prompt() renders the JSON Schema as YAML-like guidance text
   that's appended to the system prompt. This informs without enforcing.

5. **Tool Filtering by Schema Configuration**
   The schema.json_schema_extra.tools list controls which tools the agent can access:
   - Empty list [] = Agent cannot use any tools
   - Specific tools [{name: "search"}] = Only those tools are available
   - Not specified = All provided tools are available

   This enables fine-grained control over agent capabilities per-agent.

USAGE EXAMPLE
-------------

    # Load schema from YAML file
    schema = schema_from_yaml_file("schemas/query-agent.yaml")

    # Create agent runtime with tools
    runtime = await create_agent(
        schema=schema,
        model_name="openai:gpt-4.1",
        tools=[search_tool, action_tool],
        context=agent_context,
    )

    # Use the agent
    result = await runtime.agent.run("Find documents about machine learning")

FUTURE WORK
-----------
- Remote MCP tool support via FastMCP client (not yet implemented)
- Dynamic tool loading from MCP server URIs
"""

from typing import Any

from loguru import logger
from pydantic import BaseModel, create_model
from pydantic_ai import Agent

from remlight.agentic.schema import AgentSchema, AgentSchemaMetadata
from remlight.agentic.context import AgentContext
from remlight.settings import settings


class AgentRuntime(BaseModel):
    """
    Container for a configured agent with its runtime settings.

    This bundles the pydantic-ai Agent instance with the resolved configuration
    from the schema (temperature, max_iterations) and identifies which schema
    created this agent.

    The separation of AgentRuntime from Agent allows:
    - Passing around agent + config together
    - Accessing schema metadata after agent creation
    - Different runtime configs for same agent type

    Attributes:
        agent: The pydantic-ai Agent instance ready to execute
        temperature: LLM sampling temperature (from schema or settings)
        max_iterations: Max tool call iterations (prevents infinite loops)
        schema_name: Name of the agent schema that created this runtime
    """

    agent: Any  # pydantic_ai.Agent (Any to avoid type issues with generic)
    temperature: float
    max_iterations: int
    schema_name: str

    model_config = {"arbitrary_types_allowed": True}


def _build_output_model(properties: dict[str, Any], required: list[str]) -> type:
    """
    Dynamically create a Pydantic model from JSON Schema properties.

    This is the JSON Schema → Pydantic conversion that enables typed structured output.
    The function maps JSON Schema types to Python types and creates a Pydantic model
    using create_model().

    TYPE MAPPING:
        JSON Schema     →   Python Type
        "string"            str
        "number"            float
        "integer"           int
        "boolean"           bool
        "array"             list
        "object"            dict

    REQUIRED vs OPTIONAL:
        - Fields in the `required` list get ... (Ellipsis) as default = required
        - Other fields get None as default = optional

    WHY DYNAMIC MODELS?
        Agents defined in YAML have different output structures. Rather than
        pre-defining all possible output models, we generate them at runtime
        from the schema. This enables truly declarative agent definitions.

    EXAMPLE:
        properties = {
            "answer": {"type": "string", "description": "The response"},
            "confidence": {"type": "number", "description": "0.0 to 1.0"}
        }
        required = ["answer"]

        → Creates model equivalent to:
        class AgentOutput(BaseModel):
            answer: str
            confidence: float | None = None

    Args:
        properties: JSON Schema properties dict from agent schema
        required: List of required field names

    Returns:
        Dynamically created Pydantic model class
    """
    fields = {}
    for name, prop in properties.items():
        # Map JSON Schema type to Python type (default to str for unknown types)
        field_type = str  # Default: treat unknown types as strings
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

        # Required fields use ... (Ellipsis), optional fields default to None
        default = ... if name in required else None
        fields[name] = (field_type, default)

    # create_model() is Pydantic's factory for runtime model creation
    return create_model("AgentOutput", **fields)


def _create_schema_wrapper(
    result_type: type[BaseModel], strip_description: bool = True
) -> type[BaseModel]:
    """
    Create a wrapper model that strips the description field from JSON schema.

    THE DOUBLE-PASS PROBLEM
    -----------------------
    In our declarative agent system, the YAML schema's `description` field serves
    as the system prompt. When using structured output, pydantic-ai:

    1. Sends the system prompt to the LLM (from Agent's system_prompt parameter)
    2. Also sends the output JSON schema to the LLM (for structured output)

    If the output schema includes a `description` field (which Pydantic generates
    from docstrings and field descriptions), the LLM receives similar text TWICE:
    once as the system prompt, once embedded in the JSON schema.

    This wrapper prevents double-passing by overriding model_json_schema() to
    remove the description field before the schema is sent to the LLM.

    HOW IT WORKS
    ------------
    1. We subclass the dynamically-created output model
    2. Override model_json_schema() to call parent, then remove 'description'
    3. Return the wrapped model - externally identical but with cleaner schema

    BEFORE (without wrapper):
        {
            "type": "object",
            "description": "You are a helpful assistant...",  # DUPLICATE!
            "properties": {...}
        }

    AFTER (with wrapper):
        {
            "type": "object",
            "properties": {...}
        }

    Args:
        result_type: The original Pydantic model (from _build_output_model)
        strip_description: If True, removes description from generated schema

    Returns:
        Wrapped model class with description-stripped JSON schema generation
    """
    if not strip_description:
        return result_type

    class SchemaWrapper(result_type):  # type: ignore
        @classmethod
        def model_json_schema(cls, **kwargs):
            # Generate the standard JSON schema
            schema = super().model_json_schema(**kwargs)
            # Remove model-level description to avoid duplication with system prompt
            # The system prompt is already sent separately via pydantic-ai
            schema.pop("description", None)
            return schema

    # Preserve the original model name for debugging and error messages
    SchemaWrapper.__name__ = result_type.__name__
    return SchemaWrapper


def _render_schema_recursive(schema: dict[str, Any], indent: int = 0) -> list[str]:
    """
    Recursively render a JSON Schema as YAML-like text for LLM consumption.

    When structured_output is DISABLED, we still want the LLM to understand
    the expected response structure. This function converts JSON Schema into
    a human-readable YAML format that's included in the system prompt.

    WHY YAML-LIKE FORMAT?
    --------------------
    - LLMs understand YAML well (common in training data)
    - Shows exact field names the LLM should use internally
    - Includes descriptions as comments for context
    - Marks required vs optional fields clearly

    HANDLING NESTED STRUCTURES
    -------------------------
    The function handles:
    - object: Recursively renders nested properties
    - array: Shows item type, recursively renders if items are objects
    - primitives: Shows type directly, with enum values if constrained

    EXAMPLE OUTPUT:
        research_context:
          # Background information gathered
          topic: string (required)
          sources: [string]
            # List of source URLs
          confidence:
            # Confidence in the research
            score: number
            reasoning: string

    Args:
        schema: JSON Schema dict to render
        indent: Current indentation level (for recursion)

    Returns:
        List of lines (to be joined with newlines)
    """
    lines = []
    prefix = "  " * indent  # Two spaces per indent level (YAML convention)

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
                # Nested object: render field name, then recurse into properties
                lines.append(f"{prefix}{field_name}:{req_marker}")
                if field_desc:
                    lines.append(f"{prefix}  # {field_desc}")
                nested_lines = _render_schema_recursive(field_def, indent + 1)
                lines.extend(nested_lines)

            elif field_type == "array":
                # Array: show item type in brackets, recurse if items are objects
                items = field_def.get("items", {})
                items_type = items.get("type", "any")
                lines.append(f"{prefix}{field_name}: [{items_type}]{req_marker}")
                if field_desc:
                    lines.append(f"{prefix}  # {field_desc}")
                if items_type == "object":
                    # If array contains objects, show their structure
                    lines.append(f"{prefix}  # Each item has:")
                    nested_lines = _render_schema_recursive(items, indent + 2)
                    lines.extend(nested_lines)

            else:
                # Primitive type: show type (and enum values if constrained)
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
    Convert schema properties to prompt guidance text for unstructured output.

    THE UNSTRUCTURED OUTPUT PATTERN
    -------------------------------
    Most agents should NOT use structured output. Why?

    1. **Natural Conversation**: Structured output forces JSON, which is awkward
       for conversational agents. Users expect natural language, not JSON blobs.

    2. **Flexibility**: The LLM can adapt its response format to context
       (bullet points, paragraphs, code blocks, etc.)

    3. **Action Tool for Metadata**: When agents need to emit structured data
       (confidence scores, sources, etc.), they use the `action` tool instead
       of structured output. This keeps text responses natural while still
       capturing metadata.

    But we still want the LLM to organize its thinking. This function generates
    prompt guidance that describes the expected structure WITHOUT enforcing it.

    THE ANSWER FIELD PATTERN
    -----------------------
    By convention, agent schemas have an "answer" field for the user-visible output.
    Other fields are for internal tracking (confidence, sources, etc.).

    This function:
    - Separates "answer" field (OUTPUT) from other fields (INTERNAL)
    - Tells the LLM to only output the answer content as natural text
    - Documents internal fields for the LLM's tracking purposes

    EXAMPLE GENERATED TEXT:
    ----------------------
    ## Internal Thinking Structure (DO NOT output these labels)

    Use this structure to organize your thinking, but ONLY output the answer content:

    **OUTPUT (what the user sees):** Your response to the user

    **INTERNAL (for your tracking only - do NOT include in output):**

    Schema (use these EXACT field names):
    ```yaml
    confidence: number
      # Your confidence level 0.0 to 1.0
    sources: [string]
      # Entity keys you referenced
    ```

    CRITICAL: Your response must be ONLY the conversational answer text.
    Do NOT output field names like 'answer:' or JSON - just the response itself.

    Args:
        properties: Schema properties dict (field_name → field_definition)

    Returns:
        Prompt text to append to system prompt
    """
    if not properties:
        return ""

    # Separate answer (output) from other fields (internal tracking)
    # The "answer" field is special - it's what the user sees
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
    # These fields are for the LLM's internal use, not output
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
    """
    Build the complete system prompt from schema and context.

    The system prompt is the agent's "personality" and instructions. It comes
    from two sources that are combined:

    1. **User Profile Hint** (from context):
       If the user is authenticated and has a profile, include relevant
       user context at the start of the system prompt. This enables
       personalized responses without the agent needing to look up user info.

    2. **Schema System Prompt** (from YAML description):
       The main agent instructions from the schema's description field,
       plus any extended instructions from json_schema_extra.system_prompt.

    USER PROFILE INTEGRATION
    -----------------------
    The user_profile_hint is loaded during context creation (from_headers_with_profile)
    by calling the MCP resource `user://profile`. It contains info like:
    - User's name/email
    - Preferences
    - Organization context

    This is prepended to give the agent immediate user context without
    requiring a tool call.

    Args:
        schema: AgentSchema containing the base system prompt
        context: Optional AgentContext with user profile hint

    Returns:
        Complete system prompt string ready for the LLM
    """
    parts = []

    # Add user profile hint if available (enables personalized responses)
    if context and context.user_profile_hint:
        parts.append(f"## User Context\n{context.user_profile_hint}\n")

    # Add main system prompt from schema (description + optional system_prompt extension)
    parts.append(schema.get_system_prompt())

    return "\n".join(parts)


async def create_agent(
    schema: dict[str, Any] | AgentSchema,
    model_name: str | None = None,
    tools: list | None = None,
    context: AgentContext | None = None,
) -> AgentRuntime:
    """
    Create a pydantic-ai Agent from a declarative YAML/JSON schema.

    This is the MAIN ENTRY POINT for the agent factory. It takes a declarative
    schema (typically loaded from YAML) and produces an executable agent with:
    - Configured LLM model
    - System prompt from schema description
    - Structured output (if enabled) with description-stripped JSON schema
    - Filtered tool set based on schema configuration

    THE AGENT CREATION PIPELINE
    ---------------------------
    1. Parse schema (dict → AgentSchema if needed)
    2. Resolve runtime config (temperature, max_iterations from schema or settings)
    3. Build system prompt (schema.description + user profile hint)
    4. Configure output type:
       - structured_output=true: Generate Pydantic model, wrap to strip description
       - structured_output=false: Append schema guidance to prompt, use text output
    5. Filter tools based on schema.json_schema_extra.tools
    6. Create pydantic-ai Agent with all configuration
    7. Return AgentRuntime bundle

    OUTPUT TYPE SELECTION
    --------------------
    The output_type determines how pydantic-ai processes the agent's response:

    - `str` (default): Agent returns free-form text. Best for conversational agents.
    - Pydantic model: Agent returns JSON matching the schema. For data extraction.

    We default to `str` (unstructured) because:
    1. Most agents are conversational
    2. The `action` tool handles structured metadata emission
    3. Structured output constrains LLM creativity

    TOOL FILTERING
    -------------
    The schema's `tools` list controls agent capabilities:

        # No tools list = ALL provided tools available
        json_schema_extra:
          tools: []  # Empty = NO tools available

        json_schema_extra:
          tools:
            - name: search    # Only search available
            - name: action    # Only action available

    This enables creating restricted agents that can only use specific tools,
    or unrestricted agents that have access to everything.

    Args:
        schema: Agent definition as dict (raw YAML) or AgentSchema (parsed)
        model_name: LLM model identifier (e.g., "openai:gpt-4.1", "anthropic:claude-sonnet-4-5-20250929")
                   Falls back to settings.llm.default_model
        tools: List of tool functions or dict of {name: FunctionTool}
               These are filtered based on schema.tools configuration
        context: AgentContext with session info and user_profile_hint

    Returns:
        AgentRuntime containing:
        - agent: Configured pydantic-ai Agent ready to execute
        - temperature: Resolved temperature setting
        - max_iterations: Max tool call loops (prevents infinite agent loops)
        - schema_name: Name of the schema for logging/tracing

    Example:
        # Load and create a query agent
        schema = schema_from_yaml_file("schemas/query-agent.yaml")
        runtime = await create_agent(
            schema=schema,
            model_name="openai:gpt-4.1",
            tools=mcp_tools,
            context=AgentContext(user_id="user-123", session_id="sess-456"),
        )
        result = await runtime.agent.run("Find documents about AI")
    """
    # Parse schema if provided as raw dict (e.g., from YAML load)
    if isinstance(schema, dict):
        schema = AgentSchema(**schema)

    meta: AgentSchemaMetadata = schema.json_schema_extra
    model_name = model_name or settings.llm.default_model

    # Resolve runtime configuration from schema overrides or global settings
    # Schema can override defaults for specific agent types (e.g., low temp for extraction)
    temperature = meta.override_temperature or settings.llm.temperature
    max_iterations = meta.override_max_iterations or settings.llm.max_iterations

    # Build the system prompt from schema description + user context
    system_prompt = _build_system_prompt(schema, context)

    # =========================================================================
    # OUTPUT TYPE CONFIGURATION
    # =========================================================================
    # Two modes:
    # 1. structured_output=true: Generate Pydantic model, wrap to strip description
    # 2. structured_output=false (default): Append schema as prompt guidance, use str
    #
    # The key insight: schema.description is the system prompt, NOT a schema description.
    # When using structured output, we must remove it from the output schema to avoid
    # sending it to the LLM twice (once as system prompt, once in output schema).
    # =========================================================================

    output_type: type | None = str  # Default: free-form text output

    if meta.structured_output is True and schema.properties:
        # STRUCTURED OUTPUT MODE
        # Convert JSON Schema properties → Pydantic model → Wrapped model (description stripped)
        base_model = _build_output_model(schema.properties, schema.required)
        # Wrap to strip description field from JSON schema sent to LLM
        output_type = _create_schema_wrapper(base_model, strip_description=True)
        logger.debug(f"Created structured output model: {output_type.__name__} (description stripped)")

    elif schema.properties:
        # UNSTRUCTURED OUTPUT MODE (default)
        # Convert properties to prompt guidance that informs without enforcing
        # The LLM sees the expected structure but can output natural language
        properties_prompt = _convert_properties_to_prompt(schema.properties)
        if properties_prompt:
            system_prompt = system_prompt + "\n\n" + properties_prompt

    # =========================================================================
    # TOOL FILTERING
    # =========================================================================
    # Filter provided tools based on schema.json_schema_extra.tools configuration
    #
    # Three cases:
    # 1. tools not specified in schema → ALL provided tools available
    # 2. tools: [] (empty list) → NO tools available (agent can't call tools)
    # 3. tools: [{name: "x"}] → only named tools available
    # =========================================================================

    agent_tools = []
    schema_tools = meta.tools  # List of MCPToolReference or dicts
    has_tool_filter = len(schema_tools) > 0 if schema_tools else False

    # Extract allowed tool names from schema (handles both MCPToolReference and dict)
    allowed_tool_names: set[str] = set()
    for t in (schema_tools or []):
        if hasattr(t, "name"):  # MCPToolReference object
            allowed_tool_names.add(t.name)
        elif isinstance(t, dict) and "name" in t:  # Plain dict from YAML
            allowed_tool_names.add(t["name"])

    if tools:
        if isinstance(tools, dict):
            # FastMCP format: {name: FunctionTool}
            # FunctionTool has .fn attribute with the actual callable
            for name, tool in tools.items():
                # Apply filter: skip tools not in schema's allowed list
                if has_tool_filter and name not in allowed_tool_names:
                    continue
                # Extract callable from FunctionTool or use directly
                if hasattr(tool, "fn"):
                    agent_tools.append(tool.fn)
                elif callable(tool):
                    agent_tools.append(tool)

        elif isinstance(tools, list):
            # List of callables or FunctionTool objects
            for tool in tools:
                tool_name = getattr(tool, "__name__", None) or getattr(tool, "name", None)
                # Apply filter: skip tools not in schema's allowed list
                if has_tool_filter and tool_name and tool_name not in allowed_tool_names:
                    continue
                if hasattr(tool, "fn"):
                    agent_tools.append(tool.fn)
                elif callable(tool):
                    agent_tools.append(tool)

    # =========================================================================
    # AGENT CREATION
    # =========================================================================
    # Create the pydantic-ai Agent with all resolved configuration
    # This is the final step that produces the executable agent
    # =========================================================================

    agent = Agent(
        model=model_name,
        system_prompt=system_prompt,
        output_type=output_type,
        tools=agent_tools,
    )

    # Bundle agent with runtime config for downstream use
    return AgentRuntime(
        agent=agent,
        temperature=temperature,
        max_iterations=max_iterations,
        schema_name=meta.name,
    )
