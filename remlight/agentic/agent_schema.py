"""
AgentSchema - Unified Schema Abstraction for Declarative Agents
================================================================

This module centralizes the agent schema abstraction, combining:
- Schema definition (JSON Schema + extensions)
- Output model generation (Pydantic models)
- Prompt generation from properties
- Option overrides from environment or explicit values
- Usage limits for controlling token/request budgets

DESIGN PHILOSOPHY
-----------------
The AgentSchema unifies three concerns:

1. **Declaration**: Define agents in YAML/JSON with JSON Schema structure
2. **Output**: Generate Pydantic models for structured output (with description stripping)
3. **Prompts**: Convert schema properties to prompt guidance for unstructured output

USAGE
-----
    from remlight.agentic.agent_schema import AgentSchema

    # Load from YAML
    schema = AgentSchema.from_yaml_file("schemas/query-agent.yaml")

    # Get output model for structured output
    OutputModel = schema.to_output_schema()  # Description stripped

    # Get prompt guidance for unstructured output
    prompt_guidance = schema.to_prompt()

    # Override options from env/config
    schema = schema.with_options(
        model="anthropic:claude-sonnet-4-5-20250929",
        temperature=0.3,
        max_iterations=10
    )

    # Set limits
    schema = schema.with_options(
        request_limit=10,
        total_tokens_limit=50000,
        tool_calls_limit=20
    )

    # Access all options via get_options()
    options = schema.get_options()
    options.model                    # → "anthropic:claude-sonnet-4-5-20250929"
    options.temperature              # → 0.3
    options.limits                   # → AgentUsageLimits
    options.limits.to_pydantic_ai()  # → pydantic_ai.UsageLimits
    options.extras                   # → {"custom": "value"}
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml
from pydantic import BaseModel, Field, create_model

if TYPE_CHECKING:
    from pydantic_ai import UsageLimits
    from fastapi import Request


class AgentContext(BaseModel):
    """
    Minimal execution context for agent runs.

    Carries WHO (user) and WHAT (session) through agent execution.
    Used by tools that need to scope operations to user/session.

    Creation:
        # From FastAPI request
        ctx = AgentContext.from_request(request)

        # Direct (for CLI/tests)
        ctx = AgentContext(user_id="user-123", session_id="sess-456")
    """
    user_id: str | None = None
    session_id: str | None = None
    trace_id: str | None = None

    @classmethod
    def from_request(cls, request: "Request") -> "AgentContext":
        """Extract context from FastAPI request headers."""
        return cls(
            user_id=request.headers.get("x-user-id"),
            session_id=request.headers.get("x-session-id"),
            trace_id=request.headers.get("x-trace-id"),
        )


class AgentUsageLimits(BaseModel):
    """
    Usage limits for agent runs (maps to pydantic-ai UsageLimits).

    These limits are passed to agent.run() to control resource usage.
    All limits are optional - None means no limit.

    LIMIT TYPES
    -----------
    request_limit: Max requests to the model (checked before each request)
    tool_calls_limit: Max successful tool calls (checked before execution)
    input_tokens_limit: Max input/prompt tokens (checked after response)
    output_tokens_limit: Max output/response tokens (checked after response)
    total_tokens_limit: Combined input + output tokens

    Example YAML:
        json_schema_extra:
          name: my-agent
          limits:
            request_limit: 10
            total_tokens_limit: 50000
    """

    request_limit: int | None = None
    tool_calls_limit: int | None = None
    input_tokens_limit: int | None = None
    output_tokens_limit: int | None = None
    total_tokens_limit: int | None = None

    def to_pydantic_ai(self) -> "UsageLimits":
        """Convert to pydantic-ai UsageLimits for agent.run()."""
        from pydantic_ai import UsageLimits

        return UsageLimits(
            request_limit=self.request_limit,
            tool_calls_limit=self.tool_calls_limit,
            input_tokens_limit=self.input_tokens_limit,
            output_tokens_limit=self.output_tokens_limit,
            total_tokens_limit=self.total_tokens_limit,
        )

    def is_empty(self) -> bool:
        """Check if all limits are None."""
        return all(
            v is None
            for v in [
                self.request_limit,
                self.tool_calls_limit,
                self.input_tokens_limit,
                self.output_tokens_limit,
                self.total_tokens_limit,
            ]
        )


class AgentOptions(BaseModel):
    """
    Runtime options for agent execution.

    Accessed via schema.get_options() - bundles all configurable runtime
    settings in one place.

    STRUCTURE
    ---------
    model: LLM model override (e.g., "openai:gpt-4.1")
    temperature: Sampling temperature (0.0-1.0)
    max_iterations: Max tool call loops
    limits: Token/request/tool call limits (AgentUsageLimits)
    extras: Additional custom options (dict)

    Example:
        options = schema.get_options()
        options.model         # → "openai:gpt-4.1"
        options.limits        # → AgentUsageLimits
        options.limits.to_pydantic_ai()  # → pydantic_ai.UsageLimits
        options.extras        # → {"custom_key": "value"}
    """

    model: str | None = None
    temperature: float | None = None
    max_iterations: int | None = None
    limits: AgentUsageLimits = Field(default_factory=AgentUsageLimits)
    extras: dict[str, Any] = Field(default_factory=dict)

    def with_defaults(
        self,
        default_model: str | None = None,
        default_temperature: float | None = None,
        default_max_iterations: int | None = None,
    ) -> "AgentOptions":
        """
        Return options with defaults filled in for None values.

        Args:
            default_model: Fallback model if self.model is None
            default_temperature: Fallback temperature
            default_max_iterations: Fallback max_iterations

        Returns:
            New AgentOptions with defaults applied
        """
        return AgentOptions(
            model=self.model or default_model,
            temperature=self.temperature if self.temperature is not None else default_temperature,
            max_iterations=self.max_iterations if self.max_iterations is not None else default_max_iterations,
            limits=self.limits,
            extras=self.extras,
        )

    def for_agent(self, default_model: str | None = None) -> dict[str, Any]:
        """
        Return kwargs dict for pydantic-ai Agent constructor.

        Maps our options to pydantic-ai conventions:
        - model: LLM model string
        - model_settings: dict with temperature, etc.

        Note: usage_limits are passed to run/run_stream(), not Agent()

        Example:
            options = schema.get_options()
            agent = Agent(**options.for_agent(settings.llm.default_model))

        Args:
            default_model: Fallback if model not set

        Returns:
            Dict ready to unpack into Agent()
        """
        kwargs: dict[str, Any] = {}

        # Model (required for Agent)
        kwargs["model"] = self.model or default_model

        # Model settings (temperature, etc.)
        model_settings = {}
        if self.temperature is not None:
            model_settings["temperature"] = self.temperature
        if model_settings:
            kwargs["model_settings"] = model_settings

        return kwargs


class MCPToolReference(BaseModel):
    """
    Reference to an MCP tool available to the agent.

    Attributes:
        name: Tool function name (matches tool.__name__)
        server: Server alias (None = "local" = built-in)
        description: Optional description override
    """

    name: str
    server: str | None = None
    description: str | None = None


class MCPResourceReference(BaseModel):
    """
    Reference to MCP resources accessible to the agent.

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


class AgentExtraSchema(BaseModel):
    """
    Extension schema for agent-specific configuration.

    This contains REMLight-specific metadata stored in json_schema_extra.
    It's the "extension point" for custom fields beyond standard JSON Schema.

    KEY FIELDS
    ----------
    kind: Always "agent" - identifies this as an agent schema
    name: Unique identifier for loading/logging/multi-agent references
    tools: List of MCPToolReference - available tools for the agent
    resources: List of MCPResourceReference - accessible resources

    OVERRIDE OPTIONS
    ----------------
    These can be overridden via .with_options() or environment variables:

    model: Force a specific model (highest priority)
    temperature: Per-agent temperature
    max_iterations: Per-agent max tool loops
    limits: Token/request/tool call limits for runs

    Environment variable names (when using with_options with env=True):
    - AGENT_MODEL
    - AGENT_TEMPERATURE
    - AGENT_MAX_ITERATIONS
    - AGENT_REQUEST_LIMIT
    - AGENT_TOTAL_TOKENS_LIMIT
    """

    kind: str | None = "agent"
    name: str
    version: str = "1.0.0"
    system_prompt: str | None = None
    structured_output: bool | None = None
    tools: list[MCPToolReference] = Field(default_factory=list)
    resources: list[MCPResourceReference] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    author: str | None = None

    # Runtime options - accessed via schema.get_options()
    model: str | None = None
    temperature: float | None = None
    max_iterations: int | None = None
    limits: AgentUsageLimits | None = None

    model_config = {"extra": "allow"}


class PromptFormat:
    """
    Format options for .to_prompt() output.

    Attributes:
        include_answer: Whether to document the 'answer' field
        include_internal: Whether to document internal fields
        yaml_style: Use YAML-like formatting (default) vs JSON
        include_instructions: Add critical instructions about output format
    """

    def __init__(
        self,
        include_answer: bool = True,
        include_internal: bool = True,
        yaml_style: bool = True,
        include_instructions: bool = True,
    ):
        self.include_answer = include_answer
        self.include_internal = include_internal
        self.yaml_style = yaml_style
        self.include_instructions = include_instructions


class AgentSchema(BaseModel):
    """
    Unified schema for declarative agent definition.

    Extends JSON Schema with REMLight-specific extensions and provides
    methods for output model generation and prompt conversion.

    STRUCTURE
    ---------
    type: Always "object"
    description: The agent's system prompt (most important field!)
    properties: Output structure definition
    required: Required output fields
    json_schema_extra: AgentExtraSchema with extensions

    METHODS
    -------
    to_output_schema(): Generate Pydantic model with description stripped
    to_prompt(format): Convert properties to prompt guidance text
    with_options(**opts): Create copy with overridden options
    get_system_prompt(): Get combined system prompt

    Example YAML:
        type: object
        description: |
          You are a helpful assistant.

        properties:
          answer:
            type: string

        json_schema_extra:
          kind: agent
          name: my-agent
          tools:
            - name: search
    """

    type: Literal["object"] = "object"
    description: str
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)
    json_schema_extra: AgentExtraSchema
    definitions: dict[str, Any] | None = None
    additionalProperties: bool = False

    # Cache for generated output model
    _output_model_cache: type[BaseModel] | None = None

    model_config = {"arbitrary_types_allowed": True}

    def get_system_prompt(self) -> str:
        """
        Get the complete system prompt from schema.

        Combines:
        1. description (main system prompt)
        2. json_schema_extra.system_prompt (optional extension)
        """
        parts = [self.description]
        if self.json_schema_extra.system_prompt:
            parts.append(self.json_schema_extra.system_prompt)
        return "\n\n".join(parts)

    @property
    def name(self) -> str:
        """Convenience accessor for the agent name."""
        return self.json_schema_extra.name

    @property
    def tools(self) -> list[MCPToolReference]:
        """Convenience accessor for tool references."""
        return self.json_schema_extra.tools

    @property
    def structured_output(self) -> bool:
        """Whether this agent uses structured output."""
        return self.json_schema_extra.structured_output or False

    def get_options(self, **overrides) -> dict[str, Any]:
        """
        Get runtime options as a dict for pydantic-ai Agent.

        Loads defaults from schema and settings, allows runtime overrides.
        Returns dict with pydantic-ai compatible keys (model, model_settings, etc.)

        Priority: override > schema > settings default

        Example:
            options = schema.get_options()
            agent = Agent(**options, system_prompt=..., toolsets=...)

            # With overrides
            options = schema.get_options(model="anthropic:claude-sonnet-4-5-20250929", temperature=0.7)

        Args:
            **overrides: Runtime overrides (model, temperature, etc.)

        Returns:
            Dict ready to unpack into Agent()
        """
        from remlight.settings import settings

        extra = self.json_schema_extra

        # Priority: override > schema > settings default
        model = overrides.get("model") or extra.model or settings.llm.default_model
        temperature = (
            overrides.get("temperature")
            if "temperature" in overrides
            else extra.temperature if extra.temperature is not None
            else settings.llm.temperature
        )

        # Build pydantic-ai compatible kwargs
        options: dict[str, Any] = {"model": model}

        # Model settings (temperature, etc.)
        if temperature is not None:
            options["model_settings"] = {"temperature": temperature}

        return options

    def to_output_schema(self, strip_description: bool = True) -> type[BaseModel] | type[str]:
        """
        Generate a Pydantic model from schema properties.

        Returns str if structured_output is disabled or no properties defined.
        Otherwise creates a dynamic Pydantic model for structured output.

        When strip_description=True (default), the model's JSON schema
        will NOT include the description field, avoiding duplication
        with the system prompt.

        Returns:
            Pydantic model class with fields from properties, or str
        """
        # Return str if structured output disabled or no properties
        if not self.structured_output or not self.properties:
            return str  # type: ignore

        # Build the base model from properties
        fields = {}
        for name, prop in self.properties.items():
            field_type = self._json_type_to_python(prop.get("type", "string"))
            default = ... if name in self.required else None
            fields[name] = (field_type, default)

        base_model = create_model("AgentOutput", **fields)

        if not strip_description:
            return base_model

        # Create wrapper that strips description from JSON schema
        class SchemaWrapper(base_model):  # type: ignore
            @classmethod
            def model_json_schema(cls, **kwargs: Any) -> dict[str, Any]:
                schema = super().model_json_schema(**kwargs)
                schema.pop("description", None)
                return schema

        SchemaWrapper.__name__ = "AgentOutput"
        return SchemaWrapper

    def to_prompt(self, format: PromptFormat | None = None) -> str:
        """
        Convert schema properties to prompt guidance text.

        This generates human-readable documentation of the expected output
        structure for unstructured (text) output mode. The LLM sees this
        guidance but isn't forced to output JSON.

        THE ANSWER FIELD PATTERN
        ------------------------
        By convention, "answer" is the user-visible output field.
        Other fields are internal tracking (confidence, sources, etc.)

        Args:
            format: PromptFormat options (defaults to standard format)

        Returns:
            Prompt guidance text to append to system prompt
        """
        if not self.properties:
            return ""

        format = format or PromptFormat()

        # Separate answer (output) from internal fields
        answer_field = self.properties.get("answer")
        internal_fields = {k: v for k, v in self.properties.items() if k != "answer"}

        lines: list[str] = []

        if format.include_instructions:
            lines.append("## Internal Thinking Structure (DO NOT output these labels)")
            lines.append("")
            lines.append(
                "Use this structure to organize your thinking, "
                "but ONLY output the answer content:"
            )
            lines.append("")

        # Document the answer field
        if format.include_answer and answer_field:
            answer_desc = answer_field.get("description", "Your response")
            lines.append(f"**OUTPUT (what the user sees):** {answer_desc}")
            lines.append("")

        # Document internal fields
        if format.include_internal and internal_fields:
            lines.append("**INTERNAL (for your tracking only - do NOT include in output):**")
            lines.append("")
            lines.append("Schema (use these EXACT field names):")

            if format.yaml_style:
                lines.append("```yaml")
                lines.extend(self._render_properties_yaml(internal_fields))
                lines.append("```")
            else:
                lines.append("```json")
                lines.extend(self._render_properties_json(internal_fields))
                lines.append("```")

        if format.include_instructions:
            lines.append("")
            lines.append("CRITICAL: Your response must be ONLY the conversational answer text.")
            lines.append(
                "Do NOT output field names like 'answer:' or JSON - "
                "just the response itself."
            )

        return "\n".join(lines)

    def with_options(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_iterations: int | None = None,
        request_limit: int | None = None,
        tool_calls_limit: int | None = None,
        input_tokens_limit: int | None = None,
        output_tokens_limit: int | None = None,
        total_tokens_limit: int | None = None,
        from_env: bool = False,
        **extra: Any,
    ) -> AgentSchema:
        """
        Create a copy with overridden options.

        This allows customizing agent behavior without modifying the original
        schema. Options can come from explicit parameters or environment.

        PRIORITY ORDER
        --------------
        1. Explicit parameters (model, temperature, etc.)
        2. Environment variables (if from_env=True)
        3. Existing schema values

        Environment Variables (when from_env=True):
        - AGENT_MODEL → model
        - AGENT_TEMPERATURE → temperature
        - AGENT_MAX_ITERATIONS → max_iterations
        - AGENT_REQUEST_LIMIT → limits.request_limit
        - AGENT_TOTAL_TOKENS_LIMIT → limits.total_tokens_limit
        - AGENT_TOOL_CALLS_LIMIT → limits.tool_calls_limit

        Args:
            model: Override model (e.g., "openai:gpt-4.1")
            temperature: Override temperature (0.0-1.0)
            max_iterations: Override max tool iterations
            request_limit: Max requests to model
            tool_calls_limit: Max tool calls
            input_tokens_limit: Max input tokens
            output_tokens_limit: Max output tokens
            total_tokens_limit: Max total tokens
            from_env: If True, also check environment variables
            **extra: Additional metadata to add to json_schema_extra

        Returns:
            New AgentSchema with updated options

        Example:
            # Explicit override
            schema = schema.with_options(model="anthropic:claude-sonnet-4-5-20250929", temperature=0.3)

            # With limits
            schema = schema.with_options(request_limit=10, total_tokens_limit=50000)

            # From environment
            schema = schema.with_options(from_env=True)

            # Access via get_options()
            options = schema.get_options()
            options.model                    # → "anthropic:claude-sonnet-4-5-20250929"
            options.limits                   # → AgentUsageLimits(...)
            options.limits.to_pydantic_ai()  # → pydantic_ai.UsageLimits
            options.extras                   # → {...}
        """
        # Start with current metadata
        extra_dict = self.json_schema_extra.model_dump()

        # Apply environment variables if requested
        if from_env:
            env_model = os.getenv("AGENT_MODEL")
            env_temp = os.getenv("AGENT_TEMPERATURE")
            env_iters = os.getenv("AGENT_MAX_ITERATIONS")
            env_request_limit = os.getenv("AGENT_REQUEST_LIMIT")
            env_total_tokens = os.getenv("AGENT_TOTAL_TOKENS_LIMIT")
            env_tool_calls = os.getenv("AGENT_TOOL_CALLS_LIMIT")

            if env_model and not model:
                model = env_model
            if env_temp and temperature is None:
                try:
                    temperature = float(env_temp)
                except ValueError:
                    pass
            if env_iters and max_iterations is None:
                try:
                    max_iterations = int(env_iters)
                except ValueError:
                    pass
            if env_request_limit and request_limit is None:
                try:
                    request_limit = int(env_request_limit)
                except ValueError:
                    pass
            if env_total_tokens and total_tokens_limit is None:
                try:
                    total_tokens_limit = int(env_total_tokens)
                except ValueError:
                    pass
            if env_tool_calls and tool_calls_limit is None:
                try:
                    tool_calls_limit = int(env_tool_calls)
                except ValueError:
                    pass

        # Apply explicit overrides (highest priority)
        if model is not None:
            extra_dict["model"] = model
        if temperature is not None:
            extra_dict["temperature"] = temperature
        if max_iterations is not None:
            extra_dict["max_iterations"] = max_iterations

        # Build limits if any are specified
        limits_specified = any(
            v is not None
            for v in [
                request_limit,
                tool_calls_limit,
                input_tokens_limit,
                output_tokens_limit,
                total_tokens_limit,
            ]
        )
        if limits_specified:
            # Start with existing limits or empty
            existing_limits = extra_dict.get("limits") or {}
            if isinstance(existing_limits, AgentUsageLimits):
                existing_limits = existing_limits.model_dump()

            # Merge with new values (explicit takes precedence)
            if request_limit is not None:
                existing_limits["request_limit"] = request_limit
            if tool_calls_limit is not None:
                existing_limits["tool_calls_limit"] = tool_calls_limit
            if input_tokens_limit is not None:
                existing_limits["input_tokens_limit"] = input_tokens_limit
            if output_tokens_limit is not None:
                existing_limits["output_tokens_limit"] = output_tokens_limit
            if total_tokens_limit is not None:
                existing_limits["total_tokens_limit"] = total_tokens_limit

            extra_dict["limits"] = AgentUsageLimits(**existing_limits)

        # Add any extra metadata
        extra_dict.update(extra)

        # Create new schema with updated extra
        return AgentSchema(
            type=self.type,
            description=self.description,
            properties=self.properties,
            required=self.required,
            json_schema_extra=AgentExtraSchema(**extra_dict),
            definitions=self.definitions,
            additionalProperties=self.additionalProperties,
        )

    # =========================================================================
    # CLASS METHODS - Loading from various sources
    # =========================================================================

    @classmethod
    def from_yaml(cls, yaml_content: str) -> AgentSchema:
        """
        Parse agent schema from YAML string.

        Args:
            yaml_content: YAML string containing agent definition

        Returns:
            Validated AgentSchema instance
        """
        data = yaml.safe_load(yaml_content)
        return cls(**data)

    @classmethod
    def from_yaml_file(cls, file_path: str | Path) -> AgentSchema:
        """
        Load agent schema from a YAML file.

        Args:
            file_path: Path to YAML file

        Returns:
            Validated AgentSchema instance
        """
        content = Path(file_path).read_text()
        return cls.from_yaml(content)

    @classmethod
    def load(cls, name: str) -> "AgentSchema":
        """
        Load agent schema by name from file, database, or cache.

        Resolution order:
        1. YAML file in schemas/ directory
        2. Database (TODO)
        3. Cache (TODO)

        Args:
            name: Agent name (e.g., "query-agent")

        Returns:
            AgentSchema instance

        Raises:
            FileNotFoundError: If schema not found
        """
        # Try filesystem first
        schemas_dir = Path(__file__).parent.parent.parent / "schemas"
        schema_path = schemas_dir / f"{name}.yaml"

        if schema_path.exists():
            return cls.from_yaml_file(schema_path)

        # TODO: Try database - await schema_from_database(name)
        # TODO: Try cache - cache.get(name)

        raise FileNotFoundError(f"Agent schema '{name}' not found")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentSchema:
        """
        Create schema from a dictionary.

        Args:
            data: Schema as dictionary

        Returns:
            Validated AgentSchema instance
        """
        return cls(**data)

    @classmethod
    def build(
        cls,
        name: str,
        description: str,
        properties: dict[str, Any] | None = None,
        tools: list[str] | list[MCPToolReference] | None = None,
        version: str = "1.0.0",
        **extra: Any,
    ) -> AgentSchema:
        """
        Build an agent schema programmatically.

        This is a helper for creating schemas in code rather than YAML.
        Useful for unit tests, dynamic agents, and default configurations.

        Args:
            name: Unique agent identifier
            description: System prompt for the agent
            properties: Output schema properties
            tools: List of tool names or MCPToolReference objects
            version: Schema version
            **extra: Additional metadata for json_schema_extra

        Returns:
            AgentSchema instance
        """
        # Convert tool names to references if needed
        tool_refs: list[MCPToolReference] = []
        if tools:
            for t in tools:
                if isinstance(t, str):
                    tool_refs.append(MCPToolReference(name=t))
                elif isinstance(t, MCPToolReference):
                    tool_refs.append(t)
                elif isinstance(t, dict):
                    tool_refs.append(MCPToolReference(**t))

        default_properties = {
            "answer": {"type": "string", "description": "Response to the user"},
        }

        return cls(
            type="object",
            description=description,
            properties=properties or default_properties,
            required=["answer"],
            json_schema_extra=AgentExtraSchema(
                kind="agent",
                name=name,
                version=version,
                tools=tool_refs,
                **extra,
            ),
        )

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_yaml(self) -> str:
        """
        Serialize schema to YAML string.

        Returns:
            YAML string representation
        """
        return yaml.dump(
            self.model_dump(exclude_none=True),
            default_flow_style=False,
            sort_keys=False,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert schema to dictionary.

        Returns:
            Schema as dictionary
        """
        return self.model_dump(exclude_none=True)

    # =========================================================================
    # PRIVATE HELPERS
    # =========================================================================

    @staticmethod
    def _json_type_to_python(json_type: str) -> type:
        """Map JSON Schema type to Python type."""
        type_map: dict[str, type] = {
            "string": str,
            "number": float,
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        return type_map.get(json_type, str)

    def _render_properties_yaml(
        self, properties: dict[str, Any], indent: int = 0
    ) -> list[str]:
        """Render properties as YAML-like text."""
        lines: list[str] = []
        prefix = "  " * indent

        for field_name, field_def in properties.items():
            field_type = field_def.get("type", "any")
            field_desc = field_def.get("description", "")
            is_required = field_name in self.required
            req_marker = " (required)" if is_required else ""

            if field_type == "object":
                lines.append(f"{prefix}{field_name}:{req_marker}")
                if field_desc:
                    lines.append(f"{prefix}  # {field_desc}")
                nested = field_def.get("properties", {})
                if nested:
                    lines.extend(self._render_properties_yaml(nested, indent + 1))

            elif field_type == "array":
                items = field_def.get("items", {})
                items_type = items.get("type", "any")
                lines.append(f"{prefix}{field_name}: [{items_type}]{req_marker}")
                if field_desc:
                    lines.append(f"{prefix}  # {field_desc}")
                if items_type == "object":
                    nested = items.get("properties", {})
                    if nested:
                        lines.append(f"{prefix}  # Each item has:")
                        lines.extend(self._render_properties_yaml(nested, indent + 2))

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

    def _render_properties_json(self, properties: dict[str, Any]) -> list[str]:
        """Render properties as JSON example."""
        import json

        example: dict[str, Any] = {}
        for field_name, field_def in properties.items():
            field_type = field_def.get("type", "string")
            if field_type == "string":
                example[field_name] = f"<{field_def.get('description', field_name)}>"
            elif field_type == "number":
                example[field_name] = 0.0
            elif field_type == "integer":
                example[field_name] = 0
            elif field_type == "boolean":
                example[field_name] = False
            elif field_type == "array":
                example[field_name] = []
            elif field_type == "object":
                example[field_name] = {}

        return json.dumps(example, indent=2).split("\n")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_system_prompt(schema: AgentSchema | dict[str, Any]) -> str:
    """
    Extract system prompt from schema (polymorphic version).

    Works with both parsed AgentSchema objects and raw dict (from YAML load).
    """
    if isinstance(schema, AgentSchema):
        return schema.get_system_prompt()

    base = schema.get("description", "")
    extra = schema.get("json_schema_extra", {})
    custom = extra.get("system_prompt") if isinstance(extra, dict) else None

    if custom:
        return f"{base}\n\n{custom}"
    return base


def schema_from_yaml(yaml_content: str) -> AgentSchema:
    """Parse agent schema from YAML string."""
    return AgentSchema.from_yaml(yaml_content)


def schema_from_yaml_file(file_path: str | Path) -> AgentSchema:
    """Load agent schema from a YAML file."""
    return AgentSchema.from_yaml_file(file_path)


def schema_to_yaml(schema: AgentSchema) -> str:
    """Serialize agent schema to YAML string."""
    return schema.to_yaml()


def build_agent_spec(
    name: str,
    description: str,
    properties: dict[str, Any] | None = None,
    tools: list[str] | None = None,
    version: str = "1.0.0",
) -> dict[str, Any]:
    """
    Build a minimal agent spec dictionary programmatically.

    Returns dict for backwards compatibility with code expecting raw dicts.
    """
    schema = AgentSchema.build(
        name=name,
        description=description,
        properties=properties,
        tools=tools,
        version=version,
    )
    return schema.to_dict()
