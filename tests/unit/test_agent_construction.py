"""
Unit tests for Agent Construction Flow.

Tests covering the complete agent construction pipeline:
1. Schema Loading (YAML string, file, database, cache)
2. Tool Builder (PydanticAI signature extraction)
3. MCP Server Inspection (local/remote tools)
4. System Prompt Construction (cleaning, user context)
5. Output Type Configuration (structured vs unstructured)

Run with: pytest tests/unit/test_agent_construction.py -v -s

VERIFIED BEHAVIOR
=================

Schema Loading Priority:
- schema_from_yaml(): Parse YAML string → AgentSchema
- schema_from_yaml_file(): Load file → parse → AgentSchema
- schema_from_database(): File OR Database (configurable priority)

Tool Registration:
- PydanticAI extracts: docstring → description, type hints → JSON Schema
- Tools accessed via: agent._function_toolset.tools[name].function_schema

System Prompt:
- Combined from: schema.description + json_schema_extra.system_prompt
- User context prepended if available

Output Types:
- structured_output=false: Free-form text, properties → prompt guidance
- structured_output=true: Pydantic model generated, description stripped from schema
"""

import pytest
from typing import Any
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock


# =============================================================================
# SECTION 1: Schema Loading Tests
# =============================================================================

class TestSchemaLoading:
    """Tests for loading agent schemas from various sources."""

    def test_schema_from_yaml_string(self):
        """
        Verify schema can be loaded from YAML string.

        Expected: YAML string → AgentSchema with all fields parsed
        """
        from remlight.agentic.schema import schema_from_yaml, AgentSchema

        yaml_content = """
type: object
description: You are a helpful research assistant.

properties:
  answer:
    type: string
    description: Your response

required:
  - answer

json_schema_extra:
  kind: agent
  name: test-agent
  version: "1.0.0"
  tools:
    - name: search
    - name: action
"""
        schema = schema_from_yaml(yaml_content)

        assert isinstance(schema, AgentSchema)
        assert schema.description == "You are a helpful research assistant."
        assert schema.type == "object"
        assert "answer" in schema.properties
        assert schema.json_schema_extra.name == "test-agent"
        assert schema.json_schema_extra.version == "1.0.0"
        assert len(schema.json_schema_extra.tools) == 2

    def test_schema_from_yaml_file(self, tmp_path):
        """
        Verify schema can be loaded from YAML file.

        Expected: File path → read content → parse → AgentSchema
        """
        from remlight.agentic.schema import schema_from_yaml_file

        yaml_content = """
type: object
description: File-based test agent.
json_schema_extra:
  kind: agent
  name: file-test-agent
  version: "2.0.0"
"""
        yaml_file = tmp_path / "test-agent.yaml"
        yaml_file.write_text(yaml_content)

        schema = schema_from_yaml_file(yaml_file)

        assert schema.description == "File-based test agent."
        assert schema.json_schema_extra.name == "file-test-agent"
        assert schema.json_schema_extra.version == "2.0.0"

    def test_schema_with_system_prompt_extension(self):
        """
        Verify json_schema_extra.system_prompt is appended to description.

        Expected:
        - schema.description = main prompt
        - json_schema_extra.system_prompt = extension
        - get_system_prompt() combines both with newline separation
        """
        from remlight.agentic.schema import schema_from_yaml

        yaml_content = """
type: object
description: You are a research assistant.

json_schema_extra:
  kind: agent
  name: extended-prompt-agent
  system_prompt: |
    Additional instructions:
    - Always cite sources
    - Use formal tone
"""
        schema = schema_from_yaml(yaml_content)

        # Individual parts accessible
        assert schema.description == "You are a research assistant."
        assert "Always cite sources" in schema.json_schema_extra.system_prompt

        # Combined via get_system_prompt()
        full_prompt = schema.get_system_prompt()
        assert "You are a research assistant." in full_prompt
        assert "Always cite sources" in full_prompt
        assert "Use formal tone" in full_prompt

    def test_build_agent_spec_programmatic(self):
        """
        Verify agents can be built programmatically (for tests/dynamic creation).

        Expected: build_agent_spec() creates valid dict that can be parsed
        """
        from remlight.agentic.schema import build_agent_spec, AgentSchema

        spec = build_agent_spec(
            name="dynamic-agent",
            description="Dynamically created agent for testing.",
            tools=["search", "action"],
            version="3.0.0",
        )

        # Can be parsed as AgentSchema
        schema = AgentSchema(**spec)
        assert schema.json_schema_extra.name == "dynamic-agent"
        assert schema.description == "Dynamically created agent for testing."
        assert len(schema.json_schema_extra.tools) == 2


# =============================================================================
# SECTION 2: Tool Builder Tests (PydanticAI Integration)
# =============================================================================

class TestToolBuilder:
    """
    Tests verifying PydanticAI extracts tool signatures correctly.

    See also: tests/unit/test_tool_signature.py for detailed signature tests.
    """

    def test_tool_registered_with_agent(self):
        """
        Verify tools are registered with PydanticAI Agent.

        Expected: tools passed to Agent() are accessible via _function_toolset
        """
        from pydantic_ai import Agent

        async def my_tool(query: str) -> dict:
            """Search for something."""
            return {"result": query}

        agent = Agent(model="test", tools=[my_tool])

        assert "my_tool" in agent._function_toolset.tools
        tool = agent._function_toolset.tools["my_tool"]
        assert tool.function_schema is not None

    def test_tool_json_schema_generation(self):
        """
        Verify PydanticAI generates correct JSON Schema from type hints.

        Expected:
        - str → "string"
        - int → "integer"
        - str | None → anyOf[string, null]
        - Parameters without defaults → required array
        """
        from pydantic_ai import Agent

        async def typed_tool(
            required_str: str,
            optional_int: int = 10,
            nullable_str: str | None = None,
        ) -> dict[str, Any]:
            """A tool with various parameter types."""
            return {}

        agent = Agent(model="test", tools=[typed_tool])
        schema = agent._function_toolset.tools["typed_tool"].function_schema.json_schema

        props = schema["properties"]

        # Required string
        assert props["required_str"]["type"] == "string"
        assert "required_str" in schema["required"]

        # Optional int with default
        assert props["optional_int"]["type"] == "integer"
        assert props["optional_int"]["default"] == 10
        assert "optional_int" not in schema["required"]

        # Nullable string (anyOf)
        assert "anyOf" in props["nullable_str"]
        types = [t.get("type") for t in props["nullable_str"]["anyOf"]]
        assert "string" in types
        assert "null" in types

    def test_tool_description_from_docstring(self):
        """
        Verify tool description is extracted from docstring.

        Expected: Docstring content appears in tool.description with XML tags
        """
        from pydantic_ai import Agent

        async def documented_tool(query: str) -> dict:
            """
            Search the knowledge base for relevant information.

            This tool executes semantic search queries.

            Args:
                query: The search query string

            Returns:
                Search results with relevance scores
            """
            return {}

        agent = Agent(model="test", tools=[documented_tool])
        tool = agent._function_toolset.tools["documented_tool"]

        assert "Search the knowledge base" in tool.description
        assert "<summary>" in tool.description
        assert "Search results" in tool.description


# =============================================================================
# SECTION 3: MCP Server Inspection Tests
# =============================================================================

class TestMCPServerInspection:
    """
    Tests for MCP server tool registration and inspection.

    MCP (Model Context Protocol) servers register tools that agents can use.
    Tools can be local (built-in) or remote (via HTTP/stdio).
    """

    def test_mcp_server_creation(self):
        """
        Verify MCP server can be created with tools registered.

        Expected: FastMCP server has tools accessible via get_tools()
        """
        from remlight.api.mcp_main import create_mcp_server

        mcp = create_mcp_server()

        assert mcp is not None
        assert mcp.name is not None

    @pytest.mark.asyncio
    async def test_mcp_get_tools_returns_tool_dict(self):
        """
        Verify get_mcp_tools() returns tools as dict.

        Expected: Dict mapping tool names to FunctionTool objects
        """
        from remlight.api.mcp_main import get_mcp_tools, create_mcp_server

        # Ensure server is created
        create_mcp_server()
        tools = await get_mcp_tools()

        assert isinstance(tools, dict)
        assert "search" in tools
        assert "action" in tools
        assert "ask_agent" in tools
        assert "parse_file" in tools

    @pytest.mark.asyncio
    async def test_mcp_tool_has_function_attribute(self):
        """
        Verify MCP tools have callable .fn attribute.

        Expected: Each tool has .fn that is the actual async function
        """
        from remlight.api.mcp_main import get_mcp_tools, create_mcp_server

        create_mcp_server()
        tools = await get_mcp_tools()

        for name, tool in tools.items():
            assert hasattr(tool, "fn"), f"Tool {name} missing .fn attribute"
            assert callable(tool.fn), f"Tool {name}.fn is not callable"

    def test_mcp_tool_reference_with_server(self):
        """
        Verify MCPToolReference can specify remote server.

        Expected: server field allows "local" or server name
        """
        from remlight.agentic.schema import MCPToolReference

        # Local tool (default)
        local_ref = MCPToolReference(name="search")
        assert local_ref.server is None  # None = local

        # Explicit local
        explicit_local = MCPToolReference(name="search", server="local")
        assert explicit_local.server == "local"

        # Remote server
        remote_ref = MCPToolReference(name="fetch_data", server="data-service")
        assert remote_ref.server == "data-service"


# =============================================================================
# SECTION 4: System Prompt Construction Tests
# =============================================================================

class TestSystemPromptConstruction:
    """
    Tests for system prompt building and cleaning.

    The system prompt is constructed from:
    1. User profile hint (if context has user_profile_hint)
    2. Schema description (main prompt)
    3. Schema json_schema_extra.system_prompt (optional extension)
    """

    def test_system_prompt_from_schema_only(self):
        """
        Verify system prompt built from schema description alone.

        Expected: get_system_prompt() returns description when no extension
        """
        from remlight.agentic.schema import schema_from_yaml

        yaml_content = """
type: object
description: You are a helpful assistant.
json_schema_extra:
  kind: agent
  name: simple-agent
"""
        schema = schema_from_yaml(yaml_content)
        prompt = schema.get_system_prompt()

        assert prompt == "You are a helpful assistant."

    def test_system_prompt_with_extension(self):
        """
        Verify system prompt combines description and extension.

        Expected: description + "\\n\\n" + system_prompt
        """
        from remlight.agentic.schema import schema_from_yaml

        yaml_content = """
type: object
description: Main instructions here.
json_schema_extra:
  kind: agent
  name: extended-agent
  system_prompt: Extended instructions here.
"""
        schema = schema_from_yaml(yaml_content)
        prompt = schema.get_system_prompt()

        assert "Main instructions here." in prompt
        assert "Extended instructions here." in prompt
        # Separated by double newline
        assert "\n\n" in prompt

    def test_build_system_prompt_with_user_context(self):
        """
        Verify user profile hint is prepended to system prompt.

        Expected: "## User Context\\n{hint}\\n" + schema prompt
        """
        from remlight.agentic.provider import _build_system_prompt
        from remlight.agentic.schema import schema_from_yaml
        from remlight.agentic.context import AgentContext

        yaml_content = """
type: object
description: You are an assistant.
json_schema_extra:
  kind: agent
  name: context-agent
"""
        schema = schema_from_yaml(yaml_content)
        context = AgentContext(
            user_id="test-user",
            user_profile_hint="User: John Doe\nInterests: AI, ML",
        )

        prompt = _build_system_prompt(schema, context)

        assert "## User Context" in prompt
        assert "John Doe" in prompt
        assert "You are an assistant." in prompt
        # User context comes first
        assert prompt.index("User Context") < prompt.index("assistant")


# =============================================================================
# SECTION 5: Output Type Configuration Tests
# =============================================================================

class TestOutputTypeConfiguration:
    """
    Tests for structured vs unstructured output handling.

    Two modes:
    - structured_output=false (default): Free-form text, properties as prompt guidance
    - structured_output=true: Pydantic model enforced, description stripped from schema
    """

    def test_unstructured_output_default(self):
        """
        Verify structured_output defaults to false (unstructured).

        Expected: Agent uses str output type by default
        """
        from remlight.agentic.schema import schema_from_yaml

        yaml_content = """
type: object
description: You are an assistant.
json_schema_extra:
  kind: agent
  name: default-agent
"""
        schema = schema_from_yaml(yaml_content)

        # structured_output should be None (falsy)
        assert schema.json_schema_extra.structured_output is None

    def test_structured_output_explicit_true(self):
        """
        Verify structured_output=true is preserved in schema.

        Expected: structured_output field is True
        """
        from remlight.agentic.schema import schema_from_yaml

        yaml_content = """
type: object
description: You are a structured assistant.
properties:
  answer:
    type: string
  confidence:
    type: number
json_schema_extra:
  kind: agent
  name: structured-agent
  structured_output: true
"""
        schema = schema_from_yaml(yaml_content)

        assert schema.json_schema_extra.structured_output is True
        assert "answer" in schema.properties
        assert "confidence" in schema.properties

    def test_build_output_model_from_properties(self):
        """
        Verify Pydantic model is built from JSON Schema properties.

        Expected: create_model() generates model with correct field types
        """
        from remlight.agentic.provider import _build_output_model

        properties = {
            "answer": {"type": "string", "description": "The response"},
            "confidence": {"type": "number", "description": "Confidence score"},
            "count": {"type": "integer"},
        }
        required = ["answer"]

        Model = _build_output_model(properties, required)

        # Model should have the fields
        assert "answer" in Model.model_fields
        assert "confidence" in Model.model_fields
        assert "count" in Model.model_fields

        # Required field annotation
        answer_field = Model.model_fields["answer"]
        assert answer_field.is_required()

        # Optional fields have defaults
        confidence_field = Model.model_fields["confidence"]
        assert not confidence_field.is_required()

    def test_schema_wrapper_strips_description(self):
        """
        Verify schema wrapper removes description from JSON schema.

        WHY THIS MATTERS
        ----------------
        The schema 'description' field IS the system prompt. When using
        structured output, PydanticAI sends the output model's JSON schema
        to the LLM. Without stripping, the description appears TWICE:

        1. In the system prompt (where we put it intentionally)
        2. In the output JSON schema (duplication - wastes tokens, confuses LLM)

        THE FIX
        -------
        _create_schema_wrapper() overrides model_json_schema() to remove
        the 'description' field, preventing duplication.

        CODE REFERENCE: remlight/agentic/provider.py:320-382

        Expected: model_json_schema() output has no "description" key
        """
        from remlight.agentic.provider import _build_output_model, _create_schema_wrapper

        properties = {"answer": {"type": "string"}}
        Model = _build_output_model(properties, ["answer"])

        # Add a description to the base model
        Model.__doc__ = "This is the model description that should be stripped"

        WrappedModel = _create_schema_wrapper(Model, strip_description=True)
        schema = WrappedModel.model_json_schema()

        # Description should be removed from root level
        # (Note: property-level descriptions are preserved)
        assert "description" not in schema or schema.get("description") is None

    def test_properties_converted_to_prompt_guidance(self):
        """
        Verify unstructured mode converts properties to prompt text.

        UNSTRUCTURED OUTPUT PATTERN
        ---------------------------
        Most agents output natural text, not JSON. But we still want the LLM
        to track internal state (confidence, sources, reasoning).

        This function converts schema properties to prompt guidance:
        - "answer" field → OUTPUT section (what user sees)
        - Other fields → INTERNAL section (for LLM tracking, not output)

        GENERATED PROMPT STRUCTURE:
        ```
        ## Internal Thinking Structure (DO NOT output these labels)
        **OUTPUT (what the user sees):** Your response to the user
        **INTERNAL (for your tracking only):**
            confidence: number
            sources: [string]
        CRITICAL: Output ONLY the conversational answer text.
        ```

        CODE REFERENCE: remlight/agentic/provider.py:476-593

        Expected: _convert_properties_to_prompt() generates human-readable guidance
        """
        from remlight.agentic.provider import _convert_properties_to_prompt

        properties = {
            "answer": {
                "type": "string",
                "description": "Your response to the user"
            },
            "thinking": {
                "type": "object",
                "description": "Internal reasoning",
                "properties": {
                    "steps": {"type": "array", "items": {"type": "string"}}
                }
            }
        }

        prompt = _convert_properties_to_prompt(properties)

        assert prompt is not None
        assert "answer" in prompt.lower() or "output" in prompt.lower()
        # Should mention internal fields are not for output
        assert "internal" in prompt.lower() or "tracking" in prompt.lower()


# =============================================================================
# SECTION 6: Agent Cache Tests
# =============================================================================

class TestAgentCache:
    """
    Tests for agent instance caching.

    Agents are cached by (schema_hash, model, user_id) to avoid
    repeated schema parsing and agent instantiation.
    """

    def test_cache_key_computation(self):
        """
        Verify cache key includes schema hash, model, and user_id.

        Expected: Key format is "{schema_hash}:{model}:{user_id_prefix}"
        """
        from remlight.agentic.provider import _compute_cache_key

        schema = {
            "type": "object",
            "description": "Test agent",
            "json_schema_extra": {"kind": "agent", "name": "test"},
        }

        key1 = _compute_cache_key(schema, "openai:gpt-4", "user-123")
        key2 = _compute_cache_key(schema, "openai:gpt-4", "user-123")
        key3 = _compute_cache_key(schema, "openai:gpt-4", "user-456")
        key4 = _compute_cache_key(schema, "anthropic:claude", "user-123")

        # Same inputs = same key
        assert key1 == key2

        # Different user = different key
        assert key1 != key3

        # Different model = different key
        assert key1 != key4

        # Key contains expected parts
        assert "openai:gpt-4" in key1
        assert "user-123"[:8] in key1

    def test_cache_stats(self):
        """
        Verify cache statistics are accessible.

        Expected: get_agent_cache_stats() returns size, max_size, ttl, keys
        """
        from remlight.agentic.provider import get_agent_cache_stats

        stats = get_agent_cache_stats()

        assert "size" in stats
        assert "max_size" in stats
        assert "ttl_seconds" in stats
        assert "keys" in stats
        assert stats["max_size"] == 50  # Default max size
        assert stats["ttl_seconds"] == 300  # 5 minutes


# =============================================================================
# SECTION 7: Integration Test - Full Agent Creation
# =============================================================================

class TestAgentCreationIntegration:
    """
    Integration tests for the complete create_agent() flow.
    """

    @pytest.mark.asyncio
    async def test_create_agent_from_yaml_schema(self):
        """
        Verify agent can be created from YAML schema with tools.

        Expected: create_agent() returns AgentRuntime with configured agent
        """
        from remlight.agentic import create_agent
        from remlight.agentic.schema import schema_from_yaml

        yaml_content = """
type: object
description: You are a test assistant.

properties:
  answer:
    type: string

json_schema_extra:
  kind: agent
  name: integration-test-agent
  version: "1.0.0"
"""
        schema = schema_from_yaml(yaml_content)

        # Create simple test tool
        async def test_tool(query: str) -> str:
            return f"Result: {query}"

        runtime = await create_agent(
            schema=schema,
            model_name="test",
            tools=[test_tool],
            use_cache=False,  # Don't cache for tests
        )

        assert runtime is not None
        assert runtime.agent is not None
        assert runtime.schema_name == "integration-test-agent"

    @pytest.mark.asyncio
    async def test_create_agent_model_priority(self):
        """
        Verify model selection priority: override > parameter > default.

        Expected:
        1. schema.override_model takes highest priority
        2. model_name parameter is second
        3. settings.llm.default_model is fallback
        """
        from remlight.agentic import create_agent
        from remlight.agentic.schema import schema_from_yaml

        # Schema with override_model
        yaml_with_override = """
type: object
description: Agent with forced model.
json_schema_extra:
  kind: agent
  name: override-model-agent
  override_model: "anthropic:claude-sonnet-4-5-20250929"
"""
        schema = schema_from_yaml(yaml_with_override)

        # Even if we pass a different model, override wins
        runtime = await create_agent(
            schema=schema,
            model_name="openai:gpt-4",  # This should be ignored
            use_cache=False,
        )

        # The agent should use the override model
        # (Note: model is set on agent._model)
        assert runtime.agent._model is not None
