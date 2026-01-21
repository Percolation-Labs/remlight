"""Unit tests for the agentic module.

Tests cover:
1. YAML to Pydantic schema parsing
2. Structured output description stripping
3. Properties to prompt conversion
4. Context header parsing
5. MCP tool/resource references
"""

import pytest
from pydantic import BaseModel

from remlight.agentic.schema import (
    AgentSchema,
    AgentSchemaMetadata,
    MCPToolReference,
    MCPResourceReference,
    schema_from_yaml,
    get_system_prompt,
)
from remlight.agentic.provider import (
    _build_output_model,
    _create_schema_wrapper,
    _convert_properties_to_prompt,
)
from remlight.agentic.context import AgentContext


# =============================================================================
# Schema Parsing Tests
# =============================================================================


class TestSchemaFromYAML:
    """Test YAML to Pydantic schema parsing."""

    def test_basic_schema_parsing(self):
        """Test parsing a basic agent schema from YAML."""
        yaml_content = """
type: object
description: You are a helpful assistant.
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
"""
        schema = schema_from_yaml(yaml_content)

        assert schema.type == "object"
        assert "helpful assistant" in schema.description
        assert "answer" in schema.properties
        assert schema.json_schema_extra.name == "test-agent"
        assert schema.json_schema_extra.version == "1.0.0"

    def test_schema_with_tools(self):
        """Test parsing schema with MCP tool references."""
        yaml_content = """
type: object
description: Agent with tools.
properties:
  answer:
    type: string
json_schema_extra:
  name: tool-agent
  tools:
    - name: search
      description: Search the knowledge base
    - name: action
      server: rem
"""
        schema = schema_from_yaml(yaml_content)

        tools = schema.json_schema_extra.tools
        assert len(tools) == 2
        assert tools[0].name == "search"
        assert tools[0].description == "Search the knowledge base"
        assert tools[1].name == "action"
        assert tools[1].server == "rem"

    def test_schema_with_resources(self):
        """Test parsing schema with MCP resource references."""
        yaml_content = """
type: object
description: Agent with resources.
properties:
  answer:
    type: string
json_schema_extra:
  name: resource-agent
  resources:
    - uri: rem://agents
      name: Agent Schemas
      description: List available agents
    - uri_pattern: rem://resources/.*
"""
        schema = schema_from_yaml(yaml_content)

        resources = schema.json_schema_extra.resources
        assert len(resources) == 2
        assert resources[0].uri == "rem://agents"
        assert resources[0].name == "Agent Schemas"
        assert resources[1].uri_pattern == "rem://resources/.*"

    def test_schema_with_structured_output(self):
        """Test parsing schema with structured_output flag."""
        yaml_content = """
type: object
description: Structured output agent.
properties:
  answer:
    type: string
  confidence:
    type: number
json_schema_extra:
  name: structured-agent
  structured_output: true
"""
        schema = schema_from_yaml(yaml_content)

        assert schema.json_schema_extra.structured_output is True

    def test_get_system_prompt_combined(self):
        """Test combined system prompt from description + system_prompt."""
        yaml_content = """
type: object
description: Base description.
properties:
  answer:
    type: string
json_schema_extra:
  name: prompt-agent
  system_prompt: Extended instructions.
"""
        schema = schema_from_yaml(yaml_content)
        prompt = get_system_prompt(schema)

        assert "Base description" in prompt
        assert "Extended instructions" in prompt


# =============================================================================
# Provider Tests
# =============================================================================


class TestBuildOutputModel:
    """Test dynamic Pydantic model creation from schema properties."""

    def test_basic_model_creation(self):
        """Test creating a model with basic types."""
        properties = {
            "answer": {"type": "string", "description": "Response"},
            "confidence": {"type": "number"},
            "count": {"type": "integer"},
            "is_valid": {"type": "boolean"},
        }
        required = ["answer"]

        Model = _build_output_model(properties, required)

        # Check field types
        fields = Model.model_fields
        assert "answer" in fields
        assert "confidence" in fields
        assert "count" in fields
        assert "is_valid" in fields

        # Check required vs optional
        assert fields["answer"].is_required()
        assert not fields["confidence"].is_required()

    def test_array_and_object_types(self):
        """Test creating model with array and object types."""
        properties = {
            "items": {"type": "array"},
            "metadata": {"type": "object"},
        }

        Model = _build_output_model(properties, [])

        # Should create fields with list and dict types
        instance = Model(items=["a", "b"], metadata={"key": "value"})
        assert instance.items == ["a", "b"]
        assert instance.metadata == {"key": "value"}


class TestSchemaWrapper:
    """Test schema wrapper for description stripping."""

    def test_strips_description(self):
        """Test that wrapper strips description from JSON schema."""

        class TestModel(BaseModel):
            """This description should be stripped."""

            answer: str

        WrappedModel = _create_schema_wrapper(TestModel, strip_description=True)
        schema = WrappedModel.model_json_schema()

        assert "description" not in schema

    def test_preserves_description_when_disabled(self):
        """Test that wrapper preserves description when stripping disabled."""

        class TestModel(BaseModel):
            """This description should remain."""

            answer: str

        WrappedModel = _create_schema_wrapper(TestModel, strip_description=False)
        schema = WrappedModel.model_json_schema()

        assert schema.get("description") == "This description should remain."


class TestConvertPropertiesToPrompt:
    """Test conversion of schema properties to prompt guidance."""

    def test_basic_conversion(self):
        """Test converting properties to prompt text."""
        properties = {
            "answer": {"type": "string", "description": "Your response"},
            "confidence": {"type": "number", "description": "0-1 score"},
        }

        prompt = _convert_properties_to_prompt(properties)

        assert "answer" in prompt
        assert "Your response" in prompt
        assert "confidence" in prompt
        assert "0-1 score" in prompt
        assert "DO NOT output" in prompt  # Should warn against JSON output

    def test_empty_properties(self):
        """Test that empty properties returns empty string."""
        prompt = _convert_properties_to_prompt({})
        assert prompt == ""

    def test_array_type_conversion(self):
        """Test converting array type properties."""
        properties = {
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of sources",
            }
        }

        prompt = _convert_properties_to_prompt(properties)

        assert "sources" in prompt
        assert "[string]" in prompt


# =============================================================================
# Context Tests
# =============================================================================


class TestAgentContext:
    """Test AgentContext creation and methods."""

    def test_from_headers(self):
        """Test creating context from HTTP headers."""
        headers = {
            "X-User-Id": "user-123",
            "X-Session-Id": "sess-456",
            "X-Tenant-Id": "acme",
            "X-Client-Id": "web",
            "X-Agent-Schema": "research-agent",
            "X-Is-Eval": "true",
        }

        context = AgentContext.from_headers(headers)

        assert context.user_id == "user-123"
        assert context.session_id == "sess-456"
        assert context.tenant_id == "acme"
        assert context.client_id == "web"
        assert context.agent_schema_uri == "research-agent"
        assert context.is_eval is True

    def test_from_headers_case_insensitive(self):
        """Test that header lookup is case-insensitive."""
        headers = {
            "x-user-id": "user-123",
            "X-SESSION-ID": "sess-456",
        }

        context = AgentContext.from_headers(headers)

        assert context.user_id == "user-123"
        assert context.session_id == "sess-456"

    def test_from_headers_defaults(self):
        """Test default values when headers missing."""
        context = AgentContext.from_headers({})

        assert context.user_id is None
        assert context.session_id is None
        assert context.tenant_id == "default"
        assert context.is_eval is False

    def test_child_context(self):
        """Test creating child context for multi-agent."""
        parent = AgentContext(
            user_id="user-123",
            tenant_id="acme",
            session_id="sess-456",
            is_eval=True,
            client_id="web",
        )

        child = parent.child_context(
            agent_schema_uri="child-agent",
            model_override="openai:gpt-4",
        )

        # Should inherit from parent
        assert child.user_id == "user-123"
        assert child.tenant_id == "acme"
        assert child.session_id == "sess-456"
        assert child.is_eval is True
        assert child.client_id == "web"

        # Should have overrides
        assert child.agent_schema_uri == "child-agent"
        assert child.default_model == "openai:gpt-4"

    def test_get_user_id_or_default(self):
        """Test user ID fallback logic."""
        # Returns user_id if provided
        assert AgentContext.get_user_id_or_default("user-123", "test") == "user-123"

        # Returns None for anonymous (no fake IDs)
        assert AgentContext.get_user_id_or_default(None, "test") is None

        # Returns explicit default if provided
        assert AgentContext.get_user_id_or_default(None, "test", "default-user") == "default-user"


# =============================================================================
# MCP Reference Tests
# =============================================================================


class TestMCPReferences:
    """Test MCP tool and resource reference models."""

    def test_tool_reference_basic(self):
        """Test basic tool reference."""
        ref = MCPToolReference(name="search", description="Search the KB")

        assert ref.name == "search"
        assert ref.description == "Search the KB"
        assert ref.server is None

    def test_tool_reference_with_server(self):
        """Test tool reference with server specified."""
        ref = MCPToolReference(name="action", server="rem")

        assert ref.name == "action"
        assert ref.server == "rem"

    def test_resource_reference_uri(self):
        """Test resource reference with exact URI."""
        ref = MCPResourceReference(
            uri="rem://agents",
            name="Agents",
            description="List agents",
        )

        assert ref.uri == "rem://agents"
        assert ref.name == "Agents"
        assert ref.uri_pattern is None

    def test_resource_reference_pattern(self):
        """Test resource reference with URI pattern."""
        ref = MCPResourceReference(uri_pattern="rem://resources/.*")

        assert ref.uri_pattern == "rem://resources/.*"
        assert ref.uri is None
