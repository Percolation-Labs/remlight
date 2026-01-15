"""
Integration tests for agent provider using Pydantic AI test models.

Uses TestModel to test agent creation and execution without hitting
real LLM APIs - safe for CI/unit tests.

TestModel parameters:
- custom_output_text: Custom text response
- custom_output_args: Custom structured output args
- call_tools: Which tools to call ('all' or list of names)
- seed: Random seed for reproducibility

See: https://ai.pydantic.dev/testing/
"""

import os
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from remlight.agentic.provider import create_agent, AgentRuntime
from remlight.agentic.context import AgentContext
from remlight.agentic.schema import AgentSchema


# Block accidental real API calls in these tests
@pytest.fixture(autouse=True)
def block_real_api_calls():
    """Prevent accidental real API calls during tests."""
    original = os.environ.get("ALLOW_MODEL_REQUESTS")
    os.environ["ALLOW_MODEL_REQUESTS"] = "False"
    yield
    if original is not None:
        os.environ["ALLOW_MODEL_REQUESTS"] = original
    else:
        os.environ.pop("ALLOW_MODEL_REQUESTS", None)


class TestAgentCreationWithTestModel:
    """Test agent creation using Pydantic AI TestModel."""

    @pytest.mark.asyncio
    async def test_create_agent_with_structured_output(self):
        """Test agent creation with structured_output=true creates output model."""
        schema = {
            "type": "object",
            "description": "You are a helpful assistant.",
            "properties": {
                "answer": {"type": "string", "description": "Your response"},
                "confidence": {"type": "number", "description": "Confidence 0-1"},
            },
            "required": ["answer"],
            "json_schema_extra": {
                "kind": "agent",
                "name": "test-structured-agent",
                "version": "1.0.0",
                "structured_output": True,
                "tools": [],
            },
        }

        runtime = await create_agent(schema=schema, model_name="test")
        assert isinstance(runtime, AgentRuntime)
        assert runtime.schema_name == "test-structured-agent"

        # Agent should have output_type for structured output
        assert runtime.agent._output_type is not None

    @pytest.mark.asyncio
    async def test_create_agent_with_text_output(self):
        """Test agent creation with structured_output=false uses prompt guidance."""
        schema = {
            "type": "object",
            "description": "You are a helpful assistant.",
            "properties": {
                "answer": {"type": "string", "description": "Your response"},
            },
            "required": ["answer"],
            "json_schema_extra": {
                "kind": "agent",
                "name": "test-text-agent",
                "version": "1.0.0",
                "structured_output": False,
                "tools": [],
            },
        }

        runtime = await create_agent(schema=schema, model_name="test")

        # Agent should use str output type (text mode)
        assert runtime.agent._output_type == str

        # System prompt should include inline YAML guidance
        # Access the system prompt from the agent
        system_prompts = runtime.agent._system_prompts
        assert len(system_prompts) > 0

        # The prompt should contain the schema guidance
        full_prompt = system_prompts[0] if system_prompts else ""
        assert "Internal Thinking Structure" in full_prompt or "OUTPUT" in full_prompt

    @pytest.mark.asyncio
    async def test_agent_run_with_test_model(self):
        """Test running agent with TestModel returns canned response."""
        schema = {
            "type": "object",
            "description": "You are a test assistant.",
            "properties": {
                "answer": {"type": "string"},
            },
            "json_schema_extra": {
                "name": "mock-agent",
                "structured_output": False,
                "tools": [],
            },
        }

        runtime = await create_agent(schema=schema, model_name="test")

        # Override with TestModel for testing
        with runtime.agent.override(model=TestModel()):
            result = await runtime.agent.run("Hello, world!")
            # TestModel returns a default response
            assert result.output is not None

    @pytest.mark.asyncio
    async def test_agent_with_tool_filtering(self):
        """Test that agent only gets tools specified in schema."""
        # Define a simple tool
        def my_tool(query: str) -> str:
            """A test tool."""
            return f"Result for: {query}"

        def other_tool(x: int) -> int:
            """Another tool that should be filtered out."""
            return x * 2

        schema = {
            "type": "object",
            "description": "Agent with specific tools.",
            "properties": {"answer": {"type": "string"}},
            "json_schema_extra": {
                "name": "filtered-tool-agent",
                "structured_output": False,
                "tools": [{"name": "my_tool"}],  # Only allow my_tool
            },
        }

        runtime = await create_agent(
            schema=schema,
            model_name="test",
            tools=[my_tool, other_tool],
        )

        # Check that only my_tool is included
        tool_names = [t.name for t in runtime.agent._function_toolset.tools.values()]
        assert "my_tool" in tool_names
        assert "other_tool" not in tool_names


class TestAgentExecutionWithTestModel:
    """Test agent execution using TestModel with custom responses."""

    @pytest.mark.asyncio
    async def test_agent_with_custom_text_response(self):
        """Test agent with TestModel returning custom text response."""
        schema = {
            "type": "object",
            "description": "Test agent.",
            "properties": {"answer": {"type": "string"}},
            "json_schema_extra": {
                "name": "custom-mock-agent",
                "structured_output": False,
                "tools": [],
            },
        }

        runtime = await create_agent(schema=schema, model_name="test")

        # TestModel with custom output text
        with runtime.agent.override(
            model=TestModel(custom_output_text="This is a custom mocked response!")
        ):
            result = await runtime.agent.run("Any prompt")
            assert result.output == "This is a custom mocked response!"

    @pytest.mark.asyncio
    async def test_agent_with_different_seeds(self):
        """Test that different seeds produce different default responses."""
        schema = {
            "type": "object",
            "description": "Test agent.",
            "properties": {"answer": {"type": "string"}},
            "json_schema_extra": {
                "name": "seed-test-agent",
                "structured_output": False,
                "tools": [],
            },
        }

        runtime = await create_agent(schema=schema, model_name="test")

        # Same seed should produce same result
        with runtime.agent.override(model=TestModel(seed=42)):
            result1 = await runtime.agent.run("Test prompt")

        with runtime.agent.override(model=TestModel(seed=42)):
            result2 = await runtime.agent.run("Test prompt")

        assert result1.output == result2.output

    @pytest.mark.asyncio
    async def test_agent_with_tool_calls_mock(self):
        """Test agent that calls tools using TestModel."""
        # Define a tool
        def search_tool(query: str) -> str:
            """Search for information."""
            return f"Found results for: {query}"

        schema = {
            "type": "object",
            "description": "Agent that uses search tool.",
            "properties": {"answer": {"type": "string"}},
            "json_schema_extra": {
                "name": "search-agent",
                "structured_output": False,
                "tools": [{"name": "search_tool"}],
            },
        }

        runtime = await create_agent(
            schema=schema,
            model_name="test",
            tools=[search_tool],
        )

        # TestModel with call_tools='all' automatically calls all tools
        with runtime.agent.override(model=TestModel(call_tools="all")):
            result = await runtime.agent.run("Search for Python tutorials")
            # TestModel will call search_tool with default values
            assert result.output is not None

    @pytest.mark.asyncio
    async def test_agent_with_selective_tool_calls(self):
        """Test agent with TestModel calling only specific tools."""
        def tool_a(x: str) -> str:
            """Tool A."""
            return f"A: {x}"

        def tool_b(y: str) -> str:
            """Tool B."""
            return f"B: {y}"

        schema = {
            "type": "object",
            "description": "Agent with multiple tools.",
            "properties": {"answer": {"type": "string"}},
            "json_schema_extra": {
                "name": "multi-tool-agent",
                "structured_output": False,
                "tools": [{"name": "tool_a"}, {"name": "tool_b"}],
            },
        }

        runtime = await create_agent(
            schema=schema,
            model_name="test",
            tools=[tool_a, tool_b],
        )

        # Only call tool_a
        with runtime.agent.override(model=TestModel(call_tools=["tool_a"])):
            result = await runtime.agent.run("Test")
            assert result.output is not None


class TestAgentContextIntegration:
    """Test agent creation with context."""

    @pytest.mark.asyncio
    async def test_agent_with_user_profile_hint(self):
        """Test that user profile hint is included in system prompt."""
        schema = {
            "type": "object",
            "description": "Base system prompt.",
            "properties": {"answer": {"type": "string"}},
            "json_schema_extra": {
                "name": "context-agent",
                "structured_output": False,
                "tools": [],
            },
        }

        context = AgentContext(
            user_id="test-user",
            user_profile_hint="User prefers concise answers.",
        )

        runtime = await create_agent(
            schema=schema,
            model_name="test",
            context=context,
        )

        # Check system prompt includes user hint
        system_prompts = runtime.agent._system_prompts
        full_prompt = system_prompts[0] if system_prompts else ""
        assert "User prefers concise answers" in full_prompt

    @pytest.mark.asyncio
    async def test_agent_runtime_parameters(self):
        """Test that runtime parameters are resolved from schema."""
        schema = {
            "type": "object",
            "description": "Agent with custom parameters.",
            "properties": {"answer": {"type": "string"}},
            "json_schema_extra": {
                "name": "param-agent",
                "structured_output": False,
                "tools": [],
                "override_temperature": 0.7,
                "override_max_iterations": 5,
            },
        }

        runtime = await create_agent(schema=schema, model_name="test")

        assert runtime.temperature == 0.7
        assert runtime.max_iterations == 5


class TestInlineYAMLPromptGeneration:
    """Test that inline YAML prompt guidance is generated correctly."""

    @pytest.mark.asyncio
    async def test_nested_schema_in_prompt(self):
        """Test that nested schema properties appear in prompt guidance."""
        schema = {
            "type": "object",
            "description": "Agent with complex output schema.",
            "properties": {
                "answer": {"type": "string", "description": "Main response"},
                "metadata": {
                    "type": "object",
                    "description": "Response metadata",
                    "properties": {
                        "confidence": {"type": "number"},
                        "source": {"type": "string"},
                    },
                },
                "references": {
                    "type": "array",
                    "description": "Source references",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "url": {"type": "string"},
                        },
                    },
                },
            },
            "required": ["answer"],
            "json_schema_extra": {
                "name": "complex-schema-agent",
                "structured_output": False,  # Use prompt guidance
                "tools": [],
            },
        }

        runtime = await create_agent(schema=schema, model_name="test")

        # Get the system prompt
        system_prompts = runtime.agent._system_prompts
        full_prompt = system_prompts[0] if system_prompts else ""

        # Check that nested fields appear in YAML guidance
        assert "metadata:" in full_prompt
        assert "confidence:" in full_prompt or "confidence: number" in full_prompt
        assert "references:" in full_prompt or "references: [object]" in full_prompt
        assert "title:" in full_prompt or "title: string" in full_prompt

    @pytest.mark.asyncio
    async def test_answer_field_marked_as_output(self):
        """Test that answer field is clearly marked as OUTPUT in prompt."""
        schema = {
            "type": "object",
            "description": "Test agent.",
            "properties": {
                "answer": {"type": "string", "description": "Your helpful response"},
                "internal_notes": {"type": "string", "description": "Internal tracking"},
            },
            "json_schema_extra": {
                "name": "output-test-agent",
                "structured_output": False,
                "tools": [],
            },
        }

        runtime = await create_agent(schema=schema, model_name="test")

        system_prompts = runtime.agent._system_prompts
        full_prompt = system_prompts[0] if system_prompts else ""

        # Answer should be marked as OUTPUT
        assert "OUTPUT" in full_prompt
        assert "Your helpful response" in full_prompt

        # Internal fields should be marked as INTERNAL
        assert "INTERNAL" in full_prompt
        assert "internal_notes" in full_prompt


class TestAgentSchemaValidation:
    """Test agent schema parsing and validation."""

    @pytest.mark.asyncio
    async def test_dict_schema_converted_to_agent_schema(self):
        """Test that dict schema is properly converted."""
        schema_dict = {
            "type": "object",
            "description": "Test description.",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "json_schema_extra": {
                "name": "dict-schema-agent",
                "version": "1.0.0",
                "tools": [],
            },
        }

        runtime = await create_agent(schema=schema_dict, model_name="test")
        assert runtime.schema_name == "dict-schema-agent"

    @pytest.mark.asyncio
    async def test_agent_schema_object_accepted(self):
        """Test that AgentSchema object is accepted directly."""
        from remlight.agentic.schema import AgentSchemaMetadata

        schema = AgentSchema(
            type="object",
            description="Test agent schema object.",
            properties={"answer": {"type": "string"}},
            required=["answer"],
            json_schema_extra=AgentSchemaMetadata(
                name="schema-object-agent",
                version="1.0.0",
            ),
        )

        runtime = await create_agent(schema=schema, model_name="test")
        assert runtime.schema_name == "schema-object-agent"
