"""
Unit tests for Pydantic AI agent provider.

Tests the conversion of JSON Schema agent definitions to Pydantic AI agents,
including schema parsing, prompt generation, and structured output handling.
"""

import pytest

from remlight.agentic.provider import (
    _build_output_model,
    _convert_properties_to_prompt,
    _render_schema_recursive,
)


class TestRenderSchemaRecursive:
    """Test recursive schema rendering to YAML-like text."""

    def test_render_simple_object(self):
        """Test rendering simple object schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "User name"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }

        lines = _render_schema_recursive(schema)

        assert "name: string (required)" in lines
        assert "  # User name" in lines
        assert "age: integer" in lines

    def test_render_nested_object(self):
        """Test rendering nested object schema."""
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "description": "User info",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                    },
                },
            },
        }

        lines = _render_schema_recursive(schema)

        assert "user:" in lines
        assert "  # User info" in lines
        assert "  name: string" in lines
        assert "  email: string" in lines

    def test_render_array_with_object_items(self):
        """Test rendering array schema with object items."""
        schema = {
            "type": "object",
            "properties": {
                "sources": {
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
        }

        lines = _render_schema_recursive(schema)

        assert "sources: [object]" in lines
        assert "  # Source references" in lines
        assert "  # Each item has:" in lines
        assert "    title: string" in lines
        assert "    url: string" in lines

    def test_render_enum_field(self):
        """Test rendering field with enum values."""
        schema = {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["pending", "active", "completed"],
                },
            },
        }

        lines = _render_schema_recursive(schema)

        assert "status: string (one of: pending, active, completed)" in lines


class TestConvertPropertiesToPrompt:
    """Test conversion of schema properties to prompt guidance."""

    def test_answer_only_schema(self):
        """Test schema with only answer field."""
        properties = {
            "answer": {
                "type": "string",
                "description": "Your response to the user",
            },
        }

        prompt = _convert_properties_to_prompt(properties)

        assert "## Internal Thinking Structure" in prompt
        assert "**OUTPUT (what the user sees):** Your response to the user" in prompt
        assert "**INTERNAL" not in prompt  # No internal fields
        assert "CRITICAL:" in prompt

    def test_answer_with_internal_fields(self):
        """Test schema with answer and internal tracking fields."""
        properties = {
            "answer": {"type": "string", "description": "Response"},
            "confidence": {"type": "number", "description": "Confidence 0-1"},
            "reasoning": {"type": "string", "description": "Internal reasoning"},
        }

        prompt = _convert_properties_to_prompt(properties)

        assert "**OUTPUT (what the user sees):** Response" in prompt
        assert "**INTERNAL (for your tracking only" in prompt
        assert "```yaml" in prompt
        assert "confidence: number" in prompt
        assert "reasoning: string" in prompt
        assert "```" in prompt

    def test_nested_internal_fields(self):
        """Test schema with nested object internal fields."""
        properties = {
            "answer": {"type": "string"},
            "metadata": {
                "type": "object",
                "description": "Response metadata",
                "properties": {
                    "source_count": {"type": "integer"},
                    "processing_time": {"type": "number"},
                },
            },
        }

        prompt = _convert_properties_to_prompt(properties)

        assert "metadata:" in prompt
        assert "  # Response metadata" in prompt
        assert "  source_count: integer" in prompt
        assert "  processing_time: number" in prompt

    def test_array_internal_fields(self):
        """Test schema with array internal fields."""
        properties = {
            "answer": {"type": "string"},
            "sources": {
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
        }

        prompt = _convert_properties_to_prompt(properties)

        assert "sources: [object]" in prompt
        assert "  # Source references" in prompt
        assert "  # Each item has:" in prompt
        assert "    title: string" in prompt
        assert "    url: string" in prompt

    def test_empty_properties(self):
        """Test empty properties returns empty string."""
        prompt = _convert_properties_to_prompt({})
        assert prompt == ""

    def test_critical_warning_present(self):
        """Test that critical warning about not outputting field names is present."""
        properties = {"answer": {"type": "string"}}
        prompt = _convert_properties_to_prompt(properties)

        assert "CRITICAL:" in prompt
        assert "ONLY the conversational answer text" in prompt
        assert "Do NOT output field names" in prompt


class TestBuildOutputModel:
    """Test dynamic Pydantic model creation."""

    def test_simple_string_field(self):
        """Test model with string field."""
        properties = {"name": {"type": "string"}}
        model = _build_output_model(properties, required=["name"])

        instance = model(name="test")
        assert instance.name == "test"

    def test_number_field(self):
        """Test model with number field."""
        properties = {"score": {"type": "number"}}
        model = _build_output_model(properties, required=[])

        instance = model(score=0.95)
        assert instance.score == 0.95

    def test_integer_field(self):
        """Test model with integer field."""
        properties = {"count": {"type": "integer"}}
        model = _build_output_model(properties, required=[])

        instance = model(count=42)
        assert instance.count == 42

    def test_boolean_field(self):
        """Test model with boolean field."""
        properties = {"active": {"type": "boolean"}}
        model = _build_output_model(properties, required=[])

        instance = model(active=True)
        assert instance.active is True

    def test_array_field(self):
        """Test model with array field."""
        properties = {"tags": {"type": "array"}}
        model = _build_output_model(properties, required=[])

        instance = model(tags=["a", "b"])
        assert instance.tags == ["a", "b"]

    def test_object_field(self):
        """Test model with object field."""
        properties = {"metadata": {"type": "object"}}
        model = _build_output_model(properties, required=[])

        instance = model(metadata={"key": "value"})
        assert instance.metadata == {"key": "value"}

    def test_required_vs_optional(self):
        """Test required vs optional field handling."""
        properties = {
            "required_field": {"type": "string"},
            "optional_field": {"type": "string"},
        }
        model = _build_output_model(properties, required=["required_field"])

        # Optional field can be None
        instance = model(required_field="test")
        assert instance.required_field == "test"
        assert instance.optional_field is None

    def test_multiple_fields(self):
        """Test model with multiple field types."""
        properties = {
            "answer": {"type": "string"},
            "confidence": {"type": "number"},
            "sources": {"type": "array"},
        }
        model = _build_output_model(properties, required=["answer"])

        instance = model(answer="test", confidence=0.9, sources=["a", "b"])
        assert instance.answer == "test"
        assert instance.confidence == 0.9
        assert instance.sources == ["a", "b"]
