"""
Unit tests for PydanticAI tool signature extraction.

These tests verify that tool signatures are correctly extracted from Python
functions and transmitted to the LLM via PydanticAI's schema generation.

VERIFIED OUTPUT - What the LLM Actually Sees
============================================

For a tool defined as:

    async def search(
        query: str,
        limit: int = 20,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        '''
        Execute REM queries to search the knowledge base.

        Args:
            query: REM query string
            limit: Maximum results (default 20)
            user_id: Optional user scope

        Returns:
            Query results with entities and metadata
        '''

PydanticAI generates:

    DESCRIPTION:
    <summary>Execute REM queries to search the knowledge base.</summary>
    <returns>
    <description>Query results with entities and metadata</description>
    </returns>

    PARAMETERS JSON SCHEMA:
    {
      "additionalProperties": false,
      "properties": {
        "query": {
          "description": "REM query string",
          "type": "string"
        },
        "limit": {
          "default": 20,
          "description": "Maximum results (default 20)",
          "type": "integer"
        },
        "user_id": {
          "anyOf": [{"type": "string"}, {"type": "null"}],
          "default": null,
          "description": "Optional user scope"
        }
      },
      "required": ["query"],
      "type": "object"
    }

KEY FINDINGS:
- Docstring → description (wrapped in <summary> and <returns> XML tags)
- Type hints → JSON Schema types (str→string, int→integer)
- Union types → anyOf arrays (str | None → anyOf with string and null)
- Default values preserved in schema
- Args: section parsed into per-parameter descriptions
- Required = parameters without defaults

AGENT-SPECIFIC TOOL DESCRIPTIONS:
The MCPToolReference.description field in agent schemas allows appending
a short, agent-specific hint to the base tool description. This helps
agents understand how to use tools in their specific context.

Example schema YAML:
    tools:
      - name: search
        description: "Use LOOKUP for entities, SEARCH for semantic queries"
      - name: action
        description: "Always emit observation with confidence score"
"""

import pytest
from typing import Any
from pydantic_ai import Agent


# =============================================================================
# Test Tools (matching real tools.py signatures)
# =============================================================================

async def search(
    query: str,
    limit: int = 20,
    user_id: str | None = None,
) -> dict[str, Any]:
    """
    Execute REM queries to search the knowledge base.

    Query Syntax:
    - LOOKUP <key>: O(1) exact entity lookup by key
    - SEARCH <text> IN <table>: Semantic vector search

    Args:
        query: REM query string
        limit: Maximum results (default 20)
        user_id: Optional user scope

    Returns:
        Query results with entities and metadata
    """
    return {"result": query}


async def action(
    type: str,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Emit a typed action event for SSE streaming.

    Action Types:
        - "observation": Record metadata (confidence, sources)
        - "elicit": Request user input
        - "delegate": Signal delegation

    Args:
        type: Action type ("observation", "elicit", "delegate")
        payload: Action-specific data

    Returns:
        Action result with _action_event marker
    """
    return {"status": "ok"}


async def complex_tool(
    required_str: str,
    required_int: int,
    optional_str: str | None = None,
    optional_int: int = 10,
    optional_dict: dict[str, Any] | None = None,
    optional_list: list[str] | None = None,
) -> dict[str, Any]:
    """
    A tool with complex parameter types to verify schema generation.

    Args:
        required_str: A required string parameter
        required_int: A required integer parameter
        optional_str: An optional string (nullable)
        optional_int: An optional integer with default
        optional_dict: An optional dictionary (nullable)
        optional_list: An optional list of strings (nullable)

    Returns:
        Processing result
    """
    return {"processed": True}


# =============================================================================
# Tests
# =============================================================================

class TestToolSignatureExtraction:
    """Tests verifying PydanticAI extracts tool signatures correctly."""

    def test_tool_json_schema_generation(self):
        """
        Verify PydanticAI extracts tool signatures and generates JSON Schema for the LLM.

        TOOL REGISTRATION FLOW
        ======================

        1. TOOLS DEFINED in `remlight/api/routers/tools.py`:
           - `search()`, `action()`, `ask_agent()`, `parse_file()` are async functions
           - Each has type hints + Google-style docstrings with Args/Returns sections

        2. FASTMCP WRAPS tools in `remlight/api/mcp_main.py`:
           ```python
           from fastmcp import FastMCP
           mcp = FastMCP(name="REMLight MCP Server")

           # Register tools - FastMCP wraps them in FunctionTool objects
           mcp.tool(name="search")(search)
           mcp.tool(name="action")(action)
           ```

        3. GET_MCP_TOOLS() returns `dict[str, FunctionTool]`:
           ```python
           tools = await mcp.get_tools()
           # Returns: {"search": FunctionTool(fn=<function search>, ...), ...}
           ```

           IMPORTANT: FunctionTool contains:
           - `fn`: The original Python function (callable)
           - `name`, `description`, `tags`, `enabled`: MCP metadata
           - `parameters`: MCP-formatted parameter schema

        4. PROVIDER.PY EXTRACTS THE CALLABLE (`provider.py:882-886`):
           ```python
           # PydanticAI does NOT accept FunctionTool directly!
           # It expects raw callables, so we extract tool.fn

           for tool in tools.values():
               if hasattr(tool, "fn"):
                   agent_tools.append(tool.fn)  # Extract raw function
               elif callable(tool):
                   agent_tools.append(tool)

           agent = Agent(model=model, tools=agent_tools)
           ```

        5. PYDANTICAI RE-PARSES EVERYTHING from the raw function:
           - Uses Python `inspect` module for type hints and defaults
           - Uses `griffe` library to parse docstrings (Google/NumPy/Sphinx)
           - Converts to JSON Schema for the LLM

           KEY INSIGHT: PydanticAI IGNORES FastMCP's metadata (name, description,
           parameters). It re-parses everything from the function itself. FastMCP's
           role is just to store/retrieve the callable, not to provide metadata.

        TYPE COMPATIBILITY TABLE
        ========================
        | Step                  | Returns                  | PydanticAI Accepts? |
        |-----------------------|--------------------------|---------------------|
        | mcp.get_tools()       | dict[str, FunctionTool]  | NO                  |
        | tool.fn               | function (callable)      | YES                 |
        | raw function          | function (callable)      | YES                 |

        SIGNATURE EXTRACTION (what PydanticAI parses)
        =============================================
        | Source                    | Method           | Result                   |
        |---------------------------|------------------|--------------------------|
        | `query: str`              | inspect module   | {"type": "string"}       |
        | `limit: int = 20`         | inspect module   | {"type": "integer", ...} |
        | `str | None`              | inspect module   | {"anyOf": [...]}         |
        | Docstring summary         | griffe parser    | <summary>...</summary>   |
        | Docstring Args: section   | griffe parser    | per-param descriptions   |
        | Docstring Returns:        | griffe parser    | <returns>...</returns>   |

        PYDANTICAI DOCS REFERENCE
        =========================
        - Tools: https://ai.pydantic.dev/tools/
        - Function Schema: https://ai.pydantic.dev/api/tools/#pydantic_ai.tools.FunctionSchema
        - Docstring parsing uses `griffe`: https://mkdocstrings.github.io/griffe/

        EXPECTED SCHEMA OUTPUT
        ======================
        For the `search` tool, the LLM receives:

        DESCRIPTION (XML-wrapped):
            <summary>Execute REM queries to search the knowledge base.</summary>
            <returns><description>Query results with entities and metadata</description></returns>

        JSON SCHEMA:
            {
              "properties": {
                "query": {"type": "string", "description": "REM query string"},
                "limit": {"type": "integer", "default": 20, "description": "..."},
                "user_id": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": null}
              },
              "required": ["query"]
            }
        """
        agent = Agent(model="test", tools=[search])

        tools = agent._function_toolset.tools
        assert "search" in tools, "search tool should be registered"

        tool = tools["search"]
        schema = tool.function_schema.json_schema

        # Verify description is extracted
        assert tool.description is not None
        assert "Execute REM queries" in tool.description
        assert "<summary>" in tool.description  # PydanticAI wraps in XML

        # Verify parameter types
        props = schema["properties"]

        # query: str (required)
        assert props["query"]["type"] == "string"
        assert props["query"]["description"] == "REM query string"
        assert "query" in schema["required"]

        # limit: int = 20 (optional with default)
        assert props["limit"]["type"] == "integer"
        assert props["limit"]["default"] == 20
        assert "limit" not in schema["required"]

        # user_id: str | None = None (optional, nullable)
        assert "anyOf" in props["user_id"]
        types = [t.get("type") for t in props["user_id"]["anyOf"]]
        assert "string" in types
        assert "null" in types
        assert props["user_id"]["default"] is None

    def test_complex_parameter_types(self):
        """
        Verify complex parameter types are correctly converted to JSON Schema.

        Type mappings verified:
        - str → "string"
        - int → "integer"
        - str | None → anyOf[string, null]
        - dict[str, Any] | None → anyOf[object, null]
        - list[str] | None → anyOf[array, null]
        """
        agent = Agent(model="test", tools=[complex_tool])

        tool = agent._function_toolset.tools["complex_tool"]
        schema = tool.function_schema.json_schema
        props = schema["properties"]

        # Required parameters (no defaults)
        assert "required_str" in schema["required"]
        assert "required_int" in schema["required"]
        assert props["required_str"]["type"] == "string"
        assert props["required_int"]["type"] == "integer"

        # Optional with default value
        assert "optional_int" not in schema["required"]
        assert props["optional_int"]["default"] == 10

        # Nullable types (Union with None)
        for field in ["optional_str", "optional_dict", "optional_list"]:
            assert "anyOf" in props[field], f"{field} should have anyOf for nullable"
            types = [t.get("type") for t in props[field]["anyOf"]]
            assert "null" in types, f"{field} should allow null"

    def test_docstring_args_become_descriptions(self):
        """
        Verify that docstring Args: section is parsed into parameter descriptions.
        """
        agent = Agent(model="test", tools=[search, action])

        # Check search tool
        search_schema = agent._function_toolset.tools["search"].function_schema.json_schema
        assert search_schema["properties"]["query"]["description"] == "REM query string"
        assert "Maximum results" in search_schema["properties"]["limit"]["description"]

        # Check action tool
        action_schema = agent._function_toolset.tools["action"].function_schema.json_schema
        assert "Action type" in action_schema["properties"]["type"]["description"]

    def test_multiple_tools_registered(self):
        """
        Verify multiple tools can be registered and each has correct schema.
        """
        agent = Agent(model="test", tools=[search, action, complex_tool])

        tools = agent._function_toolset.tools
        assert len(tools) == 3
        assert "search" in tools
        assert "action" in tools
        assert "complex_tool" in tools

        # Each tool should have its own schema
        for name, tool in tools.items():
            assert tool.function_schema is not None
            assert tool.function_schema.json_schema is not None
            assert "properties" in tool.function_schema.json_schema

    def test_tool_description_contains_docstring_content(self):
        """
        Verify the full docstring content is available in tool description.

        PydanticAI wraps the docstring in XML tags:
        - <summary>...</summary> for main description
        - <returns><description>...</description></returns> for return docs
        """
        agent = Agent(model="test", tools=[search])
        tool = agent._function_toolset.tools["search"]

        # Key docstring content should be present
        assert "LOOKUP" in tool.description
        assert "SEARCH" in tool.description
        assert "Query Syntax" in tool.description

        # Returns section should be present
        assert "<returns>" in tool.description
        assert "Query results" in tool.description


class TestAgentSpecificToolDescriptions:
    """
    Tests for agent-specific tool description enhancement.

    Agent schemas can include a SHORT description hint for each tool
    via MCPToolReference.description. This provides agent-specific context
    for how the tool should be used by THIS agent.

    Example YAML:
        tools:
          - name: search
            description: "Prefer LOOKUP for known entities"  # SHORT hint
          - name: action
            description: "Always emit confidence scores"     # SHORT hint

    WHY SHORT DESCRIPTIONS?
    The base tool docstring is comprehensive (query syntax, examples, etc.).
    The agent-specific description should be a SHORT hint (1-2 sentences)
    that guides THIS agent's specific usage pattern.

    CURRENT STATUS:
    - MCPToolReference.description field EXISTS in schema (schema.py:129)
    - NOT YET IMPLEMENTED in provider.py to append to tool descriptions
    - These tests document expected behavior for future implementation

    TODO: Implement in provider.py around line 870-900 to:
    1. Extract description from MCPToolReference
    2. Append to tool.description before passing to Agent()
    """

    def test_tool_reference_description_field_exists(self):
        """
        Verify MCPToolReference schema supports description field.

        The description should be SHORT - a hint for this specific agent,
        not a replacement for the full tool documentation.
        """
        from remlight.agentic.schema import MCPToolReference

        ref = MCPToolReference(
            name="search",
            description="Use LOOKUP for entity keys, SEARCH for semantic queries"
        )

        assert ref.name == "search"
        assert ref.description == "Use LOOKUP for entity keys, SEARCH for semantic queries"
        # Description should be short - under 100 chars recommended
        assert len(ref.description) < 100, "Agent tool hints should be SHORT"

    def test_schema_with_tool_descriptions(self):
        """
        Verify agent schema can include tool-specific descriptions.

        These SHORT descriptions help the agent understand how to use
        tools in its specific context without replacing the full docs.
        """
        from remlight.agentic.schema import schema_from_yaml

        yaml_content = """
type: object
description: You are a research assistant.

properties:
  answer:
    type: string

json_schema_extra:
  kind: agent
  name: research-agent
  version: "1.0.0"
  tools:
    - name: search
      description: "Prefer LOOKUP for known entity keys"
    - name: action
      description: "Always emit confidence in observations"
"""

        schema = schema_from_yaml(yaml_content)
        tools = schema.json_schema_extra.tools

        assert len(tools) == 2
        assert tools[0].name == "search"
        assert tools[0].description == "Prefer LOOKUP for known entity keys"
        assert tools[1].name == "action"
        assert tools[1].description == "Always emit confidence in observations"

    def test_tool_description_extraction_for_enhancement(self):
        """
        Document how tool descriptions could be enhanced in provider.py.

        PROPOSED IMPLEMENTATION (not yet implemented):

        In provider.py, when filtering tools (~line 870-900):

            # Build map of agent-specific descriptions
            tool_descriptions: dict[str, str] = {}
            for t in (schema_tools or []):
                name = t.name if hasattr(t, 'name') else t.get('name')
                desc = t.description if hasattr(t, 'description') else t.get('description')
                if name and desc:
                    tool_descriptions[name] = desc

            # When adding tools, append description
            for tool in tools:
                tool_name = getattr(tool, '__name__', None)
                if tool_name in tool_descriptions:
                    # Wrap tool with enhanced description
                    enhanced_tool = enhance_tool_description(
                        tool,
                        suffix=f"\\n\\n[Agent hint: {tool_descriptions[tool_name]}]"
                    )
                    agent_tools.append(enhanced_tool)
                else:
                    agent_tools.append(tool)
        """
        from remlight.agentic.schema import schema_from_yaml

        yaml_content = """
type: object
description: Research agent with enhanced tool guidance.

json_schema_extra:
  kind: agent
  name: enhanced-agent
  tools:
    - name: search
      description: "Always use SEARCH for documents, LOOKUP for people"
    - name: action
      description: "Emit observation after every search with confidence 0.8+"
"""

        schema = schema_from_yaml(yaml_content)
        tools = schema.json_schema_extra.tools

        # Extract descriptions that SHOULD be appended to tool descriptions
        tool_hints = {t.name: t.description for t in tools if t.description}

        assert tool_hints == {
            "search": "Always use SEARCH for documents, LOOKUP for people",
            "action": "Emit observation after every search with confidence 0.8+",
        }

        # These hints should be SHORT
        for name, hint in tool_hints.items():
            assert len(hint) < 100, f"Tool hint for {name} should be SHORT (<100 chars)"


class TestToolSchemaInspection:
    """
    Utility tests to inspect and document actual schema output.

    Run with: pytest tests/unit/test_tool_signature.py -v -s
    to see the actual output printed.
    """

    def test_print_full_tool_schema(self, capsys):
        """
        Print the complete tool schema for documentation purposes.

        This test prints what the LLM actually sees for each tool.
        """
        import json

        agent = Agent(model="test", tools=[search, action])

        print("\n" + "=" * 70)
        print("TOOL SCHEMAS SENT TO LLM")
        print("=" * 70)

        for name, tool in agent._function_toolset.tools.items():
            print(f"\n### Tool: {name}")
            print("-" * 40)
            print(f"\nDESCRIPTION:\n{tool.description}")
            print(f"\nJSON SCHEMA:\n{json.dumps(tool.function_schema.json_schema, indent=2)}")

        print("\n" + "=" * 70)

        # This test always passes - it's for documentation
        assert True
