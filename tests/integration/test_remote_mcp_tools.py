"""
Integration tests for remote MCP tool resolution.

These tests verify that tools fetched from a remote MCP server
have proper type annotations that PydanticAI can parse.

TEST SETUP
----------
We use the local REMLight MCP server (in-process via FastMCP Client)
to simulate a "remote" server. This tests the full flow:

1. Connect to MCP server via FastMCP Client
2. Fetch tool schemas (list_tools)
3. Build annotated wrapper functions
4. Verify PydanticAI extracts correct JSON Schema

WHY THIS MATTERS
----------------
Without proper annotations, PydanticAI generates empty schemas:
    {"additionalProperties": true, "properties": {}, "type": "object"}

The LLM has NO IDEA what parameters to pass. With proper annotations:
    {
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "default": 20}
        },
        "required": ["query"]
    }

The LLM knows exactly what parameters are available and their types.
"""

import pytest
import json
from pydantic_ai import Agent

from remlight.agentic.tool_resolver import (
    fetch_mcp_tool_schemas,
    create_annotated_mcp_wrapper,
    clear_server_cache,
)


class TestRemoteMCPToolSchemas:
    """Test fetching tool schemas from MCP server."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear caches before each test."""
        clear_server_cache()

    @pytest.mark.asyncio
    async def test_fetch_schemas_from_local_mcp_server(self):
        """
        Fetch tool schemas from the local MCP server (in-process).

        This simulates connecting to a "remote" MCP server and fetching
        tool schemas via FastMCP Client.
        """
        from fastmcp import Client
        from remlight.api.mcp_main import create_mcp_server

        # Create the local MCP server
        mcp = create_mcp_server()

        # Connect via FastMCP Client (simulates remote connection)
        async with Client(mcp) as client:
            tools = await client.list_tools()

            # Verify we got tools with schemas
            assert len(tools) >= 2, "Should have at least search and action tools"

            # Find the search tool
            search_tool = next((t for t in tools if t.name == "search"), None)
            assert search_tool is not None, "search tool should exist"

            # Verify schema structure
            assert search_tool.description is not None
            assert "query" in search_tool.description.lower() or "search" in search_tool.description.lower()

            schema = search_tool.inputSchema
            assert "properties" in schema
            assert "query" in schema["properties"]
            assert schema["properties"]["query"]["type"] == "string"

            # Verify required params
            assert "required" in schema
            assert "query" in schema["required"]

    @pytest.mark.asyncio
    async def test_schema_includes_all_parameter_info(self):
        """Verify schema includes types, defaults, and descriptions."""
        from fastmcp import Client
        from remlight.api.mcp_main import create_mcp_server

        mcp = create_mcp_server()

        async with Client(mcp) as client:
            tools = await client.list_tools()
            search_tool = next((t for t in tools if t.name == "search"), None)
            schema = search_tool.inputSchema

            # Check limit parameter (optional with default)
            assert "limit" in schema["properties"]
            limit_prop = schema["properties"]["limit"]
            assert limit_prop["type"] == "integer"
            assert limit_prop["default"] == 20

            # Check user_id parameter (nullable)
            assert "user_id" in schema["properties"]
            user_id_prop = schema["properties"]["user_id"]
            assert "anyOf" in user_id_prop  # Nullable type


class TestAnnotatedWrapperCreation:
    """Test creating properly-annotated wrappers from MCP schemas."""

    def test_create_wrapper_with_proper_annotations(self):
        """
        Create a wrapper function and verify it has proper annotations.

        The wrapper should have:
        - Correct parameter names and types
        - Docstring with description and Args
        - Type annotations PydanticAI can parse
        """
        input_schema = {
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "default": 20, "description": "Max results"},
                "user_id": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                    "description": "Optional user ID",
                },
            },
            "required": ["query"],
        }

        wrapper = create_annotated_mcp_wrapper(
            tool_name="search",
            description="Search the knowledge base.",
            input_schema=input_schema,
            endpoint="http://localhost:8001/mcp",
        )

        # Verify function name
        assert wrapper.__name__ == "search"

        # Verify docstring contains description
        assert "Search the knowledge base" in wrapper.__doc__
        assert "query:" in wrapper.__doc__  # Args section

    def test_pydanticai_extracts_correct_schema_from_wrapper(self):
        """
        THE CRITICAL TEST: Verify PydanticAI extracts proper JSON Schema
        from dynamically-created remote MCP tool wrappers.

        WHY THIS MATTERS
        ----------------
        When a tool is on a REMOTE MCP server, we don't have a local Python
        function with type annotations. We only have the JSON Schema from
        the server's list_tools() response.

        Without proper handling, PydanticAI sees **kwargs and generates:
            {"additionalProperties": true, "properties": {}, "type": "object"}

        The LLM has NO IDEA what parameters to pass!

        THE SOLUTION
        ------------
        create_annotated_mcp_wrapper() dynamically creates a Python function
        with the correct signature from the JSON Schema:

            async def search(query: str, limit: int = 20) -> dict:
                '''Search description. Args: query: Search query'''
                ...

        PydanticAI then extracts the correct schema:
            {
                "properties": {"query": {"type": "string"}, ...},
                "required": ["query"]
            }

        CODE REFERENCE: remlight/agentic/tool_resolver.py:138-252
        WALKTHROUGH: code-walkthrough.md Section 2.3

        This test proves the wrapper works by:
        1. Creating a wrapper from JSON Schema
        2. Passing it to a PydanticAI Agent
        3. Inspecting agent._function_toolset.tools[name].function_schema.json_schema
        4. Verifying it has proper properties, types, and required fields
        """
        input_schema = {
            "properties": {
                "query": {"type": "string", "description": "Search query string"},
                "limit": {"type": "integer", "default": 20, "description": "Maximum results"},
            },
            "required": ["query"],
        }

        wrapper = create_annotated_mcp_wrapper(
            tool_name="remote_search",
            description="Search a remote knowledge base.",
            input_schema=input_schema,
            endpoint="http://localhost:8001/mcp",
        )

        # Create PydanticAI agent with the wrapper
        agent = Agent(model="test", tools=[wrapper])

        # Extract what PydanticAI sees
        tool = agent._function_toolset.tools["remote_search"]
        pydantic_schema = tool.function_schema.json_schema

        print("\n" + "=" * 60)
        print("PYDANTICAI SCHEMA FOR REMOTE MCP TOOL")
        print("=" * 60)
        print(json.dumps(pydantic_schema, indent=2))

        # CRITICAL ASSERTIONS - this is what the LLM sees
        assert "properties" in pydantic_schema
        assert "query" in pydantic_schema["properties"]
        assert pydantic_schema["properties"]["query"]["type"] == "string"

        # Verify required params
        assert "required" in pydantic_schema
        assert "query" in pydantic_schema["required"]

        # Verify optional param with default
        assert "limit" in pydantic_schema["properties"]
        assert pydantic_schema["properties"]["limit"]["default"] == 20

        # Verify it's NOT an empty schema
        assert pydantic_schema != {"additionalProperties": True, "properties": {}, "type": "object"}

    def test_nullable_params_handled_correctly(self):
        """Verify nullable parameters (anyOf with null) are handled."""
        input_schema = {
            "properties": {
                "required_param": {"type": "string"},
                "nullable_param": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "default": None,
                },
            },
            "required": ["required_param"],
        }

        wrapper = create_annotated_mcp_wrapper(
            tool_name="nullable_test",
            description="Test nullable params.",
            input_schema=input_schema,
            endpoint="http://localhost:8001/mcp",
        )

        agent = Agent(model="test", tools=[wrapper])
        schema = agent._function_toolset.tools["nullable_test"].function_schema.json_schema

        # Nullable param should have anyOf or allow null
        nullable_prop = schema["properties"]["nullable_param"]
        assert "anyOf" in nullable_prop or nullable_prop.get("default") is None


class TestEndToEndRemoteToolResolution:
    """End-to-end test using the actual MCP server."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear caches before each test."""
        clear_server_cache()

    @pytest.mark.asyncio
    async def test_full_flow_mcp_to_pydanticai(self):
        """
        Full integration test:
        1. Connect to local MCP server
        2. Fetch schema for 'search' tool
        3. Create annotated wrapper
        4. Verify PydanticAI schema matches original

        This proves remote MCP tools work correctly with PydanticAI.
        """
        from fastmcp import Client
        from remlight.api.mcp_main import create_mcp_server

        # Step 1: Connect to MCP server
        mcp = create_mcp_server()

        async with Client(mcp) as client:
            # Step 2: Fetch tool schema
            tools = await client.list_tools()
            search_tool = next(t for t in tools if t.name == "search")

            # Step 3: Create annotated wrapper
            wrapper = create_annotated_mcp_wrapper(
                tool_name="search",
                description=search_tool.description,
                input_schema=search_tool.inputSchema,
                endpoint="http://localhost:8001/mcp",  # Would be real endpoint
            )

            # Step 4: Verify PydanticAI schema
            agent = Agent(model="test", tools=[wrapper])
            pydantic_schema = agent._function_toolset.tools["search"].function_schema.json_schema

            print("\n" + "=" * 60)
            print("ORIGINAL MCP SCHEMA:")
            print(json.dumps(search_tool.inputSchema, indent=2))
            print("\nPYDANTICAI SCHEMA:")
            print(json.dumps(pydantic_schema, indent=2))
            print("=" * 60)

            # Verify key properties match
            original_props = search_tool.inputSchema["properties"]
            pydantic_props = pydantic_schema["properties"]

            # query should be required string in both
            assert "query" in pydantic_props
            assert pydantic_props["query"]["type"] == "string"
            assert "query" in pydantic_schema["required"]

            # limit should have default 20 in both
            assert "limit" in pydantic_props
            assert pydantic_props["limit"]["default"] == original_props["limit"]["default"]

    @pytest.mark.asyncio
    async def test_compare_local_vs_remote_tool_schemas(self):
        """
        Compare schemas: local tool.fn vs remote wrapper.

        Both should produce equivalent schemas for PydanticAI.
        """
        from fastmcp import Client
        from remlight.api.mcp_main import create_mcp_server, get_mcp_tools

        mcp = create_mcp_server()

        # Get local tool (via tool.fn)
        local_tools = await get_mcp_tools()
        local_search = local_tools["search"]

        # Create agent with LOCAL tool
        agent_local = Agent(model="test", tools=[local_search.fn])
        local_schema = agent_local._function_toolset.tools["search"].function_schema.json_schema

        # Get remote tool schema and create wrapper
        async with Client(mcp) as client:
            tools = await client.list_tools()
            search_tool = next(t for t in tools if t.name == "search")

            wrapper = create_annotated_mcp_wrapper(
                tool_name="search_remote",  # Different name to avoid conflict
                description=search_tool.description,
                input_schema=search_tool.inputSchema,
                endpoint="http://localhost:8001/mcp",
            )

        # Create agent with REMOTE wrapper
        agent_remote = Agent(model="test", tools=[wrapper])
        remote_schema = agent_remote._function_toolset.tools["search_remote"].function_schema.json_schema

        print("\n" + "=" * 60)
        print("LOCAL TOOL SCHEMA:")
        print(json.dumps(local_schema, indent=2))
        print("\nREMOTE WRAPPER SCHEMA:")
        print(json.dumps(remote_schema, indent=2))
        print("=" * 60)

        # Both should have the same properties
        local_props = set(local_schema["properties"].keys())
        remote_props = set(remote_schema["properties"].keys())
        assert local_props == remote_props, f"Properties differ: {local_props} vs {remote_props}"

        # Both should have query as required
        assert "query" in local_schema["required"]
        assert "query" in remote_schema["required"]

        # Both should have same types for query
        assert local_schema["properties"]["query"]["type"] == remote_schema["properties"]["query"]["type"]
