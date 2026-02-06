"""
Integration tests for remote MCP tool resolution using FastMCPToolset.

These tests verify that FastMCPToolset correctly creates filtered toolsets
from local and remote MCP servers.

TEST SETUP
----------
We use the local REMLight MCP server (in-process via FastMCP) to test the flow:

1. Create FastMCPToolset from MCP server
2. Apply filtering to restrict available tools
3. Verify the toolset works with PydanticAI agents

WHY THIS MATTERS
----------------
FastMCPToolset from pydantic-ai handles:
- Schema extraction from MCP servers
- Type annotation generation
- Tool wrapper creation
- Caching

We just need to verify our filtering and integration work correctly.
"""

import pytest
from pydantic_ai import Agent

from remlight.agentic.tool_resolver import (
    create_filtered_mcp_toolset,
    resolve_tools_as_toolsets,
    clear_server_cache,
)


class TestFastMCPToolsetCreation:
    """Test creating FastMCPToolset from MCP servers."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear caches before each test."""
        clear_server_cache()

    def test_create_toolset_from_fastmcp_server(self):
        """
        Create a FastMCPToolset from a FastMCP server instance.

        This tests the in-process connection (no network overhead).
        """
        from fastmcp import FastMCP

        # Create a test MCP server
        mcp = FastMCP("test-server")

        @mcp.tool()
        def test_tool(query: str, limit: int = 10) -> str:
            """A test tool for searching.

            Args:
                query: The search query
                limit: Maximum results to return
            """
            return f"Results for {query}"

        # Create toolset
        toolset = create_filtered_mcp_toolset(mcp)

        # Verify toolset was created
        assert toolset is not None
        # The toolset should be a FastMCPToolset or FilteredToolset
        assert "Toolset" in type(toolset).__name__

    def test_create_filtered_toolset(self):
        """
        Create a filtered toolset that only includes specific tools.
        """
        from fastmcp import FastMCP

        mcp = FastMCP("test-server")

        @mcp.tool()
        def allowed_tool(x: str) -> str:
            """An allowed tool."""
            return x

        @mcp.tool()
        def blocked_tool(x: str) -> str:
            """A blocked tool."""
            return x

        # Create filtered toolset - only allow 'allowed_tool'
        toolset = create_filtered_mcp_toolset(
            mcp,
            allowed_tools={"allowed_tool"}
        )

        # The toolset should be a FilteredToolset
        assert "FilteredToolset" in type(toolset).__name__

    def test_create_toolset_from_local_mcp_server(self):
        """
        Create toolset from the actual REMLight MCP server.
        """
        from remlight.api.mcp_main import create_mcp_server

        mcp = create_mcp_server()
        toolset = create_filtered_mcp_toolset(mcp)

        assert toolset is not None

    def test_filtered_toolset_from_local_mcp_server(self):
        """
        Create filtered toolset from REMLight MCP server with only 'search'.
        """
        from remlight.api.mcp_main import create_mcp_server

        mcp = create_mcp_server()
        toolset = create_filtered_mcp_toolset(
            mcp,
            allowed_tools={"search"}
        )

        assert toolset is not None
        assert "FilteredToolset" in type(toolset).__name__


class TestResolveToolsAsToolsets:
    """Test the resolve_tools_as_toolsets function."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear caches before each test."""
        clear_server_cache()

    @pytest.mark.asyncio
    async def test_resolve_local_tools(self):
        """
        Resolve local tools using the MCP server.
        """
        from remlight.api.mcp_main import create_mcp_server

        mcp = create_mcp_server()

        tool_refs = [
            {"name": "search", "server": "local"},
            {"name": "action", "server": "local"},
        ]

        toolsets, legacy_tools = await resolve_tools_as_toolsets(
            tool_refs=tool_refs,
            local_mcp_server=mcp,
        )

        # Should have one toolset for local server
        assert len(toolsets) == 1
        # Should have no legacy tools (all are MCP)
        assert len(legacy_tools) == 0

    @pytest.mark.asyncio
    async def test_resolve_no_tools_when_no_refs(self):
        """
        Resolve with empty tool refs returns empty lists.
        """
        toolsets, legacy_tools = await resolve_tools_as_toolsets(
            tool_refs=[],
            local_mcp_server=None,
        )

        assert len(toolsets) == 0
        assert len(legacy_tools) == 0

    @pytest.mark.asyncio
    async def test_resolve_warns_when_no_local_server(self):
        """
        When local tools are requested but no server provided, warn.
        """
        tool_refs = [{"name": "search", "server": "local"}]

        toolsets, legacy_tools = await resolve_tools_as_toolsets(
            tool_refs=tool_refs,
            local_mcp_server=None,  # No server!
        )

        # Should have no toolsets (couldn't create without server)
        assert len(toolsets) == 0


class TestToolsetIntegrationWithAgent:
    """Test that toolsets work with PydanticAI agents."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Clear caches before each test."""
        clear_server_cache()

    @pytest.mark.asyncio
    async def test_agent_with_fastmcp_toolset(self):
        """
        Create a PydanticAI agent with a FastMCPToolset.

        This verifies the full integration path.
        """
        from remlight.api.mcp_main import create_mcp_server

        mcp = create_mcp_server()

        # Create filtered toolset
        toolset = create_filtered_mcp_toolset(
            mcp,
            allowed_tools={"search"}
        )

        # Create agent with toolset
        agent = Agent(
            model="test",
            toolsets=[toolset],
        )

        # Agent should have been created successfully
        assert agent is not None

    @pytest.mark.asyncio
    async def test_agent_with_resolved_toolsets(self):
        """
        Create agent using resolve_tools_as_toolsets output.
        """
        from remlight.api.mcp_main import create_mcp_server

        mcp = create_mcp_server()

        tool_refs = [
            {"name": "search", "server": "local"},
        ]

        toolsets, legacy_tools = await resolve_tools_as_toolsets(
            tool_refs=tool_refs,
            local_mcp_server=mcp,
        )

        # Create agent with resolved toolsets
        agent = Agent(
            model="test",
            toolsets=toolsets,
            tools=legacy_tools if legacy_tools else [],
        )

        assert agent is not None


class TestMCPSchemaExtraction:
    """Test that MCP server schemas are accessible via FastMCP Client."""

    @pytest.mark.asyncio
    async def test_fetch_schemas_from_local_mcp_server(self):
        """
        Verify we can still fetch schemas via FastMCP Client.

        This is useful for debugging and introspection.
        """
        from fastmcp import Client
        from remlight.api.mcp_main import create_mcp_server

        mcp = create_mcp_server()

        async with Client(mcp) as client:
            tools = await client.list_tools()

            # Verify we got tools with schemas
            assert len(tools) >= 2, "Should have at least search and action tools"

            # Find the search tool
            search_tool = next((t for t in tools if t.name == "search"), None)
            assert search_tool is not None, "search tool should exist"

            # Verify schema structure
            assert search_tool.description is not None
            schema = search_tool.inputSchema
            assert "properties" in schema
            assert "query" in schema["properties"]

    @pytest.mark.asyncio
    async def test_schema_includes_parameter_info(self):
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
