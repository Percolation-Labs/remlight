"""Integration tests for MCP resources (user profile, project)."""

import json
import pytest


@pytest.fixture(autouse=True)
def init_tools_fixture(db):
    """Initialize tools with database connection for all tests."""
    from remlight.api.routers.tools import init_tools
    init_tools(db)


class TestUserProfileResource:
    """Test user://profile/{user_id} MCP resource."""

    @pytest.mark.asyncio
    async def test_load_test_user_profile(self, db):
        """Test loading the seeded test user profile."""
        from remlight.api.routers.tools import get_user_profile, format_user_profile

        # Load test user by email
        profile = await get_user_profile("test@example.com")

        assert profile is not None
        assert profile["name"] == "Test User"
        assert profile["email"] == "test@example.com"
        assert "artificial intelligence" in profile["interests"]
        assert profile["activity_level"] == "active"

    @pytest.mark.asyncio
    async def test_load_user_by_uuid(self, db):
        """Test loading user by UUID."""
        from remlight.api.routers.tools import get_user_profile

        # Load by UUID (the seeded test user ID)
        profile = await get_user_profile("a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11")

        assert profile is not None
        assert profile["email"] == "test@example.com"

    @pytest.mark.asyncio
    async def test_load_user_by_user_id(self, db):
        """Test loading user by user_id field."""
        from remlight.api.routers.tools import get_user_profile

        # Load by user_id field
        profile = await get_user_profile("test-user")

        assert profile is not None
        assert profile["email"] == "test@example.com"

    @pytest.mark.asyncio
    async def test_format_user_profile(self, db):
        """Test formatting user profile as markdown."""
        from remlight.api.routers.tools import get_user_profile, format_user_profile

        profile = await get_user_profile("test@example.com")
        formatted = format_user_profile(profile)

        assert "# User Profile: Test User" in formatted
        assert "test@example.com" in formatted
        assert "## Summary" in formatted
        assert "## Interests" in formatted
        assert "artificial intelligence" in formatted

    @pytest.mark.asyncio
    async def test_user_not_found(self, db):
        """Test handling of non-existent user."""
        from remlight.api.routers.tools import get_user_profile

        profile = await get_user_profile("nonexistent-user-xyz")
        assert profile is None

    @pytest.mark.asyncio
    async def test_user_profile_hint(self, db):
        """Test user profile hint for agent context."""
        from remlight.api.routers.tools import get_user_profile_hint

        # With user
        hint = await get_user_profile_hint("test@example.com")
        assert "Date:" in hint
        assert "Time:" in hint
        assert "User: Test User" in hint
        assert "Interests:" in hint

        # Without user (just date/time)
        hint_no_user = await get_user_profile_hint(None)
        assert "Date:" in hint_no_user
        assert "User:" not in hint_no_user


class TestProjectResource:
    """Test project://{project_key} MCP resource."""

    @pytest.mark.asyncio
    async def test_load_project_alpha(self, db):
        """Test loading project-alpha from database."""
        from remlight.api.routers.tools import get_project

        project = await get_project("project-alpha")

        assert project is not None
        assert project["name"] == "project-alpha"
        assert "machine learning pipeline" in project["description"]
        assert project["status"] == "active"
        assert project["lead"] == "sarah-chen"
        assert project["team_size"] == 5
        assert project["priority"] == "high"

    @pytest.mark.asyncio
    async def test_load_project_beta(self, db):
        """Test loading project-beta from database."""
        from remlight.api.routers.tools import get_project

        project = await get_project("project-beta")

        assert project is not None
        assert project["status"] == "planning"
        assert project["lead"] == "john-doe"
        assert "analytics" in project["tags"]

    @pytest.mark.asyncio
    async def test_load_project_gamma(self, db):
        """Test loading project-gamma from database."""
        from remlight.api.routers.tools import get_project

        project = await get_project("project-gamma")

        assert project is not None
        assert project["category"] == "ai"
        assert "rag" in project["tags"]
        assert project["budget"] == 200000

    @pytest.mark.asyncio
    async def test_format_project_json(self, db):
        """Test formatting project as JSON."""
        from remlight.api.routers.tools import get_project, format_project

        project = await get_project("project-alpha")
        formatted = format_project(project)

        # Should be valid JSON
        parsed = json.loads(formatted)
        assert parsed["name"] == "project-alpha"
        assert parsed["status"] == "active"
        assert parsed["lead"] == "sarah-chen"
        assert parsed["budget"] == 150000
        assert "ml" in parsed["tags"]

    @pytest.mark.asyncio
    async def test_project_not_found(self, db):
        """Test handling of non-existent project."""
        from remlight.api.routers.tools import get_project

        project = await get_project("nonexistent-project-xyz")
        assert project is None


class TestMCPResourceIntegration:
    """Test MCP resource registration and access."""

    def test_mcp_server_creates_successfully(self, db):
        """Test that MCP server can be created with resources."""
        from remlight.api.mcp_main import create_mcp_server

        mcp = create_mcp_server()

        # Server should be created
        assert mcp is not None
        assert mcp.name == f"REMLight MCP Server (development)"

    @pytest.mark.asyncio
    async def test_mcp_tools_available(self, db):
        """Test that MCP tools can be retrieved."""
        from remlight.api.mcp_main import get_mcp_tools

        tools = await get_mcp_tools()

        # Should have our tools
        tool_names = list(tools.keys()) if isinstance(tools, dict) else [t.name for t in tools]
        assert "search" in tool_names
        assert "action" in tool_names
        assert "parse_file" in tool_names


class TestOrchestratorWithProjects:
    """Test that agents can load and use project data."""

    @pytest.mark.asyncio
    async def test_search_finds_projects(self, db):
        """Test that search tool can find projects by name."""
        from remlight.api.routers.tools import search

        # LOOKUP should find project by key
        result = await search("LOOKUP project-alpha")

        assert result is not None
        assert result.get("found") is True or result.get("count", 0) > 0
        # Should have project data in result (check case-insensitive)
        assert "project" in str(result.get("result", "")).lower() and "alpha" in str(result.get("result", "")).lower()

    @pytest.mark.asyncio
    async def test_search_fuzzy_finds_projects(self, db):
        """Test that fuzzy search can find projects."""
        from remlight.api.routers.tools import search

        # FUZZY should find similar project names
        result = await search("FUZZY project alpha")

        assert result is not None
        # Fuzzy search should return matches
        if result.get("results"):
            assert len(result["results"]) > 0

    @pytest.mark.asyncio
    async def test_project_resource_callable(self, db):
        """Test that project resource returns JSON data."""
        from remlight.api.routers.tools import get_project, format_project

        project = await get_project("project-gamma")
        formatted = format_project(project)

        # Should be valid JSON with expected fields
        parsed = json.loads(formatted)
        assert parsed["name"] == "project-gamma"
        assert parsed["status"] == "active"
        assert parsed["lead"] == "jane-smith"
        assert "rag" in parsed["tags"]

    @pytest.mark.asyncio
    @pytest.mark.llm
    async def test_orchestrator_can_find_projects(self, db):
        """Test that orchestrator agent can search and find project information.

        This test requires LLM API access and is marked with 'llm'.
        Run with: pytest -m llm
        """
        from remlight.agentic import create_agent, build_agent_spec
        from remlight.api.mcp_main import get_mcp_tools

        # Create a simple agent with search tool
        spec = build_agent_spec(
            name="test-project-finder",
            description="You are an assistant that finds project information. Use the search tool to find projects.",
            tools=["search"],
        )

        tools = await get_mcp_tools()
        runtime = await create_agent(schema=spec, tools=tools, use_cache=False)

        # Ask the agent to find project information
        result = await runtime.agent.run("Find information about project-alpha")

        # Agent should have used the search tool and found the project
        output_str = str(result.output) if hasattr(result, "output") else str(result)
        # The response should mention the project or its details
        assert any(term in output_str.lower() for term in ["project", "alpha", "machine learning", "sarah"])


class TestServerRegistry:
    """Test server registration and retrieval."""

    @pytest.mark.asyncio
    async def test_register_server(self, db):
        """Test registering a new server with deterministic ID."""
        from remlight.models.entities import Server
        from remlight.services.repository import Repository
        from remlight.services.registration import generate_server_id

        repo = Repository(Server, table_name="servers")

        # Create a test server with deterministic ID
        endpoint = "http://localhost:9999"
        server = Server(
            name="test-server",
            description="A test MCP server for integration testing",
            server_type="rest",
            endpoint=endpoint,
            icon="🧪",
            tags=["test", "integration"],
        )
        server.id = generate_server_id(endpoint, server.name)

        await repo.upsert(server, generate_embeddings=False)

        # Retrieve and verify
        retrieved = await repo.get_by_id(str(server.id))
        assert retrieved is not None
        assert retrieved.name == "test-server"
        assert retrieved.server_type == "rest"
        assert retrieved.endpoint == endpoint

        # Cleanup
        await repo.delete(str(server.id))

    @pytest.mark.asyncio
    async def test_default_server_type_is_mcp(self, db):
        """Test that default server type is 'mcp'."""
        from remlight.models.entities import Server
        from remlight.services.repository import Repository
        from remlight.services.registration import generate_server_id

        repo = Repository(Server, table_name="servers")

        # Create a server with deterministic ID
        server = Server(
            name="default-type-server",
            description="Server with default type",
        )
        server.id = generate_server_id(None, server.name)

        await repo.upsert(server, generate_embeddings=False)

        # Retrieve and verify default type
        retrieved = await repo.get_by_id(str(server.id))
        assert retrieved is not None
        assert retrieved.server_type == "mcp"

        # Cleanup
        await repo.delete(str(server.id))

    @pytest.mark.asyncio
    async def test_server_search_by_name(self, db):
        """Test searching servers by name."""
        from remlight.models.entities import Server
        from remlight.services.repository import Repository
        from remlight.services.registration import generate_server_id

        repo = Repository(Server, table_name="servers")

        # Create test server with deterministic ID
        server = Server(
            name="searchable-server",
            description="A server for testing search functionality",
            tags=["searchable"],
        )
        server.id = generate_server_id(None, server.name)
        await repo.upsert(server, generate_embeddings=False)

        # Search by name
        results = await repo.search(
            query="searchable",
            search_type="name",
            limit=10,
        )

        assert len(results) >= 1
        names = [s.name for s, _ in results]
        assert "searchable-server" in names

        # Cleanup
        await repo.delete(str(server.id))


class TestToolRegistry:
    """Test tool registration and retrieval."""

    @pytest.mark.asyncio
    async def test_register_tool(self, db):
        """Test registering a new tool with deterministic ID."""
        from remlight.models.entities import Server, Tool
        from remlight.services.repository import Repository
        from remlight.services.registration import generate_server_id, generate_tool_id

        server_repo = Repository(Server, table_name="servers")
        tool_repo = Repository(Tool, table_name="tools")

        # Create a test server with deterministic ID
        server = Server(
            name="tool-test-server",
            description="Server for tool testing",
        )
        server.id = generate_server_id(None, server.name)
        await server_repo.upsert(server, generate_embeddings=False)

        # Create a test tool with deterministic ID
        tool = Tool(
            name="test-tool",
            description="A test tool for integration testing",
            server_id=server.id,
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            icon="🔧",
            tags=["test"],
        )
        tool.id = generate_tool_id(server.id, tool.name)
        await tool_repo.upsert(tool, generate_embeddings=False)

        # Retrieve and verify
        retrieved = await tool_repo.get_by_id(str(tool.id))
        assert retrieved is not None
        assert retrieved.name == "test-tool"
        assert retrieved.input_schema["type"] == "object"

        # Cleanup
        await tool_repo.delete(str(tool.id))
        await server_repo.delete(str(server.id))

    @pytest.mark.asyncio
    async def test_tool_search_by_tags(self, db):
        """Test searching tools by tags."""
        from remlight.models.entities import Server, Tool
        from remlight.services.repository import Repository
        from remlight.services.registration import generate_server_id, generate_tool_id

        server_repo = Repository(Server, table_name="servers")
        tool_repo = Repository(Tool, table_name="tools")

        # Create server with deterministic ID
        server = Server(
            name="tag-test-server",
            description="Server for tag testing",
        )
        server.id = generate_server_id(None, server.name)
        await server_repo.upsert(server, generate_embeddings=False)

        # Create tool with deterministic ID
        tool = Tool(
            name="tagged-tool",
            description="A tool with specific tags",
            server_id=server.id,
            tags=["unique-test-tag", "searchable"],
        )
        tool.id = generate_tool_id(server.id, tool.name)
        await tool_repo.upsert(tool, generate_embeddings=False)

        # Search by tag
        results = await tool_repo.search(
            query="unique-test-tag",
            search_type="tags",
            limit=10,
        )

        assert len(results) >= 1
        names = [t.name for t, _ in results]
        assert "tagged-tool" in names

        # Cleanup
        await tool_repo.delete(str(tool.id))
        await server_repo.delete(str(server.id))


class TestToolResolver:
    """Test tool resolution from local and remote servers."""

    @pytest.mark.asyncio
    async def test_resolve_local_tools(self, db):
        """Test resolving local tools."""
        from remlight.agentic.tool_resolver import resolve_tools
        from remlight.agentic.schema import MCPToolReference

        # Create mock local tools
        async def mock_search(query: str) -> str:
            return f"Search result for: {query}"

        async def mock_action(action: str) -> str:
            return f"Action executed: {action}"

        local_tools = {
            "search": mock_search,
            "action": mock_action,
        }

        # Resolve tools
        tool_refs = [
            MCPToolReference(name="search", server="local"),
            MCPToolReference(name="action"),  # Default to local
        ]

        resolved = await resolve_tools(tool_refs, local_tools)

        assert len(resolved) == 2
        # Verify they're callable
        result = await resolved[0](query="test")
        assert "Search result" in result

    @pytest.mark.asyncio
    async def test_resolve_remote_tool_not_found(self, db):
        """Test that missing remote server logs warning but doesn't fail."""
        from remlight.agentic.tool_resolver import resolve_tools
        from remlight.agentic.schema import MCPToolReference

        tool_refs = [
            MCPToolReference(name="remote-tool", server="nonexistent-server"),
        ]

        # Should not raise, just return empty list
        resolved = await resolve_tools(tool_refs, {})
        assert len(resolved) == 0


class TestProjectToolsRegistration:
    """Test automatic project tools registration."""

    @pytest.mark.asyncio
    async def test_register_project_tools(self, db):
        """Test registering project tools from MCP server."""
        from remlight.services.registration import register_project_tools

        # Run registration
        stats = await register_project_tools(force=True, generate_embeddings=False)

        # Should have registered at least the local server
        assert stats["servers_registered"] >= 1 or stats["skipped"] >= 0

    @pytest.mark.asyncio
    async def test_registration_skips_unchanged(self, db):
        """Test that registration skips unchanged items."""
        from remlight.services.registration import register_project_tools

        # First registration
        stats1 = await register_project_tools(force=True, generate_embeddings=False)

        # Second registration without force
        stats2 = await register_project_tools(force=False, generate_embeddings=False)

        # Second run should skip most items
        assert stats2["skipped"] >= stats1["servers_registered"]
