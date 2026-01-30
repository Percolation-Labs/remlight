"""REMLight MCP Server.

Standalone MCP server that registers tools from api/routers/tools.py.
Can run in stdio mode (CLI) or be mounted as HTTP endpoint in FastAPI.

Usage:
    # Stdio mode (for MCP clients like Claude Desktop)
    python -m remlight.api.mcp_main

    # HTTP mode (mounted in FastAPI)
    from remlight.api.mcp_main import create_mcp_server, get_mcp_tools
    mcp = create_mcp_server()
    app.mount("/mcp", mcp.http_app())
"""

from fastmcp import FastMCP

from remlight import __version__
from remlight.services.database import DatabaseService, get_db
from remlight.settings import settings

# Import tool functions from routers - these are the canonical implementations
from remlight.api.routers.tools import (
    search,
    action,
    ask_agent,
    parse_file,
    save_agent,
    analyze_pages,
    init_tools,
    get_user_profile,
    format_user_profile,
    get_project,
    format_project,
)

# Module-level MCP instance
_mcp: FastMCP | None = None


def create_mcp_server(is_local: bool = False) -> FastMCP:
    """
    Create FastMCP server with REMLight tools.

    Tools are imported from api/routers/tools.py, ensuring consistency
    between MCP server and REST API endpoints.

    Args:
        is_local: True for stdio mode, False for HTTP mode
    """
    global _mcp

    mcp = FastMCP(
        name=f"REMLight MCP Server ({settings.environment})",
        version=__version__,
        instructions="""
REMLight Query Workflow:

1. Use `search` tool to query the knowledge base:
   - LOOKUP <key>: O(1) exact key lookup
   - SEARCH <text> IN <table>: Semantic vector search
   - FUZZY <text>: Fuzzy text matching
   - TRAVERSE <key>: Graph traversal

2. Use `action` tool to emit typed events:
   - action(type='observation', payload={confidence, sources, session_name, ...})
   - Observations are streamed to the client as SSE MetadataEvent
   - Other action types: 'elicit' (request user input), 'delegate' (internal)

3. Use `ask_agent` tool to delegate to other agents:
   - Invoke specialized agents by name
   - Child agents inherit context from parent

Tables: ontologies, resources, users, messages, sessions
""",
    )

    # Register tools directly - no wrapper functions needed
    mcp.tool(name="search")(search)
    mcp.tool(name="action")(action)
    mcp.tool(name="ask_agent")(ask_agent)
    mcp.tool(name="parse_file")(parse_file)
    mcp.tool(name="save_agent")(save_agent)
    mcp.tool(name="analyze_pages")(analyze_pages)

    # Register MCP resources
    _register_resources(mcp)

    _mcp = mcp
    return mcp


def _register_resources(mcp: FastMCP) -> None:
    """Register MCP resources."""

    @mcp.resource("user://profile/{user_id}")
    async def user_profile_resource(user_id: str) -> str:
        """Load a user's profile by ID."""
        profile = await get_user_profile(user_id)
        if not profile:
            return f"# User Profile Not Found\n\nNo user found with ID: {user_id}"
        return format_user_profile(profile)

    @mcp.resource("project://{project_key}")
    async def project_resource(project_key: str) -> str:
        """
        Load project details by key.

        Returns JSON with project metadata (status, lead, team_size, budget, etc.)
        TODO: Track project lookups in database for analytics.
        """
        project = await get_project(project_key)
        if not project:
            return f'{{"error": "Project not found", "project_key": "{project_key}"}}'
        return format_project(project)

    @mcp.resource("rem://status")
    def system_status() -> str:
        """Get REM system status."""
        db_info = settings.postgres.connection_string.split('@')[1] if '@' in settings.postgres.connection_string else 'configured'
        return f"""# REMLight System Status

## Environment
- Environment: {settings.environment}

## LLM Configuration
- Default Model: {settings.llm.default_model}
- Temperature: {settings.llm.temperature}
- OpenAI API Key: {"configured" if settings.llm.openai_api_key else "not configured"}
- Anthropic API Key: {"configured" if settings.llm.anthropic_api_key else "not configured"}

## Database
- PostgreSQL: {db_info}

## Available Tables
- ontologies, resources, users, sessions, messages, kv_store

## MCP Tools
- search: Execute REM queries
- action: Emit typed action events (observation, elicit, delegate)
- ask_agent: Multi-agent orchestration
- parse_file: Parse files and extract content (PDF, DOCX, images, text)
- analyze_pages: Vision AI analysis of PDF pages and images
"""


async def get_mcp_tools():
    """Get tools from the MCP server for agent use."""
    global _mcp
    if _mcp is None:
        _mcp = create_mcp_server()
    return await _mcp.get_tools()


def get_mcp_server() -> FastMCP:
    """Get or create the MCP server instance."""
    global _mcp
    if _mcp is None:
        _mcp = create_mcp_server()
    return _mcp


def init_mcp(db: DatabaseService | None = None) -> None:
    """Initialize MCP server with database connection."""
    init_tools(db)


# Create default server instance for imports
mcp = create_mcp_server()


# CLI entry point for stdio mode
if __name__ == "__main__":
    import asyncio

    async def main():
        db = get_db()
        await db.connect()
        init_mcp(db)
        mcp.run()

    asyncio.run(main())
