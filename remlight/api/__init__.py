"""REMLight API module.

Provides:
- FastAPI application (app, create_app)
- MCP server (get_mcp_server, get_mcp_tools)
- Routers (chat_router, query_router, tools_router)
"""

from remlight.api.main import app, create_app
from remlight.api.mcp_main import get_mcp_server, get_mcp_tools, init_mcp

__all__ = [
    "app",
    "create_app",
    "get_mcp_server",
    "get_mcp_tools",
    "init_mcp",
]
